from typing import Callable, Iterable, Optional, Protocol, Tuple

from facp.models import DetectorConditionReport, PanelEvent

from models.sensor_asset import DetectorState


# =====================================================
# Digital Twin Asset -> Zone Assignment & Live FACP Runtime milestone,
# Phase 9 -- the seam LiveOrchestrator uses to actually TICK a
# production SimulatedFACP each cycle, mirroring every sibling Engine*
# Gateway's own "thin Protocol + real adapter" shape exactly (see
# live_system.evacuation_guidance_gateway/evacuation_signage_gateway).
#
# Before this milestone, live_runtime.factory.build_live_runtime()
# wired facp.current_snapshot() as a strictly READ-ONLY provider into
# EstimatorBuildingStateGateway -- nothing anywhere in production ever
# called facp.evaluate(), so a caller-supplied SimulatedFACP's alarm
# state machine never advanced at all outside the Designer's own
# BuildingStateDebugRunner (designer/building_state_debug_runner.py),
# which this gateway's own _build_detector_condition_reports() mirrors
# exactly (same SensorManager.all_statuses() + reading-provider shape,
# same DetectorConditionReport.from_status_and_reading() call) --
# re-derived here rather than imported, since live_system/ must never
# import designer/ (the reverse dependency direction).
#
# This gateway calls ONLY facp.evaluate() -- never acknowledge()/
# silence()/reset(). Those remain explicit operator actions reached
# through Designer's own debug controls (or a future Command Center
# wiring) exactly as before; production ticking must never silently
# acknowledge, silence, or reset the panel on a caller's behalf (Phase
# 10's own explicit "must not accidentally auto-reset" requirement).
# =====================================================


class FACPGateway(Protocol):

    def evaluate(self, time: float) -> Optional[Tuple[PanelEvent, ...]]: ...


# =====================================================


class EngineFACPGateway:

    # The real adapter -- never allowed to raise out of evaluate() and
    # crash the live cycle, exactly the same discipline every sibling
    # Engine*Gateway in live_system already established. sensor_manager
    # and the two reading providers are forwarded through unchanged,
    # duck-typed -- this gateway never imports sensor_manager/ itself.

    def __init__(
        self,
        facp,
        sensor_manager,
        smoke_detector_reading_provider: Optional[Callable[[float], Iterable]] = None,
        heat_detector_reading_provider: Optional[Callable[[float], Iterable]] = None,
    ):

        self._facp = facp
        self._sensor_manager = sensor_manager
        self._smoke_detector_reading_provider = smoke_detector_reading_provider
        self._heat_detector_reading_provider = heat_detector_reading_provider

    # =====================================================

    def evaluate(self, time: float) -> Optional[Tuple[PanelEvent, ...]]:

        try:

            detector_conditions = self._build_detector_condition_reports(time)

            return self._facp.evaluate(detector_conditions, time)

        except Exception:  # noqa: BLE001 -- an unexpected FACP evaluation failure must never crash the live cycle

            return None

    # =====================================================
    # Internals -- mirrors designer.building_state_debug_runner.
    # BuildingStateDebugRunner._sensor_statuses_by_type()/
    # _build_detector_condition_reports() exactly (never imported from
    # there -- live_system/ must not depend on designer/).
    # =====================================================

    def _build_detector_condition_reports(self, time: float):

        smoke_statuses = []
        heat_statuses = []
        mcp_statuses = []

        for status in self._sensor_manager.all_statuses():

            if status.sensor_type == "SmokeDetector":
                smoke_statuses.append(status)
            elif status.sensor_type == "HeatDetector":
                heat_statuses.append(status)
            elif status.sensor_type == "ManualCallPoint":
                mcp_statuses.append(status)

        smoke_readings = (
            self._smoke_detector_reading_provider(time)
            if self._smoke_detector_reading_provider is not None else ()
        )
        heat_readings = (
            self._heat_detector_reading_provider(time)
            if self._heat_detector_reading_provider is not None else ()
        )

        smoke_reading_by_id = {reading.detector_id: reading for reading in smoke_readings}
        heat_reading_by_id = {reading.detector_id: reading for reading in heat_readings}

        reports = {
            status.sensor_id: DetectorConditionReport.from_status_and_reading(
                status, smoke_reading_by_id.get(status.sensor_id),
            )
            for status in smoke_statuses
        }
        reports.update({
            status.sensor_id: DetectorConditionReport.from_status_and_reading(
                status, heat_reading_by_id.get(status.sensor_id),
            )
            for status in heat_statuses
        })

        # Manual Call Points & Emergency Lighting milestone -- an MCP
        # has no external Ground-Truth-driven reading (see models.
        # manual_call_point.ManualCallPoint's own docstring: activation
        # is intrinsic device state, a direct human action, not a
        # threshold-compared hazard reading), so there is no reading-
        # provider seam to thread through here the way smoke/heat have.
        # Its own compute_state() already IS the complete, authoritative
        # answer -- read directly off the real registered object via
        # SensorManager.get_sensor(), never re-derived or duplicated.
        reports.update({
            status.sensor_id: DetectorConditionReport(
                asset_id=status.sensor_id, asset_type=status.sensor_type,
                state=self._mcp_state(status, time),
                floor_id=status.floor_id, zone_ids=status.zone_ids,
            )
            for status in mcp_statuses
        })

        return reports

    # =====================================================

    def _mcp_state(self, status, time: float):

        mcp = self._sensor_manager.get_sensor(status.sensor_id)

        if mcp is None:
            # Honest fallback for an id SensorManager reported a status
            # for but no longer holds the live object for (should not
            # happen in practice -- both come from the same registry --
            # but never fabricated as ALARM/FAULT either way).
            return DetectorState.NORMAL

        return mcp.compute_state(time)
