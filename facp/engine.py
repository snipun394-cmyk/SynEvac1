from typing import Dict, FrozenSet, Mapping, Optional, Set, Tuple

from models.sensor_asset import DetectorState

from facp.event_log import PanelEventLog
from facp.models import (
    DetectorConditionReport,
    FACPSnapshot,
    InvalidPanelOperation,
    PanelEvent,
    PanelEventType,
    PanelState,
)


class SimulatedFACP:

    # The Simulation-mode Fire Alarm Control Panel (Phase 3) -- the
    # single place per-detector conditions get aggregated into one
    # panel-wide state and an append-only event history. Consumes only
    # already-computed DetectorConditionReport values (each one already
    # the output of SmokeDetector.compute_state()/HeatDetector.
    # compute_state() upstream, wrapped via DetectorConditionReport.
    # from_status_and_reading()) -- this class contains no smoke_level/
    # temperature/HazardSnapshot handling of any kind, and imports
    # nothing from hazard, hazard_evolution, perception, or simulator.
    # It is an aggregation/coordination layer over SensorManager-managed
    # assets, not a replacement for SensorManager (Phase 6's own explicit
    # instruction) -- it never discovers, enables/disables, or looks up
    # a Camera/Detector/SmokeDetector/HeatDetector object itself.
    #
    # ===================================================================
    # State machine (Phase 5)
    # ===================================================================
    #
    #   NORMAL
    #     -> ALARM               any detector/manual source enters ALARM
    #     -> FAULT                any detector enters FAULT (no ALARM)
    #
    #   ALARM
    #     -> ALARM_ACKNOWLEDGED  acknowledge() while no new source has
    #                            appeared since the last acknowledgement
    #     -> ALARM_SILENCED      silence() (directly, or after ack)
    #     -> ALARM               a genuinely NEW alarm source appears
    #                            while acknowledged/silenced -- a real
    #                            panel re-alerts on a new zone/point even
    #                            if a prior one was already silenced;
    #                            this is the one automatic state change
    #                            this class makes without an explicit
    #                            operator call
    #     -> ALARM               reset() called while any alarm source
    #                            is still active (clears ack/silence
    #                            flags, does NOT return to NORMAL --
    #                            Phase 5's explicit "RESET must not force
    #                            NORMAL if active alarm conditions still
    #                            exist" requirement)
    #     -> (stays ALARM-family) every alarm source restores to NORMAL
    #                            -- restoration alone never auto-clears
    #                            the panel; a real panel stays latched
    #                            until an operator explicitly presses
    #                            Reset, exactly like this class's own
    #                            reset() is required here too
    #     -> NORMAL              reset() called once no alarm/fault
    #                            source remains active
    #
    #   FAULT                    symmetric to ALARM but with no
    #                            acknowledge/silence variant in this
    #                            version (Phase 2's own event list is
    #                            worded entirely around ALARM); reset()
    #                            behaves the same way -- stays FAULT if
    #                            a fault condition remains, else NORMAL
    #
    # ACKNOWLEDGE and SILENCE never touch active_alarm_source_ids/
    # active_fault_source_ids on FACPSnapshot -- those two fields are
    # always the CURRENT raw set of alarming/faulted sources, regardless
    # of panel_state, so "the fire condition disappeared" can never be
    # read off of an acknowledged/silenced display state (Phase 5's own
    # explicit warning).

    def __init__(self, panel_id: str = "FACP-1"):

        self.panel_id = panel_id

        self._last_condition_by_asset: Dict[str, DetectorState] = {}
        self._asset_metadata: Dict[str, DetectorConditionReport] = {}
        self._manual_alarm_ids: Set[str] = set()

        self._panel_state: PanelState = PanelState.NORMAL
        self._acknowledged_source_ids: FrozenSet[str] = frozenset()
        self._silenced: bool = False

        self._active_alarm_source_ids: Tuple[str, ...] = ()
        self._active_fault_source_ids: Tuple[str, ...] = ()

        self._event_log = PanelEventLog()

    # =====================================================

    @property
    def panel_state(self) -> PanelState:

        return self._panel_state

    # =====================================================

    @property
    def active_alarm_source_ids(self) -> Tuple[str, ...]:

        return self._active_alarm_source_ids

    # =====================================================

    @property
    def active_fault_source_ids(self) -> Tuple[str, ...]:

        return self._active_fault_source_ids

    # =====================================================

    @property
    def event_log(self) -> PanelEventLog:

        return self._event_log

    # =====================================================
    # Phase 3 -- consuming already-computed detector conditions
    # =====================================================

    def evaluate(
        self, detector_conditions: Mapping[str, DetectorConditionReport], time: float,
    ) -> Tuple[PanelEvent, ...]:

        # Deterministic event ordering (Phase 9/12): iterates
        # detector_conditions in sorted asset_id order regardless of
        # the mapping's own iteration order, so the SAME input always
        # produces events in the SAME sequence.

        transitioned_reports = []

        for asset_id in sorted(detector_conditions.keys()):

            report = detector_conditions[asset_id]
            previous_state = self._last_condition_by_asset.get(asset_id, DetectorState.NORMAL)

            if report.state != previous_state:
                transitioned_reports.append(report)

            self._last_condition_by_asset[asset_id] = report.state
            self._asset_metadata[asset_id] = report

        self._recompute_panel_state()

        fired_events = tuple(
            self._build_transition_event(report, time) for report in transitioned_reports
        )

        for event in fired_events:
            self._event_log.append(event)

        return fired_events

    # =====================================================

    def manual_alarm(
        self,
        source_asset_id: str,
        time: float,
        floor_id: Optional[str] = None,
        zone_ids: Tuple[str, ...] = (),
        operator_id: Optional[str] = None,
    ) -> PanelEvent:

        # Manual Alarm / Manual Call Point activation -- model/event
        # support only (Phase 2's own scope): no Manual Call Point
        # engineering asset exists anywhere in this codebase
        # (confirmed by investigation), so this method never looks one
        # up -- a caller supplies whatever identity/location it has for
        # the point that was activated. Tracked the same way a detector
        # source is (participates in the same ALARM/re-alert logic
        # below), but recorded separately in _manual_alarm_ids so
        # reset() can clear it -- a manual pull station requires a
        # physical reset at the box itself in reality; since no such
        # box exists in this codebase, clearing it at the panel's own
        # reset() is the closest honest equivalent.

        self._last_condition_by_asset[source_asset_id] = DetectorState.ALARM
        self._manual_alarm_ids.add(source_asset_id)

        self._recompute_panel_state()

        event = PanelEvent(
            timestamp=time,
            event_type=PanelEventType.MANUAL_ALARM,
            source_asset_id=source_asset_id,
            source_asset_type="ManualCallPoint",
            floor_id=floor_id,
            zone_ids=tuple(zone_ids),
            panel_state_after=self._panel_state,
        )
        self._event_log.append(event)

        return event

    # =====================================================
    # Phase 5 -- operator actions
    # =====================================================

    def acknowledge(self, time: float, operator_id: Optional[str] = None) -> PanelEvent:

        if self._panel_state != PanelState.ALARM:

            raise InvalidPanelOperation(
                f"acknowledge() is only valid while the panel is in ALARM "
                f"(currently {self._panel_state.name})."
            )

        self._acknowledged_source_ids = frozenset(self._active_alarm_source_ids)
        self._panel_state = PanelState.ALARM_ACKNOWLEDGED

        event = PanelEvent(
            timestamp=time, event_type=PanelEventType.PANEL_ACKNOWLEDGED,
            panel_state_after=self._panel_state,
        )
        self._event_log.append(event)

        return event

    # =====================================================

    def silence(self, time: float, operator_id: Optional[str] = None) -> PanelEvent:

        if self._panel_state not in (PanelState.ALARM, PanelState.ALARM_ACKNOWLEDGED):

            raise InvalidPanelOperation(
                f"silence() is only valid while the panel is in ALARM or "
                f"ALARM_ACKNOWLEDGED (currently {self._panel_state.name})."
            )

        self._silenced = True
        self._panel_state = PanelState.ALARM_SILENCED

        event = PanelEvent(
            timestamp=time, event_type=PanelEventType.ALARM_SILENCED,
            panel_state_after=self._panel_state,
        )
        self._event_log.append(event)

        return event

    # =====================================================

    def reset(self, time: float, operator_id: Optional[str] = None) -> PanelEvent:

        # Always a legal call (unlike acknowledge()/silence()) -- a
        # real panel's Reset button is never disabled. Its RESULT
        # depends on whether any alarm/fault condition remains: clears
        # every operator-set flag and any manual-alarm latch, then
        # re-derives panel_state from whatever detector conditions
        # _recompute_panel_state() finds are still actually active --
        # never unconditionally forced to NORMAL (Phase 5's own
        # explicit requirement).

        for asset_id in self._manual_alarm_ids:
            self._last_condition_by_asset.pop(asset_id, None)

        self._manual_alarm_ids.clear()

        self._acknowledged_source_ids = frozenset()
        self._silenced = False

        self._recompute_panel_state(allow_clear_to_normal=True)

        event = PanelEvent(
            timestamp=time, event_type=PanelEventType.SYSTEM_RESET,
            panel_state_after=self._panel_state,
        )
        self._event_log.append(event)

        return event

    # =====================================================
    # Phase 6 -- read-only status for BuildingState integration
    # =====================================================

    def current_snapshot(self, time: float, recent_event_count: int = 20) -> FACPSnapshot:

        return FACPSnapshot(
            panel_id=self.panel_id,
            timestamp=time,
            panel_state=self._panel_state,
            active_alarm_source_ids=self._active_alarm_source_ids,
            active_fault_source_ids=self._active_fault_source_ids,
            acknowledged=bool(self._acknowledged_source_ids),
            silenced=self._silenced,
            recent_events=self._event_log.recent(recent_event_count),
        )

    # =====================================================
    # Internals
    # =====================================================

    def _build_transition_event(self, report: DetectorConditionReport, time: float) -> PanelEvent:

        if report.state == DetectorState.ALARM:
            event_type = PanelEventType.DETECTOR_ALARM
        elif report.state == DetectorState.FAULT:
            event_type = PanelEventType.DETECTOR_FAULT
        else:
            event_type = PanelEventType.DETECTOR_RESTORE

        return PanelEvent(
            timestamp=time,
            event_type=event_type,
            source_asset_id=report.asset_id,
            source_asset_type=report.asset_type,
            floor_id=report.floor_id,
            zone_ids=report.zone_ids,
            panel_state_after=self._panel_state,
        )

    # =====================================================

    def _recompute_panel_state(self, allow_clear_to_normal: bool = False) -> None:

        # allow_clear_to_normal distinguishes evaluate()'s own call
        # (False -- restoration alone must never auto-clear the panel,
        # it stays latched in whatever alarm/fault-family state it was
        # in) from reset()'s call (True -- Reset is the one operation
        # allowed to return the panel to NORMAL, and only when no
        # condition actually remains).

        alarm_ids = tuple(sorted(
            asset_id for asset_id, state in self._last_condition_by_asset.items()
            if state == DetectorState.ALARM
        ))
        fault_ids = tuple(sorted(
            asset_id for asset_id, state in self._last_condition_by_asset.items()
            if state == DetectorState.FAULT
        ))

        self._active_alarm_source_ids = alarm_ids
        self._active_fault_source_ids = fault_ids

        if alarm_ids:

            new_alarm_ids = set(alarm_ids) - self._acknowledged_source_ids

            if new_alarm_ids:
                self._acknowledged_source_ids = frozenset()
                self._silenced = False
                self._panel_state = PanelState.ALARM
            elif self._silenced:
                self._panel_state = PanelState.ALARM_SILENCED
            elif self._acknowledged_source_ids:
                self._panel_state = PanelState.ALARM_ACKNOWLEDGED
            else:
                self._panel_state = PanelState.ALARM

            return

        if fault_ids:
            self._panel_state = PanelState.FAULT
            return

        # No active alarm or fault condition remains. Restoration alone
        # (evaluate()'s own call, allow_clear_to_normal=False) does not
        # clear the panel back to NORMAL -- see this class's own module
        # docstring/state machine notes above; only reset() can do
        # that, and only when this branch is reached with
        # allow_clear_to_normal=True.
        if allow_clear_to_normal:
            self._panel_state = PanelState.NORMAL
