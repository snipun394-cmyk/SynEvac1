from typing import Any, Dict, Optional

from models.sensor_asset import HealthStatus

from building_state.models import BuildingState

from ai_features.feature_schema import CANONICAL_LIVE_FEATURE_NAMES


# =====================================================
# BuildingState -> canonical AI feature vector. A pure function: reads
# only fields BuildingState already carries, invents nothing, and never
# calls into camera_manager/sensor_manager/facp/building_control
# internals itself (BuildingState is already the fully-assembled
# snapshot -- see building_state/models.py's own docstring). This is
# the one extractor both a genuine live deployment and
# ai_features.simulation_extractor's simulation-side path call, so the
# two can never independently drift out of schema (Phase 2's central
# requirement).
#
# Every value emitted here traces to a field the observability audit
# (docs/architecture/ai_live_feature_parity.md) classified LIVE_
# OBSERVABLE or LIVE_ESTIMABLE -- nothing from BuildingState.hazard_
# summary or occupant_tracks[*].classification is read, per that
# audit's own SIMULATION_ONLY findings for those fields in every
# current code path.
# =====================================================


def extract_canonical_features(state: BuildingState) -> Dict[str, Any]:

    row: Dict[str, Any] = {}

    row.update(_occupancy_features(state))
    row.update(_camera_features(state))
    row.update(_sensor_features(state))
    row.update(_facp_features(state))
    row.update(_control_features(state))

    # Deterministic key order, matching CANONICAL_LIVE_SCHEMA exactly --
    # every _*_features() helper above must contribute every key it is
    # responsible for, or this assertion catches the omission immediately
    # rather than silently shipping a partial/misordered row.
    assert tuple(row.keys()) == CANONICAL_LIVE_FEATURE_NAMES, (
        f"extract_canonical_features() produced {tuple(row.keys())}, expected exactly "
        f"CANONICAL_LIVE_FEATURE_NAMES {CANONICAL_LIVE_FEATURE_NAMES} in that order."
    )

    return row


# =====================================================


def _occupancy_features(state: BuildingState) -> Dict[str, Any]:

    occupancy_observed = len(state.camera_observations) > 0

    tracks = tuple(state.occupant_tracks.values())

    total_occupant_count: Optional[int] = len(tracks) if occupancy_observed else None

    mean_confidence: Optional[float] = (
        sum(track.confidence for track in tracks) / len(tracks) if tracks else None
    )

    return {
        "total_occupant_count": total_occupant_count,
        "occupancy_observed": occupancy_observed,
        "mean_occupant_track_confidence": mean_confidence,
    }


# =====================================================


def _camera_features(state: BuildingState) -> Dict[str, Any]:

    return {
        "camera_total_count": len(state.camera_observations),
        "camera_active_count": len(state.active_assets.active_camera_ids),
        "camera_offline_count": len(state.active_assets.offline_camera_ids),
    }


# =====================================================


def _sensor_features(state: BuildingState) -> Dict[str, Any]:

    return {
        "sensor_total_count": len(state.smoke_detector_states) + len(state.heat_detector_states),
        "sensor_active_count": len(state.active_assets.active_sensor_ids),
        "sensor_offline_count": len(state.active_assets.offline_sensor_ids),
        "smoke_detector_coverage_count": len(state.smoke_detector_states),
        "smoke_detector_alarm_count": _alarm_count(state.smoke_detector_states),
        "smoke_detector_fault_count": _fault_count(state.smoke_detector_states),
        "heat_detector_coverage_count": len(state.heat_detector_states),
        "heat_detector_alarm_count": _alarm_count(state.heat_detector_states),
        "heat_detector_fault_count": _fault_count(state.heat_detector_states),
        "building_alarm_status": state.building_alarm_status.name,
    }


def _alarm_count(detector_states) -> int:

    return sum(
        1 for asset in detector_states.values()
        if asset.reading is not None and getattr(asset.reading, "alarm_active", False)
    )


def _fault_count(detector_states) -> int:

    return sum(
        1 for asset in detector_states.values()
        if asset.status.health_status != HealthStatus.OK
    )


# =====================================================


def _facp_features(state: BuildingState) -> Dict[str, Any]:

    facp = state.facp_status

    if facp is None:

        return {
            "facp_available": False,
            "facp_panel_state": None,
            "facp_active_alarm_source_count": None,
            "facp_active_fault_source_count": None,
            "facp_acknowledged": None,
            "facp_silenced": None,
        }

    return {
        "facp_available": True,
        "facp_panel_state": facp.panel_state.name,
        "facp_active_alarm_source_count": len(facp.active_alarm_source_ids),
        "facp_active_fault_source_count": len(facp.active_fault_source_ids),
        "facp_acknowledged": facp.acknowledged,
        "facp_silenced": facp.silenced,
    }


# =====================================================


def _control_features(state: BuildingState) -> Dict[str, Any]:

    control = state.control_status

    if control is None:

        return {
            "control_status_available": False,
            "control_pending_request_count": None,
            "control_confirmed_entry_count": None,
        }

    return {
        "control_status_available": True,
        "control_pending_request_count": control.pending_count,
        "control_confirmed_entry_count": len(control.entries),
    }
