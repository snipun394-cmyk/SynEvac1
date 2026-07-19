from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from hazard.severity import HazardSeverity

from building_state.models import BuildingState


class ConsistencyWarningType(Enum):

    CAMERA_REPORTS_OCCUPANTS_BUT_ZONE_EMPTY = auto()
    SMOKE_ALARM_WITHOUT_NEARBY_HAZARD = auto()
    HEAT_ALARM_WITHOUT_NEARBY_HAZARD = auto()
    HAZARD_PRESENT_BUT_DETECTOR_INACTIVE = auto()
    CAMERA_OFFLINE = auto()
    SENSOR_OFFLINE = auto()


@dataclass(frozen=True)
class ConsistencyWarning:

    # A diagnostic-only observation about one BuildingState -- never
    # fed back into the state itself (see check_consistency() below),
    # and never a decision/recommendation of any kind. asset_id is a
    # camera_id or sensor_id depending on warning_type; zone_id is the
    # Navigation Graph node the warning concerns, when there is one.

    warning_type: ConsistencyWarningType
    message: str
    zone_id: Optional[str] = None
    asset_id: Optional[str] = None


# Severities counted as "hazard present" by the checks below that need
# one. LOW is deliberately excluded: per hazard/severity.py's own
# cutoffs, LOW sits below where a threshold-based detector (Phase 3/4's
# alarm_threshold) would be expected to trip anyway, so a detector
# staying silent at LOW is expected behavior, not an inconsistency.
_HAZARD_PRESENT_SEVERITIES = (HazardSeverity.MODERATE, HazardSeverity.HIGH, HazardSeverity.CRITICAL)


def check_consistency(state: BuildingState) -> Tuple[ConsistencyWarning, ...]:

    # Pure function -- reads `state` only, produces diagnostic warnings,
    # never modifies the state and never constructs a new one (Phase 4's
    # explicit "generate diagnostic warnings without modifying the
    # state" requirement). Every check below cross-references two
    # fields BuildingState already carries; this function introduces no
    # new hazard/occupancy/detection estimation logic of its own.

    warnings = []

    warnings.extend(_check_camera_occupancy_mismatch(state))

    warnings.extend(_check_detector_alarm_without_hazard(
        state, state.smoke_detector_states, "Smoke",
        ConsistencyWarningType.SMOKE_ALARM_WITHOUT_NEARBY_HAZARD,
    ))
    warnings.extend(_check_detector_alarm_without_hazard(
        state, state.heat_detector_states, "Heat",
        ConsistencyWarningType.HEAT_ALARM_WITHOUT_NEARBY_HAZARD,
    ))

    warnings.extend(_check_hazard_without_active_detector(state))
    warnings.extend(_check_offline_assets(state))

    return tuple(warnings)


# =====================================================

def _check_camera_occupancy_mismatch(state: BuildingState):

    warnings = []

    for camera_id, asset in state.camera_observations.items():

        observation = asset.frame_observation

        if observation is None or observation.estimated_occupant_count is None:
            continue

        if observation.estimated_occupant_count <= 0:
            continue

        for zone_id in asset.status.zone_ids:

            occupancy = state.zone_occupancy.observation_at(zone_id)

            if occupancy.occupant_count is not None and occupancy.occupant_count <= 0:

                warnings.append(ConsistencyWarning(
                    warning_type=ConsistencyWarningType.CAMERA_REPORTS_OCCUPANTS_BUT_ZONE_EMPTY,
                    message=(
                        f"Camera {camera_id!r} reports "
                        f"{observation.estimated_occupant_count:g} occupant(s) for zone "
                        f"{zone_id!r}, but zone occupancy reads 0."
                    ),
                    zone_id=zone_id,
                    asset_id=camera_id,
                ))

    return warnings


# =====================================================

def _check_detector_alarm_without_hazard(state, detector_states, label, warning_type):

    warnings = []

    for sensor_id, asset in detector_states.items():

        reading = asset.reading

        if reading is None or not reading.alarm_active:
            continue

        covered_zone_ids = [zone_id for zone_id in asset.status.zone_ids if zone_id is not None]

        has_nearby_hazard = any(
            state.zone_severity(zone_id) in _HAZARD_PRESENT_SEVERITIES
            for zone_id in covered_zone_ids
        )

        if not has_nearby_hazard:

            warnings.append(ConsistencyWarning(
                warning_type=warning_type,
                message=(
                    f"{label} detector {sensor_id!r} is in ALARM, but no zone it "
                    f"covers currently reports a hazard."
                ),
                zone_id=covered_zone_ids[0] if covered_zone_ids else None,
                asset_id=sensor_id,
            ))

    return warnings


# =====================================================

def _check_hazard_without_active_detector(state: BuildingState):

    warnings = []

    all_detector_assets = tuple(state.smoke_detector_states.values()) + tuple(state.heat_detector_states.values())

    for zone_id, severity in state.hazard_summary.zone_severities.items():

        if severity not in _HAZARD_PRESENT_SEVERITIES:
            continue

        covering_detectors = [
            asset for asset in all_detector_assets if zone_id in asset.status.zone_ids
        ]

        if not covering_detectors:
            # No detector covers this zone at all -- a coverage gap,
            # not an asset-state inconsistency this check is about.
            continue

        if all(not asset.status.active for asset in covering_detectors):

            warnings.append(ConsistencyWarning(
                warning_type=ConsistencyWarningType.HAZARD_PRESENT_BUT_DETECTOR_INACTIVE,
                message=(
                    f"Zone {zone_id!r} reports hazard severity {severity.name}, but "
                    f"every detector covering it is currently inactive."
                ),
                zone_id=zone_id,
            ))

    return warnings


# =====================================================

def _check_offline_assets(state: BuildingState):

    warnings = []

    for camera_id in state.active_assets.offline_camera_ids:

        warnings.append(ConsistencyWarning(
            warning_type=ConsistencyWarningType.CAMERA_OFFLINE,
            message=f"Camera {camera_id!r} is offline.",
            asset_id=camera_id,
        ))

    for sensor_id in state.active_assets.offline_sensor_ids:

        warnings.append(ConsistencyWarning(
            warning_type=ConsistencyWarningType.SENSOR_OFFLINE,
            message=f"Sensor {sensor_id!r} is offline.",
            asset_id=sensor_id,
        ))

    return warnings
