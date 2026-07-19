from typing import Iterable, Optional, Union

from models.detector import Detector
from models.heat_detector import HeatDetector
from models.smoke_detector import SmokeDetector
from models.zone import Zone


# The legacy generic Detector's own detector_type values this module
# knows how to adapt -- "Flame"/"Gas" (see Detector.DETECTOR_TYPES) have
# no canonical SensorAsset-based counterpart yet, so a legacy Detector of
# either type is left completely alone (adapt_legacy_detector() returns
# None for it), consistent with "do not redesign the entire sensor
# architecture" -- this module only unifies identity for the two device
# kinds that actually have a canonical model today.
_LEGACY_SMOKE_TYPE = "Smoke"
_LEGACY_HEAT_TYPE = "Heat"


def adapt_legacy_detector(
    detector: Detector, zones: Iterable[Zone],
) -> Optional[Union[SmokeDetector, HeatDetector]]:

    # Converts one legacy generic Detector (models/detector.py --
    # detector_type-based, no zone_ids, no compute_state()) into the
    # canonical SensorAsset-based SmokeDetector/HeatDetector shape,
    # preserving its id/name/properties/timestamps exactly -- so every
    # downstream consumer (SensorManager, Perception, BuildingState)
    # sees the SAME stable identity for the SAME physical device
    # regardless of which Designer tool originally placed it (the
    # legacy generic "Detector" tool, or the newer dedicated "Smoke
    # Detector"/"Heat Detector" tools).
    #
    # Deliberately transient -- never persisted, never serialized, never
    # written back onto Floor.detectors or stored anywhere. Every call
    # site treats the return value as a throwaway, read-only view
    # constructed fresh for whatever cycle needs a canonical-shaped
    # object; the legacy Detector already registered on Floor.detectors
    # remains the one and only stored/serialized asset for that device,
    # so this never creates a second persisted, independently-editable
    # asset for the same physical detector (see this module's own
    # architecture note in building_state_estimator memory / the
    # engineering investigation for this change).
    #
    # zone_ids is resolved with the exact same Zone.contains() geometry
    # test the existing Ground Truth providers already use for legacy
    # Detectors (position-in-zone-bounding-box) -- legacy Detector never
    # tracked an explicit zone assignment, so this is the only honest
    # way to derive one. health_status/mode/connection are left at their
    # EngineeringAsset defaults (OK/Simulation/blank) since legacy
    # Detector never tracked them either; activation_threshold is left
    # at SmokeDetector/HeatDetector's own default (0.2 / 57.0), which
    # already matches GroundTruthSmokeDetectorProvider/
    # GroundTruthHeatDetectorProvider's own DEFAULT_ALARM_THRESHOLD
    # constants exactly -- adapting a legacy detector changes no alarm
    # behavior it already had.

    zone_ids = tuple(
        zone.id for zone in zones
        if zone.floor_id == detector.floor_id and zone.contains(*detector.position)
    )

    common_kwargs = dict(
        id=detector.id,
        name=detector.name,
        properties=dict(detector.properties),
        created_at=detector.created_at,
        modified_at=detector.modified_at,
        floor_id=detector.floor_id,
        zone_ids=zone_ids,
        position=detector.position,
        mount_height=detector.mount_height,
        active=detector.active,
    )

    if detector.detector_type == _LEGACY_SMOKE_TYPE:
        return SmokeDetector(**common_kwargs)

    if detector.detector_type == _LEGACY_HEAT_TYPE:
        return HeatDetector(**common_kwargs)

    return None
