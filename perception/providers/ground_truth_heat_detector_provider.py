from typing import List, Optional, Union

from hazard.provider import HazardProvider

from models.detector import Detector
from models.detector_migration import adapt_legacy_detector
from models.heat_detector import HeatDetector
from models.sensor_asset import DetectorState
from models.zone import Zone

from perception.models.heat_detector_observation import HeatDetectorReading
from perception.providers.heat_detector_provider import HeatDetectorProvider


class GroundTruthHeatDetectorProvider(HeatDetectorProvider):

    # The Ground Truth adapter for Heat-type Detector devices --
    # answers "what would this specific heat detector's alarm output
    # be" by reading the exact HazardNodeState.temperature Ground
    # Truth at the Zone the detector is physically placed in, and
    # comparing it against one fixed, documented activation threshold.
    # Same reasoning as GroundTruthSmokeDetectorProvider: a fixed
    # temperature threshold reproduces a real fixed-temperature heat
    # detector's own physical trigger, it does not estimate or infer a
    # hazard level, and readings from multiple detectors are never
    # combined here (no Sensor Fusion).
    #
    # SynEvac V1 supports Fixed Temperature heat detectors only --
    # Rate-of-Rise is deliberately out of scope (see
    # HeatDetectorReading, which carries no rate-of-rise field at all
    # for the same reason). This adapter produces only the
    # instantaneous fixed-temperature trip condition.
    #
    # Consumes Ground Truth exclusively through HazardProvider.
    # snapshot_at(time) -- never simulator/sandbox internals. Reads
    # only HazardNodeState.temperature, never hazard_score, for the
    # same reason the smoke adapter reads only smoke_level. No current
    # HazardSource in this codebase (FireGrowthModel, SmokePropagation
    # Model) actually populates temperature -- this adapter is
    # correctly built against the interface regardless of what
    # scenario content exists today; it will simply produce no
    # readings until a HazardSource that sets temperature exists.
    #
    # A Detector whose position falls inside no Zone on its own floor,
    # or whose Zone has no temperature Ground Truth recorded this step
    # (None -- "not modeled", never "zero"), produces no reading at
    # all this cycle.
    #
    # Detector Identity Unification -- this constructor accepts a mixed
    # list of legacy generic Detector objects (models/detector.py,
    # detector_type-based, from Floor.detectors) and/or canonical
    # HeatDetector assets (models/heat_detector.py, from
    # Floor.heat_detectors) -- same reasoning as
    # GroundTruthSmokeDetectorProvider's own identical change; see that
    # class's docstring for the full explanation. A legacy Detector is
    # adapted, once, at construction time via
    # models.detector_migration.adapt_legacy_detector() -- the same
    # shared conversion SensorManager.discover_sensors() uses, so a
    # legacy detector's id/identity is identical wherever it is
    # consumed. alarm_states_at() delegates the ALARM/NORMAL/FAULT
    # decision to HeatDetector.compute_state() rather than keeping an
    # independent copy of the activation-threshold comparison.

    HEAT_DETECTOR_TYPE = "Heat"

    # An arbitrary, documented placeholder loosely modeled on a common
    # fixed-temperature heat detector rating (~57C / 135F), in the
    # same degrees-Celsius-assumed units HazardNodeState.temperature
    # is otherwise used with in this codebase (see
    # tests/test_hazard_evolution.py). Not a validated UL 521
    # activation curve; always overridable via the alarm_threshold
    # constructor argument. Identical to HeatDetector.activation_
    # threshold's own default, so adapting a legacy Detector changes no
    # alarm behavior it already had.
    DEFAULT_ALARM_THRESHOLD = 57.0

    def __init__(
        self,
        detectors: List[Union[Detector, HeatDetector]],
        zones: List[Zone],
        hazard_provider: HazardProvider,
        alarm_threshold: float = DEFAULT_ALARM_THRESHOLD,
    ):

        self._zones = list(zones)
        self._hazard_provider = hazard_provider

        self._detectors = self._resolve_detectors(detectors, alarm_threshold)

    # =====================================================

    def _resolve_detectors(
        self, detectors: List[Union[Detector, HeatDetector]], alarm_threshold: float,
    ) -> List[HeatDetector]:

        resolved = []

        for detector in detectors:

            if isinstance(detector, HeatDetector):
                resolved.append(detector)
                continue

            if isinstance(detector, Detector) and detector.detector_type == self.HEAT_DETECTOR_TYPE:

                adapted = adapt_legacy_detector(detector, self._zones)

                if adapted is not None:
                    adapted.activation_threshold = alarm_threshold
                    resolved.append(adapted)

        return resolved

    # =====================================================

    def alarm_states_at(self, time: float) -> List[HeatDetectorReading]:

        snapshot = self._hazard_provider.snapshot_at(time)

        readings = []

        for detector in self._detectors:

            if not detector.active:
                continue

            zone = self._zone_containing(detector)

            if zone is None:
                continue

            temperature = snapshot.node_state(zone.id).temperature

            if temperature is None:
                continue

            state = detector.compute_state(temperature, time)

            if state == DetectorState.FAULT:
                continue

            readings.append(
                HeatDetectorReading(
                    detector_id=detector.id,
                    timestamp=time,
                    alarm_active=(state == DetectorState.ALARM),
                    confidence=1.0,
                )
            )

        return readings

    # =====================================================

    def _zone_containing(self, detector: Union[Detector, HeatDetector]) -> Optional[Zone]:

        x, y = detector.position

        for zone in self._zones:

            if zone.floor_id == detector.floor_id and zone.contains(x, y):
                return zone

        return None
