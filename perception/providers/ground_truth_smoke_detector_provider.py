from typing import List, Optional, Union

from hazard.provider import HazardProvider

from models.detector import Detector
from models.detector_migration import adapt_legacy_detector
from models.sensor_asset import DetectorState
from models.smoke_detector import SmokeDetector
from models.zone import Zone

from perception.models.smoke_detector_observation import SmokeDetectorReading
from perception.providers.smoke_detector_provider import SmokeDetectorProvider


class GroundTruthSmokeDetectorProvider(SmokeDetectorProvider):

    # The Ground Truth adapter for Smoke-type Detector devices --
    # answers "what would this specific smoke detector's alarm output
    # be" by reading the exact HazardNodeState.smoke_level Ground
    # Truth at the Zone the detector is physically placed in, and
    # comparing it against one fixed, documented activation threshold.
    # This reproduces a real photoelectric/ionization smoke detector's
    # own physical behavior -- it is a threshold-triggered device by
    # nature, so applying one constant threshold here mimics the
    # device's own trigger mechanism; it does not "estimate" or
    # "infer" a hazard level. Deliberately NOT Sensor Fusion: multiple
    # detectors covering the same zone are never combined here -- each
    # detector's alarm_active is computed and reported independently,
    # exactly as a real device would with no knowledge of any other
    # device on the panel.
    #
    # Consumes Ground Truth exclusively through HazardProvider.
    # snapshot_at(time) -- never simulator/sandbox internals. Reads
    # only HazardNodeState.smoke_level, deliberately never
    # hazard_score -- a smoke detector senses smoke, not an aggregate
    # hazard figure a fire-only scenario (no Smoke Propagation source
    # registered) would populate without any smoke actually present
    # (see fire_growth.model.FireGrowthModel, which sets hazard_score
    # but never smoke_level).
    #
    # A Detector whose position falls inside no Zone on its own floor,
    # or whose Zone has no smoke_level Ground Truth recorded this step
    # (None -- "not modeled", never "zero"), produces no reading at
    # all this cycle, consistent with every other provider in this
    # codebase never manufacturing a reading to fill a gap.
    #
    # Detector Identity Unification -- this constructor accepts a mixed
    # list of legacy generic Detector objects (models/detector.py,
    # detector_type-based, from Floor.detectors) and/or canonical
    # SmokeDetector assets (models/smoke_detector.py, from
    # Floor.smoke_detectors), so a caller can simply hand it
    # `floor.detectors + floor.smoke_detectors` and get correct behavior
    # for whichever kind of object represents a given physical device.
    # A legacy Detector is adapted, once, at construction time via
    # models.detector_migration.adapt_legacy_detector() -- the SAME
    # shared conversion sensor_manager.manager.SensorManager.
    # discover_sensors() also uses, so a legacy detector's id/identity
    # is identical whether it reaches SensorManager or Perception. A
    # genuine canonical SmokeDetector is used exactly as given, never
    # copied or mutated -- this adapter does not own engineering
    # assets, it only reads them.
    #
    # alarm_states_at() below delegates the actual ALARM/NORMAL/FAULT
    # decision to SmokeDetector.compute_state() -- the one place that
    # logic is allowed to live (see that method's own docstring) -- so
    # this file no longer keeps its own independent copy of the
    # activation-threshold comparison. alarm_threshold below is applied
    # only to a freshly-adapted legacy Detector (which has no threshold
    # field of its own to draw from); a genuine canonical SmokeDetector
    # always uses its own configured activation_threshold, never this
    # constructor argument, since Perception must never override an
    # engineering asset's own configuration.

    SMOKE_DETECTOR_TYPE = "Smoke"

    # An arbitrary, documented placeholder -- smoke_level is
    # normalized [0.0, 1.0] the same way hazard_score is (see
    # smoke_propagation/model.py), and real detectors trip well before
    # a space is fully smoke-logged. Not a validated UL 268 activation
    # curve; always overridable via the alarm_threshold constructor
    # argument. Identical to SmokeDetector.activation_threshold's own
    # default, so adapting a legacy Detector changes no alarm behavior
    # it already had.
    DEFAULT_ALARM_THRESHOLD = 0.2

    def __init__(
        self,
        detectors: List[Union[Detector, SmokeDetector]],
        zones: List[Zone],
        hazard_provider: HazardProvider,
        alarm_threshold: float = DEFAULT_ALARM_THRESHOLD,
    ):

        self._zones = list(zones)
        self._hazard_provider = hazard_provider

        self._detectors = self._resolve_detectors(detectors, alarm_threshold)

    # =====================================================

    def _resolve_detectors(
        self, detectors: List[Union[Detector, SmokeDetector]], alarm_threshold: float,
    ) -> List[SmokeDetector]:

        resolved = []

        for detector in detectors:

            if isinstance(detector, SmokeDetector):
                resolved.append(detector)
                continue

            if isinstance(detector, Detector) and detector.detector_type == self.SMOKE_DETECTOR_TYPE:

                adapted = adapt_legacy_detector(detector, self._zones)

                if adapted is not None:
                    adapted.activation_threshold = alarm_threshold
                    resolved.append(adapted)

        return resolved

    # =====================================================

    def alarm_states_at(self, time: float) -> List[SmokeDetectorReading]:

        snapshot = self._hazard_provider.snapshot_at(time)

        readings = []

        for detector in self._detectors:

            if not detector.active:
                continue

            zone = self._zone_containing(detector)

            if zone is None:
                continue

            smoke_level = snapshot.node_state(zone.id).smoke_level

            if smoke_level is None:
                continue

            state = detector.compute_state(smoke_level, time)

            if state == DetectorState.FAULT:
                # A faulty device cannot honestly report an alarm state
                # either way -- "no reading available", the same
                # convention every other gap in this file already
                # follows, never a fabricated "confirmed clear".
                continue

            readings.append(
                SmokeDetectorReading(
                    detector_id=detector.id,
                    timestamp=time,
                    alarm_active=(state == DetectorState.ALARM),
                    # A fixed, documented constant -- see
                    # GroundTruthCameraProvider's identical reasoning.
                    confidence=1.0,
                )
            )

        return readings

    # =====================================================

    def _zone_containing(self, detector: Union[Detector, SmokeDetector]) -> Optional[Zone]:

        x, y = detector.position

        for zone in self._zones:

            if zone.floor_id == detector.floor_id and zone.contains(x, y):
                return zone

        return None
