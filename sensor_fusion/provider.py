from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Mapping, Optional, Sequence, Tuple

from sensor_fusion.observation import Observation, ObservationKind


class ObservationProvider(ABC):

    # Sensor Fusion Engine milestone, Phase 4 -- the one seam every
    # current and future evidence source plugs into (YOLO/CCTV,
    # LiveOccupants/Behavior, smoke/heat detectors, FACP, manual
    # operator input, and Phase 3's own "future BLE/WiFi/UWB, future
    # firefighter reports" sources) -- mirrors hazard_evolution.source.
    # HazardSource.propose()/perception.providers.provider.
    # PerceptionProvider.observation_at()'s established "one method,
    # SensorFusionEngine never knows or cares which kind of source it
    # is talking to" role.

    @abstractmethod
    def collect(self, time: float) -> Tuple[Observation, ...]:

        # Must never raise for an honestly-empty result (no evidence
        # this cycle is a legitimate, common answer -- an empty tuple,
        # not an exception). May raise for a genuine provider failure
        # (a real error, not "nothing to report") -- sensor_fusion.
        # engine.SensorFusionEngine.collect() is the one place that
        # catches it, so a single failing provider never prevents every
        # other provider's evidence from being fused (Phase 9's own
        # "provider failures" test category).
        ...


class ManualObservationProvider(ObservationProvider):

    # Phase 4's own named example -- a human operator's direct report,
    # queued via report() and returned (and cleared) on the next
    # collect() call. The simplest possible provider: no sensor, no
    # camera, just an operator's own stated observation, confidence
    # supplied by the operator (or a caller's own policy) since there
    # is no device measurement to derive one from.

    def __init__(self, source_name: str = "manual-operator"):

        self.source_name = source_name
        self._pending: list = []

    # =====================================================

    def report(self, kind: ObservationKind, location: str, measurement, confidence: float, timestamp: float) -> None:

        self._pending.append(
            Observation(
                source=self.source_name, kind=kind, location=location,
                timestamp=timestamp, confidence=confidence, measurement=measurement,
            )
        )

    # =====================================================

    def collect(self, time: float) -> Tuple[Observation, ...]:

        observations = tuple(self._pending)
        self._pending = []

        return observations


class SmokeObservationProvider(ObservationProvider):

    # Adapts perception.models.smoke_detector_observation.
    # SmokeDetectorReading (detector_id, timestamp, alarm_active,
    # confidence) into SMOKE-kind Observations. A raw reading carries
    # no location of its own (see that type's own docstring -- it
    # reports only the device's native alarm bit) -- `zone_by_detector_id`
    # supplies the missing location, the same "detector asset placement,
    # resolved externally" convention camera_manager/sensor_manager
    # already establish for their own status objects.

    def __init__(self, zone_by_detector_id: Mapping[str, str], source_prefix: str = "smoke"):

        self.zone_by_detector_id = dict(zone_by_detector_id)
        self.source_prefix = source_prefix
        self._readings: list = []

    # =====================================================

    def set_readings(self, readings: Sequence) -> None:

        # Replaces the full current set of readings this provider
        # reports on the NEXT collect() call -- a caller (e.g. a Live
        # composition point reading real SmokeDetectorReading objects
        # each cycle) is expected to call this once per cycle before
        # SensorFusionEngine.collect() runs.
        self._readings = list(readings)

    # =====================================================

    def collect(self, time: float) -> Tuple[Observation, ...]:

        observations = []

        for reading in self._readings:

            zone_id = self.zone_by_detector_id.get(reading.detector_id)

            if zone_id is None:
                # An honestly unplaced detector -- no location to
                # report against, never guessed.
                continue

            observations.append(
                Observation(
                    source=f"{self.source_prefix}-{reading.detector_id}",
                    kind=ObservationKind.SMOKE,
                    location=zone_id,
                    timestamp=reading.timestamp,
                    confidence=reading.confidence if reading.confidence is not None else 1.0,
                    measurement=reading.alarm_active,
                )
            )

        return tuple(observations)


class HeatObservationProvider(ObservationProvider):

    # The HeatDetectorReading counterpart to SmokeObservationProvider --
    # identical shape/reasoning, HEAT kind instead of SMOKE.

    def __init__(self, zone_by_detector_id: Mapping[str, str], source_prefix: str = "heat"):

        self.zone_by_detector_id = dict(zone_by_detector_id)
        self.source_prefix = source_prefix
        self._readings: list = []

    # =====================================================

    def set_readings(self, readings: Sequence) -> None:

        self._readings = list(readings)

    # =====================================================

    def collect(self, time: float) -> Tuple[Observation, ...]:

        observations = []

        for reading in self._readings:

            zone_id = self.zone_by_detector_id.get(reading.detector_id)

            if zone_id is None:
                continue

            observations.append(
                Observation(
                    source=f"{self.source_prefix}-{reading.detector_id}",
                    kind=ObservationKind.HEAT,
                    location=zone_id,
                    timestamp=reading.timestamp,
                    confidence=reading.confidence if reading.confidence is not None else 1.0,
                    measurement=reading.alarm_active,
                )
            )

        return tuple(observations)


class CameraObservationProvider(ObservationProvider):

    # Adapts live_occupants.manager.LiveOccupantManager's own
    # active_occupants() into OCCUPANCY (one per zone, a headcount) and
    # BEHAVIOR (one per occupant with known behavior) Observations --
    # the "YOLO detections"/"Behavior observations" inputs this
    # milestone's own diagram names. Never imports tracking/
    # cross_camera_identity/camera_calibration directly -- only the
    # already-resolved LiveOccupant objects a caller hands it via
    # set_occupants(), keeping this provider (and this whole package)
    # independent of exactly how those occupants were produced.

    def __init__(self, source_name: str = "camera-occupants"):

        self.source_name = source_name
        self._occupants: Sequence = ()

    # =====================================================

    def set_occupants(self, occupants: Sequence) -> None:

        self._occupants = tuple(occupants)

    # =====================================================

    def collect(self, time: float) -> Tuple[Observation, ...]:

        observations = []

        counts: Dict[str, int] = defaultdict(int)
        confidence_sums: Dict[str, float] = defaultdict(float)

        for occupant in self._occupants:

            if occupant.current_zone_id is None:
                continue

            counts[occupant.current_zone_id] += 1
            confidence_sums[occupant.current_zone_id] += occupant.confidence

            if occupant.behavior is not None:

                observations.append(
                    Observation(
                        source=f"{self.source_name}-{occupant.occupant_id}",
                        kind=ObservationKind.BEHAVIOR,
                        location=occupant.current_zone_id,
                        timestamp=time,
                        confidence=occupant.confidence,
                        measurement=occupant.behavior,
                    )
                )

        for zone_id, count in counts.items():

            observations.append(
                Observation(
                    source=self.source_name,
                    kind=ObservationKind.OCCUPANCY,
                    location=zone_id,
                    timestamp=time,
                    confidence=confidence_sums[zone_id] / count,
                    measurement=float(count),
                )
            )

        return tuple(observations)


class FACPObservationProvider(ObservationProvider):

    # Adapts facp.models.FACPSnapshot's own active_alarm_source_ids
    # into ALARM-kind Observations -- the "FACP" input this milestone's
    # own diagram names. Like SmokeObservationProvider/
    # HeatObservationProvider, a raw alarm source id has no location of
    # its own (FACPSnapshot is a building-wide aggregation, not a
    # per-zone one -- see facp/models.py's own docstring), so
    # `zone_by_source_id` resolves it externally.

    def __init__(self, zone_by_source_id: Mapping[str, str], source_name: str = "facp"):

        self.zone_by_source_id = dict(zone_by_source_id)
        self.source_name = source_name
        self._snapshot: Optional[object] = None

    # =====================================================

    def set_snapshot(self, snapshot) -> None:

        self._snapshot = snapshot

    # =====================================================

    def collect(self, time: float) -> Tuple[Observation, ...]:

        if self._snapshot is None:
            return ()

        observations = []

        for source_id in self._snapshot.active_alarm_source_ids:

            zone_id = self.zone_by_source_id.get(source_id)

            if zone_id is None:
                continue

            observations.append(
                Observation(
                    source=self.source_name, kind=ObservationKind.ALARM, location=zone_id,
                    timestamp=self._snapshot.timestamp, confidence=1.0, measurement=True,
                )
            )

        return tuple(observations)
