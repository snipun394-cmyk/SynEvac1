from typing import Dict, Optional, Sequence, Tuple

from behavior_recognition.observation import RecognizedBehavior

from sensor_fusion.observation import Observation, ObservationKind
from sensor_fusion.provider import ObservationProvider


# Live Perception -> BuildingState Integration Bridge milestone, Phase
# 3 -- PRODUCTION observation providers: translate SynEvac's own
# runtime sources (live_occupants.manager.LiveOccupantManager,
# sensor_manager.manager.SensorManager, facp.engine.SimulatedFACP) into
# sensor_fusion.observation.Observation objects. This module is the
# architectural boundary Phase 2 requires: sensor_fusion/ itself stays
# completely generic (imports nothing from this repository at all --
# see tests/test_sensor_fusion_architecture_guards.py, unmodified by
# this milestone); every SynEvac-specific adaptation lives HERE
# instead. Each provider is duck-typed against the real object's own
# public shape (never a copy/re-derivation of it), consulted fresh
# every collect() call -- none of these classes owns or caches state of
# its own beyond what its constructor was handed.


class LiveOccupantObservationProvider(ObservationProvider):

    # Phase 3.1/3.2 (OCCUPANCY/BEHAVIOR) -- the CHOSEN canonical runtime
    # source for occupancy evidence feeding SensorFusionEngine is
    # live_occupants.manager.LiveOccupantManager, NOT multi_camera_fusion.
    # engine.MultiCameraFusionEngine directly (Phase 3's own "choose one
    # canonical runtime source after investigation" requirement).
    # Reasoning: LiveOccupantManager already tracks per-occupant
    # LIFECYCLE (NEW/ACTIVE/TEMPORARILY_LOST/EXITED/EXPIRED) -- only
    # currently-ACTIVE occupants (people genuinely observed THIS cycle,
    # via active_occupants()) are counted here, whereas MultiCameraFusion's
    # own FusionResult has no such distinction. Both sources ultimately
    # derive from the SAME upstream global occupant_id (ONE shared
    # CrossCameraIdentityResolver + LiveOccupantManager instance feeds
    # both MultiCameraFusionEngine, via Detection.occupant_id, AND this
    # provider, via LiveOccupant.occupant_id -- see docs/architecture/
    # live_perception_building_state_integration.md Sec 5 for the full
    # data-ownership map and Sec 8's own double-counting proof), so
    # BuildingState.occupant_tracks (from MultiCameraFusion, unchanged)
    # and BuildingState.zone_occupancy (from THIS provider, via
    # SensorFusionEngine) always agree on the same physical headcount --
    # never two independently-invented numbers.
    #
    # Canonical Live Occupancy Source of Truth milestone -- the per-zone
    # COUNT fed into each OCCUPANCY Observation now comes directly from
    # live_occupant_manager.canonical_occupancy(time).occupant_ids_by_zone
    # (Phase 5's own "must no longer independently reconstruct the same
    # zone headcount through a separate logic path" requirement) instead
    # of a second, independently-incremented counter -- this is the
    # SAME grouping crowd_intelligence/evacuation_progress/
    # emergency_response now all read too (see docs/architecture/
    # canonical_live_occupancy.md), memoized once per cycle. Only
    # occupants with a known current_zone_id contribute (Phase 7's own
    # "zone localization is honestly uncertain" convention, applied here
    # too -- an occupant with no resolved zone has nowhere honest to
    # report an OCCUPANCY observation against); confidence-averaging and
    # per-occupant BEHAVIOR observations still require the full
    # LiveOccupant objects, so a single pass over active_occupants()
    # remains here for that (genuinely provider-specific, not a
    # duplicated headcount).

    def __init__(self, live_occupant_manager, source_name: str = "live-occupants"):

        self.live_occupant_manager = live_occupant_manager
        self.source_name = source_name

    # =====================================================

    def collect(self, time: float) -> Tuple[Observation, ...]:

        facts = self.live_occupant_manager.canonical_occupancy(time)

        observations = []
        confidence_sums: Dict[str, float] = {}

        for occupant in self.live_occupant_manager.active_occupants():

            if occupant.current_zone_id is None:
                continue

            confidence_sums[occupant.current_zone_id] = confidence_sums.get(occupant.current_zone_id, 0.0) + occupant.confidence

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

        for zone_id, occupant_ids in facts.occupant_ids_by_zone.items():

            count = len(occupant_ids)

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


class _DetectorReadingObservationProvider(ObservationProvider):

    # Shared plumbing for LiveSmokeObservationProvider/
    # LiveHeatObservationProvider below -- both resolve a detector's
    # zone assignment from sensor_manager.manager.SensorManager's own
    # SensorStatus.zone_ids (Phase 9's own "use the existing canonical
    # detector identity work" -- SensorManager already reflects the
    # SAME sensor_id whether it came from a canonical SmokeDetector/
    # HeatDetector asset or a legacy Detector adapted via models.
    # detector_migration.adapt_legacy_detector(); this module never
    # re-derives or duplicates that identity mapping, only reads
    # SensorStatus.sensor_id/zone_ids as SensorManager already reports
    # them). A detector assigned to MULTIPLE zones honestly contributes
    # its one real reading to EACH assigned zone independently (never
    # picks an arbitrary "first" zone and silently drops the others).
    #
    # `reading_provider` is an OPTIONAL Callable[[float], Iterable] a
    # caller supplies once a real reading source exists (Phase 3's own
    # "no physical hardware yet" constraint means there is no honest
    # default implementation this milestone can provide -- perception.
    # providers.smoke_detector_provider.SmokeDetectorProvider/
    # heat_detector_provider.HeatDetectorProvider are themselves still
    # abstract, `raise NotImplementedError`, confirmed directly). None
    # (the default) means "no reading source configured yet" -- an
    # honest empty collect(), never a fabricated reading.

    _KIND: ObservationKind = None
    _SOURCE_PREFIX: str = ""

    def __init__(self, sensor_manager, reading_provider=None):

        self.sensor_manager = sensor_manager
        self.reading_provider = reading_provider

    # =====================================================

    def collect(self, time: float) -> Tuple[Observation, ...]:

        if self.reading_provider is None:
            return ()

        readings = self.reading_provider(time)

        zone_ids_by_sensor_id = {
            status.sensor_id: status.zone_ids
            for status in self.sensor_manager.all_statuses()
        }

        observations = []

        for reading in readings:

            zone_ids = zone_ids_by_sensor_id.get(reading.detector_id, ())

            for zone_id in zone_ids:

                observations.append(
                    Observation(
                        source=f"{self._SOURCE_PREFIX}-{reading.detector_id}",
                        kind=self._KIND,
                        location=zone_id,
                        timestamp=reading.timestamp,
                        confidence=reading.confidence if reading.confidence is not None else 1.0,
                        measurement=reading.alarm_active,
                    )
                )

        return tuple(observations)


class LiveSmokeObservationProvider(_DetectorReadingObservationProvider):

    _KIND = ObservationKind.SMOKE
    _SOURCE_PREFIX = "smoke"


class LiveHeatObservationProvider(_DetectorReadingObservationProvider):

    _KIND = ObservationKind.HEAT
    _SOURCE_PREFIX = "heat"


class LiveFACPObservationProvider(ObservationProvider):

    # Phase 3.6 (ALARM) -- adapts facp.models.FACPSnapshot's own
    # active_alarm_source_ids into ALARM-kind Observations, resolving
    # each source's zone the same SensorManager-backed way the detector
    # reading providers above do (a FACP alarm source id IS a
    # sensor_id -- see facp/models.py's own DetectorConditionReport.
    # from_status_and_reading(), which already keys by SensorStatus.
    # sensor_id). snapshot_provider is a Callable[[float], Optional[
    # FACPSnapshot]] -- the SAME facp.current_snapshot callable
    # build_live_runtime() already wires for BuildingState.facp_status
    # directly (never a second, independent FACP read).

    def __init__(self, sensor_manager, snapshot_provider, source_name: str = "facp"):

        self.sensor_manager = sensor_manager
        self.snapshot_provider = snapshot_provider
        self.source_name = source_name

    # =====================================================

    def collect(self, time: float) -> Tuple[Observation, ...]:

        if self.snapshot_provider is None:
            return ()

        snapshot = self.snapshot_provider(time)

        if snapshot is None:
            return ()

        zone_ids_by_sensor_id = {
            status.sensor_id: status.zone_ids
            for status in self.sensor_manager.all_statuses()
        }

        observations = []

        for source_id in snapshot.active_alarm_source_ids:

            for zone_id in zone_ids_by_sensor_id.get(source_id, ()):

                observations.append(
                    Observation(
                        source=self.source_name, kind=ObservationKind.ALARM, location=zone_id,
                        timestamp=snapshot.timestamp, confidence=1.0, measurement=True,
                    )
                )

        return tuple(observations)
