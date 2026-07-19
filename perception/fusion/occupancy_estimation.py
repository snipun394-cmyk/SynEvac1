from dataclasses import dataclass, field
from typing import List, Mapping, Optional

from perception.models.building_observation import ObservationState
from perception.models.camera_observation import CameraFrameObservation


@dataclass(frozen=True)
class EstimatedOccupancy:

    # One zone's occupancy estimate as of one OccupancyEstimator.
    # estimate() call -- this pipeline stage's own output shape, kept
    # deliberately distinct from perception.models.building_observation.
    # ObservedOccupancy: that type is BuildingObservation's own final,
    # assembled shape (out of scope here -- "Do NOT generate
    # BuildingObservation"), while this one is an intermediate value
    # this specific stage produces. A later, not-yet-built assembly
    # step is what turns an EstimatedOccupancy into an
    # ObservedOccupancy, not this module.
    #
    # Keyed externally by zone_id (the same OccupancyEstimator.estimate()
    # return mapping's key), never stored on the object itself -- same
    # convention HazardNodeState/ObservedNodeState/OccupancyObservation
    # already follow.
    #
    # observation_state reuses ObservationState from Phase 1 directly
    # (UNOBSERVED/OBSERVED) rather than inventing a second tri-state
    # enum -- this is the exact hard requirement this module exists to
    # satisfy: a zone with no camera coverage (or no reading from any
    # camera assigned to it this cycle) must default to UNOBSERVED,
    # never a fabricated "0 occupants" or "clear".
    #
    # contributing_camera_ids is provenance metadata -- every camera
    # whose reading was considered for this zone this cycle, not just
    # whichever one's count was ultimately used -- so a consumer (or a
    # test) can always see what this estimate was actually built from.

    observation_state: ObservationState = ObservationState.UNOBSERVED

    estimated_count: Optional[float] = None
    confidence: Optional[float] = None

    contributing_camera_ids: List[str] = field(default_factory=list)


class OccupancyEstimator:

    # The Occupancy Estimation stage -- converts raw, camera-scoped
    # CameraFrameObservations into zone-scoped EstimatedOccupancy
    # values. Deliberately NOT Sensor Fusion (which combines different
    # *sensor types* -- cameras and detectors -- into one
    # ObservedNodeState hazard picture) and NOT BuildingObservation
    # assembly (a later stage, out of scope here) -- this class only
    # ever reasons about cameras and zones, and produces only
    # EstimatedOccupancy values.
    #
    # camera_zone_assignments is an already-resolved camera_id ->
    # zone_id topology, supplied by the caller rather than derived
    # here from Camera/Zone geometry -- this keeps this module's
    # dependencies to exactly perception.models, nothing else, and
    # keeps it honest about what it does not know: which zone(s) a
    # camera geometrically covers is a Building Model / coverage-
    # geometry question (see
    # perception.providers.ground_truth_camera_provider), answered
    # once, elsewhere, before this class is ever constructed -- this
    # class is pure estimation math over an already-resolved topology,
    # nothing more.
    #
    # V1 assigns each camera to at most one zone. A camera whose real
    # coverage_polygon() spans multiple zones has no way to split its
    # single aggregate estimated_occupant_count back across them
    # without fabricating a distribution -- doing so would violate
    # "never fabricate occupancy for unseen zones" in spirit even
    # though the zones in question are technically covered. Handling
    # a camera assigned to multiple zones is left to a future version,
    # not guessed at here.

    def __init__(self, camera_zone_assignments: Mapping[str, str]):

        self._camera_zone_assignments = dict(camera_zone_assignments)

        self._camera_ids_by_zone_id: Mapping[str, List[str]] = {}

        for camera_id, zone_id in self._camera_zone_assignments.items():
            self._camera_ids_by_zone_id.setdefault(zone_id, []).append(camera_id)

    # =====================================================

    def estimate(
        self, camera_observations: List[CameraFrameObservation],
    ) -> Mapping[str, EstimatedOccupancy]:

        observations_by_camera_id = {
            observation.camera_id: observation for observation in camera_observations
        }

        return {
            zone_id: self._estimate_zone(camera_ids, observations_by_camera_id)
            for zone_id, camera_ids in self._camera_ids_by_zone_id.items()
        }

    # =====================================================
    # Total accessor -- always returns a usable estimate, never raises
    # and never returns None, mirroring HazardSnapshot.node_state()/
    # BuildingObservation.node_observation()'s "every id always has
    # *some* usable value" convention. A zone_id this estimator has no
    # camera topology for at all defaults to the same UNOBSERVED
    # EstimatedOccupancy() a known-but-unreported zone would -- there
    # is no meaningful difference, from a consumer's perspective,
    # between "no camera was ever assigned here" and "a camera was
    # assigned but reported nothing this cycle": both mean this
    # estimator currently knows nothing about this zone.
    # =====================================================

    def estimated_occupancy_for(
        self, zone_id: str, camera_observations: List[CameraFrameObservation],
    ) -> EstimatedOccupancy:

        return self.estimate(camera_observations).get(zone_id, EstimatedOccupancy())

    # =====================================================

    def _estimate_zone(self, camera_ids, observations_by_camera_id) -> EstimatedOccupancy:

        candidates = [
            observation
            for observation in (
                observations_by_camera_id.get(camera_id) for camera_id in camera_ids
            )
            if observation is not None and observation.estimated_occupant_count is not None
        ]

        if not candidates:
            # No camera assigned to this zone reported a reading this
            # cycle (or none is assigned at all) -- UNOBSERVED, never
            # a fabricated zero. See EstimatedOccupancy's own
            # docstring for why this is a hard requirement, not a
            # stylistic default.
            return EstimatedOccupancy()

        # Duplicate-detection resolution: multiple cameras covering
        # the same zone are additional *votes* on the same physical
        # people, not additional people -- summing them would double-
        # count anyone visible to more than one camera. Taking the
        # highest reported count is the one combining rule that never
        # inflates occupancy from overlapping coverage; a confidence-
        # or count-weighted average was deliberately not used here, to
        # avoid inventing a new confidence-combination formula this
        # module has no principled basis for. Ties broken by higher
        # confidence, then by camera_id, purely for determinism.
        best = max(
            candidates,
            key=lambda observation: (
                observation.estimated_occupant_count,
                observation.confidence if observation.confidence is not None else -1.0,
                observation.camera_id,
            ),
        )

        return EstimatedOccupancy(
            observation_state=ObservationState.OBSERVED,
            estimated_count=best.estimated_occupant_count,
            confidence=best.confidence,
            contributing_camera_ids=sorted(
                observation.camera_id for observation in candidates
            ),
        )
