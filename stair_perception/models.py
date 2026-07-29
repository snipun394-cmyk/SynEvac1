from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Mapping, Optional, Tuple
from uuid import uuid4


# =====================================================
# Observable Stair Perception milestone -- the smallest derived,
# READ-ONLY representation of "how many currently-tracked occupants are
# observed on Stair S1," owned by its own small package, mirroring the
# facp/building_control/fire_safety_manager/fire_water_manager
# convention: a self-contained package producing ONE immutable snapshot
# type that building_state.estimator.BuildingStateEstimator only ever
# passes through, never computes itself.
#
# Deliberately narrow, matching live_occupants.occupancy.OccupancyFacts'
# own "do not add prediction or capacity semantics -- this is observed
# occupancy only" discipline: no congestion score, no safety score, no
# panic, no capacity adequacy, no predicted bottleneck. Measured
# occupancy truth only -- see docs/architecture/live_stair_perception.md.
# =====================================================


class StairObservationStatus(Enum):

    # The OBSERVED-vs-UNKNOWN distinction the prior audit's Phase 18
    # requires as mandatory, not optional: a stair simply absent from
    # occupant groupings must never be silently read as "zero people"
    # unless it is KNOWN to be genuinely observable this cycle.

    # A calibrated camera covers this Stair's observable region this
    # cycle -- occupant_count is a genuine measurement, and MAY
    # legitimately be zero (nobody currently detected there).
    OBSERVED = auto()

    # No honest basis to report an occupant_count at all this cycle --
    # either this Stair has no observable region authored at all, or the
    # floor(s) its region(s) belong to have no calibrated camera this
    # cycle. occupant_count is always 0 in this case, but that 0 is
    # NEVER meaningful -- see StairObservation.occupant_count's own
    # docstring; a consumer must always check `status`, never read
    # occupant_count alone.
    UNKNOWN = auto()


@dataclass(frozen=True)
class StairObservation:

    # One Stair's occupancy fact at one point in time.

    stair_id: str
    status: StairObservationStatus = StairObservationStatus.UNKNOWN

    # The SAME occupant_ids live_occupants.occupancy.OccupancyFacts.
    # occupant_ids_by_stair already carries for this stair_id -- never
    # re-derived, always identity-tuple-equal to that source (Phase 10's
    # own "do not reintroduce duplicate occupancy computations"). Always
    # empty when status is UNKNOWN (there is no honest occupant_count to
    # report, so there can be no honest occupant_ids either).
    occupant_ids: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def occupant_count(self) -> int:

        # Deliberately still returns an int (never None) even for
        # UNKNOWN, mirroring occupancy.observation.OccupancyObservation's
        # own "occupant_count is whatever a producer reports... None
        # means no reading available" EXCEPT inverted: here the count
        # field itself always exists (it is just always 0 when
        # meaningless), and `status` is the field a caller must check
        # instead -- chosen because this type's own occupant_ids tuple
        # is already the honest zero-length signal for UNKNOWN (an empty
        # tuple either way), so a second None-vs-int distinction on the
        # same fact would be redundant rather than additionally honest.

        return len(self.occupant_ids)


@dataclass(frozen=True)
class StairOccupancySnapshot:

    # Mirrors occupancy.snapshot.OccupancySnapshot's own shape/
    # conventions exactly (snapshot_id/timestamp, a Mapping keyed by id,
    # MappingProxyType-wrapped, a total accessor that never raises) --
    # deliberately NOT that class or a subclass of it: OccupancySnapshot
    # is keyed by Navigation Graph node_id (Zone/Outside/AssemblyPoint
    # only -- there is no Stair node type, and the prior audit's own
    # Phase 10 explicitly rejected turning Stair into one), so a
    # genuinely separate, Stair-id-keyed type is required, not a reuse.

    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0

    observations: Mapping[str, StairObservation] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "observations", MappingProxyType(dict(self.observations)))

    # =====================================================
    # Total accessor -- always returns a usable StairObservation, never
    # raises and never returns None, mirroring OccupancySnapshot.
    # observation_at()'s own "absent means no reading" convention. A
    # stair_id entirely absent from `observations` (e.g. one this
    # snapshot's producer never even considered) defaults to UNKNOWN,
    # the same honest default as one explicitly computed as UNKNOWN.
    # =====================================================

    def observation_for(self, stair_id: str) -> StairObservation:

        return self.observations.get(stair_id, StairObservation(stair_id=stair_id))
