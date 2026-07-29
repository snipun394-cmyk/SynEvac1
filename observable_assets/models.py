from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Mapping, Optional, Tuple
from uuid import uuid4


# =====================================================
# Observable Asset Perception Framework milestone -- the generic
# "observable engineering asset" abstraction the Observable Stair
# Perception milestone's own StairObservationStatus/StairObservation/
# StairOccupancySnapshot turned out to already BE, in disguise: none of
# their actual logic ever referenced anything Stair-specific (no
# from_floor_id/to_floor_id, no travel_distance, nothing) -- only their
# NAMES did. This module is that same code, generalized and given
# honest names, so a future asset type (Door, Exit, Assembly Point,
# Refuge Area, Lift Lobby, Escalator, Corridor, Fire Door, Hose Reel
# Area, Hydrant Area, ...) reuses it directly rather than growing its
# own parallel Door-named copy. Stair is the FIRST, currently only,
# concrete registrant -- see camera_calibration.stair_lookup.
# STAIR_ASSET_KIND.
#
# Mirrors the facp/building_control/fire_safety_manager/fire_water_manager
# convention: a self-contained package producing ONE immutable snapshot
# type that building_state.estimator.BuildingStateEstimator only ever
# passes through, never computes itself.
#
# Deliberately narrow, matching live_occupants.occupancy.OccupancyFacts'
# own "do not add prediction or capacity semantics -- this is observed
# occupancy only" discipline: no congestion score, no safety score, no
# panic, no capacity adequacy, no predicted bottleneck, no hazard
# assumption of any kind. Measured occupancy truth only -- see
# docs/architecture/observable_asset_perception.md.
# =====================================================


class ObservationStatus(Enum):

    # The OBSERVED-vs-UNKNOWN distinction the Observable Stair Perception
    # audit's Phase 18 required as mandatory, not optional, for Stair --
    # equally mandatory for any future asset type: an asset simply absent
    # from occupant groupings must never be silently read as "zero
    # people" unless it is KNOWN to be genuinely observable this cycle.

    # A calibrated camera covers this asset's observable region this
    # cycle -- occupant_count is a genuine measurement, and MAY
    # legitimately be zero (nobody currently detected there).
    OBSERVED = auto()

    # No honest basis to report an occupant_count at all this cycle --
    # either this asset has no observable region authored at all, or the
    # floor(s) its region(s) belong to have no calibrated camera this
    # cycle. occupant_count is always 0 in this case, but that 0 is
    # NEVER meaningful -- see AssetObservation.occupant_count's own
    # docstring; a consumer must always check `status`, never read
    # occupant_count alone.
    UNKNOWN = auto()


@dataclass(frozen=True)
class AssetObservation:

    # One observable asset's occupancy fact at one point in time --
    # asset id, asset type, observation state, occupancy, provenance,
    # timestamp (this milestone's own Phase 2 requirement, verbatim).
    # `asset_type` is a plain string ("Stair" today; "Door"/"Exit"/...
    # would be equally plain strings later, never a closed enum this
    # package would need editing to extend -- see navigation.edge.Edge's
    # own EDGE_TYPES/crowd_intelligence.models.AssetApproachMetrics'
    # own `asset_type: str` precedent for exactly this convention
    # already established elsewhere in this codebase).

    asset_id: str
    asset_type: str

    status: ObservationStatus = ObservationStatus.UNKNOWN

    # The SAME occupant_ids the caller's own canonical occupancy source
    # already carries for this asset_id -- never re-derived here (this
    # package never scans a Building or a LiveOccupantManager itself).
    # Always empty when status is UNKNOWN (there is no honest
    # occupant_count to report, so there can be no honest occupant_ids
    # either).
    occupant_ids: Tuple[str, ...] = field(default_factory=tuple)

    timestamp: float = 0.0

    # A short, human/diagnostic-oriented note on WHY this status holds
    # (e.g. "no calibrated camera on this floor", "no observable region
    # authored") -- optional, never required, never itself a source of
    # occupancy truth. Deliberately a free-form string, not a closed
    # vocabulary: this is provenance for a person debugging a deployment,
    # not a machine-decision input (this package makes no congestion/
    # hazard/safety claim of any kind, and provenance must never grow
    # into one).
    provenance: Optional[str] = None

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
class ObservableAssetSnapshot:

    # Mirrors occupancy.snapshot.OccupancySnapshot's own shape/
    # conventions exactly (snapshot_id/timestamp, a Mapping keyed by id,
    # MappingProxyType-wrapped, a total accessor that never raises) --
    # deliberately NOT that class or a subclass of it: OccupancySnapshot
    # is keyed by Navigation Graph node_id (Zone/Outside/AssemblyPoint
    # only -- Stair, Door, Exit, and every future observable asset type
    # this milestone anticipates are traversal/portal assets, never
    # Navigation Graph NODES -- see docs/architecture/
    # stair_camera_perception_audit.md and observable_asset_perception.md),
    # so a genuinely separate, asset-id-keyed type is required, not a
    # reuse.
    #
    # One snapshot instance may hold observations for MULTIPLE asset
    # types at once (keyed by asset_id, which is globally unique across
    # all engineering assets in this codebase already -- see
    # models.base_object.BaseObject's own uuid4 id convention) --
    # `observations_of_type()` below is the convenience filter a future
    # consumer wanting only Door (or only Stair) entries would use.

    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0

    observations: Mapping[str, AssetObservation] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "observations", MappingProxyType(dict(self.observations)))

    # =====================================================
    # Total accessor -- always returns a usable AssetObservation, never
    # raises and never returns None, mirroring OccupancySnapshot.
    # observation_at()'s own "absent means no reading" convention. An
    # asset_id entirely absent from `observations` (e.g. one this
    # snapshot's producer never even considered) defaults to UNKNOWN,
    # the same honest default as one explicitly computed as UNKNOWN.
    # asset_type defaults to "" for a totally-unknown id -- there is no
    # honest type to report for an id this snapshot never saw at all.
    # =====================================================

    def observation_for(self, asset_id: str) -> AssetObservation:

        return self.observations.get(asset_id, AssetObservation(asset_id=asset_id, asset_type=""))

    # =====================================================

    def observations_of_type(self, asset_type: str) -> Tuple[AssetObservation, ...]:

        return tuple(
            observation for observation in self.observations.values()
            if observation.asset_type == asset_type
        )
