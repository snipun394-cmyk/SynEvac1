from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import List, Mapping, Optional
from uuid import uuid4

from perception.models.human_observation import HumanObservation


class ObservationState(Enum):

    # Whether *any* device currently covers a given node at all --
    # deliberately a field of its own on ObservedNodeState, never
    # inferred from whether some other field (e.g. alarm_active)
    # happens to be None. An explicit tri-state flag is what makes
    # "never observed" and "observed and currently silent"
    # impossible to confuse at the type level, rather than relying on
    # every future reader to remember a convention. A node absent from
    # BuildingObservation.node_observations means the same thing as
    # UNOBSERVED -- see ObservedNodeState's own default and
    # BuildingObservation.node_observation()'s total-accessor
    # behavior below.

    UNOBSERVED = auto()
    OBSERVED = auto()


class PerceptionSeverity(Enum):

    # Perception's own ordinal hazard classification -- deliberately a
    # separate type from hazard.severity.HazardSeverity, even though
    # the labels line up, for the same reason HazardSnapshot and
    # OccupancySnapshot share no base class: Ground Truth's severity
    # is derived from a continuous hazard_score Perception must never
    # see (see BuildingObservation's own module-level notes below), so
    # Perception needs its own classification authority. Deliberately
    # no from_score()-equivalent classmethod here yet -- deriving this
    # value from raw sensor observations is a future Sensor Fusion
    # stage's job, out of scope for this package's current phase, not
    # this enum's.

    NONE = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass(frozen=True)
class ObservedNodeState:

    # One node's perceived hazard picture -- keyed externally by
    # Node.id (see BuildingObservation), mirroring HazardNodeState's
    # own "keyed externally, never stored on Node" convention.
    # Defaults to UNOBSERVED/all-None on purpose: constructing this
    # class with no arguments represents a node no device currently
    # covers, never a node confirmed clear -- see
    # BuildingObservation.node_observation()'s total-accessor
    # behavior, which relies on exactly this default.

    observation_state: ObservationState = ObservationState.UNOBSERVED

    alarm_active: Optional[bool] = None
    alarm_source_types: List[str] = field(default_factory=list)

    visibility_estimate: Optional[str] = None
    estimated_severity: Optional[PerceptionSeverity] = None

    last_observed_time: Optional[float] = None


@dataclass(frozen=True)
class ObservedOccupancy:

    # One node's perceived headcount -- deliberately not merged into
    # ObservedNodeState (see BuildingObservation.occupancy_observations
    # below for why occupancy and hazard perception are kept in
    # separate mappings). estimated_count follows
    # occupancy.observation.OccupancyObservation's own convention
    # exactly: None means no reading available, never zero people.

    estimated_count: Optional[float] = None
    confidence: Optional[float] = None


@dataclass(frozen=True)
class ObservedEdgeState:

    # One edge's perceived traversability -- the Perception-side
    # counterpart to hazard.edge_state.HazardEdgeState.traversable,
    # always inferred (from the ObservedNodeState of the edge's
    # endpoints, or from a camera positioned to view the connector
    # directly), never a direct sensor reading -- there is no
    # dedicated edge-mounted device in the Building Model (Cameras/
    # Detectors are placed at points, not bound to edges).
    # blocked_estimate is None when neither endpoint is OBSERVED and
    # no camera happens to cover this connector -- "no opinion", not
    # "assumed passable".

    blocked_estimate: Optional[bool] = None


@dataclass(frozen=True)
class PerceptionSystemStatus:

    # Global, non-per-node metadata about the perception pipeline
    # itself -- deliberately separate from any single node/edge's
    # state, because a dropped panel or comms link is a fact about the
    # whole detection system, not about any one zone.
    #
    # panel_communication_ok defaults to True ("no failure reported"),
    # not because the system is verified healthy but because this
    # package makes no claim otherwise until something actually wires
    # it up -- distinct from ObservationState.UNOBSERVED above, which
    # is about individual sensor coverage, not about whether the
    # reporting channel itself is alive.

    panel_communication_ok: bool = True

    active_camera_count: int = 0
    active_detector_count: int = 0


@dataclass(frozen=True)
class BuildingObservation:

    # The one object every downstream consumer -- Rule-Based Decision
    # Engine, Dataset Generation, Firefighter Dashboard, Logging/
    # Replay, and (only through a future ObservationEncoder) an RL
    # Agent -- is meant to consume instead of Ground Truth. This file
    # uses only plain Python types throughout (str, float, bool, int,
    # Optional, Mapping, List, and this module's own
    # ObservationState/PerceptionSeverity enums) -- no Gymnasium, no
    # NumPy, no Torch, and no import of hazard, hazard_evolution, or
    # occupancy anywhere in this file. Producing a BuildingObservation
    # from Ground Truth is a future Sensor Fusion/Occupancy Estimation
    # stage's job, not this type's -- this file defines only the
    # shape, never how it gets filled in.
    #
    # node_observations/occupancy_observations/edge_observations are
    # kept as three separate mappings rather than one combined
    # per-node object -- hazard-side and occupancy-side perception are
    # produced by different sensor classes on different estimation
    # pipelines, and a consumer that only cares about one should not
    # need to reach through a combined type to get it. Same reasoning
    # that already keeps HazardSnapshot and OccupancySnapshot
    # structurally unrelated.
    #
    # observation_id/timestamp mirror HazardSnapshot's own
    # snapshot_id/timestamp. schema_version is additionally needed
    # here and not on HazardSnapshot: BuildingObservation is the one
    # snapshot type in this codebase meant to be logged, replayed, and
    # consumed by trained model weights across code revisions, not
    # just regenerated fresh every run -- new fields should default to
    # None/UNOBSERVED-equivalent so older consumers of a newer schema
    # simply see "unknown" for a field they predate, and this number
    # should only change when a field is removed or repurposed, never
    # for a purely additive change.

    observation_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0
    schema_version: int = 1

    node_observations: Mapping[str, ObservedNodeState] = field(default_factory=dict)
    occupancy_observations: Mapping[str, ObservedOccupancy] = field(default_factory=dict)
    edge_observations: Mapping[str, ObservedEdgeState] = field(default_factory=dict)

    # Human Perception Integration -- additive (Phase 6 of that work):
    # keyed by HumanObservation.person_id, a tracking identity, never a
    # Node.id -- unlike node_observations/occupancy_observations/
    # edge_observations, this mapping is not id-shaped after the
    # Navigation Graph at all. A person absent from this mapping means
    # "not currently observed" (see HumanObservation's own docstring),
    # the same absence convention every other mapping on this type
    # already follows. Every existing caller/test that never supplies
    # this field is unaffected -- it defaults to empty exactly like
    # schema_version's own additive-field precedent describes.
    human_observations: Mapping[str, HumanObservation] = field(default_factory=dict)

    system_status: PerceptionSystemStatus = field(default_factory=PerceptionSystemStatus)

    # =====================================================

    def __post_init__(self):

        # MappingProxyType-wrapped so the observation stays immutable
        # even though dict/dataclass equality still works -- same
        # technique HazardSnapshot/OccupancySnapshot already use.
        object.__setattr__(
            self, "node_observations", MappingProxyType(dict(self.node_observations)),
        )
        object.__setattr__(
            self, "occupancy_observations", MappingProxyType(dict(self.occupancy_observations)),
        )
        object.__setattr__(
            self, "edge_observations", MappingProxyType(dict(self.edge_observations)),
        )
        object.__setattr__(
            self, "human_observations", MappingProxyType(dict(self.human_observations)),
        )

    # =====================================================
    # Total accessors -- always return a usable state, never raise and
    # never return None, mirroring HazardSnapshot.node_state()/
    # OccupancySnapshot.observation_at()'s "every id always has *some*
    # usable value" convention. Here, absence means UNOBSERVED (an
    # ObservedNodeState()'s default observation_state), never "clear"
    # -- the one deliberate break from Ground Truth's own
    # absent-means-clear convention, and the single most important
    # behavior this type exists to guarantee.
    # =====================================================

    def node_observation(self, node_id) -> ObservedNodeState:

        return self.node_observations.get(node_id, ObservedNodeState())

    # =====================================================

    def occupancy_observation(self, node_id) -> ObservedOccupancy:

        return self.occupancy_observations.get(node_id, ObservedOccupancy())

    # =====================================================

    def edge_observation(self, edge_id) -> ObservedEdgeState:

        return self.edge_observations.get(edge_id, ObservedEdgeState())

    # =====================================================

    def human_observation(self, person_id) -> Optional[HumanObservation]:

        # Unlike node_observation()/occupancy_observation()/
        # edge_observation() above, there is no meaningful "default"
        # HumanObservation to fabricate for a person_id this cycle
        # never reported -- a tracking identity absent this cycle
        # simply is not being observed, and constructing a fresh
        # HumanObservation(person_id=...) with every other field at
        # its own default would misrepresent "never observed" as
        # "observed and found ordinary." Returns None instead, the
        # one total accessor on this type that does.
        return self.human_observations.get(person_id)
