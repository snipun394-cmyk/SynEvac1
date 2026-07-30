from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Mapping, Optional, Tuple
from uuid import uuid4

from observable_assets.models import ObservationStatus


# =====================================================
# Live Stair Flow & Movement Direction Intelligence milestone -- the
# smallest honest model for STAIR THROUGHPUT (entries/exits/rate/
# direction), a genuinely different question from Observable Stair
# Perception's own OBSERVED OCCUPANCY (observable_assets.models.
# AssetObservation/ObservableAssetSnapshot, unchanged by this milestone)
# and Camera Coverage Intelligence's own camera_coverage.models
# (camera <-> asset VISIBILITY, unchanged by this milestone).
#
# Deliberately narrow, matching observable_assets.models' own "no
# congestion score, no safety score, no predicted bottleneck" discipline:
# this package reports MEASURED throughput evidence only, never a
# judgement about whether that throughput is good, bad, or safe.
# =====================================================


class TrafficDirection(Enum):

    # Grounded in building/floor TOPOLOGY (Staircase.from_floor_id/
    # to_floor_id + Building.floor_elevation()'s own genuine vertical
    # ordering), never in screen-space movement or image-Y direction --
    # see stair_flow.direction.derive_direction()'s own docstring.

    UP = auto()
    DOWN = auto()

    # No honest basis to say UP or DOWN -- missing floor-side evidence at
    # the entry instant, an unresolvable floor, or (a defensively-handled,
    # not expected in a well-formed building) a Staircase whose two ends
    # report equal elevation. Never guessed.
    UNKNOWN = auto()


class StairFlowEventType(Enum):

    # ENTERED_STAIR: current_stair_id genuinely changed from None (or a
    # different stair_id) TO this stair_id.
    # EXITED_STAIR: current_stair_id genuinely changed FROM this stair_id
    # to None (or a different stair_id).
    # See stair_flow.events.extract_stair_flow_events() for the exact
    # evidence each requires, and for the ONE genuine exclusion (an
    # occupant's very first-ever observation already being on a stair
    # produces NO event -- there is no entry evidence, only presence
    # evidence).

    ENTERED_STAIR = auto()
    EXITED_STAIR = auto()


@dataclass(frozen=True)
class StairFlowEvent:

    # One evidence-backed traversal event for one occupant. Always
    # derived from a REAL live_occupants.history.StairTransitionRecord
    # already recorded by live_occupants.manager.LiveOccupantManager --
    # this package never invents a transition, and never re-runs
    # identity resolution (occupant_id is already the canonical, fused
    # global identity by the time it reaches here -- see docs/
    # architecture/stair_flow_intelligence.md Sec "Transition identity").

    occupant_id: str
    stair_id: str
    event_type: StairFlowEventType

    # UNKNOWN for every EXITED_STAIR event by design (this package
    # derives direction exactly once per traversal, at ENTRY, from the
    # confirmed floor-side the occupant entered from -- see
    # stair_flow.direction.derive_direction()). Only ever UP/DOWN for an
    # ENTERED_STAIR event when that derivation succeeded.
    direction: TrafficDirection

    timestamp: float

    # The floor_id this occupant was observed on at the SAME instant as
    # this transition (cross-referenced from live_occupants.history.
    # PositionSample.floor_id at the identical timestamp -- both are
    # written by the same LiveOccupantManager.update() call, so this is
    # genuine co-timed evidence, never inferred). None when no position
    # sample happened to share this exact timestamp (see docs' own
    # "genuine information gaps" section) -- direction is UNKNOWN
    # whenever this is None for an ENTERED_STAIR event.
    floor_id: Optional[str] = None

    provenance: Optional[str] = None


@dataclass(frozen=True)
class StairFlowMetrics:

    # Phase 4's model -- everything this package can honestly say about
    # one stair's current throughput, as of one timestamp. Every
    # Optional[...] field is None (never a fabricated 0) whenever
    # `status` is UNKNOWN and no recent window evidence exists either --
    # see stair_flow.compute._build_metrics()'s own gating, which mirrors
    # evacuation_progress.models.ExitFlow.recent_flow_per_minute's own
    # "report a real count if evidence exists OR position is currently
    # available, otherwise None" convention exactly.

    stair_id: str

    # Reused directly from observable_assets.models.ObservationStatus --
    # never a second, competing observation-status vocabulary. OBSERVED
    # means a calibrated camera currently covers this stair's floor(s);
    # UNKNOWN means no honest basis exists this cycle (see
    # observable_assets.facts.compute_asset_occupancy_snapshot()).
    status: ObservationStatus = ObservationStatus.UNKNOWN

    # Current occupancy on this stair right now -- the SAME number
    # crowd_intelligence.models.AssetApproachMetrics.observed_occupant_count
    # already reports for this stair (this package never recomputes it
    # independently; see stair_flow.compute.compute_stair_flow_snapshot()'s
    # own `observed_occupant_counts` parameter). A DIFFERENT measurement
    # from entries/exits below: occupancy is a point-in-time count,
    # entries/exits are a WINDOWED throughput count.
    observed_occupant_count: Optional[int] = None

    window_seconds: float = 60.0

    entries: Optional[int] = None
    exits: Optional[int] = None

    entry_rate_per_minute: Optional[float] = None
    exit_rate_per_minute: Optional[float] = None

    # entries - exits over the same window -- None whenever either is
    # None (both are always gated together, see compute.py).
    net_flow: Optional[int] = None

    upward_count: Optional[int] = None
    downward_count: Optional[int] = None

    # entries not attributable to UP or DOWN (direction UNKNOWN) --
    # upward_count + downward_count + unknown_direction_count always
    # equals entries whenever all three are not None (proven in
    # tests/test_stair_flow_direction.py).
    unknown_direction_count: Optional[int] = None

    upward_rate_per_minute: Optional[float] = None
    downward_rate_per_minute: Optional[float] = None

    timestamp: float = 0.0

    provenance: Optional[str] = None


@dataclass(frozen=True)
class StairFlowSnapshot:

    # Building-wide throughput picture for one cycle -- mirrors
    # observable_assets.models.ObservableAssetSnapshot/camera_coverage.
    # models.CameraCoverageSnapshot's own shape (snapshot_id/timestamp, a
    # Mapping keyed by id, MappingProxyType-wrapped, a total accessor
    # that never raises).

    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0
    window_seconds: float = 60.0

    by_stair: Mapping[str, StairFlowMetrics] = field(default_factory=dict)

    # The raw, evidence-backed events this snapshot's counts were built
    # from -- Phase 3/8's own auditability requirement ("prove one
    # canonical occupant produces exactly one entry event, not two").
    # Sorted by timestamp. Never itself a source of truth beyond
    # `by_stair` (which is derived from exactly this same list) -- kept
    # for debugging/proof, not as a second competing representation.
    events: Tuple[StairFlowEvent, ...] = field(default_factory=tuple)

    def __post_init__(self):

        object.__setattr__(self, "by_stair", MappingProxyType(dict(self.by_stair)))

    def for_stair(self, stair_id: str) -> StairFlowMetrics:

        return self.by_stair.get(stair_id, StairFlowMetrics(stair_id=stair_id, window_seconds=self.window_seconds))

    def events_for_stair(self, stair_id: str) -> Tuple[StairFlowEvent, ...]:

        return tuple(event for event in self.events if event.stair_id == stair_id)
