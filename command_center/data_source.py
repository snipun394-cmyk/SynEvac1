from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Protocol, Tuple

from models.building import Building

from advisory_system.recommendation_models import AdvisoryReport

from command_center.incident_data import IncidentData, IncidentFrame


# =====================================================
# Live Command Center Integration milestone -- the data-source
# abstraction Phase 2 asks for. Lives in `command_center` (not
# `live_system`) because `live_system.integration` already imports
# `command_center` (see DashboardCommandCenterGateway) -- `command_center`
# must therefore never import `live_system`, or that would be circular.
# The one concrete LIVE implementation of the Protocol below
# (`live_system.live_command_center_gateway.LiveCommandCenterDataSource`)
# lives on the `live_system` side of that boundary instead, exactly
# mirroring how `advisory_system.ai_evidence.AIDecisionEvidence` is
# defined in `advisory_system` while its live *source* adapter
# (`live_system.live_advisory_gateway`) lives in `live_system`.
#
# CommandCenterSnapshot is a PRESENTATION type, not a new state model:
# every "real" field on it is either a reference to an already-computed
# canonical object (`building_state`, `advisory_report`,
# `ai_prediction_snapshot` -- never copied, never re-derived) or a
# `frame`/`IncidentFrame` -- the one type every existing Command Center
# panel already renders, reused unchanged rather than inventing a
# second per-tick shape. Building BuildingState into an IncidentFrame is
# an adapter's job (see live_command_center_gateway.
# frame_from_building_state()), not something forced onto BuildingState
# or IncidentData themselves (Phase 2's own explicit constraint).
# =====================================================


class CommandCenterMode(Enum):

    REPLAY = auto()
    LIVE = auto()


class SnapshotConsistency(Enum):

    # Phase 14's own honest consistency marker -- never silently
    # presented as "fine" when it is not. CURRENT means every populated
    # component this cycle shares one coherent as-of time; STALE means
    # at least one component is carrying over a previous cycle's value
    # (a gateway returned None this cycle -- the same "skip this cycle"
    # convention live_ai_gateway/live_advisory_gateway already
    # established); PARTIAL means at least one expected component is
    # simply unavailable (None) this cycle, not merely old; UNAVAILABLE
    # means no live state exists at all yet (before the first cycle).

    CURRENT = auto()
    STALE = auto()
    PARTIAL = auto()
    UNAVAILABLE = auto()


@dataclass(frozen=True)
class CommandCenterSnapshot:

    mode: CommandCenterMode
    timestamp: float = 0.0

    building_name: Optional[str] = None
    building: Optional[Building] = None

    # The one per-tick shape every existing panel already renders --
    # for Replay this is exactly IncidentData.frame_at_index()'s own
    # return value; for Live this is adapted, best-effort and honest,
    # from a canonical BuildingState (never fabricated beyond what
    # BuildingState actually reports -- see frame_from_building_state()).
    frame: Optional[IncidentFrame] = None

    # advisory_system.AdvisoryReport -- reused verbatim, never
    # translated or re-shaped, in either mode.
    advisory_report: Optional[AdvisoryReport] = None

    # decision_policy.policy.DecisionPolicy or None. Typed Any (not
    # imported) for the same "typed Any, not imported for its own sake"
    # reason IncidentData.decision_policy already is. None in Live mode
    # is the honest value -- Decision Policy has no live equivalent
    # (see docs/architecture/ai_augmented_advisory_integration.md §1).
    decision_policy: Any = None

    # Live-only raw canonical references, never duplicated or reshaped
    # -- exactly what LiveOrchestrator/StateManager produced, so a
    # Live-only panel (Live AI / Live Status) can read fields
    # `IncidentFrame`/`AdvisoryReport` don't carry (e.g. `BuildingState.
    # camera_observations`, `LiveAIPredictionSnapshot.bottleneck`).
    # Always None in Replay mode -- never fabricated to look live.
    building_state: Optional[Any] = None
    ai_prediction_snapshot: Optional[Any] = None

    # Live Evacuation Progress, Flow & Clearance Intelligence milestone --
    # evacuation_progress.models.EvacuationProgressSnapshot, mirroring
    # ai_prediction_snapshot's own "typed Any, never fabricated in
    # Replay mode" convention exactly.
    evacuation_progress: Optional[Any] = None

    # Phase 14 -- per-component as-of timestamps, honestly reported.
    # None means that component has never been populated at all (not
    # "populated at time 0.0"). Always None/CURRENT in Replay mode,
    # where every field of one IncidentFrame/AdvisoryReport pair is,
    # by construction, for the exact same simulation_time.
    building_state_timestamp: Optional[float] = None
    ai_prediction_timestamp: Optional[float] = None
    advisory_timestamp: Optional[float] = None
    evacuation_progress_timestamp: Optional[float] = None
    consistency: SnapshotConsistency = SnapshotConsistency.CURRENT

    # Phase 15 -- a bounded, most-recent-last list of human-readable
    # event strings. Live-only; Replay mode keeps using the existing,
    # full RecommendationTimelinePanel/IncidentData.recommendation_history
    # instead (never duplicated here).
    recent_events: Tuple[str, ...] = field(default_factory=tuple)

    # Replay-only navigation context -- None/0 in Live mode, where
    # timeline scrubbing is honestly disabled (Phase 5).
    replay_incident: Optional[IncidentData] = None
    replay_frame_index: int = 0
    replay_frame_count: int = 0


class CommandCenterDataSource(Protocol):

    # The minimal seam Phase 2 asks for. `mode` is a plain, readable
    # attribute (never a method) so a caller can branch on it without
    # invoking anything. start()/stop() bracket "this source is actively
    # producing snapshots" -- for Replay this is a no-op (IncidentData is
    # always already fully resolved); for Live this starts/stops
    # whatever refresh mechanism (EventBus subscription or a poll timer)
    # keeps current_snapshot() up to date (see
    # live_system.live_command_center_gateway.LiveCommandCenterDataSource).

    mode: CommandCenterMode

    def current_snapshot(self) -> CommandCenterSnapshot: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


# =====================================================


class ReplayCommandCenterDataSource:

    # Phase 3's wrapper over the existing, untouched IncidentData/
    # IncidentFrame replay behavior -- IncidentData itself is not
    # rewritten, not one line of it changes. This class only ever reads
    # IncidentData's own already-public API
    # (frame_at_index/advisory_report_at_index/frame_count/building/
    # decision_policy), the same surface command_center.dashboard.
    # Dashboard's own existing set_incident()/show_frame()/
    # set_frame_index() methods already use -- those methods are left
    # completely unmodified by this milestone (see
    # docs/architecture/live_command_center_integration.md §2); this
    # class exists as an ADDITIVE, symmetric CommandCenterDataSource
    # implementation for a caller that wants the unified snapshot view,
    # not a replacement for Dashboard's existing, tested replay path.

    def __init__(self, incident_data: Optional[IncidentData] = None):

        self.mode = CommandCenterMode.REPLAY

        self._incident = incident_data
        self._frame_index = 0

    # =====================================================

    def set_incident(self, incident_data: Optional[IncidentData]) -> None:

        self._incident = incident_data
        self._frame_index = 0

    # =====================================================

    def set_frame_index(self, index: int) -> None:

        if self._incident is None:
            self._frame_index = 0
            return

        self._frame_index = max(0, min(index, self._incident.frame_count - 1))

    # =====================================================

    def start(self) -> None:

        # No-op -- IncidentData is always already fully resolved by the
        # time it is handed to this class (see IncidentData.__post_init__);
        # there is no background process for Replay mode to start.
        pass

    # =====================================================

    def stop(self) -> None:

        pass

    # =====================================================

    def current_snapshot(self) -> CommandCenterSnapshot:

        if self._incident is None:

            return CommandCenterSnapshot(mode=CommandCenterMode.REPLAY, consistency=SnapshotConsistency.UNAVAILABLE)

        frame = self._incident.frame_at_index(self._frame_index)
        report = self._incident.advisory_report_at_index(self._frame_index)

        return CommandCenterSnapshot(
            mode=CommandCenterMode.REPLAY,
            timestamp=frame.time,
            building_name=self._incident.building.name,
            building=self._incident.building,
            frame=frame,
            advisory_report=report,
            decision_policy=self._incident.decision_policy,
            consistency=SnapshotConsistency.CURRENT,
            replay_incident=self._incident,
            replay_frame_index=self._frame_index,
            replay_frame_count=self._incident.frame_count,
        )
