from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional
from uuid import uuid4


class IncidentState(Enum):

    IDLE = auto()
    ALARM = auto()
    VERIFICATION = auto()
    EVACUATION = auto()
    SUPPRESSION = auto()
    RECOVERY = auto()
    CLOSED = auto()


# =====================================================
# The legal transition graph -- a fixed, documented state machine
# (mirrors OccupantState's own small, closed set of legal states, one
# level up). Every edge is justified by what it represents
# operationally, not left to be inferred:
#
#   IDLE        -> ALARM                 a detector/panel first reports
#   ALARM       -> VERIFICATION          staff/panel begin confirming it
#               -> EVACUATION            confirmed serious enough to
#                                         evacuate without waiting on
#                                         verification
#               -> IDLE                  silenced/cleared before
#                                         verification even started
#   VERIFICATION -> EVACUATION           confirmed real
#                -> IDLE                 confirmed false alarm
#   EVACUATION  -> SUPPRESSION           active firefighting begins
#               -> RECOVERY              building cleared, nothing left
#                                         to suppress
#   SUPPRESSION -> RECOVERY              fire brought under control
#   RECOVERY    -> CLOSED                incident fully stood down
#               -> ALARM                 re-ignition / secondary event
#                                         during recovery
#   CLOSED      -> (none)                terminal for this incident;
#                                         start a new IncidentManager
#                                         for the next one
# =====================================================

_TRANSITIONS: Dict[IncidentState, FrozenSet[IncidentState]] = {
    IncidentState.IDLE: frozenset({IncidentState.ALARM}),
    IncidentState.ALARM: frozenset({
        IncidentState.VERIFICATION, IncidentState.EVACUATION, IncidentState.IDLE,
    }),
    IncidentState.VERIFICATION: frozenset({IncidentState.EVACUATION, IncidentState.IDLE}),
    IncidentState.EVACUATION: frozenset({IncidentState.SUPPRESSION, IncidentState.RECOVERY}),
    IncidentState.SUPPRESSION: frozenset({IncidentState.RECOVERY}),
    IncidentState.RECOVERY: frozenset({IncidentState.CLOSED, IncidentState.ALARM}),
    IncidentState.CLOSED: frozenset(),
}

# States in which an incident is actively unfolding, as opposed to not
# yet started or fully stood down -- IncidentManager.is_active below.
_ACTIVE_STATES = frozenset(_TRANSITIONS) - {IncidentState.IDLE, IncidentState.CLOSED}


class InvalidIncidentTransition(Exception):

    # Raised by transition_to() for any (from_state, to_state) pair not
    # present in _TRANSITIONS -- an incident's state must always be
    # explainable by a sequence of legal transitions, never silently
    # jumped (e.g. IDLE straight to SUPPRESSION).

    pass


@dataclass(frozen=True)
class IncidentTransition:

    from_state: IncidentState
    to_state: IncidentState
    time: float
    reason: Optional[str] = None


class IncidentManager:

    # Owns one incident's lifecycle, start (IDLE) to finish (CLOSED) --
    # Phase 6's own scope: track state and history, validate
    # transitions against the fixed graph above. Publishing an event
    # when a transition happens is the caller's job (see
    # live_system.orchestrator), not this class's -- IncidentManager
    # stays independently testable with no EventBus dependency at all.

    def __init__(self, incident_id: Optional[str] = None):

        self.incident_id = incident_id or str(uuid4())

        self._state = IncidentState.IDLE
        self._history: List[IncidentTransition] = []

    # =====================================================

    @property
    def state(self) -> IncidentState:

        return self._state

    # =====================================================

    @property
    def history(self) -> List[IncidentTransition]:

        return list(self._history)

    # =====================================================

    @property
    def is_active(self) -> bool:

        return self._state in _ACTIVE_STATES

    # =====================================================

    def can_transition_to(self, new_state: IncidentState) -> bool:

        return new_state in _TRANSITIONS[self._state]

    # =====================================================

    def transition_to(
        self, new_state: IncidentState, time: float, reason: Optional[str] = None,
    ) -> IncidentTransition:

        if not self.can_transition_to(new_state):

            raise InvalidIncidentTransition(
                f"Incident {self.incident_id!r} cannot transition "
                f"{self._state.name} -> {new_state.name} -- legal transitions from "
                f"{self._state.name} are "
                f"{sorted(state.name for state in _TRANSITIONS[self._state])}."
            )

        transition = IncidentTransition(
            from_state=self._state, to_state=new_state, time=time, reason=reason,
        )

        self._history.append(transition)
        self._state = new_state

        return transition
