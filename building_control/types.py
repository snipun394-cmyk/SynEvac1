from enum import Enum, auto


class ControlSystemType(Enum):

    # Only systems the existing platform can represent honestly.
    # Door/Exit already have real, mutable engineering state and a
    # real executor (simulation_interactive.ActionExecutor) behind
    # them. Stair Pressurization/Smoke Exhaust/Deluge have no backing
    # physics anywhere in this codebase (confirmed by the Phase 1
    # investigation this package was built from) -- their control
    # state is tracked honestly as state-only, never claimed to have
    # a physical hazard effect. See providers.SimulationControlProvider.

    DOOR = auto()
    EXIT = auto()
    STAIR_PRESSURIZATION = auto()
    SMOKE_EXHAUST = auto()
    DELUGE = auto()


class ControlAction(Enum):

    # No LOCK/UNLOCK: simulation_interactive.ActionExecutor's own
    # CLOSE_DOOR handler already documents that it sets DoorState.
    # LOCKED, not a separate CLOSED state -- "close" and "lock" are
    # the same action on the only Door model this platform has, and
    # Exit has no lock concept at all (models/exit.py has only
    # is_blocked). Fabricating a separate LOCK/UNLOCK action here
    # would not be backed by any real state transition, so OPEN/CLOSE
    # is the whole honest vocabulary for Door and Exit alike.

    OPEN = auto()
    CLOSE = auto()
    ACTIVATE = auto()
    DEACTIVATE = auto()


# The only (system, action) pairs a ControlRequest may honestly
# request. Enforced by BuildingControlController._validate_target
# alongside target existence -- a mismatched pair (e.g. DOOR +
# ACTIVATE) is rejected the same way an unknown target is.
VALID_ACTIONS = {
    ControlSystemType.DOOR: (ControlAction.OPEN, ControlAction.CLOSE),
    ControlSystemType.EXIT: (ControlAction.OPEN, ControlAction.CLOSE),
    ControlSystemType.STAIR_PRESSURIZATION: (ControlAction.ACTIVATE, ControlAction.DEACTIVATE),
    ControlSystemType.SMOKE_EXHAUST: (ControlAction.ACTIVATE, ControlAction.DEACTIVATE),
    ControlSystemType.DELUGE: (ControlAction.ACTIVATE, ControlAction.DEACTIVATE),
}


class RequestStatus(Enum):

    # The smallest honest state machine covering Phase 3's required
    # vocabulary (RECOMMENDED/REQUESTED collapse into PENDING_APPROVAL:
    # a ControlRequest already IS the request, so there is nothing
    # distinct to represent between "recommended" and "requested" once
    # translate_recommendation() has produced one).

    PENDING_APPROVAL = auto()
    APPROVED = auto()
    REJECTED = auto()
    CANCELLED = auto()
    DISPATCHED = auto()
    CONFIRMED = auto()
    FAILED = auto()


# Terminal statuses -- a request in one of these never transitions
# again. Used by BuildingControlController to decide when re-submitting
# an identical request should create a new one instead of being
# deduplicated (Phase 11: "preserve contradictory request history
# rather than silently overwriting it").
TERMINAL_STATUSES = (
    RequestStatus.REJECTED,
    RequestStatus.CANCELLED,
    RequestStatus.FAILED,
)


class RequestSource(Enum):

    ADVISORY_ADAPTER = auto()
    OPERATOR = auto()


class ApprovalMode(Enum):

    REQUIRES_APPROVAL = auto()

    # Simulation-only. Never implies Live mode auto-actuates equipment
    # -- BuildingControlController's constructor refuses this mode
    # unless the configured provider itself declares
    # is_simulation_only = True (see providers.BuildingControlProvider),
    # so a Live provider can never be paired with automatic approval by
    # accident.
    AUTO_APPROVE_SIMULATION = auto()
