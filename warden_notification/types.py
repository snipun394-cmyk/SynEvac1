from enum import Enum, auto


class WardenNotificationStatus(Enum):

    # The same honest state machine building_control.types.RequestStatus
    # already establishes -- restated independently here (this package
    # must not import building_control, mirroring dynamic_signage's own
    # "restate the vocabulary, never import the sibling package" rule).

    PENDING_APPROVAL = auto()
    APPROVED = auto()
    REJECTED = auto()
    CANCELLED = auto()
    DISPATCHED = auto()
    CONFIRMED = auto()
    FAILED = auto()


# Terminal statuses -- a request in one of these never transitions
# again. Mirrors building_control.types.TERMINAL_STATUSES exactly.
TERMINAL_STATUSES = (
    WardenNotificationStatus.REJECTED,
    WardenNotificationStatus.CANCELLED,
    WardenNotificationStatus.FAILED,
)
