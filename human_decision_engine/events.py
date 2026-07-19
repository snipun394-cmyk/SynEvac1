from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# =====================================================
# Phase 6 -- AI Integration. A plain, ordered, append-only log of every
# dynamic decision made during one registration session -- the event
# vocabulary the spec itself names verbatim. Each DecisionEvent is a
# frozen, serializable record (to_dict() only, plain dict -- the same
# "id-keyed plain dict, never a nested dataclass" convention
# ground_truth.labels.GroundTruth's own list-shaped fields already
# follow) so downstream consumers (ground_truth, dataset_builder,
# command_center) never need to import this package's own types.
# =====================================================

HELP_DECISION = "Help_Decision"
HELP_REJECTED = "Help_Rejected"
FIREFIGHTER_TASK = "Firefighter_Task"
GROUP_FORMED = "Group_Formed"
GROUP_DISSOLVED = "Group_Dissolved"
RESCUE_INITIATED = "Rescue_Initiated"
RESCUE_COMPLETED = "Rescue_Completed"

EVENT_TYPES = (
    HELP_DECISION, HELP_REJECTED, FIREFIGHTER_TASK,
    GROUP_FORMED, GROUP_DISSOLVED, RESCUE_INITIATED, RESCUE_COMPLETED,
)


@dataclass(frozen=True)
class DecisionEvent:

    event_type: str
    occupant_id: str
    related_occupant_id: Optional[str] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "event_type": self.event_type,
            "occupant_id": self.occupant_id,
            "related_occupant_id": self.related_occupant_id,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class DecisionEventLog:

    # A tiny, mutable append-only sink shared by HumanDecisionEngine/
    # FirefighterDecisionEngine/GroupRegistry for one registration
    # session -- kept separate from those engines themselves so a
    # caller that only wants firefighter events, say, can still share
    # one combined log across both engines (behaviour_profile_resolver.
    # dynamic_registrar does exactly this).

    def __init__(self):

        self._events = []

    # =====================================================

    def record(self, event_type, occupant_id, related_occupant_id=None, reason="", metadata=None) -> None:

        self._events.append(
            DecisionEvent(
                event_type=event_type,
                occupant_id=occupant_id,
                related_occupant_id=related_occupant_id,
                reason=reason,
                metadata=dict(metadata or {}),
            )
        )

    # =====================================================

    @property
    def events(self):

        return tuple(self._events)
