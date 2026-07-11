from dataclasses import dataclass
from typing import Optional


@dataclass
class BehaviorGroup:

    # A lightweight, caller-side grouping marker -- not consumed by
    # HumanBehaviorLayer directly. A caller assigns the same group_id
    # to every member's BehaviorProfile; group-aware DecisionStrategy/
    # RouteChoiceStrategy implementations then look up
    # `leader_occupant_id`'s already-resolved BehaviorDecision in
    # DecisionContext.decisions_so_far. This is what makes leader/
    # follower, family groups, and paired helping behavior all fall
    # out of the existing Strategy interfaces rather than needing
    # bespoke orchestration machinery.

    group_id: str
    leader_occupant_id: Optional[str] = None
