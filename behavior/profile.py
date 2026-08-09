from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional


class Role(Enum):

    INDEPENDENT = auto()
    LEADER = auto()
    FOLLOWER = auto()


@dataclass
class BehaviorProfile:

    # A per-occupant bundle of traits that DecisionStrategy/
    # RouteChoiceStrategy/PreMovementDelayStrategy implementations
    # read -- never a decision-maker itself. Not an engineering
    # model, not touching models/. Deliberately minimal: new
    # behaviors extend `traits` (or add fields here later) rather
    # than requiring new top-level classes.

    occupant_id: str

    walking_speed: Optional[float] = None

    # Edge-Type-Specific Movement Speed (Experimental Branch V1) --
    # mirrors BehaviorProfileTemplate.stair_speed (behaviour_profile_
    # resolver/template.py); None means no stair-specific speed is
    # configured for this occupant, the only value any existing
    # registration path produces today.
    stair_speed: Optional[float] = None

    # node_id -> familiarity score (0 = unknown). Keyed by Node.id,
    # never by an engineering object -- a profile never needs to
    # touch Zone/Door/Exit directly.
    familiarity: Dict[str, float] = field(default_factory=dict)

    compliance_level: float = 1.0

    role: Role = Role.INDEPENDENT
    group_id: Optional[str] = None

    # Open-ended bag for future, unanticipated traits. Safe here
    # (unlike the dynamic_state field once considered for Node/Edge)
    # because a BehaviorProfile has no rebuild-from-source-of-truth
    # lifecycle -- it is owned by the Behavior Layer for the run's
    # duration, not silently discarded and reconstructed.
    traits: Dict[str, Any] = field(default_factory=dict)
