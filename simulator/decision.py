from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, Mapping, Optional

from pathfinding.route import Route


class ActionType(Enum):

    # What an occupant intends to do -- this list is illustrative, not
    # exhaustive; new values are the natural extension point for new
    # behaviors, added here (not by growing BehaviorDecision's shape).
    # Simulation never branches on this value -- it is passed through
    # purely as descriptive metadata for reporting/dataset generation.

    EVACUATE = auto()
    WAIT = auto()
    HELP = auto()
    IGNORE = auto()
    RETURN_FOR_BELONGINGS = auto()
    FOLLOW_GROUP = auto()

    # Human Population & Assisted Evacuation Modeling Phase 3 --
    # additive, same "illustrative, not exhaustive" extension the
    # comment above already invites. ESCORT/PUSH_WHEELCHAIR/DRAG/CARRY
    # are the HELPER side of an assistance pairing (behavior_library.
    # assistance_strategies.AssistanceAwareDecisionStrategy); ASSISTED
    # is the receiving side, for any assistance type.
    ESCORT = auto()
    PUSH_WHEELCHAIR = auto()
    DRAG = auto()
    CARRY = auto()
    ASSISTED = auto()

    # Phase 4 -- Firefighter Modeling, kept in its own contiguous block
    # for discoverability (never used by, or referenced from, civilian
    # behavior_library code -- see behavior_library.firefighter_strategies).
    FIREFIGHTER_SEARCH = auto()
    FIREFIGHTER_RESCUE = auto()
    FIREFIGHTER_REPORT_HAZARD = auto()
    FIREFIGHTER_GUIDE = auto()


@dataclass(frozen=True)
class BehaviorDecision:

    # The immutable boundary contract between the Human Behavior Layer
    # and Simulation -- defined here, in simulator/, not in behavior/,
    # so that Simulation never has to import anything from behavior/
    # (same "each layer owns the contract type it accepts as input"
    # pattern as Route being owned by pathfinding/ and consumed by
    # simulator/).
    #
    # Instances are never modified after creation -- frozen=True makes
    # attribute reassignment raise, and __post_init__ below wraps
    # `metadata` in a MappingProxyType so even that nested container
    # can't be mutated in place. If an occupant's behavior changes
    # later (WAIT -> EVACUATE, EVACUATE -> FOLLOW_GROUP, ...), the
    # correct pattern is to construct a brand new BehaviorDecision and
    # submit it again -- never to mutate this one. See
    # MultiAgentSimulation.submit_decision() for how a later decision
    # for the same occupant_id supersedes an earlier one.
    #
    # Because each decision is a fully self-contained, immutable
    # snapshot, a caller that wants a decision *timeline* can simply
    # keep every BehaviorDecision ever submitted for an occupant, in
    # order -- nothing about this type needs to change to support
    # that. That history is exactly what future explainability
    # ("why did this occupant do X at time T"), dataset generation
    # (a labeled sequence of intents), and RL (a trace of actions
    # taken) would consume. None of that storage is built here --
    # only the immutability that makes it safe to build later.

    occupant_id: str
    action_type: ActionType
    start_id: str

    # None/None together means either a non-movement decision (WAIT,
    # IGNORE, ...) or a movement decision for which no route to any
    # exit exists -- route_unavailable disambiguates the two (see
    # below). Simulation treats the *absence* of a goal/route, with
    # route_unavailable False, as the signal to register the occupant
    # as stationary, never by inspecting action_type itself.
    goal_id: Optional[str] = None
    route: Optional[Route] = None

    # True only when the Navigation Stage was run (movement was
    # required) and found no route to any exit -- a structural
    # disconnection from every exit, not a choice to stay put. Set by
    # HumanBehaviorLayer.register(), the only place that still knows
    # whether movement was required once goal_id/route have already
    # collapsed to None/None. False (the default) preserves every
    # existing caller's behavior unchanged. See
    # docs/validation/technical_report.md §6.
    route_unavailable: bool = False

    walking_speed: Optional[float] = None

    # Edge-Type-Specific Movement Speed (Experimental Branch V1) --
    # mirrors BehaviorProfile.stair_speed; carried straight through
    # from profile.stair_speed by whichever registration path built
    # this decision (behavior/orchestrator.py, behaviour_profile_
    # resolver/dynamic_registrar.py), never derived here. None (the
    # default) means Simulation falls back to walking_speed for every
    # edge, including Stair -- today's exact, unchanged behavior.
    stair_speed: Optional[float] = None

    depart_time: Optional[float] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
