from dataclasses import dataclass, field
from typing import Any, Dict

from simulator.decision import ActionType


@dataclass
class ActionIntent:

    # Decision Stage's output -- ephemeral, internal to the Behavior
    # Layer's own pipeline, never crosses into Simulation (unlike
    # BehaviorDecision, which is what Navigation Stage eventually
    # produces from this).

    occupant_id: str
    action_type: ActionType

    # Set by the DecisionStrategy itself, not inferred from a
    # hardcoded action_type table -- keeps new action types free to
    # decide this for themselves without touching the orchestrator.
    requires_movement: bool

    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionStrategy:

    # Decision Stage interface: "what does this occupant intend to
    # do" -- mirrors CostModel/Heuristic/CapacityModel/CongestionModel's
    # established pattern.

    def decide(self, context) -> ActionIntent:

        raise NotImplementedError


class AlwaysEvacuateDecisionStrategy(DecisionStrategy):

    # The trivial default -- proves the interface without implementing
    # any real behavior. Every occupant simply intends to evacuate,
    # exactly what MultiAgentSimulation.evacuate() already does today
    # when nothing else is plugged in.

    def decide(self, context) -> ActionIntent:

        return ActionIntent(
            occupant_id=context.profile.occupant_id,
            action_type=ActionType.EVACUATE,
            requires_movement=True,
        )
