from dataclasses import dataclass, field
from typing import Dict, Optional

from navigation.graph import NavigationGraph
from pathfinding.engine import PathfindingEngine

from simulator.decision import BehaviorDecision
from simulator.multi_agent_result import MultiAgentSimulationResult

from behavior.profile import BehaviorProfile


@dataclass
class DecisionContext:

    # Read-only, passed into every DecisionStrategy/RouteChoiceStrategy/
    # PreMovementDelayStrategy call. Deliberately extensible: this is
    # the one place designed to grow additively (new optional fields)
    # as future Dynamic Hazard/AI systems are integrated, without
    # breaking existing strategy implementations that don't look at
    # those new fields.

    graph: NavigationGraph
    engine: PathfindingEngine
    profile: BehaviorProfile
    start_id: str

    # occupant_id -> the fully resolved BehaviorDecision already made
    # for them in this registration session, in resolution order --
    # what lets a follower's strategies see a leader's choice.
    decisions_so_far: Dict[str, BehaviorDecision] = field(default_factory=dict)

    # A previous run's result, for multi-pass/iterative behaviors
    # (e.g. approximating herding by biasing route choice using a
    # prior pass's peak_edge_occupancy) -- optional, None on a first
    # pass.
    prior_result: Optional[MultiAgentSimulationResult] = None
