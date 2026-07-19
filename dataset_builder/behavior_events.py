import json

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from scenario.scenario import Scenario
from simulator.decision import BehaviorDecision


@dataclass(frozen=True)
class BehaviorEventsRun:

    # Dataset Builder V1/V2's own sibling shape (SimulationRun/
    # TimelineRun), applied to one more already-completed run artifact:
    # the ordered BehaviorDecision objects the Human Behavior Layer
    # already produced and submitted to Simulation this run
    # (behavior/orchestrator.HumanBehaviorLayer._decisions_so_far) --
    # nothing here is generated, validated, or simulated by this
    # package; it is only ever handed in by the caller once
    # registration has already finished.

    scenario: Scenario
    behavior_decisions: Tuple[BehaviorDecision, ...]


# =====================================================
# Behavior Events -- one row per occupant's own final, resolved
# BehaviorDecision. BehaviorDecision's own docstring already invites
# exactly this ("a caller that wants a decision timeline can simply
# keep every BehaviorDecision ever submitted for an occupant... that
# history is exactly what... dataset generation... would consume") --
# this module is that caller. metadata is serialized as a JSON string
# (a CSV row cannot honestly carry a nested mapping); route length is
# reported as an edge count, never the geometry itself, matching every
# other CSV column in this package that summarizes a Route rather than
# embedding it whole.
# =====================================================


def extract_behavior_event_rows(run: BehaviorEventsRun) -> List[Dict[str, Any]]:

    scenario_id = run.scenario.metadata.scenario_id

    rows: List[Dict[str, Any]] = []

    for decision in run.behavior_decisions:

        rows.append(
            {
                "scenario_id": scenario_id,
                "occupant_id": decision.occupant_id,
                "action_type": decision.action_type.name,
                "start_id": decision.start_id,
                "goal_id": decision.goal_id,
                "route_edge_count": len(decision.route.edges) if decision.route is not None else None,
                "route_unavailable": decision.route_unavailable,
                "walking_speed": decision.walking_speed,
                "depart_time": decision.depart_time,
                "metadata": json.dumps(dict(decision.metadata), sort_keys=True),
            }
        )

    return rows
