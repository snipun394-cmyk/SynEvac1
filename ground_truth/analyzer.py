from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from models.building import Building
from scenario.scenario import Scenario
from simulation_runtime.result import TickResult
from simulator.multi_agent_result import MultiAgentSimulationResult

from ground_truth import (
    bottleneck,
    decision_events,
    evacuation_metrics,
    human_behavior,
    occupant_attribute_outcomes,
    profile_outcomes,
    recommendations,
    rescue_outcomes,
    risk_analysis,
)
from ground_truth.labels import GroundTruth


@dataclass(frozen=True)
class SimulationArtifacts:

    # Everything the AI Ground Truth Generator needs from one completed
    # simulation -- nothing here is generated, validated, or simulated
    # by this package. tick_results is optional (defaults to empty):
    # every Fire/congestion-duration label that needs it is simply left
    # absent when it isn't supplied, never estimated in its place.

    scenario: Scenario
    building: Building
    movement_result: MultiAgentSimulationResult
    tick_results: Tuple[TickResult, ...] = field(default_factory=tuple)

    # Human Decision Engine Phase 6 -- additive. A plain list of
    # human_decision_engine.events.DecisionEvent.to_dict()'s own shape,
    # supplied only by a caller that used behaviour_profile_resolver.
    # dynamic_registrar.register_population_dynamic() this run; empty
    # (the default) for every scenario using the scenario-authored
    # assistance/rescue path, exactly like tick_results' own "simply
    # absent when not supplied" convention above.
    decision_events: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)


# =====================================================
# Deterministic Building enumeration -- same convention
# dataset_builder/schema.py already establishes (walk
# Building.ordered_floors(), then each Floor's own collection in
# insertion order), reimplemented here rather than imported so this
# package has no dependency on its sibling.
# =====================================================


def ordered_zones(building: Building) -> List:

    return [zone for floor in building.ordered_floors() for zone in floor.zones]


def ordered_doors(building: Building) -> List:

    return [door for floor in building.ordered_floors() for door in floor.doors]


def ordered_exits(building: Building) -> List:

    return [exit_obj for floor in building.ordered_floors() for exit_obj in floor.exits]


def ordered_stairs(building: Building) -> List:

    return [stair for floor in building.ordered_floors() for stair in floor.stairs]


def ordered_detectors(building: Building) -> List:

    return [detector for floor in building.ordered_floors() for detector in floor.detectors]


def ordered_cameras(building: Building) -> List:

    return [camera for floor in building.ordered_floors() for camera in floor.cameras]


# =====================================================


def analyze(artifacts: SimulationArtifacts) -> GroundTruth:

    scenario = artifacts.scenario
    building = artifacts.building
    movement_result = artifacts.movement_result
    tick_results = artifacts.tick_results

    zones = ordered_zones(building)
    doors = ordered_doors(building)
    exits = ordered_exits(building)
    stairs = ordered_stairs(building)
    detectors = ordered_detectors(building)
    cameras = ordered_cameras(building)

    fields = {
        "scenario_id": scenario.metadata.scenario_id,
        "definition_id": scenario.metadata.definition_id,
    }

    fields.update(evacuation_metrics.compute_evacuation_metrics(scenario, movement_result))

    zone_route_stats = evacuation_metrics.compute_zone_route_stats(
        scenario, movement_result, zones,
    )
    fields["zone_route_stats"] = zone_route_stats

    congestion = bottleneck.compute_congestion(movement_result, zones, doors, exits, stairs)
    fields.update(congestion)

    engineering = bottleneck.compute_engineering_findings(movement_result, doors, exits, stairs)
    fields.update(engineering)

    zone_risk_scores = risk_analysis.compute_zone_risk_scores(
        scenario, movement_result, tick_results, zones,
    )
    exit_risk_scores = risk_analysis.compute_exit_risk_scores(movement_result, exits)
    stair_risk_scores = risk_analysis.compute_stair_risk_scores(movement_result, stairs)

    fields["zone_risk_scores"] = zone_risk_scores
    fields["exit_risk_scores"] = exit_risk_scores
    fields["stair_risk_scores"] = stair_risk_scores

    fire_summary = risk_analysis.compute_fire_summary(tick_results, zones, zone_risk_scores)
    fields.update(fire_summary)

    fields["recommendations"] = recommendations.generate_recommendations(
        zones=zones,
        doors=doors,
        detectors=detectors,
        cameras=cameras,
        zone_route_stats=zone_route_stats,
        engineering=engineering,
        congestion=congestion,
        zone_risk_scores=zone_risk_scores,
        stair_risk_scores=stair_risk_scores,
        maximum_hazard_zone=fire_summary["maximum_hazard_zone"],
        people_trapped=fields["people_trapped"],
        exit_open_lookup=_exit_open_lookup(scenario, exits),
    )

    behavior_records = human_behavior.compute_occupant_behavior(scenario, movement_result, tick_results)
    fields["occupant_behavior"] = [
        {
            "occupant_id": record.occupant_id,
            "dynamic_state": record.dynamic_state.name,
            "ever_helping": record.ever_helping,
            "ever_assisted": record.ever_assisted,
            "ever_fallen": record.ever_fallen,
            "ever_possible_injury": record.ever_possible_injury,
        }
        for record in behavior_records.values()
    ]
    fields.update(human_behavior.summarize_behavior_counts(behavior_records))

    fields.update(profile_outcomes.compute_profile_outcomes(scenario, movement_result))

    fields.update(
        occupant_attribute_outcomes.compute_occupant_attribute_outcomes(scenario, movement_result),
    )

    rescue_records = rescue_outcomes.compute_rescue_outcomes(scenario, movement_result)
    fields.update(rescue_outcomes.summarize_rescue_outcomes(rescue_records))

    fields.update(decision_events.summarize_decision_events(artifacts.decision_events))

    return GroundTruth(**fields)


# =====================================================


def _exit_open_lookup(scenario, exits) -> dict:

    lookup = {exit_obj.id: not exit_obj.is_blocked for exit_obj in exits}

    for state in scenario.exit_states:
        lookup[state.exit_id] = state.is_open

    return lookup
