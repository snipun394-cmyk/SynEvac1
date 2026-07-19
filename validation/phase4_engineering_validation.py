"""
Phase 4 -- Engineering Validation.

Hand-designed scenarios with a known-correct expected answer (same
idiom the existing pytest suite already uses), run through the real
production pipeline, asserting the platform's own outputs match what
the building's geometry/topology dictates. Does not modify any
existing package -- only calls its public API and inspects results.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from scenario import Scenario, ScenarioEvent, ScenarioFire, ScenarioMetadata, ScenarioOccupant
from scenario_runner import run as run_scenario
from behaviour_profile_resolver import register_occupants

from ai_decision.engine import AIDecisionEngine

from simulation_runtime import SimulationRuntime

from ground_truth.analyzer import SimulationArtifacts, analyze
from dataset_builder.timeline import TimelineRun, extract_timeline_rows

from decision_policy.policy import DecisionInputs, generate_policy
from decision_policy import zone_policy, exit_policy, stair_policy

CHECKS = []


def check(name):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


def _metadata(**overrides):
    defaults = dict(
        scenario_id="val-scn", definition_id="val-def", definition_content_hash="hash",
        generation_version="validation/1", seed=1, created_at="2026-07-15T00:00:00",
    )
    defaults.update(overrides)
    return ScenarioMetadata(**defaults)


# =====================================================
# Check 1: recommended exit matches the geometrically closer exit.
# =====================================================


@check("recommended_exit_matches_closer_exit")
def check_recommended_exit():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Start", x=0.0, y=0.0, width=2.0, height=2.0),
            Zone(id="zone-near", name="Near", x=10.0, y=0.0, width=2.0, height=2.0),
            Zone(id="zone-far", name="Far", x=100.0, y=0.0, width=2.0, height=2.0),
        ],
        doors=[
            Door(id="door-near", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-near"),
            Door(id="door-far", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-far"),
        ],
        exits=[Exit(id="exit-near", zone_id="zone-near"), Exit(id="exit-far", zone_id="zone-far")],
    )
    building = Building(name="Check1", id="b1", floors=[floor1])

    scenario = Scenario(
        metadata=_metadata(scenario_id="check1"),
        occupants=(
            ScenarioOccupant(occupant_id="o1", zone_id="zone-1", floor_id="floor-1",
                              position=(1.0, 1.0), behaviour_profile_id="Staff_Default"),
        ),
        fire=ScenarioFire(ignition_zone_id="zone-far", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 200.0}),
        events=(),
    )

    context = run_scenario(scenario, building)
    register_occupants(context)
    engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, engine, dt=5.0)
    tick_results = runtime.run()

    gt = analyze(SimulationArtifacts(scenario=scenario, building=building,
                                      movement_result=runtime.movement_result, tick_results=tick_results))
    timeline_rows = extract_timeline_rows(TimelineRun(scenario=scenario, building=building,
                                                       movement_result=runtime.movement_result,
                                                       tick_results=tick_results))
    policy = generate_policy(DecisionInputs(building=building, scenario=scenario, ground_truth=gt,
                                             timeline_rows=tuple(timeline_rows)))

    route_stats = {e["zone_id"]: e for e in gt.zone_route_stats}
    preferred_exit = route_stats.get("zone-1", {}).get("preferred_exit")

    zone_decision = next((d for d in policy.zone_decisions if d["zone_id"] == "zone-1"), None)
    recommended_exit = zone_decision.get("recommended_exit") if zone_decision else None

    ok = preferred_exit == "exit-near" and recommended_exit == "exit-near"
    return ok, {"preferred_exit": preferred_exit, "recommended_exit": recommended_exit}


# =====================================================
# Check 2: recommended stair matches the only reachable route (single
# floor-2 exit only reachable via one stair).
# =====================================================


@check("recommended_stair_matches_only_viable_route")
def check_recommended_stair():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Start", x=0.0, y=0.0, width=2.0, height=2.0)],
        stairs=[Staircase(id="stair-1", from_zone_id="zone-1", to_zone_id="zone-2", to_floor_id="floor-2")],
    )
    floor2 = Floor(
        name="Upper", id="floor-2",
        zones=[Zone(id="zone-2", name="Landing", x=0.0, y=0.0, width=2.0, height=2.0)],
        exits=[Exit(id="exit-1", zone_id="zone-2")],
    )
    building = Building(name="Check2", id="b2", floors=[floor1, floor2])

    scenario = Scenario(
        metadata=_metadata(scenario_id="check2"),
        occupants=(
            ScenarioOccupant(occupant_id="o1", zone_id="zone-1", floor_id="floor-1",
                              position=(1.0, 1.0), behaviour_profile_id="Staff_Default"),
        ),
        fire=ScenarioFire(ignition_zone_id="zone-1", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 400.0}),
        events=(),
    )

    context = run_scenario(scenario, building)
    register_occupants(context)
    engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, engine, dt=5.0)
    tick_results = runtime.run()

    gt = analyze(SimulationArtifacts(scenario=scenario, building=building,
                                      movement_result=runtime.movement_result, tick_results=tick_results))
    timeline_rows = extract_timeline_rows(TimelineRun(scenario=scenario, building=building,
                                                       movement_result=runtime.movement_result,
                                                       tick_results=tick_results))
    policy = generate_policy(DecisionInputs(building=building, scenario=scenario, ground_truth=gt,
                                             timeline_rows=tuple(timeline_rows)))

    route_stats = {e["zone_id"]: e for e in gt.zone_route_stats}
    preferred_stair = route_stats.get("zone-1", {}).get("preferred_stair")
    zone_decision = next((d for d in policy.zone_decisions if d["zone_id"] == "zone-1"), None)
    recommended_stair = zone_decision.get("recommended_stair") if zone_decision else None

    ok = preferred_stair == "stair-1" and recommended_stair == "stair-1"
    return ok, {"preferred_stair": preferred_stair, "recommended_stair": recommended_stair}


# =====================================================
# Check 3: bottleneck prediction identifies a deliberately narrow,
# heavily-used door as a bottleneck.
# =====================================================


@check("bottleneck_prediction_identifies_narrow_door")
def check_bottleneck():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Start", x=0.0, y=0.0, width=2.0, height=2.0),
            Zone(id="zone-2", name="Egress", x=10.0, y=0.0, width=2.0, height=2.0),
        ],
        doors=[Door(id="door-narrow", normally_open=True, width=0.1, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-2")],
    )
    building = Building(name="Check3", id="b3", floors=[floor1])

    occupants = tuple(
        ScenarioOccupant(occupant_id=f"o{i}", zone_id="zone-1", floor_id="floor-1",
                          position=(1.0, 1.0), behaviour_profile_id="Staff_Default")
        for i in range(15)
    )

    scenario = Scenario(
        metadata=_metadata(scenario_id="check3"),
        occupants=occupants,
        fire=ScenarioFire(ignition_zone_id="zone-2", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 400.0}),
        events=(),
    )

    context = run_scenario(scenario, building)
    register_occupants(context)
    engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, engine, dt=2.0)
    tick_results = runtime.run()

    gt = analyze(SimulationArtifacts(scenario=scenario, building=building,
                                      movement_result=runtime.movement_result, tick_results=tick_results))

    ok = "door-narrow" in gt.doors_that_became_bottlenecks or gt.worst_door == "door-narrow"
    return ok, {
        "doors_that_became_bottlenecks": list(gt.doors_that_became_bottlenecks),
        "worst_door": gt.worst_door,
        "peak_congestion_location_id": gt.peak_congestion_location_id,
    }


# =====================================================
# Check 4: hazard spreads outward from the ignition zone -- the
# ignition zone itself must be the first hazardous zone.
# =====================================================


@check("hazard_spread_starts_at_ignition_zone")
def check_hazard_spread():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-origin", name="Origin", x=0.0, y=0.0, width=2.0, height=2.0),
            Zone(id="zone-adjacent", name="Adjacent", x=10.0, y=0.0, width=2.0, height=2.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-origin", zone_b_id="zone-adjacent")],
        exits=[Exit(id="exit-1", zone_id="zone-adjacent")],
    )
    building = Building(name="Check4", id="b4", floors=[floor1])

    scenario = Scenario(
        metadata=_metadata(scenario_id="check4"),
        occupants=(
            ScenarioOccupant(occupant_id="o1", zone_id="zone-adjacent", floor_id="floor-1",
                              position=(1.0, 1.0), behaviour_profile_id="Staff_Default"),
        ),
        fire=ScenarioFire(ignition_zone_id="zone-origin", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 60.0}),
        events=(),
    )

    context = run_scenario(scenario, building)
    register_occupants(context)
    engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, engine, dt=5.0, end_time=300.0)
    tick_results = runtime.run()

    gt = analyze(SimulationArtifacts(scenario=scenario, building=building,
                                      movement_result=runtime.movement_result, tick_results=tick_results))

    ok = gt.first_hazardous_zone == "zone-origin"
    return ok, {
        "first_hazardous_zone": gt.first_hazardous_zone,
        "hazard_spread_order": list(gt.hazard_spread_order),
        "maximum_hazard_zone": gt.maximum_hazard_zone,
    }


# =====================================================
# Check 5: ground truth internal consistency (occupant accounting,
# total_evacuation_time honesty).
# =====================================================


@check("ground_truth_occupant_accounting_is_consistent")
def check_ground_truth_consistency():

    building = _phase2_style_building()

    occupants = tuple(
        ScenarioOccupant(occupant_id=f"o{i}", zone_id="zone-1", floor_id="floor-1",
                          position=(1.0, 1.0), behaviour_profile_id="Staff_Default")
        for i in range(8)
    )

    scenario = Scenario(
        metadata=_metadata(scenario_id="check5"),
        occupants=occupants,
        fire=ScenarioFire(ignition_zone_id="zone-2", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 200.0}),
        events=(),
    )

    context = run_scenario(scenario, building)
    register_occupants(context)
    engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, engine, dt=5.0)
    tick_results = runtime.run()

    gt = analyze(SimulationArtifacts(scenario=scenario, building=building,
                                      movement_result=runtime.movement_result, tick_results=tick_results))

    total_occupants = len(occupants)
    accounting_ok = (gt.reachable_occupants + gt.unreachable_occupants) == total_occupants
    evacuated_ok = gt.people_evacuated <= gt.reachable_occupants

    arrival_times = [
        timeline.arrival_time for timeline in runtime.movement_result.occupants.values()
        if timeline.arrival_time is not None
    ]
    expected_total_time = max(arrival_times) if arrival_times else None
    time_ok = gt.total_evacuation_time == expected_total_time

    ok = accounting_ok and evacuated_ok and time_ok
    return ok, {
        "reachable_occupants": gt.reachable_occupants, "unreachable_occupants": gt.unreachable_occupants,
        "total_occupants": total_occupants, "people_evacuated": gt.people_evacuated,
        "total_evacuation_time": gt.total_evacuation_time, "expected_total_evacuation_time": expected_total_time,
    }


def _phase2_style_building():

    from models.camera import Camera
    from models.detector import Detector
    from models.obstacle import Obstacle

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
        obstacles=[Obstacle(id="obs-1", active=False)],
        cameras=[Camera(id="cam-1", active=True)],
        detectors=[Detector(id="det-1", active=True)],
        stairs=[Staircase(id="stair-1", from_zone_id="zone-1", to_zone_id="zone-3", to_floor_id="floor-2")],
    )
    floor2 = Floor(name="Upper", id="floor-2", zones=[Zone(id="zone-3", name="Attic")])
    return Building(name="Check5", id="b5", floors=[floor1, floor2])


# =====================================================
# Check 6: decision policy's own documented risk thresholds are
# actually the rule applied -- exercised over many risk_score values
# by directly probing zone_policy._zone_action's public sibling,
# compute_zone_decisions, via a synthetic GroundTruth-shaped object.
# =====================================================


@check("decision_policy_threshold_rule_matches_its_own_documented_constants")
def check_decision_policy_thresholds():

    class FakeGroundTruth:
        def __init__(self, risk_score, hazardous=False):
            self.zone_route_stats = ({"zone_id": "zone-1", "preferred_exit": "exit-1", "preferred_stair": None},)
            self.zone_risk_scores = ({"zone_id": "zone-1", "risk_score": risk_score},)
            self.hazard_spread_order = ("zone-1",) if hazardous else ()
            self.maximum_hazard_zone = None
            self.first_hazardous_zone = None
            self.exits_exceeding_capacity = ()
            self.worst_exit = None

    class FakeScenario:
        occupants = (
            ScenarioOccupant(occupant_id="o1", zone_id="zone-1", floor_id="floor-1",
                              position=(0, 0), behaviour_profile_id="Staff_Default"),
        )

    class FakeZone:
        id = "zone-1"

    results = {}
    all_ok = True

    for risk_score, hazardous, expected in [
        (0.9, False, zone_policy.SHELTER_IN_PLACE),
        (0.5, True, zone_policy.EVACUATE_IMMEDIATELY),
        (0.5, False, zone_policy.EVACUATE_IMMEDIATELY),
        (0.1, False, zone_policy.EVACUATE_IMMEDIATELY),
    ]:
        decisions = zone_policy.compute_zone_decisions(
            scenario=FakeScenario(), ground_truth=FakeGroundTruth(risk_score, hazardous), zones=[FakeZone()],
        )
        actual = decisions[0]["action"]
        results[f"risk={risk_score},hazardous={hazardous}"] = {"expected": expected, "actual": actual}
        all_ok = all_ok and (actual == expected)

    return all_ok, results


def main():

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    report = []

    for name, fn in CHECKS:
        try:
            ok, details = fn()
        except Exception as exc:
            ok, details = False, {"exception": repr(exc)}

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {json.dumps(details)}")
        report.append({"check": name, "status": status, "details": details})

    with open(os.path.join(output_dir, "phase4_engineering_validation.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    failed = [r for r in report if r["status"] == "FAIL"]
    print(f"\n{len(report) - len(failed)}/{len(report)} checks passed.")


if __name__ == "__main__":
    main()
