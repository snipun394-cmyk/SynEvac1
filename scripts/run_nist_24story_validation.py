"""
Published Scenario Validation Campaign V3 -- NIST 24-story office building.

Recreates both instrumented stairs from Peacock, Hoskins & Kuligowski,
"Overall and local movement speeds during fire drill evacuations in
buildings up to 31 stories," Safety Science 50 (2012), Section 2.4,
Table 2/3:

- Stair A: exits at floor 2 into a front lobby (via a door), then out
  the front of the building -- a lobby-merge topology, like the
  18-story building's Stairs 3/7/12.
- Stair B: continues past floor 2 down to floor 1 and exits directly
  outside -- a direct-to-exit topology, like Stair 1 of the 18-story
  building and the entire 10-story building.

Unlike the 18-story building, this building's own published geometry
happens to have BOTH stairs agree on the floor-3-to-2 transition
(30 steps for both), so no shared-floor-height conflict arises the way
it did between Stairs 1 and 7 in the 18-story recreation.

Repeats V1/V2's exact methodology unchanged (scenario_pipeline,
calibration_benchmark.simulation_seam with production defaults,
research_framework.statistics), including V2's own disclosed
methodology improvement (FireDefinition.allowed_ignition_zone_ids
restricted to one isolated, unoccupied zone).

No stair model, movement model, or capacity model source file is
modified by this script.
"""

import json
import os
import statistics as pystats

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from scenario_definition import FireDefinition, FixedValue, OccupantDefinition, ScenarioDefinition
from scenario_pipeline import run_batch_pipeline

from calibration_benchmark.simulation_seam import run_with_overrides
from calibration_benchmark.metrics import extract_metrics

from simulator.capacity import StairCapacityModel
from research_framework.statistics import confidence_interval


# =====================================================
# PHASE 1 -- published parameters (Known), disclosed assumptions (Estimated).
# =====================================================

FLOOR_COUNT = 24                        # Known
STAIR_WIDTH_M = 1.12                    # Known -- Table 2 (44 in)
EXIT_WIDTH_M = 0.91                     # Known -- Table 2
LOBBY_DOOR_WIDTH_M = 0.91                # Estimated -- not separately published for this
                                          # building's own Stair-A lobby door; reuses the same
                                          # assumption already made for the 18-story building.
RISER_M = 0.178                          # Known -- Table 2 (178 mm)

STEPS_UPPER = 20                         # Known -- floors 24 to 4 (both stairs)
STEPS_FINAL = 30                         # Known -- floor 3 to 2 (both stairs agree, unlike the
                                          # 18-story case)

RISE_UPPER_M = STEPS_UPPER * RISER_M     # 3.56 m -- floors 3..23
RISE_FINAL_M = STEPS_FINAL * RISER_M     # 5.34 m -- floor 2 (shared by both stairs)

STAIRA_OCCUPANTS = 249                   # Known -- Table 2
STAIRB_OCCUPANTS = 356                   # Known -- Table 2

STAIRA_OCCUPIED_FLOORS = list(range(3, FLOOR_COUNT + 1))   # floors 3-24 (22 floors) -- Known
                                                             # topology (floor 2 is Stair A's own
                                                             # lobby-exit landing, not occupied)
STAIRB_OCCUPIED_FLOORS = list(range(2, FLOOR_COUNT + 1))   # floors 2-24 (23 floors) -- Known
                                                             # topology (Stair B continues past
                                                             # floor 2 to its own direct exit at
                                                             # floor 1)


def _split_evenly(total, n):
    base, remainder = divmod(total, n)
    return [base + 1 if i < remainder else base for i in range(n)]


def build_nist_24story_building() -> Building:

    floors = []

    for floor_number in range(1, FLOOR_COUNT + 1):

        floor_id = f"floor-{floor_number}"
        display_order = floor_number - 1

        zones, exits, doors, staircases = [], [], [], []

        if floor_number == 1:

            zones.append(Zone(id="zone-fire-only", name="(Unoccupied, ignition-only zone)", x=-50.0, y=-50.0, width=1.0, height=1.0))

            # Stair B's own direct exit landing.
            zone_b1 = Zone(id="zone-b-1", name="Stair B Area F1", x=0.0, y=0.0, width=5.0, height=5.0)
            zones.append(zone_b1)
            exits.append(Exit(id="exit-b", zone_id="zone-b-1", width=EXIT_WIDTH_M))

        if floor_number >= 2:

            zone_a = Zone(id=f"zone-a-{floor_number}", name=f"Stair A Area F{floor_number}", x=0.0, y=0.0, width=5.0, height=5.0)
            zones.append(zone_a)

            if floor_number == 2:
                # Stair A's own lobby-exit landing.
                lobby_zone = Zone(id="zone-lobby", name="Front Lobby (Floor 2)", x=40.0, y=0.0, width=8.0, height=8.0)
                zones.append(lobby_zone)
                doors.append(Door(id="door-a-lobby", floor_id=floor_id, zone_a_id="zone-a-2", zone_b_id="zone-lobby", width=LOBBY_DOOR_WIDTH_M, normally_open=True))
                exits.append(Exit(id="exit-lobby", zone_id="zone-lobby", width=EXIT_WIDTH_M))
            else:
                staircases.append(Staircase(id=f"stair-a-{floor_number}", from_floor_id=floor_id, to_floor_id=f"floor-{floor_number - 1}", from_zone_id=f"zone-a-{floor_number}", to_zone_id=f"zone-a-{floor_number - 1}", width=STAIR_WIDTH_M))

            zone_b = Zone(id=f"zone-b-{floor_number}", name=f"Stair B Area F{floor_number}", x=20.0, y=0.0, width=5.0, height=5.0)
            zones.append(zone_b)
            staircases.append(Staircase(id=f"stair-b-{floor_number}", from_floor_id=floor_id, to_floor_id=f"floor-{floor_number - 1}", from_zone_id=f"zone-b-{floor_number}", to_zone_id=f"zone-b-{floor_number - 1}", width=STAIR_WIDTH_M))

        if floor_number == 1:
            height = RISE_FINAL_M
        elif floor_number == 2:
            height = RISE_FINAL_M
        elif floor_number < FLOOR_COUNT:
            height = RISE_UPPER_M
        else:
            height = 3.0

        floors.append(Floor(name=f"Floor {floor_number}", id=floor_id, display_order=display_order, height=height, zones=zones, exits=exits, doors=doors, stairs=staircases))

    return Building(name="NIST 24-Story Office Building (Peacock et al. 2012)", id="nist-24story-v1", floors=floors)


def build_nist_24story_definition() -> ScenarioDefinition:

    occupancy, behaviour = {}, {}

    for index, floor_number in enumerate(STAIRA_OCCUPIED_FLOORS):
        zone_id = f"zone-a-{floor_number}"
        occupancy[zone_id] = FixedValue(_split_evenly(STAIRA_OCCUPANTS, len(STAIRA_OCCUPIED_FLOORS))[index])
        behaviour[zone_id] = FixedValue("Adult_Default")

    for index, floor_number in enumerate(STAIRB_OCCUPIED_FLOORS):
        zone_id = f"zone-b-{floor_number}"
        occupancy[zone_id] = FixedValue(_split_evenly(STAIRB_OCCUPANTS, len(STAIRB_OCCUPIED_FLOORS))[index])
        behaviour[zone_id] = FixedValue("Adult_Default")

    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(300.0), allowed_ignition_zone_ids=frozenset(["zone-fire-only"])),
        occupant=OccupantDefinition(occupancy_distribution=occupancy, behaviour_profile_distribution=behaviour),
    )


DEFINITION_ID = "nist-24story-validation-v1"
MASTER_SEED = 20260804


def run_campaign(n_seeds: int, dt: float = 1.0, capacity_model=None, congestion_model=None, registry=None):

    building = build_nist_24story_building()
    definition = build_nist_24story_definition()
    batch = run_batch_pipeline(definition, DEFINITION_ID, building, MASTER_SEED, n_seeds)

    results = []
    for scenario in batch.scenarios:
        movement_result, ground_truth, building_copy = run_with_overrides(scenario, building, registry=registry, capacity_model=capacity_model, congestion_model=congestion_model, dt=dt)
        metric_capacity_model = capacity_model or StairCapacityModel()
        sample = extract_metrics(scenario.metadata.scenario_id, ground_truth, movement_result, building_copy, metric_capacity_model)
        results.append({
            "scenario_id": scenario.metadata.scenario_id,
            "total_evacuation_time": ground_truth.total_evacuation_time,
            "people_evacuated": ground_truth.people_evacuated,
            "people_trapped": ground_truth.people_trapped,
            "unreachable_occupants": ground_truth.unreachable_occupants,
            "peak_congestion_value": ground_truth.peak_congestion_value,
            "congestion_duration": ground_truth.congestion_duration,
            "peak_occupancy_ratio": sample.peak_occupancy_ratio,
            "exit_utilization_balance": sample.exit_utilization_balance,
            "exits_underutilized": list(ground_truth.exits_underutilized),
            "exits_exceeding_capacity": list(ground_truth.exits_exceeding_capacity),
            "doors_that_became_bottlenecks": list(ground_truth.doors_that_became_bottlenecks),
        })
    return results


def summarize(results):

    evac_times = [r["total_evacuation_time"] for r in results if r["total_evacuation_time"] is not None]
    ci = confidence_interval(evac_times)
    return {
        "n_runs": len(results), "n_with_evacuation_time": len(evac_times),
        "evacuation_time_mean": ci.mean, "evacuation_time_ci_lower": ci.lower, "evacuation_time_ci_upper": ci.upper,
        "evacuation_time_min": min(evac_times) if evac_times else None, "evacuation_time_max": max(evac_times) if evac_times else None,
        "evacuation_time_stdev": pystats.pstdev(evac_times) if len(evac_times) > 1 else None,
        "mean_people_evacuated": pystats.fmean(r["people_evacuated"] for r in results),
        "mean_people_trapped": pystats.fmean(r["people_trapped"] for r in results),
        "mean_peak_congestion_value": pystats.fmean(r["peak_congestion_value"] for r in results if r["peak_congestion_value"] is not None) if any(r["peak_congestion_value"] is not None for r in results) else None,
        "mean_peak_occupancy_ratio": pystats.fmean(r["peak_occupancy_ratio"] for r in results if r["peak_occupancy_ratio"] is not None) if any(r["peak_occupancy_ratio"] is not None for r in results) else None,
    }


def trace_worst_occupant(zone_prefix: str, dt: float = 1.0):

    building = build_nist_24story_building()
    definition = build_nist_24story_definition()
    batch = run_batch_pipeline(definition, DEFINITION_ID, building, MASTER_SEED, 1)
    scenario = batch.scenarios[0]
    movement_result, ground_truth, _ = run_with_overrides(scenario, building, dt=dt)

    candidates = [t for t in movement_result.occupants.values() if t.state.name == "ARRIVED" and t.route is not None and t.route.edges and t.route.edges[0].id.startswith(f"stair-{zone_prefix}-")]
    worst = max(candidates, key=lambda t: t.steps[-1].end_time if t.steps else 0)

    trace = [{"edge_id": s.edge.id, "edge_type": str(s.edge.edge_type), "start_time": s.start_time, "end_time": s.end_time, "queue_wait_time": getattr(s, "queue_wait_time", 0.0) or 0.0} for s in worst.steps]

    return {"final_arrival_time": worst.steps[-1].end_time, "n_steps": len(worst.steps), "total_queue_wait": sum(s["queue_wait_time"] for s in trace), "trace": trace}


def main():

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    official_results = run_campaign(n_seeds=40, dt=1.0)
    official_summary = summarize(official_results)

    from calibration_benchmark.candidates import CapacityWidthCandidate
    diagnostic_candidate = CapacityWidthCandidate(candidate_people_per_meter_of_width=8.0, dataset_source="Diagnostic-only, root-cause re-check (same value as V1/V2)", rationale="Repeat the same diagnostic on the 24-story building.", stair_specific=True)
    diagnostic_results = run_campaign(n_seeds=15, dt=1.0, capacity_model=diagnostic_candidate.candidate_capacity_model())
    diagnostic_summary = summarize(diagnostic_results)

    stairA_trace = trace_worst_occupant("a")
    stairB_trace = trace_worst_occupant("b")

    with open(os.path.join(output_dir, "nist_24story_validation_v1_raw_results.json"), "w", encoding="utf-8") as handle:
        json.dump({"official_summary": official_summary, "official_runs": official_results, "diagnostic_summary": diagnostic_summary, "diagnostic_runs": diagnostic_results, "stairA_trace": stairA_trace, "stairB_trace": stairB_trace}, handle, indent=2)

    print("OFFICIAL:", json.dumps(official_summary, indent=2))
    print("DIAGNOSTIC:", json.dumps(diagnostic_summary, indent=2))
    print("Stair A trace final/queue_wait:", stairA_trace["final_arrival_time"], stairA_trace["total_queue_wait"])
    print("Stair B trace final/queue_wait:", stairB_trace["final_arrival_time"], stairB_trace["total_queue_wait"])


if __name__ == "__main__":
    main()
