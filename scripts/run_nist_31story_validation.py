"""
Published Scenario Validation Campaign V3 -- NIST 31-story office building.

Recreates both instrumented stairs (North, South) from Peacock, Hoskins
& Kuligowski, "Overall and local movement speeds during fire drill
evacuations in buildings up to 31 stories," Safety Science 50 (2012),
Section 2.5, Table 2/3.

Both stairs exit at floor 2 (this building's own internal numbering has
"street level" at floor 2, not floor 1 -- a real slope/entry-level
quirk, the same kind already encountered and disclosed for the
18-story building's floor-5 lobby). The North stair additionally
requires "a horizontal travel distance... to the exit of the building"
after reaching floor 2 -- modeled here as an extra Door + lobby zone,
the same lobby-transition pattern already used for the 18-/24-story
buildings' own lobby-exit stairs. The South stair exits directly from
its own floor-2 landing.

Disclosed, unavoidable approximation: the paper states the floor 4-to-3
transition is "larger" than the standard 18-step pattern because of a
horizontal transfer corridor around a mechanical floor, but gives no
exact step count or distance for it. No better published figure exists
to use in its place, so this recreation uses the SAME 18-step (3.24 m)
value as every other upper-floor transition for floor 3's own height --
an explicit, disclosed likely UNDERSTATEMENT of that one transition's
true distance, not a fabricated number.

Repeats V1/V2/V3's exact methodology unchanged. No stair, movement, or
capacity model source file is modified by this script.
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
# PHASE 1 -- published parameters (Known), disclosed assumptions (Estimated/Unavailable).
# Building modeled with floor "2" as the lowest Floor object (display_order 0),
# matching this building's own real ground/street level -- there is no
# occupied or exited "floor 1" in this recreation, consistent with the
# published topology (both stairs exit AT floor 2).
# =====================================================

FLOOR_COUNT = 31                         # Known -- floors 2 through 31 inclusive = 30 Floor objects
LOWEST_FLOOR_NUMBER = 2

STAIR_WIDTH_M = 1.38                     # Known -- Table 2 (54.25 in)
EXIT_WIDTH_M = 0.91                      # Known -- Table 2
NORTH_LOBBY_DOOR_WIDTH_M = 0.91          # Estimated -- the paper states the North stair needs
                                          # "a horizontal travel distance... to the exit" but gives
                                          # no door/corridor width; reuses the established
                                          # convention from the 18-/24-story recreations.
RISER_M = 0.178                          # Known -- Table 2 (178 mm)

STEPS_UPPER = 18                         # Known -- floors 31 to 4 (both stairs)
STEPS_FINAL = 27                         # Known -- floor 3 to 2 (both stairs)

RISE_UPPER_M = STEPS_UPPER * RISER_M     # 3.204 m -- floors 4..30
RISE_FINAL_M = STEPS_FINAL * RISER_M     # 4.806 m -- floor 2 (the shared final segment)
RISE_FLOOR3_APPROXIMATION_M = RISE_UPPER_M  # Estimated/Unavailable -- see module docstring;
                                             # the true 4-to-3 transition is disclosed by the
                                             # paper as "larger" than this, with no exact figure
                                             # published to replace it with.

NORTH_OCCUPANTS = 704                    # Known -- Table 2
SOUTH_OCCUPANTS = 538                    # Known -- Table 2

OCCUPIED_FLOORS = list(range(3, FLOOR_COUNT + 1))   # floors 3-31 (29 floors) -- Known topology,
                                                      # both stairs (floor 2 is the shared exit
                                                      # level, not occupied)


def _split_evenly(total, n):
    base, remainder = divmod(total, n)
    return [base + 1 if i < remainder else base for i in range(n)]


def build_nist_31story_building() -> Building:

    floors = []
    floor_numbers = list(range(LOWEST_FLOOR_NUMBER, FLOOR_COUNT + 1))

    for display_order, floor_number in enumerate(floor_numbers):

        floor_id = f"floor-{floor_number}"
        zones, exits, doors, staircases = [], [], [], []

        if floor_number == LOWEST_FLOOR_NUMBER:

            zones.append(Zone(id="zone-fire-only", name="(Unoccupied, ignition-only zone)", x=-50.0, y=-50.0, width=1.0, height=1.0))

            # South stair: direct exit at floor 2.
            zone_s = Zone(id="zone-south-2", name="South Stair Area F2", x=20.0, y=0.0, width=5.0, height=5.0)
            zones.append(zone_s)
            exits.append(Exit(id="exit-south", zone_id="zone-south-2", width=EXIT_WIDTH_M))

            # North stair: lands at floor 2, then a disclosed extra
            # horizontal travel segment (Door + lobby) to the actual exit.
            zone_n = Zone(id="zone-north-2", name="North Stair Area F2", x=0.0, y=0.0, width=5.0, height=5.0)
            zones.append(zone_n)
            lobby_zone = Zone(id="zone-north-lobby", name="North Stair Exit Corridor", x=-20.0, y=0.0, width=8.0, height=4.0)
            zones.append(lobby_zone)
            doors.append(Door(id="door-north-lobby", floor_id=floor_id, zone_a_id="zone-north-2", zone_b_id="zone-north-lobby", width=NORTH_LOBBY_DOOR_WIDTH_M, normally_open=True))
            exits.append(Exit(id="exit-north", zone_id="zone-north-lobby", width=EXIT_WIDTH_M))

        else:

            zone_n = Zone(id=f"zone-north-{floor_number}", name=f"North Stair Area F{floor_number}", x=0.0, y=0.0, width=5.0, height=5.0)
            zone_s = Zone(id=f"zone-south-{floor_number}", name=f"South Stair Area F{floor_number}", x=20.0, y=0.0, width=5.0, height=5.0)
            zones.extend([zone_n, zone_s])

            lower_floor_number = floor_numbers[display_order - 1]
            staircases.append(Staircase(id=f"stair-north-{floor_number}", from_floor_id=floor_id, to_floor_id=f"floor-{lower_floor_number}", from_zone_id=f"zone-north-{floor_number}", to_zone_id=f"zone-north-{lower_floor_number}", width=STAIR_WIDTH_M))
            staircases.append(Staircase(id=f"stair-south-{floor_number}", from_floor_id=floor_id, to_floor_id=f"floor-{lower_floor_number}", from_zone_id=f"zone-south-{floor_number}", to_zone_id=f"zone-south-{lower_floor_number}", width=STAIR_WIDTH_M))

        if floor_number == LOWEST_FLOOR_NUMBER:
            height = RISE_FINAL_M
        elif floor_number == 3:
            height = RISE_FLOOR3_APPROXIMATION_M
        elif floor_number < FLOOR_COUNT:
            height = RISE_UPPER_M
        else:
            height = 3.0

        floors.append(Floor(name=f"Floor {floor_number}", id=floor_id, display_order=display_order, height=height, zones=zones, exits=exits, doors=doors, stairs=staircases))

    return Building(name="NIST 31-Story Office Building (Peacock et al. 2012)", id="nist-31story-v1", floors=floors)


def build_nist_31story_definition() -> ScenarioDefinition:

    occupancy, behaviour = {}, {}

    north_per_floor = _split_evenly(NORTH_OCCUPANTS, len(OCCUPIED_FLOORS))
    south_per_floor = _split_evenly(SOUTH_OCCUPANTS, len(OCCUPIED_FLOORS))

    for index, floor_number in enumerate(OCCUPIED_FLOORS):
        occupancy[f"zone-north-{floor_number}"] = FixedValue(north_per_floor[index])
        behaviour[f"zone-north-{floor_number}"] = FixedValue("Adult_Default")
        occupancy[f"zone-south-{floor_number}"] = FixedValue(south_per_floor[index])
        behaviour[f"zone-south-{floor_number}"] = FixedValue("Adult_Default")

    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(300.0), allowed_ignition_zone_ids=frozenset(["zone-fire-only"])),
        occupant=OccupantDefinition(occupancy_distribution=occupancy, behaviour_profile_distribution=behaviour),
    )


DEFINITION_ID = "nist-31story-validation-v1"
MASTER_SEED = 20260805


def run_campaign(n_seeds: int, dt: float = 1.0, capacity_model=None, congestion_model=None, registry=None):

    building = build_nist_31story_building()
    definition = build_nist_31story_definition()
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

    building = build_nist_31story_building()
    definition = build_nist_31story_definition()
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
    diagnostic_candidate = CapacityWidthCandidate(candidate_people_per_meter_of_width=8.0, dataset_source="Diagnostic-only, root-cause re-check (same value as V1/V2/V3)", rationale="Repeat the same diagnostic on the 31-story building.", stair_specific=True)
    diagnostic_results = run_campaign(n_seeds=15, dt=1.0, capacity_model=diagnostic_candidate.candidate_capacity_model())
    diagnostic_summary = summarize(diagnostic_results)

    north_trace = trace_worst_occupant("north")
    south_trace = trace_worst_occupant("south")

    with open(os.path.join(output_dir, "nist_31story_validation_v1_raw_results.json"), "w", encoding="utf-8") as handle:
        json.dump({"official_summary": official_summary, "official_runs": official_results, "diagnostic_summary": diagnostic_summary, "diagnostic_runs": diagnostic_results, "north_trace": north_trace, "south_trace": south_trace}, handle, indent=2)

    print("OFFICIAL:", json.dumps(official_summary, indent=2))
    print("DIAGNOSTIC:", json.dumps(diagnostic_summary, indent=2))
    print("North trace final/queue_wait:", north_trace["final_arrival_time"], north_trace["total_queue_wait"])
    print("South trace final/queue_wait:", south_trace["final_arrival_time"], south_trace["total_queue_wait"])


if __name__ == "__main__":
    main()
