"""
Published Scenario Validation Campaign V2 -- NIST 18-story office building.

Recreates two of the four instrumented stairs from Peacock, Hoskins &
Kuligowski, "Overall and local movement speeds during fire drill
evacuations in buildings up to 31 stories," Safety Science 50 (2012),
Section 2.3, Table 2/3, Figure 8:

- Stair 1: direct-to-exit topology (continues to floor 1, exits directly
  out the rear of the building) -- the same topological SHAPE as the
  entire 10-story building validated in V1.
- Stair 7: lobby-merge topology (terminates at floor 5, exits through a
  0.91 m door into a shared ground-floor lobby, then out) -- a
  DIFFERENT topological shape, deliberately included to test whether
  V1's queueing finding is specific to a pure stair-only chain or also
  appears when a door/lobby merge is involved.

Stairs 3 and 12 (the other two instrumented stairs, sharing Stair 7's
own lobby-merge topology) are explicitly NOT recreated -- a disclosed
scope reduction to keep this campaign tractable; both are expected to
behave like Stair 7, not like Stair 1.

Repeats V1's exact methodology (scenario_pipeline.run_batch_pipeline +
calibration_benchmark.simulation_seam.run_with_overrides with SynEvac's
own unmodified production defaults + research_framework.statistics),
with one disclosed, non-code methodology improvement: FireDefinition.
forbidden_ignition_zone_ids (an already-existing field) is populated
with every occupied zone, avoiding V1's own 91% scenario-rejection rate
without touching any SynEvac source file.

No stair model, movement model, or any other SynEvac source file is
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
# PHASE 1 -- published parameters (Known), explicitly disclosed
# assumptions (Estimated), used to construct the Building/Scenario.
# =====================================================

FLOOR_COUNT = 18                        # Known -- "18-Story Office Building"
STAIR_WIDTH_M = 1.12                    # Known -- Table 2 (44 in, full width incl. handrails)
LOBBY_DOOR_WIDTH_M = 0.91                # Known -- "each through a 0.91 m (36 in) wide door" (Stairs 3/7/12)
FRONT_EXIT_WIDTH_M = 0.83                # Known -- Table 2, "Exit width" for the 18-story building
REAR_EXIT_WIDTH_M = 0.83                 # Estimated -- the paper gives one building-level exit width
                                          # (0.83 m) in Table 2 without separately breaking out Stair 1's
                                          # own rear exit from Stairs 3/7/12's front lobby exit; the same
                                          # value is used for both, disclosed as an assumption.
RISER_M = 0.191                          # Known -- Table 2 (191 mm)

STEPS_UPPER = 16                         # Known -- floors 18 to 4 (Stair 1) / floors 18 to 6 (Stair 7)
STEPS_STAIR1_LOWER = 18                  # Known -- Stair 1, floors 4-3, 3-2, 2-1
STEPS_STAIR7_FINAL = 19                  # Known -- Stair 7, floor 6 to 5 (transition into the lobby)

RISE_UPPER_M = STEPS_UPPER * RISER_M               # 3.056 m -- floors 6..17 (shared by both stairs)
RISE_STAIR1_LOWER_M = STEPS_STAIR1_LOWER * RISER_M  # 3.438 m -- Stair 1's own floors 1..3
RISE_STAIR7_FINAL_M = STEPS_STAIR7_FINAL * RISER_M  # 3.629 m -- Stair 7's true final segment (6 to 5)

# Known-required approximation (disclosed, not a stair-model change):
# floor_5.height is a single, shared Building-wide value. Stair 1's own
# published geometry treats floors 5-4 as part of its uniform 16-step
# ("floors 18 to 4") pattern (3.056 m), while Stair 7's true final
# segment (floor 6 to 5) is a distinct 19-step, 3.629 m rise. SynEvac's
# Floor.height/Building.floor_elevation model has no way to give the
# same floor two different heights depending on which stair references
# it. floor_5.height is set to the more common 16-step value (3.056 m,
# matching Stair 1's own real geometry exactly); Stair 7's own final
# flight is therefore modeled about 0.57 m (~16%) shorter than its
# published true rise -- a small, disclosed geometric approximation
# affecting exactly one of Stair 7's 13 flights, not a stair-model
# change.

STAIR1_OCCUPANTS = 255                   # Known -- Table 2
STAIR7_OCCUPANTS = 340                   # Known -- Table 2

STAIR1_OCCUPIED_FLOORS = list(range(2, FLOOR_COUNT + 1))   # floors 2-18 (17 floors) -- Known topology
STAIR7_OCCUPIED_FLOORS = list(range(6, FLOOR_COUNT + 1))   # floors 6-18 (13 floors) -- Known topology
                                                             # (Stair 7 does not extend below floor 5)


def _split_evenly(total, n):

    base, remainder = divmod(total, n)
    return [base + 1 if i < remainder else base for i in range(n)]


def build_nist_18story_building() -> Building:

    floors = []

    for floor_number in range(1, FLOOR_COUNT + 1):

        floor_id = f"floor-{floor_number}"
        display_order = floor_number - 1

        zones = []
        exits = []
        doors = []
        staircases = []

        # ---- Stair 1 (direct-to-exit) -- present on every floor 1-18 ----
        zone_1 = Zone(
            id=f"zone-1-{floor_number}", name=f"Stair 1 Area F{floor_number}",
            x=0.0, y=0.0, width=5.0, height=5.0,
        )
        zones.append(zone_1)

        if floor_number == 1:
            # A deliberately isolated, unoccupied, unconnected zone --
            # this recreation's topology has zero redundant routing
            # (every real zone is on someone's sole critical path), so
            # SOMETHING must be eligible for FireDefinition's required
            # ignition-zone sample (an empty allowed set is itself a
            # generator error). Adult_Default never uses hazard-aware
            # routing, so where a structurally-required, otherwise
            # inert fire ignites has no effect on any occupant's
            # behaviour in this scenario -- restricting ignition to
            # this one disconnected zone (via allowed_ignition_zone_ids
            # below) is a data-authoring choice, not a new capability.
            zones.append(Zone(id="zone-fire-only", name="(Unoccupied, ignition-only zone)", x=-50.0, y=-50.0, width=1.0, height=1.0))

        if floor_number == 1:
            exits.append(Exit(id="exit-1", zone_id="zone-1-1", width=REAR_EXIT_WIDTH_M))
        else:
            staircases.append(
                Staircase(
                    id=f"stair-1-{floor_number}", from_floor_id=floor_id, to_floor_id=f"floor-{floor_number - 1}",
                    from_zone_id=f"zone-1-{floor_number}", to_zone_id=f"zone-1-{floor_number - 1}",
                    width=STAIR_WIDTH_M,
                )
            )

        # ---- Stair 7 (lobby-merge) -- present on floors 5-18 only ----
        if floor_number >= 5:

            zone_7 = Zone(
                id=f"zone-7-{floor_number}", name=f"Stair 7 Area F{floor_number}",
                x=20.0, y=0.0, width=5.0, height=5.0,
            )
            zones.append(zone_7)

            if floor_number == 5:
                # Known -- "the fifth floor (ground level in front of
                # the building)": the lobby Stair 7 exits into is
                # itself AT floor 5 (this building's front ground
                # level sits at a different absolute floor number than
                # its rear ground level at floor 1, a real slope-sited
                # building's own internal numbering, not a modeling
                # error) -- so the lobby zone, its connecting Door, and
                # its own Exit are all placed on floor 5, a same-floor
                # connection, exactly as Door requires.
                lobby_zone = Zone(id="zone-lobby", name="Ground Floor Lobby (Front)", x=40.0, y=0.0, width=8.0, height=8.0)
                zones.append(lobby_zone)
                doors.append(
                    Door(
                        id="door-7-lobby", floor_id=floor_id, zone_a_id="zone-7-5", zone_b_id="zone-lobby",
                        width=LOBBY_DOOR_WIDTH_M, normally_open=True,
                    )
                )
                exits.append(Exit(id="exit-lobby", zone_id="zone-lobby", width=FRONT_EXIT_WIDTH_M))
            else:
                staircases.append(
                    Staircase(
                        id=f"stair-7-{floor_number}", from_floor_id=floor_id, to_floor_id=f"floor-{floor_number - 1}",
                        from_zone_id=f"zone-7-{floor_number}", to_zone_id=f"zone-7-{floor_number - 1}",
                        width=STAIR_WIDTH_M,
                    )
                )

        if floor_number == 1:
            height = RISE_STAIR1_LOWER_M
        elif floor_number in (2, 3):
            height = RISE_STAIR1_LOWER_M
        elif floor_number == 5:
            # Approximation disclosed above -- shared floor, Stair 1's
            # own value used.
            height = RISE_UPPER_M
        elif floor_number < FLOOR_COUNT:
            height = RISE_UPPER_M
        else:
            height = 3.0  # unused -- topmost floor

        floors.append(
            Floor(
                name=f"Floor {floor_number}", id=floor_id, display_order=display_order, height=height,
                zones=zones, exits=exits, doors=doors, stairs=staircases,
            )
        )

    return Building(name="NIST 18-Story Office Building (Peacock et al. 2012)", id="nist-18story-v1", floors=floors)


def build_nist_18story_definition() -> ScenarioDefinition:

    occupancy = {}
    behaviour = {}
    occupied_zone_ids = []

    stair1_per_floor = _split_evenly(STAIR1_OCCUPANTS, len(STAIR1_OCCUPIED_FLOORS))
    for index, floor_number in enumerate(STAIR1_OCCUPIED_FLOORS):
        zone_id = f"zone-1-{floor_number}"
        occupancy[zone_id] = FixedValue(stair1_per_floor[index])
        behaviour[zone_id] = FixedValue("Adult_Default")
        occupied_zone_ids.append(zone_id)

    stair7_per_floor = _split_evenly(STAIR7_OCCUPANTS, len(STAIR7_OCCUPIED_FLOORS))
    for index, floor_number in enumerate(STAIR7_OCCUPIED_FLOORS):
        zone_id = f"zone-7-{floor_number}"
        occupancy[zone_id] = FixedValue(stair7_per_floor[index])
        behaviour[zone_id] = FixedValue("Adult_Default")
        occupied_zone_ids.append(zone_id)

    # Methodology improvement over V1 (disclosed, no code changed --
    # FireDefinition.allowed_ignition_zone_ids already existed):
    # restricts ignition to the single isolated "zone-fire-only" zone
    # (see build_nist_18story_building()'s own comment) -- this
    # topology has no redundant routing, so any other zone risks
    # FIRE_ORIGIN_BLOCKS_EVACUATION the way V1's 10-story campaign
    # experienced at a 91% rejection rate.
    return ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=FixedValue(300.0),
            allowed_ignition_zone_ids=frozenset(["zone-fire-only"]),
        ),
        occupant=OccupantDefinition(occupancy_distribution=occupancy, behaviour_profile_distribution=behaviour),
    )


DEFINITION_ID = "nist-18story-validation-v1"
MASTER_SEED = 20260803


def run_campaign(n_seeds: int, dt: float = 1.0, capacity_model=None, congestion_model=None, registry=None):

    building = build_nist_18story_building()
    definition = build_nist_18story_definition()

    batch = run_batch_pipeline(definition, DEFINITION_ID, building, MASTER_SEED, n_seeds)

    results = []

    for scenario in batch.scenarios:

        movement_result, ground_truth, building_copy = run_with_overrides(
            scenario, building, registry=registry, capacity_model=capacity_model,
            congestion_model=congestion_model, dt=dt,
        )

        metric_capacity_model = capacity_model or StairCapacityModel()
        sample = extract_metrics(
            scenario.metadata.scenario_id, ground_truth, movement_result, building_copy, metric_capacity_model,
        )

        results.append({
            "scenario_id": scenario.metadata.scenario_id,
            "total_evacuation_time": ground_truth.total_evacuation_time,
            "people_evacuated": ground_truth.people_evacuated,
            "people_trapped": ground_truth.people_trapped,
            "reachable_occupants": ground_truth.reachable_occupants,
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
        "n_runs": len(results),
        "n_with_evacuation_time": len(evac_times),
        "evacuation_time_mean": ci.mean,
        "evacuation_time_ci_lower": ci.lower,
        "evacuation_time_ci_upper": ci.upper,
        "evacuation_time_min": min(evac_times) if evac_times else None,
        "evacuation_time_max": max(evac_times) if evac_times else None,
        "evacuation_time_stdev": pystats.pstdev(evac_times) if len(evac_times) > 1 else None,
        "mean_people_evacuated": pystats.fmean(r["people_evacuated"] for r in results),
        "mean_people_trapped": pystats.fmean(r["people_trapped"] for r in results),
        "mean_peak_congestion_value": pystats.fmean(
            r["peak_congestion_value"] for r in results if r["peak_congestion_value"] is not None
        ) if any(r["peak_congestion_value"] is not None for r in results) else None,
        "mean_peak_occupancy_ratio": pystats.fmean(
            r["peak_occupancy_ratio"] for r in results if r["peak_occupancy_ratio"] is not None
        ) if any(r["peak_occupancy_ratio"] is not None for r in results) else None,
    }


def trace_worst_occupant(zone_prefix: str, dt: float = 1.0):

    # Phase 4 diagnostic -- per-flight queue-wait trace for one
    # representative occupant on the given stair ("1" or "7"), same
    # methodology as V1's own single-occupant trace.

    building = build_nist_18story_building()
    definition = build_nist_18story_definition()
    batch = run_batch_pipeline(definition, DEFINITION_ID, building, MASTER_SEED, 1)
    scenario = batch.scenarios[0]
    movement_result, ground_truth, _ = run_with_overrides(scenario, building, dt=dt)

    candidates = [
        t for t in movement_result.occupants.values()
        if t.state.name == "ARRIVED" and t.route is not None and t.route.edges
        and t.route.edges[0].id.startswith(f"stair-{zone_prefix}-")
    ]
    worst = max(candidates, key=lambda t: t.steps[-1].end_time if t.steps else 0)

    trace = []
    for step in worst.steps:
        trace.append({
            "edge_id": step.edge.id,
            "edge_type": str(step.edge.edge_type),
            "start_time": step.start_time,
            "end_time": step.end_time,
            "queue_wait_time": getattr(step, "queue_wait_time", 0.0) or 0.0,
        })

    return {
        "final_arrival_time": worst.steps[-1].end_time,
        "n_steps": len(worst.steps),
        "total_queue_wait": sum(s["queue_wait_time"] for s in trace),
        "trace": trace,
    }


def main():

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    official_results = run_campaign(n_seeds=40, dt=1.0)
    official_summary = summarize(official_results)

    from calibration_benchmark.candidates import CapacityWidthCandidate

    diagnostic_candidate = CapacityWidthCandidate(
        candidate_people_per_meter_of_width=8.0,
        dataset_source="Diagnostic-only, root-cause re-check (same value as the V1 10-story diagnostic for direct comparability)",
        rationale="Repeat V1's own diagnostic on the 18-story building to check whether the same non-result holds.",
        stair_specific=True,
    )
    diagnostic_results = run_campaign(n_seeds=15, dt=1.0, capacity_model=diagnostic_candidate.candidate_capacity_model())
    diagnostic_summary = summarize(diagnostic_results)

    stair1_trace = trace_worst_occupant("1")
    stair7_trace = trace_worst_occupant("7")

    with open(os.path.join(output_dir, "nist_18story_validation_v1_raw_results.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "official_summary": official_summary, "official_runs": official_results,
                "diagnostic_summary": diagnostic_summary, "diagnostic_runs": diagnostic_results,
                "stair1_trace": stair1_trace, "stair7_trace": stair7_trace,
            },
            handle, indent=2,
        )

    print("OFFICIAL (SynEvac defaults):")
    print(json.dumps(official_summary, indent=2))
    print("\nDIAGNOSTIC (stair capacity raised 1 -> ~9, root-cause check only):")
    print(json.dumps(diagnostic_summary, indent=2))
    print("\nSTAIR 1 trace: final_arrival_time =", stair1_trace["final_arrival_time"], "total_queue_wait =", stair1_trace["total_queue_wait"])
    print("STAIR 7 trace: final_arrival_time =", stair7_trace["final_arrival_time"], "total_queue_wait =", stair7_trace["total_queue_wait"])


if __name__ == "__main__":
    main()
