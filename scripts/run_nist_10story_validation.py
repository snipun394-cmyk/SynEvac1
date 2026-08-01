"""
Published Scenario Validation Campaign V1 -- NIST 10-story office building.

Recreates the 10-story office building fire drill from Peacock, Hoskins &
Kuligowski, "Overall and local movement speeds during fire drill
evacuations in buildings up to 31 stories," Safety Science 50 (2012)
1655-1664 (Section 2.2 / Table 2 / Table 3), using ONLY published
engineering parameters plus explicitly disclosed assumptions (see
docs/architecture/nist_10story_validation_v1.md Phase 1 table for the
full Known/Estimated/Unavailable breakdown).

This script builds no new SynEvac subsystem: it constructs a Building
directly from the same model classes SynEvac Builder itself produces
(models.building.Building/Floor/Zone/Exit/Staircase), configures a
Scenario via the existing scenario_definition schema, and runs the
campaign entirely through already-existing, already-tested machinery
(scenario_pipeline.run_batch_pipeline, calibration_benchmark.
simulation_seam.run_with_overrides -- called with every override left at
its SynEvac production default, i.e. no calibration candidate is applied
here at all -- and research_framework.statistics for the paired/
descriptive statistics). No SynEvac source file is modified.

Every occupant runs under SynEvac's own UNMODIFIED current defaults
(Adult_Default profile, DefaultCapacityModel/StairCapacityModel,
DefaultCongestionModel/StairAwareCongestionModel) -- this is a validation
of SynEvac's existing behavior against a real published drill, not a
pre-calibrated best-case run.
"""

import json
import os
import statistics as pystats

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from scenario_definition import FireDefinition, FixedValue, OccupantDefinition, ScenarioDefinition
from scenario_pipeline import run_batch_pipeline

from calibration_benchmark.simulation_seam import run_with_overrides
from calibration_benchmark.metrics import extract_metrics
from calibration_benchmark.metrics import compute_peak_occupancy_ratio  # noqa: F401 (used implicitly via extract_metrics)

from simulator.capacity import StairCapacityModel
from research_framework.statistics import confidence_interval


# =====================================================
# PHASE 1/2 -- published parameters (Known), explicitly disclosed
# assumptions (Estimated), and the resulting Building/Scenario. See the
# companion report for the full field-by-field Known/Estimated/
# Unavailable table; only the numbers actually used in construction are
# restated here, each tagged.
# =====================================================

FLOOR_COUNT = 10                       # Known -- "10-Story office Building"
STAIR_WIDTH_M = 1.27                   # Known -- Table 2, full width incl. handrails
EXIT_WIDTH_M = 0.91                    # Known -- Table 2
STEPS_UPPER = 22                       # Known -- floors 10 to 2, per transition
STEPS_LOWEST = 27                      # Known -- floor 2 to 1
RISER_M = 0.178                        # Known -- Table 2 (178 mm)

RISE_UPPER_M = STEPS_UPPER * RISER_M   # Known-derived: 3.916 m per floor, floors 3..10
RISE_LOWEST_M = STEPS_LOWEST * RISER_M  # Known-derived: 4.806 m, floor 1 to 2

STAIR_A_OCCUPANTS = 436                # Known -- Table 2 ("Evacuees (for each exit stair)")
STAIR_B_OCCUPANTS = 368                # Known -- Table 2

OCCUPIED_FLOORS = list(range(2, FLOOR_COUNT + 1))  # Estimated -- floor 1 occupants (if any)
# are not part of a stairwell descent and are excluded, consistent with
# the paper's own camera placement starting at the exit floor.


def _split_evenly(total, n):

    # Estimated -- the paper gives a per-stair TOTAL (436 / 368) for the
    # whole building but no per-floor breakdown for the 10-story case.
    # Splitting as evenly as an integer count allows (remainder assigned
    # to the first floors) is the least-assumption-injecting choice
    # available; it is not a claim about the real per-floor population.

    base, remainder = divmod(total, n)
    return [base + 1 if i < remainder else base for i in range(n)]


def build_nist_10story_building() -> Building:

    floors = []

    for floor_number in range(1, FLOOR_COUNT + 1):

        floor_id = f"floor-{floor_number}"
        display_order = floor_number - 1  # floor 1 = ground = display_order 0

        zone_a = Zone(
            id=f"zone-a-{floor_number}", name=f"Stair A Area F{floor_number}",
            x=0.0, y=0.0, width=5.0, height=5.0,
        )
        zone_b = Zone(
            id=f"zone-b-{floor_number}", name=f"Stair B Area F{floor_number}",
            x=20.0, y=0.0, width=5.0, height=5.0,
        )

        exits = []
        if floor_number == 1:
            # Known -- "both of which exit directly out of the building
            # on the first (or ground) floor."
            exits = [
                Exit(id="exit-a", zone_id="zone-a-1", width=EXIT_WIDTH_M),
                Exit(id="exit-b", zone_id="zone-b-1", width=EXIT_WIDTH_M),
            ]

        staircases = []
        if floor_number > 1:

            lower_floor_id = f"floor-{floor_number - 1}"

            staircases = [
                Staircase(
                    id=f"stair-a-{floor_number}", from_floor_id=floor_id, to_floor_id=lower_floor_id,
                    from_zone_id=f"zone-a-{floor_number}", to_zone_id=f"zone-a-{floor_number - 1}",
                    width=STAIR_WIDTH_M,
                ),
                Staircase(
                    id=f"stair-b-{floor_number}", from_floor_id=floor_id, to_floor_id=lower_floor_id,
                    from_zone_id=f"zone-b-{floor_number}", to_zone_id=f"zone-b-{floor_number - 1}",
                    width=STAIR_WIDTH_M,
                ),
            ]

        # Known-derived: floor_number's own height determines the RISE
        # UP FROM this floor to the next one (Building.floor_elevation()
        # sums every lower floor's own height) -- floor 1's height is the
        # 27-step (4.806 m) segment to floor 2; floors 2-9's height is
        # the 22-step (3.916 m) segment to the floor above. Floor 10's
        # height is unused (nothing sits above it) and left at the
        # model's own default.
        if floor_number == 1:
            height = RISE_LOWEST_M
        elif floor_number < FLOOR_COUNT:
            height = RISE_UPPER_M
        else:
            height = 3.0  # unused -- topmost floor

        floors.append(
            Floor(
                name=f"Floor {floor_number}", id=floor_id, display_order=display_order, height=height,
                zones=[zone_a, zone_b], exits=exits, stairs=staircases,
            )
        )

    return Building(name="NIST 10-Story Office Building (Peacock et al. 2012)", id="nist-10story-v1", floors=floors)


def build_nist_10story_definition() -> ScenarioDefinition:

    occupancy = {}
    behaviour = {}

    stair_a_per_floor = _split_evenly(STAIR_A_OCCUPANTS, len(OCCUPIED_FLOORS))
    stair_b_per_floor = _split_evenly(STAIR_B_OCCUPANTS, len(OCCUPIED_FLOORS))

    for index, floor_number in enumerate(OCCUPIED_FLOORS):

        occupancy[f"zone-a-{floor_number}"] = FixedValue(stair_a_per_floor[index])
        occupancy[f"zone-b-{floor_number}"] = FixedValue(stair_b_per_floor[index])

        # Estimated -- the paper reports gender mix and a handful of
        # other per-occupant attributes but no formal "behaviour
        # profile" concept; Adult_Default (SynEvac's own generic,
        # able-bodied adult profile, unmodified default parameters) is
        # used for 100% of occupants, deliberately NOT hand-tuned to
        # this paper's own measured speed/delay values -- that would
        # defeat the purpose of an honest validation run.
        behaviour[f"zone-a-{floor_number}"] = FixedValue("Adult_Default")
        behaviour[f"zone-b-{floor_number}"] = FixedValue("Adult_Default")

    return ScenarioDefinition(
        # Unavoidable modelling assumption -- SynEvac's Scenario schema
        # requires a FireDefinition even for a hazard-free fire drill.
        # A slow, arbitrary growth rate is supplied purely to satisfy
        # this structural requirement; Adult_Default does not use
        # HazardAwareRouteChoiceStrategy, so no route-choice or
        # behavioural decision in this run is actually influenced by it.
        fire=FireDefinition(growth_parameter_distribution=FixedValue(300.0)),
        occupant=OccupantDefinition(occupancy_distribution=occupancy, behaviour_profile_distribution=behaviour),
    )


DEFINITION_ID = "nist-10story-validation-v1"
MASTER_SEED = 20260802


def run_campaign(n_seeds: int, dt: float = 1.0, capacity_model=None, congestion_model=None, registry=None):

    # capacity_model/congestion_model/registry default to None, which
    # run_with_overrides() (calibration_benchmark/simulation_seam.py,
    # already-existing, unmodified code) resolves to SynEvac's own
    # unmodified production defaults -- the OFFICIAL Phase 4/5 run below
    # calls this with every override left at None. The one DIAGNOSTIC
    # run in main() (Phase 6 root-cause confirmation) supplies a
    # calibration_benchmark.candidates.CapacityWidthCandidate's own
    # candidate_capacity_model() here -- reusing the already-built,
    # already-tested Calibration Benchmark Framework exactly as
    # intended, adding no new capability of its own.

    building = build_nist_10story_building()
    definition = build_nist_10story_definition()

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


def main():

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    # OFFICIAL Phase 4/5 run -- SynEvac's own unmodified production
    # defaults, no calibration candidate applied.
    official_results = run_campaign(n_seeds=40, dt=1.0)
    official_summary = summarize(official_results)

    # Phase 6 diagnostic-only run -- reuses the already-existing,
    # already-tested Calibration Benchmark Framework's own
    # CapacityWidthCandidate to test whether raising
    # StairCapacityModel.PEOPLE_PER_METER_OF_WIDTH (1.2 -> 8.0, giving a
    # 1.27 m stair a capacity of 10 instead of 1) moves SynEvac's
    # simulated evacuation time toward the published 1022 s -- a
    # root-cause CONFIRMATION step, never the official validation result,
    # and never applied back to any SynEvac default (see
    # calibration_benchmark/candidates.py's own non-mutation guarantee).
    from calibration_benchmark.candidates import CapacityWidthCandidate

    diagnostic_candidate = CapacityWidthCandidate(
        candidate_people_per_meter_of_width=8.0,
        dataset_source="Diagnostic-only, this validation run's own root-cause check -- not dataset-derived, not a calibration recommendation",
        rationale=(
            "Root-cause confirmation for the NIST 10-story validation discrepancy: does raising "
            "stair capacity from 1.27m*1.2=1 to 1.27m*8.0=10 move simulated evacuation time toward "
            "the published 1022s value."
        ),
        stair_specific=True,
    )
    diagnostic_results = run_campaign(
        n_seeds=15, dt=1.0, capacity_model=diagnostic_candidate.candidate_capacity_model(),
    )
    diagnostic_summary = summarize(diagnostic_results)

    with open(os.path.join(output_dir, "nist_10story_validation_v1_raw_results.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "official_summary": official_summary, "official_runs": official_results,
                "diagnostic_summary": diagnostic_summary, "diagnostic_runs": diagnostic_results,
            },
            handle, indent=2,
        )

    print("OFFICIAL (SynEvac defaults):")
    print(json.dumps(official_summary, indent=2))
    print("\nDIAGNOSTIC (stair capacity raised 1 -> 10, root-cause check only):")
    print(json.dumps(diagnostic_summary, indent=2))


if __name__ == "__main__":
    main()
