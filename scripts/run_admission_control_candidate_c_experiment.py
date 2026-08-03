"""
Admission Control V3, Phase 1 -- Candidate C Experimental Validation.

Reuses the four already-completed NIST office-building recreations
(Published Scenario Validation Campaigns V1-V3) EXACTLY as built --
same Building construction functions, same ScenarioDefinition, same
DEFINITION_ID, same MASTER_SEED -- and runs each one FOUR times per
seeded scenario through the already-existing, unmodified
calibration_benchmark.simulation_seam.run_with_overrides():

- BASELINE: StairCapacityModel + StairAwareCongestionModel,
  use_flow_regions=False -- today's production admission control.
- V1: FlowRegionCapacityModel (area x jam-density) + FlowRegionCongestionModel,
  use_flow_regions=True.
- V2: FlowRegionCapacityModelV2 (min-cut/max-flow) + FlowRegionCongestionModel,
  use_flow_regions=True.
- CANDIDATE C: CapacityModelCandidateC (derived server-count:
  discharge_rate x service_time) + FlowRegionCongestionModel,
  use_flow_regions=True.

No SynEvac production default is touched by this script. No default is
promoted. This is read-only evaluation of an isolated, opt-in
experimental capacity model, per the Admission Control V3 investigation's
own Candidate C recommendation.
"""

import json
import os
import statistics as pystats
import sys

sys.path.insert(0, os.path.dirname(__file__))

from run_nist_10story_validation import (  # noqa: E402
    build_nist_10story_building, build_nist_10story_definition,
    DEFINITION_ID as DEFINITION_ID_10, MASTER_SEED as MASTER_SEED_10,
)
from run_nist_18story_validation import (  # noqa: E402
    build_nist_18story_building, build_nist_18story_definition,
    DEFINITION_ID as DEFINITION_ID_18, MASTER_SEED as MASTER_SEED_18,
)
from run_nist_24story_validation import (  # noqa: E402
    build_nist_24story_building, build_nist_24story_definition,
    DEFINITION_ID as DEFINITION_ID_24, MASTER_SEED as MASTER_SEED_24,
)
from run_nist_31story_validation import (  # noqa: E402
    build_nist_31story_building, build_nist_31story_definition,
    DEFINITION_ID as DEFINITION_ID_31, MASTER_SEED as MASTER_SEED_31,
)

from navigation.edge import Edge  # noqa: E402

from scenario_pipeline import run_batch_pipeline  # noqa: E402

from calibration_benchmark.metrics import compute_exit_utilization_balance  # noqa: E402
from calibration_benchmark.simulation_seam import run_with_overrides  # noqa: E402

from simulator.capacity import StairCapacityModel  # noqa: E402
from simulator.capacity_candidate_c import CapacityModelCandidateC  # noqa: E402
from simulator.congestion import StairAwareCongestionModel  # noqa: E402
from simulator.flow_region_capacity import FlowRegionCapacityModel, FlowRegionCapacityModelV2  # noqa: E402
from simulator.flow_region_congestion import FlowRegionCongestionModel  # noqa: E402

from research_framework.statistics import (  # noqa: E402
    confidence_interval, effect_size_cohens_d, paired_comparison,
)


PUBLISHED_EVACUATION_TIME_S = {
    "10-story": 1022.0,
    "18-story": 1192.0,
    "24-story": 1090.0,
    "31-story": 1002.0,
}

BUILDINGS = {
    "10-story": (build_nist_10story_building, build_nist_10story_definition, DEFINITION_ID_10, MASTER_SEED_10),
    "18-story": (build_nist_18story_building, build_nist_18story_definition, DEFINITION_ID_18, MASTER_SEED_18),
    "24-story": (build_nist_24story_building, build_nist_24story_definition, DEFINITION_ID_24, MASTER_SEED_24),
    "31-story": (build_nist_31story_building, build_nist_31story_definition, DEFINITION_ID_31, MASTER_SEED_31),
}

ARMS = ("baseline", "v1", "v2", "candidate_c")


def _capacity_congestion_for_arm(arm):

    if arm == "baseline":
        return StairCapacityModel(), StairAwareCongestionModel(), False

    if arm == "v1":
        return FlowRegionCapacityModel(), FlowRegionCongestionModel(), True

    if arm == "v2":
        return FlowRegionCapacityModelV2(), FlowRegionCongestionModel(), True

    if arm == "candidate_c":
        return CapacityModelCandidateC(), FlowRegionCongestionModel(), True

    raise ValueError(arm)


# =====================================================
# Per-run metric extraction -- identical methodology to
# run_flow_region_nist_validation.py (Milestone 5) and
# run_flow_region_formula_v2_merge_experiment.py, so all four arms are
# directly comparable to the already-published V1/V2 findings.
# =====================================================


def _all_steps(movement_result):

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            yield step


def _queue_wait_stats(movement_result):

    waits = [getattr(step, "queue_wait_time", 0.0) or 0.0 for step in _all_steps(movement_result)]

    if not waits:
        return {"mean": 0.0, "max": 0.0}

    return {"mean": pystats.fmean(waits), "max": max(waits)}


def _queue_wait_percentage_population_mean(movement_result):

    percentages = []

    for timeline in movement_result.occupants.values():

        if timeline.arrival_time is None or not timeline.steps:
            continue

        total_travel_time = timeline.arrival_time - timeline.depart_time

        if total_travel_time <= 0:
            continue

        total_wait = sum(getattr(s, "queue_wait_time", 0.0) or 0.0 for s in timeline.steps)
        percentages.append(100.0 * total_wait / total_travel_time)

    return pystats.fmean(percentages) if percentages else None


def _per_edge_queue_wait_totals(movement_result):

    totals = {}

    for step in _all_steps(movement_result):

        wait = getattr(step, "queue_wait_time", 0.0) or 0.0
        totals[step.edge.id] = totals.get(step.edge.id, 0.0) + wait

    return totals


def _per_edge_crossing_counts(movement_result, edge_type):

    counts = {}

    for step in _all_steps(movement_result):

        if step.edge.edge_type == edge_type:
            counts[step.edge.id] = counts.get(step.edge.id, 0) + 1

    return counts


def _edge_label(edge_id, building_copy):

    for floor in building_copy.ordered_floors():

        for door in floor.doors:
            if door.id == edge_id:
                return f"Door '{door.name}' ({edge_id})"

        for exit_obj in floor.exits:
            if exit_obj.id == edge_id:
                return f"Exit '{exit_obj.name}' ({edge_id})"

        for stair in floor.stairs:
            if stair.id == edge_id:
                return f"Stair '{stair.name}' ({edge_id})"

    return edge_id


def _record(scenario, ground_truth, movement_result, building_copy):

    queue_wait_stats = _queue_wait_stats(movement_result)
    per_edge_wait = _per_edge_queue_wait_totals(movement_result)

    dominant_bottleneck_id = max(per_edge_wait, key=per_edge_wait.get) if per_edge_wait else None

    return {
        "scenario_id": scenario.metadata.scenario_id,
        "total_evacuation_time": ground_truth.total_evacuation_time,
        "people_evacuated": ground_truth.people_evacuated,
        "people_trapped": ground_truth.people_trapped,
        "peak_congestion_value": ground_truth.peak_congestion_value,
        "congestion_duration": ground_truth.congestion_duration,
        "exit_utilization_balance": compute_exit_utilization_balance(ground_truth, building_copy),
        "avg_queue_wait_time": queue_wait_stats["mean"],
        "max_queue_wait_time": queue_wait_stats["max"],
        "queue_wait_percentage": _queue_wait_percentage_population_mean(movement_result),
        "dominant_bottleneck_id": dominant_bottleneck_id,
        "dominant_bottleneck_label": (
            _edge_label(dominant_bottleneck_id, building_copy) if dominant_bottleneck_id else None
        ),
        "exit_flow": _per_edge_crossing_counts(movement_result, Edge.EXIT),
        "stair_flow": _per_edge_crossing_counts(movement_result, Edge.STAIR),
    }


# =====================================================
# Per-building campaign -- one seeded scenario batch, four arms each.
# =====================================================


def run_building_comparison(name, n_seeds, dt=1.0):

    build_building, build_definition, definition_id, master_seed = BUILDINGS[name]

    building = build_building()
    definition = build_definition()

    batch = run_batch_pipeline(definition, definition_id, building, master_seed, n_seeds)

    records = {arm: [] for arm in ARMS}

    for scenario in batch.scenarios:

        for arm in ARMS:

            capacity_model, congestion_model, use_flow_regions = _capacity_congestion_for_arm(arm)

            movement_result, ground_truth, building_copy = run_with_overrides(
                scenario, building,
                capacity_model=capacity_model, congestion_model=congestion_model,
                dt=dt, use_flow_regions=use_flow_regions,
            )
            records[arm].append(_record(scenario, ground_truth, movement_result, building_copy))

    return records


# =====================================================
# Aggregation and paired statistics -- reuses
# research_framework.statistics exactly as calibration_benchmark/
# harness.py and both prior Flow Region experiments do, for direct
# methodological consistency. No new statistical method is introduced.
# =====================================================


NUMERIC_FIELDS = (
    "total_evacuation_time", "avg_queue_wait_time", "max_queue_wait_time",
    "queue_wait_percentage", "peak_congestion_value", "congestion_duration",
    "exit_utilization_balance", "people_evacuated", "people_trapped",
)


def _paired_non_none(a_values, b_values):

    paired_a, paired_b = [], []

    for a, b in zip(a_values, b_values):

        if a is not None and b is not None:
            paired_a.append(float(a))
            paired_b.append(float(b))

    return paired_a, paired_b


def _compare(field_name, records_a, records_b):

    a_values = [r[field_name] for r in records_a]
    b_values = [r[field_name] for r in records_b]

    paired_a, paired_b = _paired_non_none(a_values, b_values)
    n = len(paired_a)

    return {
        "n_pairs": n,
        "a_mean": (sum(paired_a) / n) if n else None,
        "b_mean": (sum(paired_b) / n) if n else None,
        "a_ci": confidence_interval(paired_a).to_dict() if n else None,
        "b_ci": confidence_interval(paired_b).to_dict() if n else None,
        "paired": paired_comparison(paired_a, paired_b).to_dict() if n >= 2 else None,
        "effect_size_cohens_d": effect_size_cohens_d(paired_b, paired_a).to_dict() if n >= 2 else None,
    }


def _dominant_bottleneck_mode(records):

    labels = [r["dominant_bottleneck_label"] for r in records if r["dominant_bottleneck_label"]]

    return pystats.mode(labels) if labels else None


def _aggregate_flow(records, key):

    totals = {}

    for record in records:
        for edge_id, count in record[key].items():
            totals[edge_id] = totals.get(edge_id, 0) + count

    n = len(records) or 1

    return {edge_id: total / n for edge_id, total in totals.items()}


def summarize_building(name, records):

    published = PUBLISHED_EVACUATION_TIME_S[name]

    arm_summaries = {}

    for arm in ARMS:

        arm_records = records[arm]
        evac_times = [r["total_evacuation_time"] for r in arm_records if r["total_evacuation_time"] is not None]
        evac_mean = pystats.fmean(evac_times) if evac_times else None

        arm_summaries[arm] = {
            "n_seeds": len(arm_records),
            "evacuation_time_mean": evac_mean,
            "overprediction_ratio": (evac_mean / published) if evac_mean is not None else None,
            "avg_queue_wait_time_mean": pystats.fmean(r["avg_queue_wait_time"] for r in arm_records),
            "max_queue_wait_time_mean": pystats.fmean(r["max_queue_wait_time"] for r in arm_records),
            "queue_wait_percentage_mean": pystats.fmean(
                r["queue_wait_percentage"] for r in arm_records if r["queue_wait_percentage"] is not None
            ) if any(r["queue_wait_percentage"] is not None for r in arm_records) else None,
            "peak_congestion_mean": pystats.fmean(
                r["peak_congestion_value"] for r in arm_records if r["peak_congestion_value"] is not None
            ) if any(r["peak_congestion_value"] is not None for r in arm_records) else None,
            "congestion_duration_mean": pystats.fmean(
                r["congestion_duration"] for r in arm_records if r["congestion_duration"] is not None
            ) if any(r["congestion_duration"] is not None for r in arm_records) else None,
            "exit_utilization_balance_mean": pystats.fmean(
                r["exit_utilization_balance"] for r in arm_records if r["exit_utilization_balance"] is not None
            ) if any(r["exit_utilization_balance"] is not None for r in arm_records) else None,
            "people_evacuated_mean": pystats.fmean(r["people_evacuated"] for r in arm_records),
            "people_trapped_mean": pystats.fmean(r["people_trapped"] for r in arm_records),
            "dominant_bottleneck": _dominant_bottleneck_mode(arm_records),
            "exit_flow_mean": _aggregate_flow(arm_records, "exit_flow"),
            "stair_flow_mean": _aggregate_flow(arm_records, "stair_flow"),
        }

    comparisons = {
        "baseline_vs_v1": {field: _compare(field, records["baseline"], records["v1"]) for field in NUMERIC_FIELDS},
        "baseline_vs_v2": {field: _compare(field, records["baseline"], records["v2"]) for field in NUMERIC_FIELDS},
        "baseline_vs_candidate_c": {
            field: _compare(field, records["baseline"], records["candidate_c"]) for field in NUMERIC_FIELDS
        },
        "v1_vs_candidate_c": {
            field: _compare(field, records["v1"], records["candidate_c"]) for field in NUMERIC_FIELDS
        },
        "v2_vs_candidate_c": {
            field: _compare(field, records["v2"], records["candidate_c"]) for field in NUMERIC_FIELDS
        },
    }

    return {
        "building": name,
        "published_evacuation_time_s": published,
        "arms": arm_summaries,
        "comparisons": comparisons,
    }


def main():

    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    dt = 1.0

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for name in BUILDINGS:

        print(f"Running {name} ({n_seeds} seeds x 4 arms: baseline/v1/v2/candidate_c)...", flush=True)

        records = run_building_comparison(name, n_seeds, dt=dt)
        summary = summarize_building(name, records)

        all_results[name] = {
            "summary": summary,
            "records": records,
        }

        print(json.dumps(summary["arms"], indent=2, default=str))

    with open(
        os.path.join(output_dir, "admission_control_candidate_c_experiment_raw_results.json"), "w", encoding="utf-8",
    ) as handle:
        json.dump(all_results, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
