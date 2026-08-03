"""
Admission Control V3, Phase 2 -- Candidate C Generalization Validation.

Stress-tests Candidate C (simulator/capacity_candidate_c.py, validated in
Phase 1 against the four standard NIST office towers) against every
INDEPENDENT validation source already in this repository -- buildings
and topologies NEVER used to develop or tune Candidate C's formula:

- The real, already-committed 18-story COMPLETE MERGE NIST recreation
  (scripts/run_nist_18story_complete_merge_validation.py) -- Phase 1
  only used the STANDARD 18-story building; the complete-merge variant
  (Stairs 3, 7, 12 converging on one shared lobby, a genuine MERGE flow
  region) was never touched by Candidate C's own validation.
- All 24 synthetic structural-diversity topology variants from the
  Predictive Dataset V3/V4 campaigns (predictive_dataset.topologies_v4.
  all_structural_variants_v4()) -- 6 families (single_exit_lowrise,
  twin_stair_highrise, multi_exit_wide, v1_topology_fixed, multi_wing,
  ring_corridor), none of them office towers, none of them used by any
  prior Candidate C experiment.

NOT applicable / not available, disclosed rather than fabricated:
- Julich -- no real pedestrian trajectory dataset exists anywhere in
  this repo; "Julich" appears only as an illustrative dataset_source
  citation string on calibration_benchmark.candidates.WalkingSpeedCandidate,
  never as loaded data, and WalkingSpeedCandidate concerns walking speed,
  not admission capacity, so it is out of scope for Candidate C regardless.
- Calibration Studio's PublishedBenchmarkLibrary -- grep-verified: every
  PublishedBenchmark(...) instantiation in this repo lives in a test
  file; no real benchmark has ever been registered into it.

Unlike Phase 1 (which had a published evacuation time to compute an
overprediction ratio against), NONE of these sources have a published
ground-truth evacuation time -- they are synthetic or augmented
topologies, not real fire drills. Every comparison here is therefore
BASELINE vs CANDIDATE C directly (paired, same scenario batch), never
an "overprediction ratio" against a number that does not exist.

No SynEvac production default is touched by this script.
"""

import json
import os
import statistics as pystats
import sys

sys.path.insert(0, os.path.dirname(__file__))

from run_nist_18story_complete_merge_validation import (  # noqa: E402
    build_nist_18story_complete_building, build_nist_18story_complete_definition,
    DEFINITION_ID as MERGE_DEFINITION_ID, MASTER_SEED as MERGE_MASTER_SEED,
)

from navigation.edge import Edge  # noqa: E402

from predictive_dataset.topologies_v4 import all_structural_variants_v4  # noqa: E402
from predictive_dataset.campaign_config_v3 import MASTER_SEED_V3  # noqa: E402

from scenario_pipeline import run_batch_pipeline  # noqa: E402

from calibration_benchmark.metrics import compute_exit_utilization_balance  # noqa: E402
from calibration_benchmark.simulation_seam import run_with_overrides  # noqa: E402

from simulator.capacity import StairCapacityModel  # noqa: E402
from simulator.capacity_candidate_c import CapacityModelCandidateC  # noqa: E402
from simulator.congestion import StairAwareCongestionModel  # noqa: E402
from simulator.flow_region_congestion import FlowRegionCongestionModel  # noqa: E402

from research_framework.statistics import (  # noqa: E402
    confidence_interval, effect_size_cohens_d, paired_comparison,
)


ARMS = ("baseline", "candidate_c")


def _capacity_congestion_for_arm(arm):

    if arm == "baseline":
        return StairCapacityModel(), StairAwareCongestionModel(), False

    if arm == "candidate_c":
        return CapacityModelCandidateC(), FlowRegionCongestionModel(), True

    raise ValueError(arm)


# =====================================================
# Per-run metric extraction -- identical methodology to Phase 1's
# run_admission_control_candidate_c_experiment.py, plus an explicit
# aggregate discharge/flow-rate measure the Phase 2 brief asks for.
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

    evac_time = ground_truth.total_evacuation_time
    people_evacuated = ground_truth.people_evacuated

    return {
        "scenario_id": scenario.metadata.scenario_id,
        "total_evacuation_time": evac_time,
        "people_evacuated": people_evacuated,
        "people_trapped": ground_truth.people_trapped,
        "peak_congestion_value": ground_truth.peak_congestion_value,
        "congestion_duration": ground_truth.congestion_duration,
        "exit_utilization_balance": compute_exit_utilization_balance(ground_truth, building_copy),
        "avg_queue_wait_time": queue_wait_stats["mean"],
        "max_queue_wait_time": queue_wait_stats["max"],
        "queue_wait_percentage": _queue_wait_percentage_population_mean(movement_result),
        "mean_flow_rate_people_per_s": (
            people_evacuated / evac_time if evac_time else None
        ),
        "dominant_bottleneck_id": dominant_bottleneck_id,
        "dominant_bottleneck_label": (
            _edge_label(dominant_bottleneck_id, building_copy) if dominant_bottleneck_id else None
        ),
        "exit_flow": _per_edge_crossing_counts(movement_result, Edge.EXIT),
        "stair_flow": _per_edge_crossing_counts(movement_result, Edge.STAIR),
    }


# =====================================================
# Per-topology campaign -- one seeded scenario batch, two arms each.
# =====================================================


def run_topology_comparison(building, definition, definition_id, master_seed, n_seeds, dt=1.0):

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


NUMERIC_FIELDS = (
    "total_evacuation_time", "avg_queue_wait_time", "max_queue_wait_time",
    "queue_wait_percentage", "peak_congestion_value", "congestion_duration",
    "exit_utilization_balance", "people_evacuated", "people_trapped",
    "mean_flow_rate_people_per_s",
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


def summarize_topology(name, family, records):

    arm_summaries = {}

    for arm in ARMS:

        arm_records = records[arm]
        evac_times = [r["total_evacuation_time"] for r in arm_records if r["total_evacuation_time"] is not None]

        arm_summaries[arm] = {
            "n_seeds": len(arm_records),
            "evacuation_time_mean": pystats.fmean(evac_times) if evac_times else None,
            "avg_queue_wait_time_mean": pystats.fmean(r["avg_queue_wait_time"] for r in arm_records),
            "max_queue_wait_time_mean": pystats.fmean(r["max_queue_wait_time"] for r in arm_records),
            "queue_wait_percentage_mean": pystats.fmean(
                r["queue_wait_percentage"] for r in arm_records if r["queue_wait_percentage"] is not None
            ) if any(r["queue_wait_percentage"] is not None for r in arm_records) else None,
            "peak_congestion_mean": pystats.fmean(
                r["peak_congestion_value"] for r in arm_records if r["peak_congestion_value"] is not None
            ) if any(r["peak_congestion_value"] is not None for r in arm_records) else None,
            "mean_flow_rate_people_per_s": pystats.fmean(
                r["mean_flow_rate_people_per_s"] for r in arm_records if r["mean_flow_rate_people_per_s"] is not None
            ) if any(r["mean_flow_rate_people_per_s"] is not None for r in arm_records) else None,
            "people_evacuated_mean": pystats.fmean(r["people_evacuated"] for r in arm_records),
            "people_trapped_mean": pystats.fmean(r["people_trapped"] for r in arm_records),
            "dominant_bottleneck": _dominant_bottleneck_mode(arm_records),
        }

    evac_a = arm_summaries["baseline"]["evacuation_time_mean"]
    evac_b = arm_summaries["candidate_c"]["evacuation_time_mean"]

    comparisons = {field: _compare(field, records["baseline"], records["candidate_c"]) for field in NUMERIC_FIELDS}

    return {
        "topology": name,
        "family": family,
        "arms": arm_summaries,
        "candidate_c_evac_time_ratio_vs_baseline": (evac_b / evac_a) if (evac_a and evac_b is not None) else None,
        "comparisons": comparisons,
        "bottleneck_identity_changed": (
            arm_summaries["baseline"]["dominant_bottleneck"] != arm_summaries["candidate_c"]["dominant_bottleneck"]
        ),
    }


def main():

    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    dt = 1.0

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # -- Independent source 1: real NIST-derived complete-merge building --
    print("Running 18-story-complete-merge (held out from Phase 1)...", flush=True)

    merge_building = build_nist_18story_complete_building()
    merge_definition = build_nist_18story_complete_definition()

    merge_records = run_topology_comparison(
        merge_building, merge_definition, MERGE_DEFINITION_ID, MERGE_MASTER_SEED, n_seeds, dt=dt,
    )
    merge_summary = summarize_topology("18story_complete_merge", "nist_merge_holdout", merge_records)
    all_results["18story_complete_merge"] = {"summary": merge_summary, "records": merge_records}
    print(json.dumps(merge_summary["arms"], indent=2, default=str))

    # -- Independent source 2: 24 synthetic structural-diversity variants --
    for variant in all_structural_variants_v4():

        print(f"Running {variant.variant_id} (family={variant.family})...", flush=True)

        spec = variant.topology
        definition_id = f"admission-control-v3-generalization-{variant.variant_id}"

        records = run_topology_comparison(
            spec.building, spec.definition, definition_id, MASTER_SEED_V3, n_seeds, dt=dt,
        )
        summary = summarize_topology(variant.variant_id, variant.family, records)
        all_results[variant.variant_id] = {"summary": summary, "records": records}
        print(json.dumps(summary["arms"], indent=2, default=str))

    with open(
        os.path.join(output_dir, "admission_control_candidate_c_generalization_raw_results.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(all_results, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
