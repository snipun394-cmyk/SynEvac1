"""
Flow Region Capacity Formula V2 -- Experimental Implementation & Merge
Validation.

Runs three arms of the SAME seeded scenario batch, on the completed
18-story merge benchmark (Stairs 3, 7, and 12 sharing one fifth-floor
lobby -- scripts/run_nist_18story_complete_merge_validation.py):

- BASELINE: today's unmodified production simulator (StairCapacityModel
  + StairAwareCongestionModel, no flow_region_map at all -- exactly
  scenario_runner's own default).
- V1: the existing (Milestone 2) FlowRegionCapacityModel, the
  area-summing formula Milestone 5 found to overcorrect into severe
  underprediction.
- V2: the new FlowRegionCapacityModelV2, the min-cut/max-flow formula
  designed in the Flow Region Capacity Formula V2 investigation and
  implemented this milestone.

Both experimental arms use the SAME (unmodified) FlowRegionCongestionModel
-- this milestone changes only the capacity calculation, per its own
explicit scope. No SynEvac production default is touched by this
script; scenario_runner/Designer are never imported here.
"""

import json
import os
import statistics as pystats
import sys

sys.path.insert(0, os.path.dirname(__file__))

from run_nist_18story_complete_merge_validation import (  # noqa: E402
    build_nist_18story_complete_building, build_nist_18story_complete_definition,
    DEFINITION_ID, MASTER_SEED,
)

from navigation.edge import Edge  # noqa: E402
from navigation.flow_region import FlowRegion  # noqa: E402
from navigation.graph_builder import NavigationGraphGenerator  # noqa: E402

from scenario_pipeline import run_batch_pipeline  # noqa: E402

from calibration_benchmark.metrics import compute_exit_utilization_balance  # noqa: E402
from calibration_benchmark.simulation_seam import run_with_overrides  # noqa: E402

from simulator.capacity import StairCapacityModel  # noqa: E402
from simulator.congestion import StairAwareCongestionModel  # noqa: E402
from simulator.flow_region_capacity import FlowRegionCapacityModel, FlowRegionCapacityModelV2  # noqa: E402
from simulator.flow_region_congestion import FlowRegionCongestionModel  # noqa: E402

from research_framework.statistics import (  # noqa: E402
    confidence_interval, effect_size_cohens_d, paired_comparison,
)


PUBLISHED_EVACUATION_TIME_S = 1192.0  # 18-story building, all four stairs combined


# =====================================================
# Per-run metric extraction -- identical methodology to
# run_flow_region_nist_validation.py's own (Milestone 5), extended with
# the merge-specific measures this milestone asks for (discharge rate,
# per-branch usage).
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


def _per_edge_crossing_counts(movement_result, edge_type=None, edge_id_prefix=None):

    counts = {}

    for step in _all_steps(movement_result):

        if edge_type is not None and step.edge.edge_type != edge_type:
            continue

        if edge_id_prefix is not None and not step.edge.id.startswith(edge_id_prefix):
            continue

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

    lobby_exit_crossings = _per_edge_crossing_counts(movement_result, edge_type=Edge.EXIT, edge_id_prefix="exit-lobby")
    lobby_discharge_count = lobby_exit_crossings.get("exit-lobby", 0)

    door_crossings = {}
    for prefix in ("3", "7", "12"):
        counts = _per_edge_crossing_counts(movement_result, edge_id_prefix=f"door-{prefix}-lobby")
        door_crossings[f"door-{prefix}-lobby"] = counts.get(f"door-{prefix}-lobby", 0)

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
        "lobby_discharge_count": lobby_discharge_count,
        "lobby_discharge_rate": (
            lobby_discharge_count / ground_truth.total_evacuation_time
            if ground_truth.total_evacuation_time else None
        ),
        "door_3_lobby_crossings": door_crossings["door-3-lobby"],
        "door_7_lobby_crossings": door_crossings["door-7-lobby"],
        "door_12_lobby_crossings": door_crossings["door-12-lobby"],
    }


# =====================================================
# Three-arm campaign
# =====================================================


ARMS = ("baseline", "v1", "v2")


def _capacity_congestion_for_arm(arm):

    if arm == "baseline":
        return StairCapacityModel(), StairAwareCongestionModel(), False

    if arm == "v1":
        return FlowRegionCapacityModel(), FlowRegionCongestionModel(), True

    if arm == "v2":
        return FlowRegionCapacityModelV2(), FlowRegionCongestionModel(), True

    raise ValueError(arm)


def run_three_arm_campaign(n_seeds, dt=1.0):

    building = build_nist_18story_complete_building()
    definition = build_nist_18story_complete_definition()

    batch = run_batch_pipeline(definition, DEFINITION_ID, building, MASTER_SEED, n_seeds)

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
    "lobby_discharge_count", "lobby_discharge_rate",
    "door_3_lobby_crossings", "door_7_lobby_crossings", "door_12_lobby_crossings",
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


def summarize(records):

    summary = {}

    for arm in ARMS:

        arm_records = records[arm]
        evac_times = [r["total_evacuation_time"] for r in arm_records if r["total_evacuation_time"] is not None]

        summary[arm] = {
            "n_seeds": len(arm_records),
            "evacuation_time_mean": pystats.fmean(evac_times) if evac_times else None,
            "overprediction_ratio": (
                pystats.fmean(evac_times) / PUBLISHED_EVACUATION_TIME_S if evac_times else None
            ),
            "avg_queue_wait_time_mean": pystats.fmean(r["avg_queue_wait_time"] for r in arm_records),
            "max_queue_wait_time_mean": pystats.fmean(r["max_queue_wait_time"] for r in arm_records),
            "queue_wait_percentage_mean": pystats.fmean(
                r["queue_wait_percentage"] for r in arm_records if r["queue_wait_percentage"] is not None
            ) if any(r["queue_wait_percentage"] is not None for r in arm_records) else None,
            "peak_congestion_mean": pystats.fmean(
                r["peak_congestion_value"] for r in arm_records if r["peak_congestion_value"] is not None
            ) if any(r["peak_congestion_value"] is not None for r in arm_records) else None,
            "people_evacuated_mean": pystats.fmean(r["people_evacuated"] for r in arm_records),
            "people_trapped_mean": pystats.fmean(r["people_trapped"] for r in arm_records),
            "lobby_discharge_rate_mean": pystats.fmean(
                r["lobby_discharge_rate"] for r in arm_records if r["lobby_discharge_rate"] is not None
            ) if any(r["lobby_discharge_rate"] is not None for r in arm_records) else None,
            "dominant_bottleneck": _dominant_bottleneck_mode(arm_records),
            "door_3_lobby_crossings_mean": pystats.fmean(r["door_3_lobby_crossings"] for r in arm_records),
            "door_7_lobby_crossings_mean": pystats.fmean(r["door_7_lobby_crossings"] for r in arm_records),
            "door_12_lobby_crossings_mean": pystats.fmean(r["door_12_lobby_crossings"] for r in arm_records),
        }

    comparisons = {
        "baseline_vs_v1": {field: _compare(field, records["baseline"], records["v1"]) for field in NUMERIC_FIELDS},
        "baseline_vs_v2": {field: _compare(field, records["baseline"], records["v2"]) for field in NUMERIC_FIELDS},
        "v1_vs_v2": {field: _compare(field, records["v1"], records["v2"]) for field in NUMERIC_FIELDS},
    }

    return summary, comparisons


def main():

    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    dt = 1.0

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running {n_seeds} seeds x 3 arms (baseline, v1, v2) on the complete 18-story merge benchmark...", flush=True)

    records = run_three_arm_campaign(n_seeds, dt=dt)
    summary, comparisons = summarize(records)

    print(json.dumps(summary, indent=2, default=str))

    with open(
        os.path.join(output_dir, "flow_region_formula_v2_merge_experiment_raw_results.json"), "w", encoding="utf-8",
    ) as handle:
        json.dump({"summary": summary, "comparisons": comparisons, "records": records}, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
