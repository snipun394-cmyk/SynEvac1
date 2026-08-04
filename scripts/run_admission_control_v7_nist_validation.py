"""
Admission Control V7 -- Hybrid Buffer-Service Architecture. NIST
10-story and 18-story validation.

Compares OLD (storage-only -- today's exact production admission
control) against NEW (the SAME storage constraint, unchanged, plus
Admission Control V4's DischargeModel and this milestone's own new
BufferModel together -- the complete Hybrid Buffer-Service architecture
V6 recommended) on the real, already-committed 10-story and 18-story
NIST recreations.

calibration_benchmark.simulation_seam.run_with_overrides() has no
discharge_model/buffer_model parameters, and this milestone is
forbidden from modifying calibration_benchmark -- this script therefore
restates that same production composition itself, exactly the same
"restate, don't touch the frozen entry point" pattern run_with_overrides()
itself already uses relative to scenario_runner.run(), and the same
pattern Admission Control V4's own NIST validation script already
established. No SynEvac production default is changed by running this
script.
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

from ai_decision.engine import AIDecisionEngine  # noqa: E402

from behaviour_profile_resolver.registrar import register_occupants  # noqa: E402

from ground_truth import SimulationArtifacts  # noqa: E402
from ground_truth import analyze as analyze_ground_truth  # noqa: E402

from navigation.edge import Edge  # noqa: E402

from scenario_pipeline import run_batch_pipeline  # noqa: E402

from scenario_runner.building_initializer import build_initialized_building  # noqa: E402
from scenario_runner.context import SimulationContext  # noqa: E402
from scenario_runner.event_initializer import build_scheduled_events  # noqa: E402
from scenario_runner.fire_initializer import build_hazard_engine  # noqa: E402
from scenario_runner.navigation_initializer import build_navigation  # noqa: E402
from scenario_runner.occupant_initializer import build_firefighters, build_occupants  # noqa: E402

from simulation_runtime import SimulationRuntime  # noqa: E402

from simulator.buffer import DefaultBufferModel  # noqa: E402
from simulator.capacity import StairCapacityModel  # noqa: E402
from simulator.congestion import StairAwareCongestionModel  # noqa: E402
from simulator.coordinator import MultiAgentSimulation  # noqa: E402
from simulator.discharge import DefaultDischargeModel  # noqa: E402

from calibration_benchmark.metrics import compute_exit_utilization_balance  # noqa: E402

from research_framework.statistics import (  # noqa: E402
    confidence_interval, effect_size_cohens_d, paired_comparison,
)


PUBLISHED_EVACUATION_TIME_S = {
    "10-story": 1022.0,
    "18-story": 1192.0,
}

BUILDINGS = {
    "10-story": (build_nist_10story_building, build_nist_10story_definition, DEFINITION_ID_10, MASTER_SEED_10),
    "18-story": (build_nist_18story_building, build_nist_18story_definition, DEFINITION_ID_18, MASTER_SEED_18),
}


def run_with_hybrid_architecture(scenario, building, *, discharge_model=None, buffer_model=None, dt=1.0):

    building_copy = build_initialized_building(scenario, building)
    graph, engine = build_navigation(scenario, building_copy)

    simulation = MultiAgentSimulation(
        engine,
        capacity_model=StairCapacityModel(),
        congestion_model=StairAwareCongestionModel(),
        discharge_model=discharge_model,
        buffer_model=buffer_model,
    )
    occupants = build_occupants(scenario)
    firefighters = build_firefighters(scenario)

    hazard_engine, initial_hazard_snapshot = build_hazard_engine(scenario, graph)

    scheduled_events = build_scheduled_events(scenario)

    context = SimulationContext(
        building=building_copy,
        graph=graph,
        engine=engine,
        simulation=simulation,
        hazard_engine=hazard_engine,
        initial_hazard_snapshot=initial_hazard_snapshot,
        occupants=occupants,
        scheduled_events=scheduled_events,
        firefighters=firefighters,
        metadata=scenario.metadata,
    )

    register_occupants(context, registry=None)

    decision_engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, decision_engine, dt=dt, perception_provider=None)
    runtime.run()

    movement_result = runtime.movement_result

    ground_truth = analyze_ground_truth(
        SimulationArtifacts(scenario=scenario, building=building_copy, movement_result=movement_result),
    )

    return movement_result, ground_truth, building_copy


# =====================================================
# Per-run metric extraction -- identical methodology to every prior
# NIST validation script in this repo, plus a "spillback" measure:
# how many DISTINCT landings ever reached their own buffer capacity
# (only meaningful for the NEW arm; always 0 for OLD since no buffer
# tracking exists there).
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


def _per_edge_crossing_counts(movement_result, edge_type):

    counts = {}

    for step in _all_steps(movement_result):

        if step.edge.edge_type == edge_type:
            counts[step.edge.id] = counts.get(step.edge.id, 0) + 1

    return counts


def _record(scenario, ground_truth, movement_result, building_copy, buffer_model):

    queue_wait_stats = _queue_wait_stats(movement_result)
    evac_time = ground_truth.total_evacuation_time
    people_evacuated = ground_truth.people_evacuated

    peak_node_values = list(movement_result.peak_node_occupancy.values())

    # Spillback proxy: how many landings' own peak occupancy reached or
    # exceeded what the buffer model would allow there -- only
    # meaningful when a buffer_model is active; a real, structural
    # measurement (peak_node_occupancy is already produced by every
    # arm, buffer or not), not a synthetic metric invented for this
    # script alone.
    landings_at_buffer_capacity = 0

    if buffer_model is not None:

        for node_id, peak in movement_result.peak_node_occupancy.items():

            # peak_node_occupancy has no direct Node object attached --
            # this reports the count of landings whose peak occupancy
            # is itself notably high (>= 2), a simple, disclosed proxy
            # for "this landing experienced real backpressure", not a
            # re-derivation of each node's own buffer_capacity (which
            # would require re-resolving Node objects this record-level
            # function does not have access to).
            if peak >= 2:
                landings_at_buffer_capacity += 1

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
        "peak_node_occupancy_mean": pystats.fmean(peak_node_values) if peak_node_values else 0.0,
        "peak_node_occupancy_max": max(peak_node_values) if peak_node_values else 0,
        "landings_with_peak_occupancy_2plus": landings_at_buffer_capacity,
        "mean_flow_rate_people_per_s": (people_evacuated / evac_time) if evac_time else None,
        "exit_flow": _per_edge_crossing_counts(movement_result, Edge.EXIT),
        "stair_flow": _per_edge_crossing_counts(movement_result, Edge.STAIR),
    }


# =====================================================
# Two-arm campaign -- OLD (storage-only) vs NEW (storage + discharge +
# buffer, the complete Hybrid Buffer-Service architecture).
# =====================================================


def run_comparison(building_name, n_seeds, dt=1.0):

    build_building, build_definition, definition_id, master_seed = BUILDINGS[building_name]

    building = build_building()
    definition = build_definition()

    batch = run_batch_pipeline(definition, definition_id, building, master_seed, n_seeds)

    old_records, new_records = [], []

    for scenario in batch.scenarios:

        old_movement, old_gt, old_building_copy = run_with_hybrid_architecture(
            scenario, building, discharge_model=None, buffer_model=None, dt=dt,
        )
        old_records.append(_record(scenario, old_gt, old_movement, old_building_copy, buffer_model=None))

        new_movement, new_gt, new_building_copy = run_with_hybrid_architecture(
            scenario, building, discharge_model=DefaultDischargeModel(), buffer_model=DefaultBufferModel(), dt=dt,
        )
        new_records.append(_record(
            scenario, new_gt, new_movement, new_building_copy, buffer_model=DefaultBufferModel(),
        ))

    return old_records, new_records


NUMERIC_FIELDS = (
    "total_evacuation_time", "avg_queue_wait_time", "max_queue_wait_time",
    "queue_wait_percentage", "peak_congestion_value", "congestion_duration",
    "exit_utilization_balance", "people_evacuated", "people_trapped",
    "peak_node_occupancy_mean", "peak_node_occupancy_max", "landings_with_peak_occupancy_2plus",
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
        "old_mean": (sum(paired_a) / n) if n else None,
        "new_mean": (sum(paired_b) / n) if n else None,
        "old_ci": confidence_interval(paired_a).to_dict() if n else None,
        "new_ci": confidence_interval(paired_b).to_dict() if n else None,
        "paired": paired_comparison(paired_a, paired_b).to_dict() if n >= 2 else None,
        "effect_size_cohens_d": effect_size_cohens_d(paired_b, paired_a).to_dict() if n >= 2 else None,
    }


def summarize(building_name, old_records, new_records):

    comparisons = {field: _compare(field, old_records, new_records) for field in NUMERIC_FIELDS}

    old_evac_mean = comparisons["total_evacuation_time"]["old_mean"]
    new_evac_mean = comparisons["total_evacuation_time"]["new_mean"]
    published = PUBLISHED_EVACUATION_TIME_S[building_name]

    return {
        "building": building_name,
        "published_evacuation_time_s": published,
        "n_seeds": len(old_records),
        "old_overprediction_ratio": (old_evac_mean / published) if old_evac_mean is not None else None,
        "new_overprediction_ratio": (new_evac_mean / published) if new_evac_mean is not None else None,
        "comparisons": comparisons,
    }


def main():

    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    dt = 1.0

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for building_name in BUILDINGS:

        print(f"Running Admission Control V7 validation for {building_name} ({n_seeds} seeds x 2 arms: old/new)...", flush=True)

        old_records, new_records = run_comparison(building_name, n_seeds, dt=dt)
        summary = summarize(building_name, old_records, new_records)

        all_results[building_name] = {
            "summary": summary,
            "old_records": old_records,
            "new_records": new_records,
        }

        print(json.dumps(summary, indent=2, default=str))

    with open(
        os.path.join(output_dir, "admission_control_v7_nist_validation_raw_results.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(all_results, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
