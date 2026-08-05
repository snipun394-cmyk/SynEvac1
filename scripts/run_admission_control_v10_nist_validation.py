"""
Admission Control V10 -- Storage-Throughput Separation. NIST validation.

Tests the implemented V10 architecture (simulator/coordinator.py,
simulator/flow_region_capacity.py) against the same four published NIST
buildings and the same seeds every prior validation script in this
family has used, composing:

  - use_flow_regions=True, flow_region_map=graph.flow_regions
  - capacity_model=FlowRegionCapacityModelV2() -- under V10 this is
    consulted for STORAGE exactly like any other CapacityModel (always
    called with the edge itself, delegating to its own base_model,
    identical to StairCapacityModel); what V2 actually contributes here
    is bottleneck_edges(), which the coordinator uses to decide WHERE
    throughput applies.
  - congestion_model=StairAwareCongestionModel(base_model=<MINIMUM_SPEED_FACTOR=0.10>)
    -- the SAME calibrated congestion floor Campaign V2 already
    established as the best available storage+congestion calibration
    (0.934x/0.918x on 10-/18-story). V10 does not change congestion
    mechanics from that already-validated per-edge behavior at all, so
    this campaign deliberately reuses it rather than re-running a new
    grid search -- the point of this validation is to test whether
    ADDING V10's correctly-scoped throughput layer on top of the
    already-best-known storage+congestion calibration finally closes
    the 24-/31-story gap, not to re-derive congestion calibration from
    scratch.
  - registry=Adult_Default.walking_speed=1.4 m/s -- same reason,
    Campaign V2's own established best value.
  - discharge_model=DefaultDischargeModel() -- the real, literature-
    grounded (SFPE/Nelson-Mowrer) throughput model, composed for the
    first time under the corrected V10 semantics (bottleneck-only
    gating, never on internal continuation).
  - buffer_model=None -- not part of this comparison; Admission Control
    V9 already closed the Discharge+Buffer composition question
    independently of this milestone.

No calibration search is run by this script -- congestion floor and
walking speed are fixed, reused constants, exactly like Admission
Control V9's own script reused Campaign V2's values. Buildings/
constructors/master seeds are the same run_nist_*_validation.py modules
every prior script in this family imports from.
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

from ai_decision.engine import AIDecisionEngine  # noqa: E402

from behaviour_profile_resolver.registrar import register_occupants  # noqa: E402

from calibration_benchmark.candidates import _registry_with_walking_speed  # noqa: E402
from calibration_benchmark.metrics import compute_exit_utilization_balance  # noqa: E402

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

from simulator.congestion import DefaultCongestionModel, StairAwareCongestionModel  # noqa: E402
from simulator.coordinator import MultiAgentSimulation  # noqa: E402
from simulator.discharge import DefaultDischargeModel  # noqa: E402
from simulator.flow_region_capacity import FlowRegionCapacityModelV2  # noqa: E402

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

# Campaign V2's own already-established best storage+congestion
# calibration -- reused, not recalibrated.
CONGESTION_FLOOR = 0.10
WALKING_SPEED = 1.4

N_SCENARIOS = 15
DT = 1.0


def _congestion_model():

    candidate_default_cls = type(
        "V10CalibratedCongestionModel", (DefaultCongestionModel,),
        {"MINIMUM_SPEED_FACTOR": CONGESTION_FLOOR},
    )

    return StairAwareCongestionModel(base_model=candidate_default_cls())


def run_with_v10(scenario, building, dt=DT):

    building_copy = build_initialized_building(scenario, building)
    graph, engine = build_navigation(scenario, building_copy)

    simulation = MultiAgentSimulation(
        engine,
        capacity_model=FlowRegionCapacityModelV2(),
        congestion_model=_congestion_model(),
        flow_region_map=graph.flow_regions,
        discharge_model=DefaultDischargeModel(),
        buffer_model=None,
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

    register_occupants(context, registry=_registry_with_walking_speed("Adult_Default", WALKING_SPEED))

    decision_engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, decision_engine, dt=dt, perception_provider=None)
    runtime.run()

    movement_result = runtime.movement_result

    ground_truth = analyze_ground_truth(
        SimulationArtifacts(scenario=scenario, building=building_copy, movement_result=movement_result),
    )

    return movement_result, ground_truth, building_copy, graph


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


def _mean_movement_time(movement_result):

    movement_times = []

    for timeline in movement_result.occupants.values():

        if timeline.arrival_time is None or not timeline.steps:
            continue

        total_travel_time = timeline.arrival_time - timeline.depart_time

        if total_travel_time <= 0:
            continue

        total_wait = sum(getattr(s, "queue_wait_time", 0.0) or 0.0 for s in timeline.steps)
        movement_times.append(total_travel_time - total_wait)

    return pystats.fmean(movement_times) if movement_times else None


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


def _bottleneck_utilization(dominant_bottleneck_id, all_edge_crossings, graph, discharge_model, evac_time):

    # Read-only diagnostic: achieved crossing rate at the dominant
    # bottleneck edge vs. the discharge model's own theoretical rate
    # for whatever admission object actually governs it in this run
    # (the edge's own FlowRegion if one is mapped and this edge is a
    # member, otherwise the edge itself) -- reports how saturated the
    # true throughput constraint actually was.

    if dominant_bottleneck_id is None or not evac_time:
        return None, None, None

    admission_object = graph.flow_regions.get(dominant_bottleneck_id)

    if admission_object is None:
        admission_object = next((e for e in graph.edges if e.id == dominant_bottleneck_id), None)

    if admission_object is None:
        return None, None, None

    theoretical_rate = discharge_model.discharge_rate(admission_object)
    achieved_rate = all_edge_crossings.get(dominant_bottleneck_id, 0) / evac_time
    utilization = (achieved_rate / theoretical_rate) if theoretical_rate else None

    return achieved_rate, theoretical_rate, utilization


def _record(scenario, ground_truth, movement_result, building_copy, graph):

    queue_wait_stats = _queue_wait_stats(movement_result)
    evac_time = ground_truth.total_evacuation_time
    people_evacuated = ground_truth.people_evacuated

    per_edge_wait = _per_edge_queue_wait_totals(movement_result)
    dominant_bottleneck_id = max(per_edge_wait, key=per_edge_wait.get) if per_edge_wait else None

    all_edge_crossings = {
        **_per_edge_crossing_counts(movement_result, Edge.EXIT),
        **_per_edge_crossing_counts(movement_result, Edge.STAIR),
        **_per_edge_crossing_counts(movement_result, Edge.DOOR),
    }

    achieved_rate, theoretical_rate, bottleneck_utilization = _bottleneck_utilization(
        dominant_bottleneck_id, all_edge_crossings, graph, DefaultDischargeModel(), evac_time,
    )

    peak_edge_values = list(movement_result.peak_edge_occupancy.values())
    peak_node_values = list(movement_result.peak_node_occupancy.values())

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
        "mean_movement_time_s": _mean_movement_time(movement_result),
        "peak_edge_occupancy_mean": pystats.fmean(peak_edge_values) if peak_edge_values else 0.0,
        "peak_edge_occupancy_max": max(peak_edge_values) if peak_edge_values else 0,
        "peak_node_occupancy_mean": pystats.fmean(peak_node_values) if peak_node_values else 0.0,
        "peak_node_occupancy_max": max(peak_node_values) if peak_node_values else 0,
        "dominant_bottleneck_id": dominant_bottleneck_id,
        "dominant_bottleneck_label": (
            _edge_label(dominant_bottleneck_id, building_copy) if dominant_bottleneck_id else None
        ),
        "dominant_bottleneck_achieved_rate_people_per_s": achieved_rate,
        "dominant_bottleneck_discharge_rate_people_per_s": theoretical_rate,
        "dominant_bottleneck_discharge_utilization": bottleneck_utilization,
        "mean_flow_rate_people_per_s": (people_evacuated / evac_time) if evac_time else None,
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


NUMERIC_FIELDS = (
    "total_evacuation_time", "avg_queue_wait_time", "max_queue_wait_time", "queue_wait_percentage",
    "mean_movement_time_s", "peak_congestion_value", "congestion_duration", "exit_utilization_balance",
    "peak_edge_occupancy_mean", "peak_edge_occupancy_max", "peak_node_occupancy_mean", "peak_node_occupancy_max",
    "dominant_bottleneck_achieved_rate_people_per_s", "dominant_bottleneck_discharge_rate_people_per_s",
    "dominant_bottleneck_discharge_utilization", "mean_flow_rate_people_per_s",
    "people_evacuated", "people_trapped",
)


def _dominant_bottleneck_mode(records):

    labels = [r["dominant_bottleneck_label"] for r in records if r["dominant_bottleneck_label"]]

    return pystats.mode(labels) if labels else None


def _output_dir():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _checkpoint_path():
    return os.path.join(_output_dir(), "admission_control_v10_nist_validation_checkpoint.json")


def _load_checkpoint():
    path = _checkpoint_path()
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _save_checkpoint(data):
    with open(_checkpoint_path(), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def run_building(building_name, n_scenarios=N_SCENARIOS, dt=DT):

    build_building, build_definition, definition_id, master_seed = BUILDINGS[building_name]

    building = build_building()
    definition = build_definition()

    print(f"[V10] {building_name} -- ({n_scenarios} seeds)...", flush=True)

    batch = run_batch_pipeline(definition, definition_id, building, master_seed, n_scenarios)

    records = []

    for scenario in batch.scenarios:

        movement_result, ground_truth, building_copy, graph = run_with_v10(scenario, building, dt=dt)
        records.append(_record(scenario, ground_truth, movement_result, building_copy, graph))

    published = PUBLISHED_EVACUATION_TIME_S[building_name]

    field_stats = {}

    for field in NUMERIC_FIELDS:

        values = [r[field] for r in records if r[field] is not None]
        field_stats[field] = {
            "mean": _mean(values),
            "ci": confidence_interval(values).to_dict() if len(values) >= 2 else None,
        }

    evac_times = [r["total_evacuation_time"] for r in records if r["total_evacuation_time"] is not None]
    mean_evac_time = _mean(evac_times)

    summary = {
        "building": building_name,
        "published_evacuation_time_s": published,
        "n_scenarios": len(records),
        "congestion_floor": CONGESTION_FLOOR,
        "walking_speed": WALKING_SPEED,
        "mean_evacuation_time": mean_evac_time,
        "overprediction_ratio": (mean_evac_time / published) if mean_evac_time is not None else None,
        "deviation_from_published_s": (mean_evac_time - published) if mean_evac_time is not None else None,
        "deviation_from_published_pct": (
            100.0 * (mean_evac_time - published) / published if mean_evac_time is not None else None
        ),
        "field_stats": field_stats,
        "dominant_bottleneck": _dominant_bottleneck_mode(records),
    }

    return summary, records


def main():

    n_scenarios = int(sys.argv[1]) if len(sys.argv) > 1 else N_SCENARIOS

    checkpoint = _load_checkpoint()

    for building_name in BUILDINGS:

        if building_name in checkpoint:
            print(f"[V10] {building_name} already checkpointed -- skipping.", flush=True)
            continue

        summary, records = run_building(building_name, n_scenarios=n_scenarios, dt=DT)
        checkpoint[building_name] = {"summary": summary, "records": records}
        _save_checkpoint(checkpoint)

        print(json.dumps({building_name: summary}, indent=2, default=str), flush=True)

    with open(
        os.path.join(_output_dir(), "admission_control_v10_nist_validation_raw_results.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(checkpoint, handle, indent=2, default=str)

    print(json.dumps({b: checkpoint[b]["summary"]["overprediction_ratio"] for b in checkpoint}, indent=2))


if __name__ == "__main__":
    main()
