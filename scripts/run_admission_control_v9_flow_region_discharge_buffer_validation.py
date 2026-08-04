"""
Admission Control V9 -- Flow Region + Discharge + Buffer Composition.

The Cross-Building Calibration Residual Investigation (this session, no
commit of its own -- an in-chat report) concluded that Campaign V2's
4-building residual pattern (10-/18-story fit within ~7-8% of published
evacuation time; 24-story underpredicts by ~53%; 31-story overpredicts
by ~50%) traces to FlowRegionCapacityModel (V1)'s area-summing formula
over merged stair chains, which inflates admission capacity for long
chains far beyond their true bottleneck width -- and that
MultiAgentSimulation (simulator/coordinator.py) already accepts
flow_region_map, discharge_model, and buffer_model as fully independent
constructor parameters, with DefaultDischargeModel already dual-
accepting a plain Edge or a FlowRegion (via representative_width). That
investigation's own recommendation was to test this exact composition --
Flow Regions + calibrated congestion + calibrated speed + Discharge +
Buffer -- before touching the capacity model itself. This script is that
test.

This script modifies NO production source file. FlowRegionCapacityModel,
FlowRegionCongestionModel, DefaultDischargeModel, DefaultBufferModel,
Calibration Studio, and Automatic Calibration Engine are all used
exactly as-is and none of their own calibration/search machinery is
invoked -- this is a fixed 5-arm comparison at ALREADY-calibrated
constants (Campaign V2's own selected MINIMUM_SPEED_FACTOR=0.10,
walking_speed=1.4 m/s), read the same way Campaign V2's own Phase 3 and
scripts/run_admission_control_v4_nist_10story_validation.py /
run_admission_control_v7_nist_validation.py already read their own
fixed-arm comparisons. No parameter is recalibrated here.

calibration_benchmark.simulation_seam.run_with_overrides() has no
discharge_model/buffer_model parameters (and no flow_region_map +
discharge/buffer combination), and this milestone is forbidden from
modifying calibration_benchmark -- this script therefore restates that
same production composition itself, exactly the "restate, don't touch
the frozen entry point" pattern every prior Admission Control validation
script in this repo already uses.

The one FlowRegionCongestionModel wrinkle Campaign V2's own script
already worked around (FlowRegionCongestionModel.__init__ hardcodes
self._region_model = DefaultCongestionModel() with no constructor
override for the merged-region code path) is reused verbatim by
importing run_automatic_calibration_campaign_v2._flow_region_congestion_model
rather than redefining it a second time -- one already-tested client-side
workaround, not a second drifting copy.

Five arms, exactly the escalation chain this milestone specifies:

  1. original_simulator                                    -- legacy per-edge, no discharge/buffer
  2. flow_region_default                                    -- Flow Regions on, uncalibrated congestion (MINIMUM_SPEED_FACTOR=0.3 production default)
  3. flow_region_calibrated_congestion                       -- + MINIMUM_SPEED_FACTOR=0.10
  4. flow_region_calibrated_congestion_and_speed              -- + walking_speed=1.4 m/s (Campaign V2's own best arm)
  5. flow_region_calibrated_congestion_speed_discharge_buffer -- + DefaultDischargeModel + DefaultBufferModel

Same four NIST buildings, same MASTER_SEED per building, same
n_scenarios=15, as Campaign V2 Phase 3 -- run_batch_pipeline() is
deterministic given (definition, definition_id, building, master_seed,
n_scenarios), so this reproduces the exact same seeded scenario batches.
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

from run_automatic_calibration_campaign_v2 import _flow_region_congestion_model  # noqa: E402

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

from simulator.buffer import DefaultBufferModel  # noqa: E402
from simulator.capacity import StairCapacityModel  # noqa: E402
from simulator.congestion import StairAwareCongestionModel  # noqa: E402
from simulator.coordinator import MultiAgentSimulation  # noqa: E402
from simulator.discharge import DefaultDischargeModel  # noqa: E402
from simulator.flow_region_capacity import FlowRegionCapacityModel  # noqa: E402
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

# Campaign V2's own selected values. Fixed, not recalibrated here.
CONGESTION_FLOOR = 0.10
WALKING_SPEED = 1.4

N_SCENARIOS = 15
DT = 1.0

ARM_NAMES = (
    "original_simulator",
    "flow_region_default",
    "flow_region_calibrated_congestion",
    "flow_region_calibrated_congestion_and_speed",
    "flow_region_calibrated_congestion_speed_discharge_buffer",
)

# Used only as a read-only MEASUREMENT lens (never as the active
# discharge_model/buffer_model) on arms 1-4, so every arm's dominant
# bottleneck and busiest landing can be compared against the same fixed,
# literature-grounded throughput/density ceiling -- answering "how close
# to this constraint was the simulation already running" even where the
# constraint was not yet an active admission gate. Arm 5 additionally
# uses fresh instances of these same classes as the ACTIVE
# discharge_model/buffer_model.
_MEASUREMENT_DISCHARGE_MODEL = DefaultDischargeModel()
_MEASUREMENT_BUFFER_MODEL = DefaultBufferModel()


def _arm_kwargs(arm_name):

    if arm_name == "original_simulator":
        return dict(
            registry=None, capacity_model=StairCapacityModel(), congestion_model=StairAwareCongestionModel(),
            use_flow_regions=False, discharge_model=None, buffer_model=None,
        )

    if arm_name == "flow_region_default":
        return dict(
            registry=None, capacity_model=FlowRegionCapacityModel(), congestion_model=FlowRegionCongestionModel(),
            use_flow_regions=True, discharge_model=None, buffer_model=None,
        )

    if arm_name == "flow_region_calibrated_congestion":
        return dict(
            registry=None, capacity_model=FlowRegionCapacityModel(),
            congestion_model=_flow_region_congestion_model(CONGESTION_FLOOR),
            use_flow_regions=True, discharge_model=None, buffer_model=None,
        )

    if arm_name == "flow_region_calibrated_congestion_and_speed":
        return dict(
            registry=_registry_with_walking_speed("Adult_Default", WALKING_SPEED),
            capacity_model=FlowRegionCapacityModel(),
            congestion_model=_flow_region_congestion_model(CONGESTION_FLOOR),
            use_flow_regions=True, discharge_model=None, buffer_model=None,
        )

    if arm_name == "flow_region_calibrated_congestion_speed_discharge_buffer":
        return dict(
            registry=_registry_with_walking_speed("Adult_Default", WALKING_SPEED),
            capacity_model=FlowRegionCapacityModel(),
            congestion_model=_flow_region_congestion_model(CONGESTION_FLOOR),
            use_flow_regions=True,
            discharge_model=DefaultDischargeModel(), buffer_model=DefaultBufferModel(),
        )

    raise ValueError(arm_name)


# =====================================================
# Composition seam -- restates run_with_overrides()'s own composition
# (calibration_benchmark/simulation_seam.py) plus flow_region_map +
# discharge_model/buffer_model, exactly the same "restate, don't touch
# the frozen entry point" discipline run_admission_control_v4/v7's own
# scripts already established, extended to also carry use_flow_regions
# (which run_with_overrides has, but discharge_model/buffer_model do
# not exist there at all).
# =====================================================


def run_with_full_composition(
    scenario, building, *, registry=None, capacity_model=None, congestion_model=None,
    use_flow_regions=False, discharge_model=None, buffer_model=None, dt=1.0,
):

    building_copy = build_initialized_building(scenario, building)
    graph, engine = build_navigation(scenario, building_copy)

    simulation = MultiAgentSimulation(
        engine,
        capacity_model=capacity_model or StairCapacityModel(),
        congestion_model=congestion_model or StairAwareCongestionModel(),
        flow_region_map=graph.flow_regions if use_flow_regions else None,
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

    register_occupants(context, registry=registry)

    decision_engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, decision_engine, dt=dt, perception_provider=None)
    runtime.run()

    movement_result = runtime.movement_result

    ground_truth = analyze_ground_truth(
        SimulationArtifacts(scenario=scenario, building=building_copy, movement_result=movement_result),
    )

    return movement_result, ground_truth, building_copy, graph


# =====================================================
# Per-run metric extraction -- read-only, from already-produced fields
# (movement_result/ground_truth/graph), same methodology as every prior
# NIST validation script in this repo, plus two new read-only
# measurements this milestone's own metric list requires that no prior
# script computed: discharge utilization (at the dominant bottleneck)
# and landing/buffer occupancy (already partially covered by
# run_admission_control_v7_nist_validation.py's own
# landings_with_peak_occupancy_2plus, extended here with an explicit
# buffer_capacity-relative utilization ratio).
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


def _mean_movement_time(movement_result):

    # Pure movement time = (arrival - depart) - own total queue wait, per
    # occupant. Already-produced fields only (timeline.arrival_time/
    # depart_time, step.queue_wait_time) -- no new simulator
    # instrumentation, exactly the same "read back already-produced
    # fields" discipline scripts/run_flow_region_nist_validation.py's
    # own module docstring establishes for this file family.

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


def _per_edge_crossing_counts(movement_result, edge_type=None):

    counts = {}

    for step in _all_steps(movement_result):

        if edge_type is None or step.edge.edge_type == edge_type:
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


def _dominant_bottleneck_discharge_utilization(dominant_bottleneck_id, all_edge_crossings, graph, evac_time):

    # Read-only measurement lens: resolves the dominant-bottleneck edge
    # id to whatever admission object actually governed it in THIS run
    # (its FlowRegion, when one is mapped; otherwise the plain Edge
    # itself) and asks _MEASUREMENT_DISCHARGE_MODEL (a fresh, unused
    # DefaultDischargeModel instance -- never the arm's own active
    # discharge_model, so this is comparable across every arm including
    # the four that never gate on discharge at all) what its literature-
    # grounded throughput ceiling would be. "Achieved rate" is a coarse
    # whole-scenario average (crossings / total_evacuation_time), not a
    # tight active-window rate -- disclosed, not hidden.

    if dominant_bottleneck_id is None or not evac_time:
        return None, None, None

    admission_object = graph.flow_regions.get(dominant_bottleneck_id)

    if admission_object is None:
        admission_object = next((e for e in graph.edges if e.id == dominant_bottleneck_id), None)

    if admission_object is None:
        return None, None, None

    theoretical_rate = _MEASUREMENT_DISCHARGE_MODEL.discharge_rate(admission_object)

    achieved_rate = all_edge_crossings.get(dominant_bottleneck_id, 0) / evac_time

    utilization = (achieved_rate / theoretical_rate) if theoretical_rate else None

    return achieved_rate, theoretical_rate, utilization


def _landing_buffer_utilization(movement_result, graph):

    # Same read-only measurement-lens discipline as discharge above:
    # every node's OWN peak_node_occupancy (already produced by every
    # arm) against a fresh, unused DefaultBufferModel's derived
    # buffer_capacity for that node -- comparable across all five arms,
    # not only the one arm where buffer_model is actually active.

    ratios = []

    for node_id, peak in movement_result.peak_node_occupancy.items():

        node = graph.nodes.get(node_id)

        if node is None:
            continue

        capacity = _MEASUREMENT_BUFFER_MODEL.buffer_capacity(node)

        if capacity:
            ratios.append(peak / capacity)

    return pystats.fmean(ratios) if ratios else None, max(ratios) if ratios else None


def _record(scenario, ground_truth, movement_result, building_copy, graph):

    queue_wait_stats = _queue_wait_stats(movement_result)
    evac_time = ground_truth.total_evacuation_time
    people_evacuated = ground_truth.people_evacuated

    per_edge_wait = _per_edge_queue_wait_totals(movement_result)
    dominant_bottleneck_id = max(per_edge_wait, key=per_edge_wait.get) if per_edge_wait else None

    all_edge_crossings = _per_edge_crossing_counts(movement_result, edge_type=None)

    achieved_rate, theoretical_rate, discharge_utilization = _dominant_bottleneck_discharge_utilization(
        dominant_bottleneck_id, all_edge_crossings, graph, evac_time,
    )

    peak_edge_values = list(movement_result.peak_edge_occupancy.values())
    peak_node_values = list(movement_result.peak_node_occupancy.values())

    landings_at_2plus = sum(1 for peak in movement_result.peak_node_occupancy.values() if peak >= 2)

    landing_buffer_utilization_mean, landing_buffer_utilization_max = _landing_buffer_utilization(
        movement_result, graph,
    )

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
        "landings_with_peak_occupancy_2plus": landings_at_2plus,
        "landing_buffer_utilization_mean": landing_buffer_utilization_mean,
        "landing_buffer_utilization_max": landing_buffer_utilization_max,
        "dominant_bottleneck_id": dominant_bottleneck_id,
        "dominant_bottleneck_label": (
            _edge_label(dominant_bottleneck_id, building_copy) if dominant_bottleneck_id else None
        ),
        "dominant_bottleneck_total_wait": per_edge_wait.get(dominant_bottleneck_id) if dominant_bottleneck_id else None,
        "dominant_bottleneck_achieved_rate_people_per_s": achieved_rate,
        "dominant_bottleneck_discharge_rate_people_per_s": theoretical_rate,
        "dominant_bottleneck_discharge_utilization": discharge_utilization,
        "mean_flow_rate_people_per_s": (people_evacuated / evac_time) if evac_time else None,
        "exit_flow": _per_edge_crossing_counts(movement_result, Edge.EXIT),
        "stair_flow": _per_edge_crossing_counts(movement_result, Edge.STAIR),
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


NUMERIC_FIELDS = (
    "total_evacuation_time", "avg_queue_wait_time", "max_queue_wait_time", "queue_wait_percentage",
    "mean_movement_time_s", "peak_congestion_value", "congestion_duration", "exit_utilization_balance",
    "peak_edge_occupancy_mean", "peak_edge_occupancy_max", "peak_node_occupancy_mean", "peak_node_occupancy_max",
    "landings_with_peak_occupancy_2plus", "landing_buffer_utilization_mean", "landing_buffer_utilization_max",
    "dominant_bottleneck_achieved_rate_people_per_s", "dominant_bottleneck_discharge_rate_people_per_s",
    "dominant_bottleneck_discharge_utilization", "mean_flow_rate_people_per_s",
    "people_evacuated", "people_trapped",
)


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


def run_building(building_name, n_scenarios=N_SCENARIOS, dt=DT):

    build_building, build_definition, definition_id, master_seed = BUILDINGS[building_name]

    building = build_building()
    definition = build_definition()

    print(f"[V9] {building_name} -- 5-arm composition comparison ({n_scenarios} seeds)...", flush=True)

    batch = run_batch_pipeline(definition, definition_id, building, master_seed, n_scenarios)

    per_arm_records = {arm: [] for arm in ARM_NAMES}

    for scenario in batch.scenarios:

        for arm in ARM_NAMES:

            kwargs = _arm_kwargs(arm)
            movement_result, ground_truth, building_copy, graph = run_with_full_composition(
                scenario, building, dt=dt, **kwargs,
            )
            per_arm_records[arm].append(_record(scenario, ground_truth, movement_result, building_copy, graph))

    published = PUBLISHED_EVACUATION_TIME_S[building_name]

    arm_summaries = {}

    for arm in ARM_NAMES:

        records = per_arm_records[arm]

        field_stats = {}

        for field in NUMERIC_FIELDS:

            values = [r[field] for r in records if r[field] is not None]
            field_stats[field] = {
                "mean": _mean(values),
                "ci": confidence_interval(values).to_dict() if len(values) >= 2 else None,
            }

        evac_times = [r["total_evacuation_time"] for r in records if r["total_evacuation_time"] is not None]
        mean_evac_time = _mean(evac_times)

        arm_summaries[arm] = {
            "n_scenarios": len(records),
            "mean_evacuation_time": mean_evac_time,
            "overprediction_ratio": (mean_evac_time / published) if mean_evac_time is not None else None,
            "residual_error_s": (mean_evac_time - published) if mean_evac_time is not None else None,
            "field_stats": field_stats,
            "dominant_bottleneck": _dominant_bottleneck_mode(records),
            "exit_flow_mean": _aggregate_flow(records, "exit_flow"),
            "stair_flow_mean": _aggregate_flow(records, "stair_flow"),
        }

    # Paired significance between each successive arm in the escalation
    # chain, same methodology as run_automatic_calibration_campaign_v2's
    # own Phase 3.
    pairwise = {}

    for a, b in zip(ARM_NAMES, ARM_NAMES[1:]):

        a_vals = [r["total_evacuation_time"] for r in per_arm_records[a] if r["total_evacuation_time"] is not None]
        b_vals = [r["total_evacuation_time"] for r in per_arm_records[b] if r["total_evacuation_time"] is not None]
        n = min(len(a_vals), len(b_vals))

        paired = paired_comparison(a_vals[:n], b_vals[:n]) if n >= 2 else None
        effect = effect_size_cohens_d(b_vals[:n], a_vals[:n]) if n >= 2 else None

        pairwise[f"{a}_vs_{b}"] = {
            "paired": paired.to_dict() if paired else None,
            "effect_size_cohens_d": effect.to_dict() if effect else None,
        }

    # Direct final-comparison significance -- arm 4 (before discharge/
    # buffer) vs arm 5 (after) is this milestone's own central question,
    # so it is reported explicitly rather than only implicitly via the
    # adjacent-pair chain above (which already covers it as its last
    # entry, but this key is easier to find).
    arm4_vals = [
        r["total_evacuation_time"]
        for r in per_arm_records["flow_region_calibrated_congestion_and_speed"] if r["total_evacuation_time"] is not None
    ]
    arm5_vals = [
        r["total_evacuation_time"]
        for r in per_arm_records["flow_region_calibrated_congestion_speed_discharge_buffer"]
        if r["total_evacuation_time"] is not None
    ]
    n = min(len(arm4_vals), len(arm5_vals))

    calibrated_vs_discharge_buffer = {
        "paired": paired_comparison(arm4_vals[:n], arm5_vals[:n]).to_dict() if n >= 2 else None,
        "effect_size_cohens_d": effect_size_cohens_d(arm5_vals[:n], arm4_vals[:n]).to_dict() if n >= 2 else None,
    }

    return {
        "building": building_name,
        "published_evacuation_time_s": published,
        "n_scenarios": n_scenarios,
        "congestion_floor": CONGESTION_FLOOR,
        "walking_speed": WALKING_SPEED,
        "arms": arm_summaries,
        "pairwise_significance": pairwise,
        "calibrated_vs_discharge_buffer_significance": calibrated_vs_discharge_buffer,
    }, per_arm_records


def main():

    n_scenarios = int(sys.argv[1]) if len(sys.argv) > 1 else N_SCENARIOS

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    raw_records = {}

    for building_name in BUILDINGS:

        summary, per_arm_records = run_building(building_name, n_scenarios=n_scenarios, dt=DT)
        results[building_name] = summary
        raw_records[building_name] = per_arm_records

        print(json.dumps(
            {b: results[b]["arms"] for b in results if b == building_name},
            indent=2, default=str,
        ), flush=True)

    with open(
        os.path.join(output_dir, "admission_control_v9_flow_region_discharge_buffer_validation_raw_results.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump({"summary": results, "raw_records": raw_records}, handle, indent=2, default=str)

    print(json.dumps({b: results[b]["arms"] for b in results}, indent=2, default=str))


if __name__ == "__main__":
    main()
