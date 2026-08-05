"""
Admission Control V10 -- Automatic Calibration Campaign V3.

The first calibration campaign run specifically for V10 (Storage-
Throughput Separation), rather than reusing FlowRegion V1's own
already-published congestion floor (0.10) and walking speed (1.4) --
those were calibrated for V1's generous, pooled-region storage, and the
V10 implementation report's own diagnostic evidence (queue_wait_percentage
93-95% while dominant_bottleneck_discharge_utilization sits at only
2-15%) already showed V10's local per-edge storage behaves like a
fundamentally different regime, closer to the legacy per-edge
simulator's own. This campaign treats V10 as a genuinely new
architecture and calibrates it from scratch, exactly the discipline
Campaign V2 already established for V1 and the FlowRegionCapacityModel
V2 Automatic Calibration Campaign already established for V2.

STRUCTURAL CONSTRAINT, disclosed up front, not discovered mid-script:
calibration_benchmark.simulation_seam.run_with_overrides() -- the seam
underneath CalibrationStudio/AutoCalibrationEngine/GridSearchStrategy,
none of which this script is permitted to modify -- has no
discharge_model parameter at all (confirmed directly in this repo's own
Cross-Building Calibration Residual Investigation and reconfirmed by
Admission Control V9's own script, which had to bypass run_with_overrides()
entirely to compose a discharge_model). This means Phase 1/2's own grid
search, which MUST go through the unmodified Automatic Calibration
Engine, cannot express V10's discharge_model at all.

This is not a blocking problem, for a specific, mechanically verified
reason: under V10, `capacity_model.capacity(edge)` is now ALWAYS called
with the edge itself (simulator/coordinator.py's own `_resolve_admission()`
always returns (edge, edge.id), regardless of flow_region_map -- see
that file's own V10 comments). FlowRegionCapacityModelV2, when hqnded a
plain Edge, delegates straight to its own base_model (StairCapacityModel
by default) -- IDENTICAL to what plain StairCapacityModel already gives.
Congestion is likewise always evaluated per edge under V10. The ONLY
thing flow_region_map + FlowRegionCapacityModelV2 add under V10 that a
plain per-edge configuration does not is the throughput/bottleneck
layer -- which requires a discharge_model to have ANY effect at all
(see _resolve_throughput(): `if self.discharge_model is None: return True`
unconditionally). Therefore Phase 1/2's own grid search over
MINIMUM_SPEED_FACTOR/walking_speed, run WITHOUT a discharge_model
(because it structurally cannot include one), calibrates EXACTLY the
same storage+congestion behavior V10's own coordinator will exhibit in
Phase 3 once discharge_model is added back in -- discharge only ever
ADDS a throughput constraint on top, at the identified bottleneck edge,
never changes what storage/congestion alone already computed. Phase 3
composes the full V10 architecture (capacity_model=FlowRegionCapacityModelV2,
flow_region_map=graph.flow_regions, discharge_model=DefaultDischargeModel())
directly, the same "restate the composition, don't touch the frozen
entry point" pattern every Admission Control validation script in this
repo already uses, so the discharge layer IS genuinely exercised in the
final validation -- only the calibration SEARCH itself is discharge-free,
for a reason grounded in V10's own mechanics, not a shortcut.

Phase 1/2 reuse scripts.run_automatic_calibration_campaign_v2's own
_run_grid_campaign()/_select_best_aggregate() UNMODIFIED (the same
"restate, don't touch the frozen entry point" discipline the
FlowRegionCapacityModel V2 Automatic Calibration Campaign already
established) -- only the ParameterCandidate subclasses below differ,
using FlowRegionCapacityModelV2 in place of V1's own FlowRegionCapacityModel.
Because _run_grid_campaign() is a closure over that module's own
BUILDINGS/N_SCENARIOS/DT/PUBLISHED_EVACUATION_TIME_S globals, this
script's Phase 1/2 runs against exactly the same four buildings, master
seeds, n_scenarios=8, and dt=1.0 every prior campaign in this family has
used -- the SAME grids too (CONGESTION_FLOOR_GRID, WALKING_SPEED_GRID),
reused unmodified rather than narrowed, since V10's local per-edge
storage does not exhibit V2's pathological capacity=1 cost blowup (no
narrowing was needed empirically -- see this script's own progress log
for confirmation, not assumption).
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

from run_automatic_calibration_campaign_v2 import (  # noqa: E402
    _run_grid_campaign, _select_best_aggregate,
    CONGESTION_FLOOR_GRID, WALKING_SPEED_GRID,
)

from ai_decision.engine import AIDecisionEngine  # noqa: E402

from behaviour_profile_resolver.registrar import register_occupants  # noqa: E402
from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY  # noqa: E402

from calibration_benchmark import ParameterCandidate  # noqa: E402
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

from simulator.capacity import StairCapacityModel  # noqa: E402
from simulator.congestion import DefaultCongestionModel, StairAwareCongestionModel  # noqa: E402
from simulator.coordinator import MultiAgentSimulation  # noqa: E402
from simulator.discharge import DefaultDischargeModel  # noqa: E402
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

# Already-published reference values, reused as fixed comparison rows
# in Phase 3 (not recomputed) -- Campaign V2's own V1 calibration.
V1_BEST_CONGESTION_FLOOR = 0.10
V1_BEST_WALKING_SPEED = 1.4

PRODUCTION_WALKING_SPEED = DEFAULT_PROFILE_REGISTRY["Adult_Default"].walking_speed  # 1.2 m/s
PRODUCTION_CONGESTION_FLOOR = DefaultCongestionModel.MINIMUM_SPEED_FACTOR  # 0.3

N_SCENARIOS_PHASE3 = 15
DT = 1.0


def _v10_congestion_model(minimum_speed_factor):

    # Same client-side subclass-and-reassign workaround Campaign V2's
    # own _flow_region_congestion_model() established, restated here
    # because V10 uses StairAwareCongestionModel directly (never
    # FlowRegionCongestionModel, which is a behavioral no-op under V10
    # -- see simulator/coordinator.py's own V10 comments: congestion is
    # always called with the edge itself now, so FlowRegionCongestionModel
    # would only ever take its own edge-delegating path).

    candidate_default_cls = type(
        "V10CandidateCongestionModel", (DefaultCongestionModel,),
        {"MINIMUM_SPEED_FACTOR": minimum_speed_factor},
    )

    return StairAwareCongestionModel(base_model=candidate_default_cls())


# =====================================================
# Phase 1 candidate -- congestion-degradation curve, V10's own capacity
# model (FlowRegionCapacityModelV2) and flow_region_map fixed ON in
# both arms (matching every prior campaign's own "keep every other
# parameter fixed" discipline) -- no discharge_model in either arm,
# per this file's own module docstring (the Engine cannot express one;
# storage/congestion calibrated here is identical to what V10's full
# composition exhibits regardless, since discharge only adds a
# constraint on top, never changes the storage/congestion baseline).
# =====================================================


class AdmissionControlV10CongestionFloorCandidate(ParameterCandidate):

    def __init__(self, candidate_minimum_speed_factor, dataset_source, rationale):

        self.candidate_minimum_speed_factor = candidate_minimum_speed_factor

        super().__init__(
            name="StairAwareCongestionModel.MINIMUM_SPEED_FACTOR (Admission Control V10 local storage)",
            subsystem="Congestion Model (Admission Control V10 architecture)",
            calibration_tier="Tier 2",
            dataset_source=dataset_source,
            current_value=PRODUCTION_CONGESTION_FLOOR,
            candidate_value=candidate_minimum_speed_factor,
            unit="dimensionless speed factor",
            rationale=rationale,
        )

    def baseline_capacity_model(self):
        return FlowRegionCapacityModelV2()

    def candidate_capacity_model(self):
        return FlowRegionCapacityModelV2()

    def baseline_congestion_model(self):
        return StairAwareCongestionModel()  # production default floor, 0.3

    def candidate_congestion_model(self):
        return _v10_congestion_model(self.candidate_minimum_speed_factor)

    def baseline_use_flow_regions(self):
        return True

    def candidate_use_flow_regions(self):
        return True


# =====================================================
# Phase 2 candidate -- walking speed, V10's capacity model ON and
# Phase 1's selected congestion floor FIXED IDENTICALLY on both arms.
# =====================================================


class AdmissionControlV10CalibratedWalkingSpeedCandidate(ParameterCandidate):

    def __init__(self, profile_id, candidate_speed, best_minimum_speed_factor, dataset_source, rationale):

        self.profile_id = profile_id
        self.candidate_speed = candidate_speed
        self.best_minimum_speed_factor = best_minimum_speed_factor

        super().__init__(
            name=f"{profile_id}.walking_speed (Admission Control V10 + calibrated congestion baseline)",
            subsystem="Walking Model",
            calibration_tier="Tier 2",
            dataset_source=dataset_source,
            current_value=DEFAULT_PROFILE_REGISTRY[profile_id].walking_speed,
            candidate_value=candidate_speed,
            unit="m/s",
            rationale=rationale,
        )

    def baseline_registry(self):
        return DEFAULT_PROFILE_REGISTRY

    def candidate_registry(self):
        return _registry_with_walking_speed(self.profile_id, self.candidate_speed)

    def baseline_capacity_model(self):
        return FlowRegionCapacityModelV2()

    def candidate_capacity_model(self):
        return FlowRegionCapacityModelV2()

    def baseline_congestion_model(self):
        return _v10_congestion_model(self.best_minimum_speed_factor)

    def candidate_congestion_model(self):
        return _v10_congestion_model(self.best_minimum_speed_factor)

    def baseline_use_flow_regions(self):
        return True

    def candidate_use_flow_regions(self):
        return True


# =====================================================
# Checkpointing -- same disciplined resumability the FlowRegionCapacityModel
# V2 Automatic Calibration Campaign needed, kept here as cheap insurance
# even though V10's local per-edge storage is not expected to reproduce
# V2's pathological capacity=1 cost blowup (this script's own progress
# log confirms whether that expectation held, rather than assuming it).
# =====================================================


def _output_dir():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _checkpoint_path(name):
    return os.path.join(_output_dir(), f"admission_control_v10_automatic_calibration_campaign_{name}_checkpoint.json")


def _load_checkpoint(name):
    path = _checkpoint_path(name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _save_checkpoint(name, data):
    with open(_checkpoint_path(name), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def _final_path(name):
    return os.path.join(_output_dir(), f"admission_control_v10_automatic_calibration_campaign_{name}_raw_results.json")


def _load_final(name):
    path = _final_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist yet -- run this script's earlier phase(s) first.",
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_phase1():

    results = _load_checkpoint("phase1")

    for building_name in BUILDINGS:

        if building_name in results:
            print(f"[V10 Campaign][Phase 1] {building_name} already checkpointed -- skipping.", flush=True)
            continue

        print(f"[V10 Campaign][Phase 1] Congestion-floor grid search -- {building_name} "
              f"({len(CONGESTION_FLOOR_GRID)} grid points, Admission Control V10 local storage, "
              f"walking speed fixed at production default {PRODUCTION_WALKING_SPEED} m/s)...", flush=True)

        results[building_name] = _run_grid_campaign(
            building_name,
            "StairAwareCongestionModel.MINIMUM_SPEED_FACTOR",
            lambda v: AdmissionControlV10CongestionFloorCandidate(
                v, "admission-control-v10-automatic-calibration-campaign-phase1",
                "Phase 1: calibrate the congestion-degradation floor from scratch for Admission Control V10's own local-storage architecture.",
            ),
            CONGESTION_FLOOR_GRID,
            "Admission Control V10 Automatic Calibration Campaign Phase 1 (congestion floor)",
        )

        _save_checkpoint("phase1", results)

    return results


def run_phase2(best_congestion_floor):

    results = _load_checkpoint("phase2")

    for building_name in BUILDINGS:

        if building_name in results:
            print(f"[V10 Campaign][Phase 2] {building_name} already checkpointed -- skipping.", flush=True)
            continue

        print(f"[V10 Campaign][Phase 2] Walking-speed grid search -- {building_name} "
              f"({len(WALKING_SPEED_GRID)} grid points, Admission Control V10 local storage, "
              f"congestion floor fixed at {best_congestion_floor})...", flush=True)

        results[building_name] = _run_grid_campaign(
            building_name,
            "Adult_Default.walking_speed",
            lambda v: AdmissionControlV10CalibratedWalkingSpeedCandidate(
                "Adult_Default", v, best_congestion_floor,
                "admission-control-v10-automatic-calibration-campaign-phase2",
                "Phase 2: calibrate walking speed on top of Phase 1's selected V10 congestion floor.",
            ),
            WALKING_SPEED_GRID,
            "Admission Control V10 Automatic Calibration Campaign Phase 2 (walking speed)",
        )

        _save_checkpoint("phase2", results)

    return results


# =====================================================
# Phase 3 -- 5-arm comparison. Arms 1-3 (Original Simulator, FlowRegion
# V1 default, FlowRegion V1 + Calibration) reuse ALREADY-PUBLISHED
# historical numbers from Campaign V2 / the FlowRegionCapacityModel V2
# Automatic Calibration Campaign / the Admission Control V10
# implementation report -- not recomputed, per this milestone's own
# comparison-table instruction to reuse established data where
# available. Only arms 4-5 (Admission Control V10, uncalibrated and
# calibrated) are fresh, since they've never been run at THIS
# campaign's own from-scratch-calibrated constants before.
# =====================================================


ARM_NAMES = (
    "admission_control_v10_default",
    "admission_control_v10_calibrated",
)


def _arm_kwargs(arm_name, v10_best_congestion_floor, v10_best_walking_speed):

    if arm_name == "admission_control_v10_default":
        return dict(
            registry=None,
            capacity_model=FlowRegionCapacityModelV2(),
            congestion_model=StairAwareCongestionModel(),  # production default floor, 0.3
            use_flow_regions=True,
            discharge_model=DefaultDischargeModel(),
        )

    if arm_name == "admission_control_v10_calibrated":
        return dict(
            registry=_registry_with_walking_speed("Adult_Default", v10_best_walking_speed),
            capacity_model=FlowRegionCapacityModelV2(),
            congestion_model=_v10_congestion_model(v10_best_congestion_floor),
            use_flow_regions=True,
            discharge_model=DefaultDischargeModel(),
        )

    raise ValueError(arm_name)


def run_with_full_composition(
    scenario, building, *, registry=None, capacity_model=None, congestion_model=None,
    use_flow_regions=False, discharge_model=None, dt=1.0,
):

    building_copy = build_initialized_building(scenario, building)
    graph, engine = build_navigation(scenario, building_copy)

    simulation = MultiAgentSimulation(
        engine,
        capacity_model=capacity_model or StairCapacityModel(),
        congestion_model=congestion_model or StairAwareCongestionModel(),
        flow_region_map=graph.flow_regions if use_flow_regions else None,
        discharge_model=discharge_model,
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

    register_occupants(context, registry=registry)

    decision_engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, decision_engine, dt=dt, perception_provider=None)
    runtime.run()

    movement_result = runtime.movement_result

    ground_truth = analyze_ground_truth(
        SimulationArtifacts(scenario=scenario, building=building_copy, movement_result=movement_result),
    )

    return movement_result, ground_truth, building_copy, graph


_MEASUREMENT_DISCHARGE_MODEL = DefaultDischargeModel()


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


def _bottleneck_utilization(dominant_bottleneck_id, all_edge_crossings, graph, evac_time):

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
        dominant_bottleneck_id, all_edge_crossings, graph, evac_time,
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


def run_building_phase3(building_name, v10_best_congestion_floor, v10_best_walking_speed, n_scenarios=N_SCENARIOS_PHASE3, dt=DT):

    build_building, build_definition, definition_id, master_seed = BUILDINGS[building_name]

    building = build_building()
    definition = build_definition()

    print(f"[V10 Campaign][Phase 3] {building_name} -- 2-arm comparison ({n_scenarios} seeds)...", flush=True)

    batch = run_batch_pipeline(definition, definition_id, building, master_seed, n_scenarios)

    phase3_arms_checkpoint = _load_checkpoint("phase3_arms")
    building_arm_checkpoint = phase3_arms_checkpoint.get(building_name, {})

    per_arm_records = {}

    for arm in ARM_NAMES:

        if arm in building_arm_checkpoint:
            print(f"    arm '{arm}' already checkpointed -- skipping.", flush=True)
            per_arm_records[arm] = building_arm_checkpoint[arm]
            continue

        print(f"    running arm '{arm}' ({n_scenarios} seeds)...", flush=True)

        arm_records = []

        for scenario in batch.scenarios:

            kwargs = _arm_kwargs(arm, v10_best_congestion_floor, v10_best_walking_speed)
            movement_result, ground_truth, building_copy, graph = run_with_full_composition(
                scenario, building, dt=dt, **kwargs,
            )
            arm_records.append(_record(scenario, ground_truth, movement_result, building_copy, graph))

        per_arm_records[arm] = arm_records
        building_arm_checkpoint[arm] = arm_records
        phase3_arms_checkpoint[building_name] = building_arm_checkpoint
        _save_checkpoint("phase3_arms", phase3_arms_checkpoint)

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
            "deviation_from_published_s": (mean_evac_time - published) if mean_evac_time is not None else None,
            "deviation_from_published_pct": (
                100.0 * (mean_evac_time - published) / published if mean_evac_time is not None else None
            ),
            "field_stats": field_stats,
            "dominant_bottleneck": _dominant_bottleneck_mode(records),
            "exit_flow_mean": _aggregate_flow(records, "exit_flow"),
            "stair_flow_mean": _aggregate_flow(records, "stair_flow"),
        }

    a_vals = [r["total_evacuation_time"] for r in per_arm_records[ARM_NAMES[0]] if r["total_evacuation_time"] is not None]
    b_vals = [r["total_evacuation_time"] for r in per_arm_records[ARM_NAMES[1]] if r["total_evacuation_time"] is not None]
    n = min(len(a_vals), len(b_vals))

    default_vs_calibrated_significance = {
        "paired": paired_comparison(a_vals[:n], b_vals[:n]).to_dict() if n >= 2 else None,
        "effect_size_cohens_d": effect_size_cohens_d(b_vals[:n], a_vals[:n]).to_dict() if n >= 2 else None,
    }

    return {
        "building": building_name,
        "published_evacuation_time_s": published,
        "n_scenarios": n_scenarios,
        "v10_congestion_floor": v10_best_congestion_floor,
        "v10_walking_speed": v10_best_walking_speed,
        "arms": arm_summaries,
        "default_vs_calibrated_significance": default_vs_calibrated_significance,
    }, per_arm_records


def run_phase3(v10_best_congestion_floor, v10_best_walking_speed, n_scenarios=N_SCENARIOS_PHASE3, dt=DT):

    summaries = _load_checkpoint("phase3_summary")
    raw_records = _load_checkpoint("phase3_raw")

    for building_name in BUILDINGS:

        if building_name in summaries:
            print(f"[V10 Campaign][Phase 3] {building_name} already checkpointed -- skipping.", flush=True)
            continue

        summary, per_arm_records = run_building_phase3(
            building_name, v10_best_congestion_floor, v10_best_walking_speed, n_scenarios=n_scenarios, dt=dt,
        )
        summaries[building_name] = summary
        raw_records[building_name] = per_arm_records

        _save_checkpoint("phase3_summary", summaries)
        _save_checkpoint("phase3_raw", raw_records)

        print(json.dumps({building_name: summary["arms"]}, indent=2, default=str), flush=True)

    return summaries, raw_records


def _run_phase1_to_completion():

    print("=== Phase 1: V10 congestion-degradation curve ===", flush=True)
    phase1_results = run_phase1()
    v10_best_congestion_floor = _select_best_aggregate(phase1_results, CONGESTION_FLOOR_GRID)
    print(f"[Phase 1] Selected aggregate-best V10 MINIMUM_SPEED_FACTOR = {v10_best_congestion_floor}", flush=True)

    with open(_final_path("phase1"), "w", encoding="utf-8") as handle:
        json.dump(
            {"per_building": phase1_results, "selected_congestion_floor": v10_best_congestion_floor},
            handle, indent=2, default=str,
        )

    return v10_best_congestion_floor


def _run_phase2_to_completion(v10_best_congestion_floor):

    print("=== Phase 2: V10 walking speed (congestion floor fixed) ===", flush=True)
    phase2_results = run_phase2(v10_best_congestion_floor)
    v10_best_walking_speed = _select_best_aggregate(phase2_results, WALKING_SPEED_GRID)
    print(f"[Phase 2] Selected aggregate-best V10 walking_speed = {v10_best_walking_speed}", flush=True)

    with open(_final_path("phase2"), "w", encoding="utf-8") as handle:
        json.dump(
            {"per_building": phase2_results, "selected_walking_speed": v10_best_walking_speed},
            handle, indent=2, default=str,
        )

    return v10_best_walking_speed


def _run_phase3_to_completion(v10_best_congestion_floor, v10_best_walking_speed, n_scenarios_phase3):

    print("=== Phase 3: Admission Control V10 default vs. calibrated (4-building) ===", flush=True)
    phase3_results, phase3_raw_records = run_phase3(
        v10_best_congestion_floor, v10_best_walking_speed, n_scenarios=n_scenarios_phase3, dt=DT,
    )

    with open(_final_path("phase3"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_congestion_floor": v10_best_congestion_floor,
                "selected_walking_speed": v10_best_walking_speed,
                "summary": phase3_results,
                "raw_records": phase3_raw_records,
            },
            handle, indent=2, default=str,
        )


def main():

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_scenarios_phase3 = int(sys.argv[2]) if len(sys.argv) > 2 else N_SCENARIOS_PHASE3

    if mode not in ("all", "phase1", "phase2", "phase3"):
        raise ValueError(f"Unknown mode {mode!r} -- expected one of: all, phase1, phase2, phase3")

    if mode == "phase1":
        _run_phase1_to_completion()
        return

    if mode == "phase2":
        v10_best_congestion_floor = _load_final("phase1")["selected_congestion_floor"]
        _run_phase2_to_completion(v10_best_congestion_floor)
        return

    if mode == "phase3":
        v10_best_congestion_floor = _load_final("phase1")["selected_congestion_floor"]
        v10_best_walking_speed = _load_final("phase2")["selected_walking_speed"]
        _run_phase3_to_completion(v10_best_congestion_floor, v10_best_walking_speed, n_scenarios_phase3)
        return

    # mode == "all"
    v10_best_congestion_floor = _run_phase1_to_completion()
    v10_best_walking_speed = _run_phase2_to_completion(v10_best_congestion_floor)
    _run_phase3_to_completion(v10_best_congestion_floor, v10_best_walking_speed, n_scenarios_phase3)

    print(json.dumps({
        "v10_selected_congestion_floor": v10_best_congestion_floor,
        "v10_selected_walking_speed": v10_best_walking_speed,
    }, indent=2))


if __name__ == "__main__":
    main()
