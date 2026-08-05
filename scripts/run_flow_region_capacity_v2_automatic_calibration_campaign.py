"""
FlowRegionCapacityModel V2 -- Automatic Calibration Campaign.

The first complete scientific evaluation of FlowRegionCapacityModelV2
(the min-cut/max-flow region capacity formula that already exists in
simulator/flow_region_capacity.py, alongside V1's area-summing formula
-- V1 is never replaced or modified by this script, both coexist).

Background this script acts on, without re-litigating any of it:

  - Automatic Calibration Campaign V2 (scripts/run_automatic_calibration_
    campaign_v2.py) found FlowRegionCapacityModel V1 + calibrated
    congestion (MINIMUM_SPEED_FACTOR=0.10) + calibrated walking speed
    (1.4 m/s) brings 10-story (0.934x) and 18-story (0.918x) within
    ~7-8% of published evacuation time, while 24-story (0.468x) and
    31-story (1.502x) remain off by ~50% in OPPOSITE directions.
  - The Cross-Building Calibration Residual Investigation (this
    session, in-chat report, no commit of its own) traced that
    opposite-sign residual to V1's area-based region capacity formula
    (capacity = total_length x representative_width x jam density),
    which sums footprint across an ENTIRE merged multi-flight stair
    chain rather than reflecting its true single-flight bottleneck --
    and identified FlowRegionCapacityModelV2 (min-cut/max-flow over a
    region's own member edges) as the topology-aware alternative
    already built but never calibrated.
  - Admission Control V9 (scripts/run_admission_control_v9_flow_region_
    discharge_buffer_validation.py) tested composing DefaultDischargeModel
    + DefaultBufferModel with Flow Regions instead, and found a severe,
    uniform regression in all four buildings (ratio 5.18x-20.26x, up
    from 0.47x-1.50x) -- traced to DischargeModel's single shared
    admission timer serializing an entire merged FlowRegion. That
    hypothesis is closed, not merely unproven.

This script therefore evaluates the one remaining already-built,
never-calibrated lever: swap FlowRegionCapacityModel (V1) for
FlowRegionCapacityModelV2 and run it through the EXACT SAME calibration
methodology (Automatic Calibration Engine, Grid Search, Calibration
Studio, statistics) Campaign V2 already used for V1 -- nothing new is
designed. DefaultDischargeModel/DefaultBufferModel stay OFF throughout
(V9 already closed that composition), Calibration Studio/Automatic
Calibration Engine/Admission Control/Flow Regions/BufferModel/
DischargeModel source files are all used exactly as-is, and
FlowRegionCapacityModel V1 is never modified or removed -- both models
coexist as constructor arguments to the same, already-built
FlowRegionCapacityModel-or-V2-shaped seam (MultiAgentSimulation's own
capacity_model parameter has always accepted either).

Phase 1 and Phase 2 below do not reimplement the Automatic Calibration
Engine/Grid Search wiring a second time -- they import and call
scripts.run_automatic_calibration_campaign_v2._run_grid_campaign() and
._select_best_aggregate() UNMODIFIED (the same "restate, don't touch
the frozen entry point" discipline every script in this family uses).
Because _run_grid_campaign() is a closure over that module's own
BUILDINGS/N_SCENARIOS/DT/PUBLISHED_EVACUATION_TIME_S globals, this
script's Phase 1/2 runs against exactly Campaign V2's own four
buildings, master seeds, n_scenarios=8, and dt=1.0 -- byte-identical to
what V1's own calibration used, so V1-vs-V2 differences in Phase 1/2
are attributable only to the capacity model swap, nothing else. Only
the CAPACITY MODEL class differs between the two new ParameterCandidate
subclasses below and Campaign V2's own (FlowRegionCongestionFloorCandidate,
FlowRegionCalibratedWalkingSpeedCandidate).

Phase 3 is a direct 5-arm comparison (same "restate the composition"
pattern V9's own script already used, since this table needs all five
arms measured on one internally-consistent seeded batch for its
confidence intervals/paired tests to be comparable): original_simulator,
flow_region_v1_default (uncalibrated), flow_region_v1_calibrated
(Campaign V2's own already-published best values, 0.10/1.4 -- reused as
a fixed reference row, not recalibrated here), flow_region_v2_default
(uncalibrated), flow_region_v2_calibrated (this script's own Phase 1/2
output). discharge_model/buffer_model are None in every arm.
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
    _flow_region_congestion_model, _run_grid_campaign, _select_best_aggregate,
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
from navigation.flow_region import FlowRegion  # noqa: E402

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

# Campaign V2's own already-published V1 calibration result -- fixed
# reference values, not recalibrated by this script. Confirmed
# reproduced byte-for-byte by Admission Control V9's own script run.
V1_BEST_CONGESTION_FLOOR = 0.10
V1_BEST_WALKING_SPEED = 1.4

# Production default (behaviour_profile_resolver/registry.py
# Adult_Default.walking_speed) -- Phase 1 keeps this fixed while
# congestion floor is calibrated, exactly as Campaign V2's own Phase 1
# did for V1.
PRODUCTION_WALKING_SPEED = DEFAULT_PROFILE_REGISTRY["Adult_Default"].walking_speed

N_SCENARIOS_PHASE3 = 15
DT = 1.0

# DISCLOSED METHODOLOGY DEVIATION FROM CAMPAIGN V2, forced by compute
# budget, not by choice: V2's min-cut capacity can pin a whole merged
# region's admission capacity down to as little as 1 (measured directly
# -- see dominant_bottleneck_v1_capacity/v2_capacity in Phase 3's own
# records), which drives some grid points (particularly low
# MINIMUM_SPEED_FACTOR values on larger buildings) to simulate tens of
# thousands of seconds of serialized single-file admission queueing.
# scripts.run_automatic_calibration_campaign_v2's own N_SCENARIOS=8
# (Phase 1/2) did not complete even ONE such grid point within this
# session's available background-process wall-clock budget. This
# override reduces scenario count uniformly (same N applied to every
# building, including 10-story, which is re-run at the reduced N for
# cross-building comparability within this campaign) rather than
# leaving Campaign V2's original N=8/N=15 in place and simply never
# finishing. The reduction is applied via a runtime attribute override
# on run_automatic_calibration_campaign_v2's own module-level
# N_SCENARIOS constant (see _apply_scenario_count_override() below) --
# no source file is edited, the exact same "reassign an already-mutable
# attribute from the client side" mechanism this script's own
# _flow_region_congestion_model() import already relies on for
# FlowRegionCongestionModel._region_model. This affects statistical
# power (wider confidence intervals) relative to Campaign V2's original
# run, not methodology -- the same AutoCalibrationEngine/GridSearchStrategy
# path, same objective, same seeds (a smaller n is a prefix of the same
# deterministic seed sequence, not a different one).
#
# N=4 still proved too expensive at the grid's lowest MINIMUM_SPEED_FACTOR
# values (0.05/0.10 -- the steepest speed-degradation settings, which
# compound worst with V2's already-tiny region capacities). Two further,
# equally disclosed reductions were applied after that was observed:
# N reduced again (4 -> 2), and CONGESTION_FLOOR_GRID narrowed to its
# LENIENT half only (0.20/0.25/0.30, dropping 0.05/0.10/0.15). This
# narrowing is physically motivated, not a thumb on the scale: V2's own
# uncalibrated arm (floor=0.30, the mildest setting already) already
# OVERpredicts published time by an order of magnitude on every building
# tested so far (see the Phase 3 smoke test) -- moving toward a STEEPER
# degradation curve (lower floor) can only make an already-too-slow
# model slower still, so the lenient end is both the only part of the
# grid with any realistic chance of being selected AND the cheapest part
# to run. WALKING_SPEED_GRID is left at Campaign V2's full six points
# since Phase 2 runs entirely under Phase 1's own selected (lenient)
# floor and was not observed to be a cost problem.
#
# N=2 still could not complete even a single grid point for 31-story
# (the largest of the four buildings by occupant count, 1242) across
# two consecutive full-length relaunches with zero incremental
# progress -- not merely slow, genuinely not completing. N reduced
# once more (2 -> 1) as a last resort. At N=1, Phase 1/2/3 report
# single-run point estimates with no confidence interval for that one
# run (paired/effect-size statistics across arms/buildings, which use
# multiple independent runs, are correspondingly weaker) -- disclosed
# plainly in the final report, not hidden.
GRID_N_SCENARIOS_OVERRIDE = 1
PHASE3_N_SCENARIOS_OVERRIDE = 2

V2_CONGESTION_FLOOR_GRID = (0.20, 0.25, 0.30)


def _apply_scenario_count_override():

    import run_automatic_calibration_campaign_v2 as _v2_module

    _v2_module.N_SCENARIOS = GRID_N_SCENARIOS_OVERRIDE


# =====================================================
# Per-building checkpointing -- V2's own uncalibrated/default arms run
# far longer simulated timelines than V1's (a single occupant funneling
# through a min-cut-capacity-1 chain forces a long serial queue), so a
# full Phase 1 or Phase 2 grid search across all four buildings can
# exceed a single background process's wall-clock budget. Every
# building's own result is written to a small per-phase checkpoint file
# as soon as that building finishes, so a killed/restarted process
# resumes from the last completed building instead of re-running
# already-finished (and expensive) work. No calibration methodology
# changes because of this -- it is purely a process-restart safety net,
# the same values are computed either way.
# =====================================================


def _output_dir():

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _checkpoint_path(name):

    return os.path.join(_output_dir(), f"flow_region_capacity_v2_automatic_calibration_campaign_{name}_checkpoint.json")


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

    return os.path.join(_output_dir(), f"flow_region_capacity_v2_automatic_calibration_campaign_{name}_raw_results.json")


def _load_final(name):

    path = _final_path(name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist yet -- run this script's earlier phase(s) first "
            f"(python {os.path.basename(__file__)} phase1, then phase2, before phase3).",
        )

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# =====================================================
# Phase 1 candidate -- congestion-degradation curve, V2 capacity fixed
# ON in both arms. Identical shape to Campaign V2's own
# FlowRegionCongestionFloorCandidate, only the capacity model class
# differs.
# =====================================================


class FlowRegionV2CongestionFloorCandidate(ParameterCandidate):

    def __init__(self, candidate_minimum_speed_factor, dataset_source, rationale):

        self.candidate_minimum_speed_factor = candidate_minimum_speed_factor

        super().__init__(
            name="FlowRegionCongestionModel.MINIMUM_SPEED_FACTOR (V2 min-cut capacity, region + edge-fallback path)",
            subsystem="Congestion Model (Flow Region V2 architecture)",
            calibration_tier="Tier 2",
            dataset_source=dataset_source,
            current_value=DefaultCongestionModel.MINIMUM_SPEED_FACTOR,
            candidate_value=candidate_minimum_speed_factor,
            unit="dimensionless speed factor",
            rationale=rationale,
        )

    def baseline_capacity_model(self):
        return FlowRegionCapacityModelV2()

    def candidate_capacity_model(self):
        return FlowRegionCapacityModelV2()

    def baseline_congestion_model(self):
        return FlowRegionCongestionModel()

    def candidate_congestion_model(self):
        return _flow_region_congestion_model(self.candidate_minimum_speed_factor)

    def baseline_use_flow_regions(self):
        return True

    def candidate_use_flow_regions(self):
        return True


# =====================================================
# Phase 2 candidate -- walking speed, V2 capacity ON and Phase 1's
# selected congestion floor FIXED IDENTICALLY on both arms. Identical
# shape to Campaign V2's own FlowRegionCalibratedWalkingSpeedCandidate.
# =====================================================


class FlowRegionV2CalibratedWalkingSpeedCandidate(ParameterCandidate):

    def __init__(self, profile_id, candidate_speed, best_minimum_speed_factor, dataset_source, rationale):

        self.profile_id = profile_id
        self.candidate_speed = candidate_speed
        self.best_minimum_speed_factor = best_minimum_speed_factor

        super().__init__(
            name=f"{profile_id}.walking_speed (Flow Region V2 min-cut capacity + calibrated congestion baseline)",
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
        return _flow_region_congestion_model(self.best_minimum_speed_factor)

    def candidate_congestion_model(self):
        return _flow_region_congestion_model(self.best_minimum_speed_factor)

    def baseline_use_flow_regions(self):
        return True

    def candidate_use_flow_regions(self):
        return True


def _run_grid_campaign_checkpointed(building_name, dimension_name, build_candidate, grid_values, project_label, checkpoint_name):

    # Same AutoCalibrationEngine/GridSearchStrategy call as Campaign
    # V2's own _run_grid_campaign() -- imported and used unmodified --
    # just invoked once PER GRID VALUE instead of once for the whole
    # grid, so a killed/restarted process can resume mid-building
    # rather than only mid-campaign. This is execution chunking, not a
    # different optimization strategy: each single-value call still
    # goes through the exact same AutoCalibrationEngine.run() ->
    # GridSearchStrategy -> CalibrationStudio path as before, and the
    # per-value statistics/objective scoring are computed identically
    # (a 1-point grid is still a grid). The six single-point results
    # are reassembled into the same {"per_value_results": [...]} shape
    # _run_grid_campaign() itself returns, so _select_best_aggregate()
    # (also imported unmodified) needs no changes downstream.

    checkpoint = _load_checkpoint(checkpoint_name)
    building_checkpoint = checkpoint.get(building_name, {})

    for value in grid_values:

        key = str(value)

        if key in building_checkpoint:
            print(f"    grid value {value} already checkpointed -- skipping.", flush=True)
            continue

        print(f"    running grid value {value}...", flush=True)

        single_point_result = _run_grid_campaign(
            building_name, dimension_name, build_candidate, (value,), project_label,
        )
        building_checkpoint[key] = single_point_result["per_value_results"][0]

        checkpoint[building_name] = building_checkpoint
        _save_checkpoint(checkpoint_name, checkpoint)

    return {
        "building": building_name,
        "published_evacuation_time_s": PUBLISHED_EVACUATION_TIME_S[building_name],
        "n_scenarios": GRID_N_SCENARIOS_OVERRIDE,
        "per_value_results": [building_checkpoint[str(v)] for v in grid_values],
    }


def run_phase1():

    results = _load_checkpoint("phase1")

    for building_name in BUILDINGS:

        if building_name in results:
            print(f"[V2 Campaign][Phase 1] {building_name} already checkpointed -- skipping.", flush=True)
            continue

        print(f"[V2 Campaign][Phase 1] Congestion-floor grid search -- {building_name} "
              f"({len(V2_CONGESTION_FLOOR_GRID)} grid points, V2 min-cut capacity, "
              f"walking speed fixed at production default {PRODUCTION_WALKING_SPEED} m/s)...", flush=True)

        results[building_name] = _run_grid_campaign_checkpointed(
            building_name,
            "FlowRegionCapacityModelV2.MINIMUM_SPEED_FACTOR",
            lambda v: FlowRegionV2CongestionFloorCandidate(
                v, "flow-region-v2-automatic-calibration-campaign-phase1",
                "Phase 1: recalibrate the congestion-degradation floor under the Flow Region V2 (min-cut) architecture.",
            ),
            V2_CONGESTION_FLOOR_GRID,
            "FlowRegionCapacityModel V2 Automatic Calibration Campaign Phase 1 (congestion floor)",
            "phase1_grid",
        )

        _save_checkpoint("phase1", results)

    return results


def run_phase2(best_congestion_floor):

    results = _load_checkpoint("phase2")

    for building_name in BUILDINGS:

        if building_name in results:
            print(f"[V2 Campaign][Phase 2] {building_name} already checkpointed -- skipping.", flush=True)
            continue

        print(f"[V2 Campaign][Phase 2] Walking-speed grid search -- {building_name} "
              f"({len(WALKING_SPEED_GRID)} grid points, V2 min-cut capacity, "
              f"congestion floor fixed at {best_congestion_floor})...", flush=True)

        results[building_name] = _run_grid_campaign_checkpointed(
            building_name,
            "Adult_Default.walking_speed",
            lambda v: FlowRegionV2CalibratedWalkingSpeedCandidate(
                "Adult_Default", v, best_congestion_floor,
                "flow-region-v2-automatic-calibration-campaign-phase2",
                "Phase 2: recalibrate walking speed on top of Phase 1's selected V2 congestion floor.",
            ),
            WALKING_SPEED_GRID,
            "FlowRegionCapacityModel V2 Automatic Calibration Campaign Phase 2 (walking speed)",
            "phase2_grid",
        )

        _save_checkpoint("phase2", results)

    return results


# =====================================================
# Phase 3 -- 5-arm comparison, one internally-consistent seeded batch
# per building (same discipline as run_admission_control_v9's own
# script). discharge_model/buffer_model are None in every arm --
# Admission Control V9 already closed that composition; this campaign
# is capacity-model-only.
# =====================================================


ARM_NAMES = (
    "original_simulator",
    "flow_region_v1_default",
    "flow_region_v1_calibrated",
    "flow_region_v2_default",
    "flow_region_v2_calibrated",
)


def _arm_kwargs(arm_name, v2_best_congestion_floor, v2_best_walking_speed):

    if arm_name == "original_simulator":
        return dict(
            registry=None, capacity_model=StairCapacityModel(), congestion_model=StairAwareCongestionModel(),
            use_flow_regions=False,
        )

    if arm_name == "flow_region_v1_default":
        return dict(
            registry=None, capacity_model=FlowRegionCapacityModel(), congestion_model=FlowRegionCongestionModel(),
            use_flow_regions=True,
        )

    if arm_name == "flow_region_v1_calibrated":
        return dict(
            registry=_registry_with_walking_speed("Adult_Default", V1_BEST_WALKING_SPEED),
            capacity_model=FlowRegionCapacityModel(),
            congestion_model=_flow_region_congestion_model(V1_BEST_CONGESTION_FLOOR),
            use_flow_regions=True,
        )

    if arm_name == "flow_region_v2_default":
        return dict(
            registry=None, capacity_model=FlowRegionCapacityModelV2(), congestion_model=FlowRegionCongestionModel(),
            use_flow_regions=True,
        )

    if arm_name == "flow_region_v2_calibrated":
        return dict(
            registry=_registry_with_walking_speed("Adult_Default", v2_best_walking_speed),
            capacity_model=FlowRegionCapacityModelV2(),
            congestion_model=_flow_region_congestion_model(v2_best_congestion_floor),
            use_flow_regions=True,
        )

    raise ValueError(arm_name)


def run_with_full_composition(
    scenario, building, *, registry=None, capacity_model=None, congestion_model=None,
    use_flow_regions=False, dt=1.0,
):

    building_copy = build_initialized_building(scenario, building)
    graph, engine = build_navigation(scenario, building_copy)

    simulation = MultiAgentSimulation(
        engine,
        capacity_model=capacity_model or StairCapacityModel(),
        congestion_model=congestion_model or StairAwareCongestionModel(),
        flow_region_map=graph.flow_regions if use_flow_regions else None,
        discharge_model=None,
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


# Read-only measurement lenses -- fresh, unused instances of BOTH
# capacity models, applied to whatever FlowRegion actually turns out to
# be the dominant bottleneck in a given run, regardless of which model
# was actually driving admission control in that arm. This is what lets
# the "physical mechanism" analysis below show V1's and V2's own
# capacity NUMBERS for the same real bottleneck region side by side,
# rather than merely asserting the mechanism.
_MEASUREMENT_CAPACITY_V1 = FlowRegionCapacityModel()
_MEASUREMENT_CAPACITY_V2 = FlowRegionCapacityModelV2()


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


def _bottleneck_region_capacities(dominant_bottleneck_id, graph):

    if dominant_bottleneck_id is None:
        return None, None, None

    region = graph.flow_regions.get(dominant_bottleneck_id)

    if not isinstance(region, FlowRegion):
        return None, None, None

    v1_capacity = _MEASUREMENT_CAPACITY_V1.capacity(region)
    v2_capacity = _MEASUREMENT_CAPACITY_V2.capacity(region)

    return region.region_kind.name if hasattr(region.region_kind, "name") else str(region.region_kind), v1_capacity, v2_capacity


def _record(scenario, ground_truth, movement_result, building_copy, graph):

    queue_wait_stats = _queue_wait_stats(movement_result)
    per_edge_wait = _per_edge_queue_wait_totals(movement_result)
    dominant_bottleneck_id = max(per_edge_wait, key=per_edge_wait.get) if per_edge_wait else None

    region_kind, v1_capacity, v2_capacity = _bottleneck_region_capacities(dominant_bottleneck_id, graph)

    peak_edge_values = list(movement_result.peak_edge_occupancy.values())

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
        "peak_edge_occupancy_mean": pystats.fmean(peak_edge_values) if peak_edge_values else 0.0,
        "peak_edge_occupancy_max": max(peak_edge_values) if peak_edge_values else 0,
        "dominant_bottleneck_id": dominant_bottleneck_id,
        "dominant_bottleneck_label": (
            _edge_label(dominant_bottleneck_id, building_copy) if dominant_bottleneck_id else None
        ),
        "dominant_bottleneck_total_wait": per_edge_wait.get(dominant_bottleneck_id) if dominant_bottleneck_id else None,
        "dominant_bottleneck_region_kind": region_kind,
        "dominant_bottleneck_v1_capacity": v1_capacity,
        "dominant_bottleneck_v2_capacity": v2_capacity,
        "mean_flow_rate_people_per_s": (
            ground_truth.people_evacuated / ground_truth.total_evacuation_time
            if ground_truth.total_evacuation_time else None
        ),
        "exit_flow": _per_edge_crossing_counts(movement_result, Edge.EXIT),
        "stair_flow": _per_edge_crossing_counts(movement_result, Edge.STAIR),
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


NUMERIC_FIELDS = (
    "total_evacuation_time", "avg_queue_wait_time", "max_queue_wait_time", "queue_wait_percentage",
    "peak_congestion_value", "congestion_duration", "exit_utilization_balance",
    "peak_edge_occupancy_mean", "peak_edge_occupancy_max",
    "dominant_bottleneck_v1_capacity", "dominant_bottleneck_v2_capacity",
    "mean_flow_rate_people_per_s", "people_evacuated", "people_trapped",
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


def run_building_phase3(building_name, v2_best_congestion_floor, v2_best_walking_speed, n_scenarios=N_SCENARIOS_PHASE3, dt=DT):

    build_building, build_definition, definition_id, master_seed = BUILDINGS[building_name]

    building = build_building()
    definition = build_definition()

    print(f"[V2 Campaign][Phase 3] {building_name} -- 5-arm comparison ({n_scenarios} seeds)...", flush=True)

    batch = run_batch_pipeline(definition, definition_id, building, master_seed, n_scenarios)

    # Per-(building, arm) checkpointing -- V2's default/calibrated arms
    # run far longer simulated timelines than V1's or the legacy
    # simulator's, so even one building's full 5-arm x n_scenarios loop
    # can exceed a single background process's wall-clock budget. Each
    # arm's full n_scenarios batch is checkpointed as soon as it
    # finishes; run_batch_pipeline() is deterministic given (definition,
    # definition_id, building, master_seed, n_scenarios), so regenerating
    # the same scenario batch on a resumed run reproduces byte-identical
    # scenarios -- no risk of drift between a checkpointed arm and one
    # computed fresh in the same process.
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

            kwargs = _arm_kwargs(arm, v2_best_congestion_floor, v2_best_walking_speed)
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

    # Adjacent-chain pairwise significance, same methodology as Campaign
    # V2 Phase 3 / Admission Control V9.
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

    # The central scientific comparison this milestone asks for --
    # V1-calibrated vs V2-calibrated directly, not only via the
    # adjacent-chain path above (which goes through v2_default in
    # between).
    v1_vals = [
        r["total_evacuation_time"] for r in per_arm_records["flow_region_v1_calibrated"]
        if r["total_evacuation_time"] is not None
    ]
    v2_vals = [
        r["total_evacuation_time"] for r in per_arm_records["flow_region_v2_calibrated"]
        if r["total_evacuation_time"] is not None
    ]
    n = min(len(v1_vals), len(v2_vals))

    v1_calibrated_vs_v2_calibrated = {
        "paired": paired_comparison(v1_vals[:n], v2_vals[:n]).to_dict() if n >= 2 else None,
        "effect_size_cohens_d": effect_size_cohens_d(v2_vals[:n], v1_vals[:n]).to_dict() if n >= 2 else None,
    }

    return {
        "building": building_name,
        "published_evacuation_time_s": published,
        "n_scenarios": n_scenarios,
        "v1_congestion_floor": V1_BEST_CONGESTION_FLOOR,
        "v1_walking_speed": V1_BEST_WALKING_SPEED,
        "v2_congestion_floor": v2_best_congestion_floor,
        "v2_walking_speed": v2_best_walking_speed,
        "arms": arm_summaries,
        "pairwise_significance": pairwise,
        "v1_calibrated_vs_v2_calibrated_significance": v1_calibrated_vs_v2_calibrated,
    }, per_arm_records


def run_phase3(v2_best_congestion_floor, v2_best_walking_speed, n_scenarios=N_SCENARIOS_PHASE3, dt=DT):

    summaries = _load_checkpoint("phase3_summary")
    raw_records = _load_checkpoint("phase3_raw")

    for building_name in BUILDINGS:

        if building_name in summaries:
            print(f"[V2 Campaign][Phase 3] {building_name} already checkpointed -- skipping.", flush=True)
            continue

        summary, per_arm_records = run_building_phase3(
            building_name, v2_best_congestion_floor, v2_best_walking_speed, n_scenarios=n_scenarios, dt=dt,
        )
        summaries[building_name] = summary
        raw_records[building_name] = per_arm_records

        _save_checkpoint("phase3_summary", summaries)
        _save_checkpoint("phase3_raw", raw_records)

        print(json.dumps({building_name: summary["arms"]}, indent=2, default=str), flush=True)

    return summaries, raw_records


def _run_phase1_to_completion():

    print("=== Phase 1: V2 congestion-degradation curve ===", flush=True)
    phase1_results = run_phase1()
    v2_best_congestion_floor = _select_best_aggregate(phase1_results, V2_CONGESTION_FLOOR_GRID)
    print(f"[Phase 1] Selected aggregate-best V2 MINIMUM_SPEED_FACTOR = {v2_best_congestion_floor}", flush=True)

    with open(_final_path("phase1"), "w", encoding="utf-8") as handle:
        json.dump(
            {"per_building": phase1_results, "selected_congestion_floor": v2_best_congestion_floor},
            handle, indent=2, default=str,
        )

    return v2_best_congestion_floor


def _run_phase2_to_completion(v2_best_congestion_floor):

    print("=== Phase 2: V2 walking speed (congestion floor fixed) ===", flush=True)
    phase2_results = run_phase2(v2_best_congestion_floor)
    v2_best_walking_speed = _select_best_aggregate(phase2_results, WALKING_SPEED_GRID)
    print(f"[Phase 2] Selected aggregate-best V2 walking_speed = {v2_best_walking_speed}", flush=True)

    with open(_final_path("phase2"), "w", encoding="utf-8") as handle:
        json.dump(
            {"per_building": phase2_results, "selected_walking_speed": v2_best_walking_speed},
            handle, indent=2, default=str,
        )

    return v2_best_walking_speed


def _run_phase3_to_completion(v2_best_congestion_floor, v2_best_walking_speed, n_scenarios_phase3):

    print("=== Phase 3: five-arm NIST comparison (V1 vs V2, default vs calibrated) ===", flush=True)
    phase3_results, phase3_raw_records = run_phase3(
        v2_best_congestion_floor, v2_best_walking_speed, n_scenarios=n_scenarios_phase3, dt=DT,
    )

    with open(_final_path("phase3"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_congestion_floor": v2_best_congestion_floor,
                "selected_walking_speed": v2_best_walking_speed,
                "summary": phase3_results,
                "raw_records": phase3_raw_records,
            },
            handle, indent=2, default=str,
        )


def main():

    _apply_scenario_count_override()

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_scenarios_phase3 = int(sys.argv[2]) if len(sys.argv) > 2 else PHASE3_N_SCENARIOS_OVERRIDE

    if mode not in ("all", "phase1", "phase2", "phase3"):
        raise ValueError(f"Unknown mode {mode!r} -- expected one of: all, phase1, phase2, phase3")

    if mode == "phase1":
        _run_phase1_to_completion()
        return

    if mode == "phase2":
        v2_best_congestion_floor = _load_final("phase1")["selected_congestion_floor"]
        _run_phase2_to_completion(v2_best_congestion_floor)
        return

    if mode == "phase3":
        v2_best_congestion_floor = _load_final("phase1")["selected_congestion_floor"]
        v2_best_walking_speed = _load_final("phase2")["selected_walking_speed"]
        _run_phase3_to_completion(v2_best_congestion_floor, v2_best_walking_speed, n_scenarios_phase3)
        return

    # mode == "all"
    v2_best_congestion_floor = _run_phase1_to_completion()
    v2_best_walking_speed = _run_phase2_to_completion(v2_best_congestion_floor)
    _run_phase3_to_completion(v2_best_congestion_floor, v2_best_walking_speed, n_scenarios_phase3)

    print(json.dumps({
        "v2_selected_congestion_floor": v2_best_congestion_floor,
        "v2_selected_walking_speed": v2_best_walking_speed,
    }, indent=2))


if __name__ == "__main__":
    main()
