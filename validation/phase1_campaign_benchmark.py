"""
Phase 1 -- Large Campaign Validation.

Runs the real production pipeline (scenario_pipeline -> scenario_runner
-> behaviour_profile_resolver -> ai_decision -> simulation_runtime ->
dataset_builder / ground_truth / decision_policy) at increasing scenario
counts, without going through CampaignWorker's QThread/Qt signal
machinery (pure compute cost, not GUI event-loop overhead). Measures
wall-clock time per phase and process memory (RSS) via psutil.

Does not modify any existing package -- reads only.
"""

import gc
import json
import os
import sys
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.building import Building
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.obstacle import Obstacle
from models.staircase import Staircase
from models.zone import Zone

from scenario_definition import FireDefinition, OccupantDefinition, ScenarioDefinition, UniformRange, FixedValue

from scenario_pipeline import run_batch_pipeline
from scenario_runner import run as run_scenario
from behaviour_profile_resolver import register_occupants

from ai_decision.engine import AIDecisionEngine

from simulation_runtime import SimulationRuntime

from dataset_builder.builder import DatasetBuilder, SimulationRun
from dataset_builder.timeline import TimelineRun, extract_timeline_rows

from ground_truth.analyzer import SimulationArtifacts, analyze

from decision_policy.policy import DecisionInputs, generate_policy


DEFINITION_ID = "validation-def-1"
MASTER_SEED = 900001
DT = 10.0


def make_building() -> Building:

    # Mirrors tests/training_dataset_fixtures.py's own shape (kept as a
    # standalone copy here so this benchmark script never depends on
    # PyQt6/QApplication -- a pure-compute measurement, no GUI needed).

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
        obstacles=[Obstacle(id="obs-1", active=False)],
        cameras=[Camera(id="cam-1", active=True)],
        detectors=[Detector(id="det-1", active=True)],
        stairs=[Staircase(id="stair-1", from_zone_id="zone-1", to_zone_id="zone-3", to_floor_id="floor-2")],
    )
    floor2 = Floor(name="Upper", id="floor-2", zones=[Zone(id="zone-3", name="Attic")])

    return Building(name="Validation Benchmark Building", id="validation-building-1", floors=[floor1, floor2])


def make_definition() -> ScenarioDefinition:

    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "zone-1": UniformRange(1, 3, discrete=True),
                "zone-2": UniformRange(1, 3, discrete=True),
            },
            behaviour_profile_distribution={
                "zone-1": FixedValue("Staff_Default"),
                "zone-2": FixedValue("Adult_Default"),
            },
        ),
    )


def _rss_mb() -> float:

    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_benchmark(count: int, output_dir: str) -> dict:

    building = make_building()
    definition = make_definition()

    gc.collect()
    mem_before = _rss_mb()

    t_gen_start = time.perf_counter()
    batch = run_batch_pipeline(definition, DEFINITION_ID, building, MASTER_SEED, count)
    t_gen_end = time.perf_counter()

    sim_runs = []
    tick_results_by_scenario = {}
    failures = 0

    t_sim_start = time.perf_counter()

    for scenario in batch.scenarios:

        try:
            context = run_scenario(scenario, building)
            register_occupants(context)
            decision_engine = AIDecisionEngine(base_engine=context.engine)
            runtime = SimulationRuntime(context, decision_engine, dt=DT)
            tick_results = runtime.run()

            sim_runs.append(
                SimulationRun(
                    scenario=scenario, building=building,
                    movement_result=runtime.movement_result, simulation_finished=True,
                )
            )
            tick_results_by_scenario[scenario.metadata.scenario_id] = tick_results

        except Exception:
            failures += 1

    t_sim_end = time.perf_counter()

    t_dataset_start = time.perf_counter()
    dataset_dir = os.path.join(output_dir, f"dataset_{count}")
    DatasetBuilder(sim_runs).export_all(dataset_dir)
    t_dataset_end = time.perf_counter()

    t_gt_start = time.perf_counter()
    ground_truth_failures = 0
    decision_policy_failures = 0

    for run in sim_runs:

        tick_results = tick_results_by_scenario.get(run.scenario.metadata.scenario_id, ())

        try:
            gt = analyze(
                SimulationArtifacts(
                    scenario=run.scenario, building=run.building,
                    movement_result=run.movement_result, tick_results=tick_results,
                )
            )
        except Exception:
            ground_truth_failures += 1
            continue

        try:
            timeline_rows = extract_timeline_rows(
                TimelineRun(
                    scenario=run.scenario, building=run.building,
                    movement_result=run.movement_result, tick_results=tick_results,
                )
            )
            generate_policy(
                DecisionInputs(
                    building=run.building, scenario=run.scenario,
                    ground_truth=gt, timeline_rows=tuple(timeline_rows),
                )
            )
        except Exception:
            decision_policy_failures += 1

    t_gt_end = time.perf_counter()

    mem_after = _rss_mb()

    return {
        "requested_count": count,
        "accepted_count": len(batch.scenarios),
        "rejected_count": count - len(batch.scenarios),
        "simulation_failures": failures,
        "ground_truth_failures": ground_truth_failures,
        "decision_policy_failures": decision_policy_failures,
        "generation_time_seconds": t_gen_end - t_gen_start,
        "simulation_time_seconds": t_sim_end - t_sim_start,
        "dataset_export_time_seconds": t_dataset_end - t_dataset_start,
        "ground_truth_and_decision_policy_time_seconds": t_gt_end - t_gt_start,
        "total_time_seconds": (t_gt_end - t_gen_start),
        "generation_time_per_scenario_ms": 1000 * (t_gen_end - t_gen_start) / max(count, 1),
        "simulation_time_per_scenario_ms": 1000 * (t_sim_end - t_sim_start) / max(len(batch.scenarios), 1),
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_delta_mb": mem_after - mem_before,
    }


def main():

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    counts = [10, 50, 100, 300, 1000, 5000, 10000]
    results = []

    for count in counts:
        print(f"Running benchmark for count={count} ...", flush=True)
        result = run_benchmark(count, output_dir)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    with open(os.path.join(output_dir, "phase1_campaign_benchmark.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
