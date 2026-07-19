"""
Phase 6 -- Performance Profiling.

cProfile + tracemalloc over each phase of the pipeline (generation,
simulation, dataset export, ground_truth + decision_policy) at a
moderate scale, to identify the slowest functions and largest memory
users. Read-only: no behavior is changed anywhere.
"""

import cProfile
import io
import os
import pstats
import sys
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.phase1_campaign_benchmark import make_building, make_definition, DEFINITION_ID, DT

from scenario_pipeline import run_batch_pipeline
from scenario_runner import run as run_scenario
from behaviour_profile_resolver import register_occupants

from ai_decision.engine import AIDecisionEngine

from simulation_runtime import SimulationRuntime

from dataset_builder.builder import DatasetBuilder, SimulationRun
from dataset_builder.timeline import TimelineRun, extract_timeline_rows

from ground_truth.analyzer import SimulationArtifacts, analyze

from decision_policy.policy import DecisionInputs, generate_policy

MASTER_SEED = 900003
COUNT = 500
TOP_N = 15


def _profile_block(label, fn):

    tracemalloc.start()
    profiler = cProfile.Profile()

    profiler.enable()
    result = fn()
    profiler.disable()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(TOP_N)

    return result, {
        "label": label,
        "peak_memory_mb": peak / (1024 * 1024),
        "top_functions": stream.getvalue(),
    }


def main():

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    building = make_building()
    definition = make_definition()

    reports = []

    batch, gen_report = _profile_block(
        "generation",
        lambda: run_batch_pipeline(definition, DEFINITION_ID, building, MASTER_SEED, COUNT),
    )
    reports.append(gen_report)

    sim_state = {}

    def _simulate():
        sim_runs = []
        tick_results_by_scenario = {}
        for scenario in batch.scenarios:
            context = run_scenario(scenario, building)
            register_occupants(context)
            engine = AIDecisionEngine(base_engine=context.engine)
            runtime = SimulationRuntime(context, engine, dt=DT)
            tick_results = runtime.run()
            sim_runs.append(SimulationRun(scenario=scenario, building=building,
                                           movement_result=runtime.movement_result, simulation_finished=True))
            tick_results_by_scenario[scenario.metadata.scenario_id] = tick_results
        sim_state["sim_runs"] = sim_runs
        sim_state["tick_results_by_scenario"] = tick_results_by_scenario

    _, sim_report = _profile_block("simulation", _simulate)
    reports.append(sim_report)

    sim_runs = sim_state["sim_runs"]
    tick_results_by_scenario = sim_state["tick_results_by_scenario"]

    dataset_dir = os.path.join(output_dir, "phase6_dataset")
    _, dataset_report = _profile_block(
        "dataset_export", lambda: DatasetBuilder(sim_runs).export_all(dataset_dir),
    )
    reports.append(dataset_report)

    def _ground_truth_and_policy():
        for run in sim_runs:
            tick_results = tick_results_by_scenario[run.scenario.metadata.scenario_id]
            gt = analyze(SimulationArtifacts(scenario=run.scenario, building=run.building,
                                              movement_result=run.movement_result, tick_results=tick_results))
            timeline_rows = extract_timeline_rows(TimelineRun(scenario=run.scenario, building=run.building,
                                                                movement_result=run.movement_result,
                                                                tick_results=tick_results))
            generate_policy(DecisionInputs(building=run.building, scenario=run.scenario, ground_truth=gt,
                                            timeline_rows=tuple(timeline_rows)))

    _, gt_report = _profile_block("ground_truth_and_decision_policy", _ground_truth_and_policy)
    reports.append(gt_report)

    text_path = os.path.join(output_dir, "phase6_profiling.txt")
    with open(text_path, "w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(f"===== {report['label']} (peak memory: {report['peak_memory_mb']:.2f} MB) =====\n")
            handle.write(report["top_functions"])
            handle.write("\n\n")

    print(f"Profiling report written to {text_path}")
    for report in reports:
        print(f"{report['label']}: peak memory {report['peak_memory_mb']:.2f} MB")


if __name__ == "__main__":
    main()
