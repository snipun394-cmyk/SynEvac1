"""Per-Candidate Predictive AI Data Foundation milestone, Phase 15/17/18.

Runs a meaningful offline scenario campaign end-to-end (Scenario
Generator -> scenario_runner -> SimulationRuntime -> predictive_dataset),
then reports:

  - scenario count vs. candidate-time row count (Phase 15 -- "10,000
    highly correlated rows from a few scenarios are not equivalent to
    10,000 independent scenarios")
  - class balance overall / by candidate type / by horizon / by
    observation-time bucket (Phase 15)
  - horizon comparison across 10/20/30/60s and a recommended first
    training horizon (Phase 17)
  - dataset EXTRACTION performance, kept separate from simulation
    EXECUTION time (Phase 18)

NOT a pytest test (this is a multi-scenario campaign, seconds-to-
minutes of simulation, not a unit test) and NOT a training script --
no model is fit here. Run manually:
    python scripts/generate_predictive_dataset_campaign.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_decision.engine import AIDecisionEngine

from behaviour_profile_resolver import register_occupants

from scenario_generator.batch_generator import iter_batch
from scenario_generator.request import BatchGenerationRequest

from scenario_runner import run as run_scenario_context

from simulation_runtime import SimulationRuntime

from dataset_builder.timeline import TimelineRun

from ai_registry.training_scenario import make_training_building, make_training_definition

from predictive_dataset.analysis import class_balance_report, horizon_analysis, recommend_first_horizon
from predictive_dataset.dataset_builder import DEFAULT_HORIZONS, build_candidate_dataset_rows, export_candidate_dataset_csv


SCENARIO_COUNT = 40
MASTER_SEED = 20260725
TICK_DT = 5.0  # seconds -- this campaign's chosen default candidate observation interval, see
               # docs/architecture/localized_predictive_ai_dataset.md for the full justification


def main():

    building = make_training_building()
    definition = make_training_definition()

    request = BatchGenerationRequest(
        definition=definition, definition_id="def-predictive-dataset-campaign",
        building=building, master_seed=MASTER_SEED, count=SCENARIO_COUNT,
    )

    all_rows = []
    accepted = 0
    failed = 0
    failure_reasons = {}

    simulation_wall_seconds = 0.0
    extraction_wall_seconds = 0.0

    for scenario in iter_batch(request):

        try:

            sim_start = time.perf_counter()

            context = run_scenario_context(scenario, building)
            register_occupants(context)

            decision_engine = AIDecisionEngine(base_engine=context.engine)
            runtime = SimulationRuntime(context, decision_engine, dt=TICK_DT)
            tick_results = runtime.run()

            simulation_wall_seconds += time.perf_counter() - sim_start

            # context.building (the scenario-initialized copy), not the shared
            # template `building` -- see scripts/run_predictive_dataset_campaign_v1.py's
            # own comment here for why: only context.building has this
            # scenario's actual door/exit blocked/locked state applied.
            timeline_run = TimelineRun(
                scenario=scenario, building=context.building,
                movement_result=runtime.movement_result, tick_results=tick_results,
            )

            extraction_start = time.perf_counter()
            rows = build_candidate_dataset_rows(timeline_run, horizons=DEFAULT_HORIZONS)
            extraction_wall_seconds += time.perf_counter() - extraction_start

            all_rows.extend(rows)
            accepted += 1

        except Exception as exc:  # noqa: BLE001 -- a campaign-generation script must not abort on one bad scenario

            failed += 1
            failure_reasons[type(exc).__name__] = failure_reasons.get(type(exc).__name__, 0) + 1

    balance = class_balance_report(all_rows)
    horizons = horizon_analysis(all_rows)
    recommended_horizon = recommend_first_horizon(horizons) if horizons else None

    rows_per_second = (len(all_rows) / extraction_wall_seconds) if extraction_wall_seconds > 0 else None

    report = {
        "scenario_campaign": {
            "requested": SCENARIO_COUNT,
            "accepted": accepted,
            "failed": failed,
            "failure_reasons": failure_reasons,
        },
        "row_counts": {
            "candidate_time_rows": len(all_rows),
            "distinct_scenario_count": balance["distinct_scenario_count"],
        },
        "class_balance": balance,
        "horizon_analysis": horizons,
        "recommended_first_horizon_seconds": recommended_horizon,
        "performance": {
            "simulation_execution_wall_seconds": simulation_wall_seconds,
            "dataset_extraction_wall_seconds": extraction_wall_seconds,
            "rows_generated": len(all_rows),
            "extraction_rows_per_second": rows_per_second,
        },
        "sampling_interval_seconds": TICK_DT,
        "horizons_evaluated": list(DEFAULT_HORIZONS),
    }

    output_dir = Path(__file__).resolve().parent.parent / "data" / "predictive_dataset_campaign"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "campaign_report.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, default=str)

    csv_path = export_candidate_dataset_csv(all_rows, str(output_dir / "candidate_dataset.csv"))

    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote report to {report_path}")
    print(f"Wrote dataset CSV to {csv_path}")


if __name__ == "__main__":
    main()
