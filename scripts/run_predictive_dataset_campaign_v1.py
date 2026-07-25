"""Large-Scale Predictive Dataset Campaign & Validation milestone.

Generates a production-scale predictive_dataset campaign (default 1000
scenarios -- see module docstring below for why that number, not 5000,
was chosen as the practical default) and runs every Phase 3-10 data-
quality analysis over it: dataset statistics, scenario diversity,
feature distributions, label analysis, temporal coverage, correlation
checks, data-quality checks, and operational coverage.

NOT a training script -- nothing here imports ai_training or sklearn.
NOT a pytest test -- run manually:
    python scripts/run_predictive_dataset_campaign_v1.py [scenario_count]

Runtime/scale note: the prior (validation-scale) 40-scenario campaign
extracted 41,940 rows in under a second. This milestone's own Phase 2
allows "the largest practical number within reasonable runtime" instead
of insisting on the top of the 1,000-5,000 range -- 1000 scenarios
(~1M candidate-time rows) was chosen as the default specifically to
keep both wall-clock runtime and in-memory row-list size practical for
a single-machine, single-process run; pass a higher count explicitly if
a larger campaign is wanted.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_decision.engine import AIDecisionEngine

from behaviour_profile_resolver import register_occupants

from scenario.engineering_state import DoorState, StairAvailability

from scenario_generator.batch_generator import iter_batch
from scenario_generator.request import BatchGenerationRequest

from scenario_runner import run as run_scenario_context

from simulation_runtime import SimulationRuntime

from dataset_builder.timeline import TimelineRun

from ai_registry.training_scenario import make_training_building, make_training_definition

from predictive_dataset.analysis import class_balance_report, horizon_analysis, recommend_first_horizon
from predictive_dataset.campaign_config import build_campaign_config
from predictive_dataset.candidate import enumerate_candidates
from predictive_dataset.correlation import categorical_target_association, feature_target_correlations, redundant_feature_pairs
from predictive_dataset.dataset_builder import build_candidate_dataset_rows, export_candidate_dataset_csv
from predictive_dataset.diversity import candidate_utilization_report, scenario_diversity_report
from predictive_dataset.feature_statistics import feature_distribution_report
from predictive_dataset.label_analysis import label_bias_report, temporal_coverage_report
from predictive_dataset.operational_coverage import operational_coverage_report
from predictive_dataset.quality_checks import duplicate_scenario_ids, run_quality_checks
from predictive_dataset.versioning import dataset_version


DEFAULT_SCENARIO_COUNT = 1000
MASTER_SEED = 20260726


def main():

    scenario_count = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCENARIO_COUNT

    config = build_campaign_config(scenario_count, MASTER_SEED)

    building = make_training_building()
    definition = make_training_definition()
    known_candidate_ids = {candidate.candidate_id for candidate in enumerate_candidates(building)}

    request = BatchGenerationRequest(
        definition=definition, definition_id=config.definition_id,
        building=building, master_seed=config.master_seed, count=config.scenario_count,
    )

    all_rows = []
    scenario_metadata = []
    accepted = 0
    failed = 0
    failure_reasons = Counter()
    zero_row_scenario_ids = []

    simulation_wall_seconds = 0.0
    extraction_wall_seconds = 0.0

    campaign_start = time.perf_counter()

    for scenario in iter_batch(request):

        try:

            sim_start = time.perf_counter()

            context = run_scenario_context(scenario, building)
            register_occupants(context)

            decision_engine = AIDecisionEngine(base_engine=context.engine)
            runtime = SimulationRuntime(context, decision_engine, dt=config.tick_dt_seconds)
            tick_results = runtime.run()

            simulation_wall_seconds += time.perf_counter() - sim_start

            # context.building -- NOT the shared template `building` -- is the
            # scenario-initialized copy scenario_runner.building_initializer.
            # build_initialized_building() produces (Door.locked/normally_open,
            # Exit.is_blocked already mutated onto it per this scenario's own
            # resolved door_states/exit_states). Passing the pristine template
            # here would silently read Edge.traversable off Door/Exit objects
            # that never received this scenario's blocked/locked state at all --
            # exactly the bug this campaign's own Phase 5 feature-distribution
            # check caught (candidate_traversable reporting constant True across
            # every row). context.simulation/context.graph were already built
            # from this SAME copy inside scenario_runner.run(), so movement/
            # congestion results were never affected -- only this one derived
            # feature was reading the wrong object.
            timeline_run = TimelineRun(
                scenario=scenario, building=context.building,
                movement_result=runtime.movement_result, tick_results=tick_results,
            )

            extraction_start = time.perf_counter()
            rows = build_candidate_dataset_rows(timeline_run, horizons=config.horizons_seconds)
            extraction_wall_seconds += time.perf_counter() - extraction_start

            all_rows.extend(rows)
            scenario_metadata.append(_scenario_metadata_entry(scenario, runtime.movement_result))
            accepted += 1

            if not rows:
                zero_row_scenario_ids.append(scenario.metadata.scenario_id)

        except Exception as exc:  # noqa: BLE001 -- a campaign script must survive one bad scenario

            failed += 1
            failure_reasons[type(exc).__name__] += 1

    campaign_wall_seconds = time.perf_counter() - campaign_start

    # =====================================================
    # Phases 3-10 -- data-quality analysis over the assembled campaign.
    # =====================================================

    balance = class_balance_report(all_rows)
    horizons = horizon_analysis(all_rows)
    recommended_horizon = recommend_first_horizon(horizons) if horizons else None

    candidate_type_row_counts = dict(Counter(row["candidate_type"] for row in all_rows))

    zero_row_scenario_id_set = set(zero_row_scenario_ids)
    zero_row_both_exits_blocked_count = sum(
        1 for entry in scenario_metadata
        if entry["scenario_id"] in zero_row_scenario_id_set and entry["blocked_exit_count"] >= 2
    )

    report = {
        "campaign_config": config.to_dict(),
        "scenario_campaign": {
            "requested": config.scenario_count,
            "accepted": accepted,
            "failed": failed,
            "failure_reasons": dict(failure_reasons),
            "discarded_or_invalid": (
                "N/A -- no scenario_validator/ package exists yet in this codebase "
                "(scenario_generator.generator's own docstring: 'no accept/reject branch "
                "exists anywhere in this module... every attempt is attempt 0, always "
                "accepted'). 'failed' above covers every scenario that raised an exception "
                "anywhere in generation/simulation/extraction; there is no separate "
                "discard/invalid category to report."
            ),
            "zero_row_scenarios": {
                "count": len(zero_row_scenario_ids),
                "of_which_both_exits_blocked": zero_row_both_exits_blocked_count,
                "scenario_ids_sample": zero_row_scenario_ids[:10],
                "explanation": (
                    "Scenarios that ran without error but contributed ZERO candidate-time rows "
                    "-- discovered root cause (see docs/architecture/predictive_dataset_campaign_v1.md "
                    "known limitations): when EngineeringConstraints.min_open_exits is not honored by "
                    "scenario_generator.generator (that package's own docstring: it is a pure sampler, "
                    "not a validator -- min_open_exits is Definition metadata a not-yet-built Scenario "
                    "Validator would enforce) a scenario can draw BOTH exits closed simultaneously. "
                    "With nobody able to ever arrive and no scheduled events, simulation_runtime.clock."
                    "resolve_default_end_time()'s own formula (max(last_arrival or 0.0, last_event_time "
                    "or 0.0)) resolves to 0.0, so SimulationRuntime.run() produces zero ticks -- not a "
                    "predictive_dataset bug, a genuine upstream edge case this campaign surfaced."
                ),
            },
        },
        "row_counts": {
            "candidate_time_rows": len(all_rows),
            "distinct_scenario_count": balance["distinct_scenario_count"],
            "rows_per_scenario_mean": (len(all_rows) / accepted) if accepted else None,
        },
        "class_balance": balance,
        "horizon_analysis": horizons,
        "recommended_first_horizon_seconds": recommended_horizon,
        "scenario_diversity": scenario_diversity_report(scenario_metadata),
        "candidate_utilization": candidate_utilization_report(all_rows),
        "feature_distributions": feature_distribution_report(all_rows),
        "label_bias": label_bias_report(all_rows, scenario_metadata),
        "temporal_coverage": temporal_coverage_report(all_rows, scenario_metadata),
        "feature_target_correlations": feature_target_correlations(all_rows),
        "candidate_type_target_association": categorical_target_association(all_rows, "candidate_type"),
        "congestion_level_target_association": categorical_target_association(all_rows, "candidate_congestion_level"),
        "redundant_feature_pairs": redundant_feature_pairs(all_rows),
        "quality_checks": run_quality_checks(all_rows, known_candidate_ids),
        "duplicate_scenario_ids": duplicate_scenario_ids(scenario_metadata),
        "operational_coverage": operational_coverage_report(all_rows, scenario_metadata, candidate_type_row_counts),
        "performance": {
            "campaign_wall_seconds": campaign_wall_seconds,
            "simulation_execution_wall_seconds": simulation_wall_seconds,
            "dataset_extraction_wall_seconds": extraction_wall_seconds,
            "rows_generated": len(all_rows),
            "extraction_rows_per_second": (len(all_rows) / extraction_wall_seconds) if extraction_wall_seconds > 0 else None,
        },
        "dataset_version": dataset_version(recommended_horizon or 0.0).to_dict(),
    }

    output_dir = Path(__file__).resolve().parent.parent / "data" / "predictive_dataset_campaign_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "campaign_v1_report.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, default=str)

    scenario_metadata_path = output_dir / "scenario_metadata.json"
    with open(scenario_metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(scenario_metadata, metadata_file, indent=2, default=str)

    csv_path = export_candidate_dataset_csv(all_rows, str(output_dir / "candidate_dataset_v1.csv"))

    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote report to {report_path}")
    print(f"Wrote scenario metadata to {scenario_metadata_path}")
    print(f"Wrote dataset CSV to {csv_path}")


# =====================================================


def _scenario_metadata_entry(scenario, movement_result) -> dict:

    fire = scenario.fire

    blocked_door_count = sum(1 for state in scenario.door_states if state.state != DoorState.OPEN)
    blocked_exit_count = sum(1 for state in scenario.exit_states if not state.is_open)
    unavailable_stair_count = sum(1 for state in scenario.stair_states if state.availability == StairAvailability.CLOSED)

    return {
        "scenario_id": scenario.metadata.scenario_id,
        "total_occupants": len(scenario.occupants),
        "ignition_zone_id": fire.ignition_zone_id if fire is not None else None,
        "fire_growth_time_seconds": fire.growth_parameters.get("growth_time") if fire is not None else None,
        "fire_profile": fire.fire_profile if fire is not None else None,
        "blocked_door_count": blocked_door_count,
        "blocked_exit_count": blocked_exit_count,
        "unavailable_stair_count": unavailable_stair_count,
        "evacuation_duration": movement_result.total_evacuation_time,
        "unreachable_occupant_count": len(movement_result.unreachable_occupant_ids),
    }


if __name__ == "__main__":
    main()
