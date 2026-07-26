"""Predictive Dataset V2 milestone -- multi-topology campaign generation & analysis.

Runs a substantial (~2500 scenario) campaign across FOUR genuinely different
building/navigation topologies (predictive_dataset.topologies_v2) specifically
designed to close the coverage gaps Model V1's error analysis found: stair
prediction failure, single-exit representation, multi-floor stair contention,
multiple-simultaneous-bottleneck coverage, high-occupancy stress, and
total-lockout semantics (Phase 9 -- every scenario's SimulationRuntime gets an
explicit minimum end_time floor so a total-lockout scenario still contributes
real, non-fabricated candidate-time rows instead of zero).

NOT a training script -- nothing here imports predictive_model or sklearn.
NOT a pytest test -- run manually:
    python scripts/run_predictive_dataset_campaign_v2.py

V1's own campaign (data/predictive_dataset_campaign_v1/, docs/architecture/
predictive_dataset_campaign_v1.md) is never read, modified, or overwritten by
this script -- V1 remains reproducible from ai_registry.training_scenario,
completely untouched.
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

from predictive_dataset.analysis import class_balance_report, horizon_analysis, recommend_first_horizon
from predictive_dataset.campaign_config_v2 import COVERAGE_TARGETS, build_campaign_config_v2
from predictive_dataset.candidate import enumerate_candidates
from predictive_dataset.correlation import categorical_target_association, feature_target_correlations, redundant_feature_pairs
from predictive_dataset.dataset_builder import build_candidate_dataset_rows, export_candidate_dataset_csv
from predictive_dataset.diversity import candidate_utilization_report, scenario_diversity_report
from predictive_dataset.feature_statistics import feature_distribution_report
from predictive_dataset.label_analysis import label_bias_report, temporal_coverage_report
from predictive_dataset.operational_coverage import operational_coverage_report
from predictive_dataset.quality_checks import duplicate_scenario_ids, run_quality_checks, zero_walking_distance_candidates
from predictive_dataset.topologies_v2 import all_topology_specs
from predictive_dataset.topology_analysis_v2 import (
    multi_bottleneck_candidate_type_combinations,
    occupancy_strata,
    stair_feature_repair_report,
    structural_distribution,
    topology_family_distribution,
    total_lockout_scenarios,
    verify_coverage_targets,
)
from predictive_dataset.versioning import dataset_version


def main():

    specs = all_topology_specs()
    config = build_campaign_config_v2(specs)

    known_candidate_ids = set()
    for spec in specs:
        known_candidate_ids |= {c.candidate_id for c in enumerate_candidates(spec.building)}

    all_rows = []
    scenario_metadata = []
    accepted = 0
    failed = 0
    failure_reasons = Counter()

    simulation_wall_seconds = 0.0
    extraction_wall_seconds = 0.0

    campaign_start = time.perf_counter()

    for spec in specs:

        floor_count = len(spec.building.floors)
        exit_count = sum(len(floor.exits) for floor in spec.building.floors)
        stair_count = sum(len(floor.stairs) for floor in spec.building.floors)
        door_count = sum(len(floor.doors) for floor in spec.building.floors)

        request = BatchGenerationRequest(
            definition=spec.definition, definition_id=f"predictive-dataset-v2-{spec.name}",
            building=spec.building, master_seed=config.master_seed, count=spec.scenario_count,
        )

        family_scenario_index = 0
        for scenario in iter_batch(request):

            family_scenario_index += 1
            if family_scenario_index % 100 == 0:
                print(f"  [{spec.name}] {family_scenario_index}/{spec.scenario_count} "
                      f"(elapsed {time.perf_counter() - campaign_start:.1f}s, rows so far {len(all_rows)})", flush=True)

            try:

                sim_start = time.perf_counter()

                context = run_scenario_context(scenario, spec.building)
                register_occupants(context)

                decision_engine = AIDecisionEngine(base_engine=context.engine)
                runtime = SimulationRuntime(context, decision_engine, dt=config.tick_dt_seconds)

                # Phase 9 -- total-lockout semantics (Option B, documented in
                # docs/architecture/predictive_dataset_campaign_v2.md Section 9):
                # applied UNIFORMLY to every scenario, not just detected
                # lockouts -- resolve_default_end_time()'s own natural formula
                # already exceeds this floor for any scenario with real
                # movement/events, so this only ever changes behavior for the
                # genuine "nobody can ever move" case. SimulationClock is a
                # mutable dataclass (simulation_runtime/clock.py) specifically
                # so this kind of post-construction adjustment, made before
                # .run() is ever called, is safe.
                runtime.clock.end_time = max(runtime.clock.end_time, config.minimum_end_time_seconds)

                tick_results = runtime.run()

                simulation_wall_seconds += time.perf_counter() - sim_start

                timeline_run = TimelineRun(
                    scenario=scenario, building=context.building,
                    movement_result=runtime.movement_result, tick_results=tick_results,
                )

                extraction_start = time.perf_counter()
                rows = build_candidate_dataset_rows(timeline_run, horizons=config.horizons_seconds)
                extraction_wall_seconds += time.perf_counter() - extraction_start

                all_rows.extend(rows)
                scenario_metadata.append(
                    _scenario_metadata_entry(
                        scenario, runtime.movement_result, topology_family=spec.name,
                        floor_count=floor_count, exit_count=exit_count, stair_count=stair_count, door_count=door_count,
                        contributed_rows=bool(rows),
                    )
                )
                accepted += 1

            except Exception as exc:  # noqa: BLE001 -- a campaign script must survive one bad scenario

                failed += 1
                failure_reasons[type(exc).__name__] += 1

    campaign_wall_seconds = time.perf_counter() - campaign_start
    print(f"Generation+simulation+extraction done: {campaign_wall_seconds:.1f}s, {len(all_rows)} rows, "
          f"{accepted} accepted, {failed} failed", flush=True)

    # =====================================================
    # Phases 13-16 -- data-quality and coverage analysis.
    # =====================================================

    def _timed(label, fn, *args):
        t0 = time.perf_counter()
        result = fn(*args)
        print(f"  analysis[{label}]: {time.perf_counter() - t0:.1f}s", flush=True)
        return result

    balance = _timed("class_balance_report", class_balance_report, all_rows)
    horizons = _timed("horizon_analysis", horizon_analysis, all_rows)
    recommended_horizon = recommend_first_horizon(horizons) if horizons else None

    candidate_type_row_counts = dict(Counter(row["candidate_type"] for row in all_rows))

    stair_report = _timed("stair_feature_repair_report", stair_feature_repair_report, all_rows)
    bottleneck_combinations = _timed(
        "multi_bottleneck_candidate_type_combinations", multi_bottleneck_candidate_type_combinations, all_rows,
    )
    lockout_report = _timed("total_lockout_scenarios", total_lockout_scenarios, scenario_metadata)

    # multiple_simultaneous_bottleneck_rows counts ROWS belonging to any
    # (scenario, time, horizon) bucket with 2+ distinct positive candidates.
    multi_bottleneck_row_count = 0
    bucket_sizes: dict = {}
    for row in all_rows:
        if row["target"] is True:
            key = (row["scenario_id"], row["observation_time"], row["prediction_horizon"])
            bucket_sizes[key] = bucket_sizes.get(key, 0) + 1
    for row in all_rows:
        if row["target"] is True:
            key = (row["scenario_id"], row["observation_time"], row["prediction_horizon"])
            if bucket_sizes.get(key, 0) >= 2:
                multi_bottleneck_row_count += 1

    family_counts = topology_family_distribution(scenario_metadata)

    actual_coverage_counts = {
        "single_exit_scenarios": family_counts.get("single_exit_lowrise", 0),
        "multi_floor_scenarios": sum(
            count for family, count in family_counts.items()
            if family in ("twin_stair_highrise", "v1_topology_fixed")
        ),
        "stair_candidate_rows_with_real_demand": sum(
            1 for row in all_rows
            if row["candidate_type"] == "Stair" and (row["candidate_queue_length"] > 0 or row["candidate_approaching_count"] > 0)
        ),
        "high_occupancy_scenarios": occupancy_strata(scenario_metadata).get("HIGH", 0),
        "multiple_simultaneous_bottleneck_rows": multi_bottleneck_row_count,
        "total_lockout_scenarios_with_rows": lockout_report["total_lockout_with_candidate_rows"],
    }

    coverage_verification = verify_coverage_targets(actual_coverage_counts, COVERAGE_TARGETS)

    scenario_diversity = _timed("scenario_diversity_report", scenario_diversity_report, scenario_metadata)
    candidate_utilization = _timed("candidate_utilization_report", candidate_utilization_report, all_rows)
    feature_distributions = _timed("feature_distribution_report", feature_distribution_report, all_rows)
    label_bias = _timed("label_bias_report", label_bias_report, all_rows, scenario_metadata)
    temporal_coverage = _timed("temporal_coverage_report", temporal_coverage_report, all_rows, scenario_metadata)
    feature_correlations = _timed("feature_target_correlations", feature_target_correlations, all_rows)
    candidate_type_association = _timed(
        "categorical_target_association[candidate_type]", categorical_target_association, all_rows, "candidate_type",
    )
    congestion_level_association = _timed(
        "categorical_target_association[congestion_level]",
        categorical_target_association, all_rows, "candidate_congestion_level",
    )
    redundant_pairs = _timed("redundant_feature_pairs", redundant_feature_pairs, all_rows)
    quality = _timed("run_quality_checks", run_quality_checks, all_rows, known_candidate_ids)
    zero_distance = _timed("zero_walking_distance_candidates", zero_walking_distance_candidates, all_rows)
    duplicate_ids = _timed("duplicate_scenario_ids", duplicate_scenario_ids, scenario_metadata)
    operational = _timed(
        "operational_coverage_report", operational_coverage_report, all_rows, scenario_metadata, candidate_type_row_counts,
    )

    report = {
        "campaign_config": config.to_dict(),
        "scenario_campaign": {
            "requested": sum(spec.scenario_count for spec in specs),
            "accepted": accepted,
            "failed": failed,
            "failure_reasons": dict(failure_reasons),
        },
        "row_counts": {
            "candidate_time_rows": len(all_rows),
            "distinct_scenario_count": balance["distinct_scenario_count"],
            "rows_per_scenario_mean": (len(all_rows) / accepted) if accepted else None,
            "candidate_type_row_counts": candidate_type_row_counts,
        },
        "class_balance": balance,
        "horizon_analysis": horizons,
        "recommended_first_horizon_seconds": recommended_horizon,
        "topology_family_distribution": topology_family_distribution(scenario_metadata),
        "structural_distribution": structural_distribution(scenario_metadata),
        "occupancy_strata": occupancy_strata(scenario_metadata),
        "scenario_diversity": scenario_diversity,
        "candidate_utilization": candidate_utilization,
        "feature_distributions": feature_distributions,
        "label_bias": label_bias,
        "temporal_coverage": temporal_coverage,
        "feature_target_correlations": feature_correlations,
        "candidate_type_target_association": candidate_type_association,
        "congestion_level_target_association": congestion_level_association,
        "redundant_feature_pairs": redundant_pairs,
        "quality_checks": quality,
        "zero_walking_distance_candidates": zero_distance,
        "duplicate_scenario_ids": duplicate_ids,
        "operational_coverage": operational,
        "stair_feature_repair": stair_report,
        "multi_bottleneck_candidate_type_combinations": bottleneck_combinations,
        "total_lockout": lockout_report,
        "coverage_target_verification": coverage_verification,
        "performance": {
            "campaign_wall_seconds": campaign_wall_seconds,
            "simulation_execution_wall_seconds": simulation_wall_seconds,
            "dataset_extraction_wall_seconds": extraction_wall_seconds,
            "rows_generated": len(all_rows),
            "extraction_rows_per_second": (len(all_rows) / extraction_wall_seconds) if extraction_wall_seconds > 0 else None,
        },
        "dataset_version": dataset_version(recommended_horizon or 0.0, campaign_version=config.campaign_version).to_dict(),
    }

    output_dir = Path(__file__).resolve().parent.parent / "data" / "predictive_dataset_campaign_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "campaign_v2_report.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, default=str)

    scenario_metadata_path = output_dir / "scenario_metadata.json"
    with open(scenario_metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(scenario_metadata, metadata_file, indent=2, default=str)

    csv_path = export_candidate_dataset_csv(all_rows, str(output_dir / "candidate_dataset_v2.csv"))

    print(f"Accepted {accepted}, failed {failed}. Rows: {len(all_rows)}. Wall: {campaign_wall_seconds:.1f}s")
    print(f"Wrote report to {report_path}")
    print(f"Wrote scenario metadata to {scenario_metadata_path}")
    print(f"Wrote dataset CSV to {csv_path}")


# =====================================================


def _scenario_metadata_entry(
    scenario, movement_result, *, topology_family: str,
    floor_count: int, exit_count: int, stair_count: int, door_count: int, contributed_rows: bool,
) -> dict:

    fire = scenario.fire

    blocked_door_count = sum(1 for state in scenario.door_states if state.state != DoorState.OPEN)
    blocked_exit_count = sum(1 for state in scenario.exit_states if not state.is_open)
    unavailable_stair_count = sum(1 for state in scenario.stair_states if state.availability == StairAvailability.CLOSED)

    return {
        "scenario_id": scenario.metadata.scenario_id,
        "topology_family": topology_family,
        "floor_count": floor_count,
        "exit_count": exit_count,
        "stair_count": stair_count,
        "door_count": door_count,
        "total_occupants": len(scenario.occupants),
        "ignition_zone_id": fire.ignition_zone_id if fire is not None else None,
        "fire_growth_time_seconds": fire.growth_parameters.get("growth_time") if fire is not None else None,
        "fire_profile": fire.fire_profile if fire is not None else None,
        "blocked_door_count": blocked_door_count,
        "blocked_exit_count": blocked_exit_count,
        "unavailable_stair_count": unavailable_stair_count,
        "evacuation_duration": movement_result.total_evacuation_time,
        "unreachable_occupant_count": len(movement_result.unreachable_occupant_ids),
        "contributed_rows": contributed_rows,
    }


if __name__ == "__main__":
    main()
