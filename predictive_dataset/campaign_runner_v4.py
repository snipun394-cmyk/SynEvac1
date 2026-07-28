import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil

from behaviour_profile_resolver import register_occupants
from scenario_generator.batch_generator import iter_batch
from scenario_generator.request import BatchGenerationRequest
from scenario_runner import run as run_scenario_context
from simulation_runtime import SimulationRuntime
from ai_decision.engine import AIDecisionEngine

from scenario.engineering_state import DoorState, StairAvailability

from predictive_dataset.campaign_config_v4 import CampaignConfigV4
from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.graph_context_v4 import compute_graph_context_for_building
from predictive_dataset.schema_v4 import CANDIDATE_FEATURE_NAMES_V4
from predictive_dataset.simulation_extractor_v2_1 import build_alternative_route_counts
from predictive_dataset.simulation_extractor_v4 import extract_v4_candidate_features
from predictive_dataset.target_generator_v2 import compute_qualifying_onsets, generate_candidate_label_v2
from predictive_dataset.topology_signature_v4 import compute_structural_signature_v4


# =====================================================
# Predictive Dataset V4 milestone, Phase 17/18 -- shared, memory-safe
# campaign execution loop. Directly adapted from
# predictive_dataset/campaign_runner_v3.py (streams rows to CSV inside
# the tick/candidate loop, never accumulates in memory; periodic
# psutil availability check aborts loudly) -- the ONLY functional
# difference is emitting CANDIDATE_FEATURE_NAMES_V4's full 15-column
# feature set (via extract_v4_candidate_features) instead of V2.1's 12,
# and precomputing graph_context ONCE per variant (compute_graph_
# context_for_building, Phase 1's shared sim/live function) alongside
# the already-existing alt_route_counts precomputation.
#
# Target V2 ONLY, unchanged (Phase 10's freeze) -- v3's own
# campaign_runner_v3.py is NOT modified, still produces Dataset V3
# exactly as it always did.
# =====================================================

IDENTITY_COLUMNS = ("scenario_id", "observation_time", "candidate_id", "candidate_type")
LABEL_COLUMNS = ("currently_congested_v2", "had_any_activity_in_window_v2", "target_v2", "lead_time_seconds_v2")

# CANDIDATE_FEATURE_NAMES_V4 legitimately includes "candidate_type"
# (predictive_dataset/schema.py's own documented "listed both as row
# identity AND as a feature" convention) -- IDENTITY_COLUMNS already
# carries it once, so it is excluded here to avoid a duplicate CSV
# column (the exact discipline campaign_runner_v3.py's own hand-picked
# BASE_FEATURE_NAMES already follows, reused rather than re-litigated).
FEATURE_COLUMNS_FOR_CSV = tuple(name for name in CANDIDATE_FEATURE_NAMES_V4 if name != "candidate_type")

CSV_COLUMNS = (
    IDENTITY_COLUMNS + ("topology_family", "structural_variant_id")
    + FEATURE_COLUMNS_FOR_CSV + LABEL_COLUMNS
)

MIN_AVAILABLE_MEMORY_BYTES = 300_000_000


def _check_memory(label: str) -> None:
    vm = psutil.virtual_memory()
    if vm.available < MIN_AVAILABLE_MEMORY_BYTES:
        raise MemoryError(f"Available memory critically low ({vm.available / 1e6:.0f}MB) at {label!r} -- aborting.")


def _min_lead_time(onsets, time_: float, horizon: float):
    leads = [onset - time_ for onset, _ in onsets if time_ < onset <= time_ + horizon]
    return min(leads) if leads else None


def run_campaign_v4(variants, config: CampaignConfigV4, output_dir: Path, *, log_every: int = 50) -> Dict[str, Any]:
    """Runs one campaign (pilot or full-scale) over `variants`
    (predictive_dataset.topologies_v4.StructuralVariant, with
    scenario_count already overridden via with_scenario_count by the
    caller). Writes candidate_dataset_v4.csv (streamed) and returns the
    scenario_metadata list plus a summary report dict."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "candidate_dataset_v4.csv"

    scenario_metadata: List[Dict[str, Any]] = []
    accepted = 0
    failed = 0
    row_count = 0

    campaign_start = time.perf_counter()

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for variant in variants:

            spec = variant.topology
            candidates = enumerate_candidates(spec.building)
            edges = edges_by_candidate_id(spec.building)
            alt_route_counts = build_alternative_route_counts(candidates)
            graph_context = compute_graph_context_for_building(spec.building)
            signature = compute_structural_signature_v4(variant.family, variant.variant_id, spec.building)

            request = BatchGenerationRequest(
                definition=spec.definition, definition_id=f"predictive-dataset-v4-{variant.variant_id}",
                building=spec.building, master_seed=config.master_seed, count=spec.scenario_count,
            )

            print(f"[{variant.variant_id}] generating {spec.scenario_count} scenarios "
                  f"(family={variant.family}, candidates={len(candidates)})", flush=True)

            variant_index = 0

            for scenario in iter_batch(request):

                variant_index += 1
                if variant_index % log_every == 0:
                    print(f"  [{variant.variant_id}] {variant_index}/{spec.scenario_count} "
                          f"(elapsed {time.perf_counter() - campaign_start:.1f}s, rows so far {row_count})", flush=True)
                    _check_memory(f"{variant.variant_id} {variant_index}/{spec.scenario_count}")

                try:
                    context = run_scenario_context(scenario, spec.building)
                    register_occupants(context)

                    decision_engine = AIDecisionEngine(base_engine=context.engine)
                    runtime = SimulationRuntime(context, decision_engine, dt=config.tick_dt_seconds)
                    runtime.clock.end_time = max(runtime.clock.end_time, config.minimum_end_time_seconds)

                    tick_results = runtime.run()
                    movement_result = runtime.movement_result
                    building = context.building

                    total_occupants = len(movement_result.occupants)

                    onsets_by_candidate = {
                        candidate.candidate_id: compute_qualifying_onsets(movement_result, candidate.candidate_id)
                        for candidate in candidates
                    }

                    scenario_row_count = 0

                    for tick in tick_results:

                        for candidate in candidates:

                            edge = edges[candidate.candidate_id]
                            onsets = onsets_by_candidate[candidate.candidate_id]

                            features = extract_v4_candidate_features(
                                candidate, edge, tick.time,
                                building=building, movement_result=movement_result,
                                occupancy_snapshot=tick.occupancy_snapshot,
                                alternative_route_counts=alt_route_counts,
                                graph_context=graph_context,
                            )

                            label = generate_candidate_label_v2(
                                candidate.candidate_id, movement_result, tick.time, config.horizon_seconds, onsets=onsets,
                            )

                            row = {
                                "scenario_id": scenario.metadata.scenario_id,
                                "observation_time": tick.time,
                                "candidate_id": candidate.candidate_id,
                                "candidate_type": candidate.candidate_type,
                                "topology_family": variant.family,
                                "structural_variant_id": variant.variant_id,
                            }
                            for name in FEATURE_COLUMNS_FOR_CSV:
                                row[name] = features[name]

                            row["currently_congested_v2"] = label.currently_congested
                            row["had_any_activity_in_window_v2"] = label.had_any_activity_in_window
                            row["target_v2"] = label.target
                            row["lead_time_seconds_v2"] = (
                                _min_lead_time(onsets, tick.time, config.horizon_seconds) if label.target else None
                            )

                            writer.writerow(row)
                            scenario_row_count += 1

                    row_count += scenario_row_count
                    accepted += 1

                    fire = scenario.fire
                    blocked_door_count = sum(1 for state in scenario.door_states if state.state != DoorState.OPEN)
                    blocked_exit_count = sum(1 for state in scenario.exit_states if not state.is_open)
                    unavailable_stair_count = sum(
                        1 for state in scenario.stair_states if state.availability == StairAvailability.CLOSED
                    )

                    scenario_metadata.append({
                        "scenario_id": scenario.metadata.scenario_id,
                        "topology_family": variant.family,
                        "structural_variant_id": variant.variant_id,
                        "floor_count": signature.v3_signature.floor_count, "zone_count": signature.v3_signature.zone_count,
                        "exit_count": signature.v3_signature.exit_count, "stair_count": signature.v3_signature.stair_count,
                        "door_count": signature.v3_signature.door_count, "candidate_count": signature.v3_signature.candidate_count,
                        "mean_alternative_route_count": signature.v3_signature.mean_alternative_route_count,
                        "has_cycle": signature.has_cycle,
                        "bridge_edge_fraction": signature.bridge_edge_fraction,
                        "max_upstream_catchment": signature.max_upstream_catchment,
                        "total_occupants": total_occupants,
                        "ignition_zone_id": fire.ignition_zone_id if fire is not None else None,
                        "fire_growth_time_seconds": fire.growth_parameters.get("growth_time") if fire is not None else None,
                        "fire_profile": fire.fire_profile if fire is not None else None,
                        "blocked_door_count": blocked_door_count,
                        "blocked_exit_count": blocked_exit_count,
                        "unavailable_stair_count": unavailable_stair_count,
                        "evacuation_duration": movement_result.total_evacuation_time,
                        "unreachable_occupant_count": len(movement_result.unreachable_occupant_ids),
                        "contributed_rows": scenario_row_count > 0,
                    })

                except Exception as exc:  # noqa: BLE001 -- a campaign must survive one bad scenario
                    failed += 1
                    print(f"  [{variant.variant_id}] scenario failed: {type(exc).__name__}: {exc}", flush=True)

    elapsed = time.perf_counter() - campaign_start
    print(f"Accepted {accepted}, failed {failed}. Rows: {row_count}. Wall: {elapsed:.1f}s "
          f"({row_count / elapsed:.0f} rows/sec)" if elapsed > 0 else "", flush=True)

    with open(output_dir / "scenario_metadata.json", "w", encoding="utf-8") as f:
        json.dump(scenario_metadata, f, indent=2, default=str)

    return {
        "csv_path": str(csv_path),
        "scenario_metadata": scenario_metadata,
        "accepted_scenarios": accepted,
        "failed_scenarios": failed,
        "row_count": row_count,
        "wall_seconds": elapsed,
        "rows_per_second": (row_count / elapsed) if elapsed > 0 else None,
        "csv_columns": list(CSV_COLUMNS),
    }
