"""Localized Predictive Model V2.2 milestone, Phase 7/8 -- FULL-SCALE
target-semantics audit and counterfactual-target analysis. Re-simulates
the SAME 2,500 scenarios (same master_seed=20270115, same topology
definitions, byte-for-byte deterministic reproduction -- see
run_predictive_dataset_campaign_v2_2_fullscale.py's own docstring for
why this is re-extraction, not "new data") and, for every candidate,
walks its edge's real event timeline directly -- NOT from the flat CSV
(which has no start/end timestamps) -- to answer two questions at full
scale that the V2.1 investigation only answered on a 15-25 scenario
sample:

  1. How much of the current congestion target (>=2 concurrent
     occupants) is a zero-duration/near-zero timestamp-boundary
     handoff, per candidate type?
  2. Under counterfactual "require sustained overlap >= N seconds"
     target definitions (analysis-only, NEVER written back as the
     production target), how does each candidate type's positive rate
     change?

Does NOT modify predictive_dataset/target_generator.py or any frozen
schema. Writes one report JSON; no per-row feature CSV (this is a
statistics-only pass, deliberately lighter than the feature-extraction
campaign).

Usage: python scripts/run_target_semantics_audit_v2_2.py
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil

from behaviour_profile_resolver import register_occupants
from scenario_generator.batch_generator import iter_batch
from scenario_generator.request import BatchGenerationRequest
from scenario_runner import run as run_scenario_context
from simulation_runtime import SimulationRuntime
from ai_decision.engine import AIDecisionEngine

from predictive_dataset.campaign_config_v2 import MASTER_SEED_V2, MINIMUM_END_TIME_SECONDS
from predictive_dataset.candidate import enumerate_candidates
from predictive_dataset.target_generator import CONGESTION_THRESHOLD, generate_candidate_label
from predictive_dataset.target_semantics_analysis import counterfactual_positive, episode_durations_and_gaps
from predictive_dataset.topologies_v2 import all_topology_specs

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v2_2"

HORIZON = 20.0
COUNTERFACTUAL_MIN_DURATIONS = (0.0, 1.0, 5.0, 10.0)  # seconds -- 0.0 reproduces the current, unmodified target
MIN_AVAILABLE_MEMORY_BYTES = 300_000_000


def _check_memory(label: str) -> None:
    vm = psutil.virtual_memory()
    if vm.available < MIN_AVAILABLE_MEMORY_BYTES:
        raise MemoryError(f"Available memory critically low ({vm.available/1e6:.0f}MB) at {label!r} -- aborting.")


def main() -> None:

    specs = all_topology_specs()

    durations_by_type = defaultdict(list)
    gaps_by_type = defaultdict(list)

    trainable_counts = defaultdict(int)
    counterfactual_positive_counts = {d: defaultdict(int) for d in COUNTERFACTUAL_MIN_DURATIONS}
    production_positive_counts = defaultdict(int)
    consistency_mismatches = 0

    accepted = 0
    failed = 0

    campaign_start = time.perf_counter()

    for spec in specs:

        candidates = enumerate_candidates(spec.building)

        request = BatchGenerationRequest(
            definition=spec.definition, definition_id=f"predictive-dataset-v2-2-audit-{spec.name}",
            building=spec.building, master_seed=MASTER_SEED_V2, count=spec.scenario_count,
        )

        print(f"[{spec.name}] auditing {spec.scenario_count} scenarios", flush=True)
        family_index = 0

        for scenario in iter_batch(request):

            family_index += 1
            if family_index % 200 == 0:
                print(f"  [{spec.name}] {family_index}/{spec.scenario_count} "
                      f"(elapsed {time.perf_counter() - campaign_start:.1f}s)", flush=True)
                _check_memory(f"{spec.name} {family_index}")

            try:
                context = run_scenario_context(scenario, spec.building)
                register_occupants(context)

                decision_engine = AIDecisionEngine(base_engine=context.engine)
                runtime = SimulationRuntime(context, decision_engine, dt=5.0)
                runtime.clock.end_time = max(runtime.clock.end_time, MINIMUM_END_TIME_SECONDS)

                tick_results = runtime.run()
                movement_result = runtime.movement_result

                for candidate in candidates:

                    durations, gaps = episode_durations_and_gaps(movement_result, candidate.candidate_id)
                    durations_by_type[candidate.candidate_type].extend(durations)
                    gaps_by_type[candidate.candidate_type].extend(gaps)

                    for tick in tick_results:

                        label = generate_candidate_label(candidate.candidate_id, movement_result, tick.time, HORIZON)
                        if label.target is None:
                            continue  # currently_congested -- not applicable, matches production semantics

                        trainable_counts[candidate.candidate_type] += 1
                        if label.target:
                            production_positive_counts[candidate.candidate_type] += 1

                        for min_duration in COUNTERFACTUAL_MIN_DURATIONS:

                            is_positive = counterfactual_positive(
                                movement_result, candidate.candidate_id, tick.time, HORIZON, min_duration,
                            )
                            if is_positive:
                                counterfactual_positive_counts[min_duration][candidate.candidate_type] += 1

                            if min_duration == 0.0 and is_positive != bool(label.target):
                                consistency_mismatches += 1

                accepted += 1

            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  [{spec.name}] scenario failed: {exc}", flush=True)

    elapsed = time.perf_counter() - campaign_start
    print(f"Accepted {accepted}, failed {failed}. Wall: {elapsed:.1f}s")
    print(f"Consistency mismatches (min_duration=0.0 vs production target): {consistency_mismatches}")

    report = {
        "accepted_scenarios": accepted,
        "failed_scenarios": failed,
        "wall_seconds": elapsed,
        "congestion_threshold": CONGESTION_THRESHOLD,
        "horizon_seconds": HORIZON,
        "consistency_mismatches_vs_production_target": consistency_mismatches,
        "episode_duration_stats": {},
        "adjacent_gap_stats": {},
        "trainable_counts": dict(trainable_counts),
        "production_positive_counts": dict(production_positive_counts),
        "production_positive_rate": {},
        "counterfactual_positive_counts": {str(d): dict(v) for d, v in counterfactual_positive_counts.items()},
        "counterfactual_positive_rate": {},
    }

    for ctype in ("Door", "Exit", "Stair"):

        durations = np.array(durations_by_type.get(ctype, []))
        gaps = np.array(gaps_by_type.get(ctype, []))

        if len(durations) > 0:
            report["episode_duration_stats"][ctype] = {
                "n_episodes": int(len(durations)),
                "mean": float(durations.mean()),
                "median": float(np.median(durations)),
                "p95": float(np.percentile(durations, 95)),
                "zero_duration_count": int((durations <= 1e-9).sum()),
                "zero_duration_fraction": float((durations <= 1e-9).mean()),
                "near_zero_le_1s_fraction": float((durations <= 1.0).mean()),
                "sustained_ge_5s_fraction": float((durations >= 5.0).mean()),
                "sustained_ge_10s_fraction": float((durations >= 10.0).mean()),
            }
        else:
            report["episode_duration_stats"][ctype] = {"n_episodes": 0}

        if len(gaps) > 0:
            report["adjacent_gap_stats"][ctype] = {
                "n_adjacent_pairs": int(len(gaps)),
                "exact_zero_gap_fraction": float((np.abs(gaps) < 1e-9).mean()),
                "mean_gap": float(gaps.mean()),
                "median_gap": float(np.median(gaps)),
            }
        else:
            report["adjacent_gap_stats"][ctype] = {"n_adjacent_pairs": 0}

        n = trainable_counts.get(ctype, 0)
        if n:
            report["production_positive_rate"][ctype] = production_positive_counts.get(ctype, 0) / n
            report["counterfactual_positive_rate"][ctype] = {
                str(d): counterfactual_positive_counts[d].get(ctype, 0) / n for d in COUNTERFACTUAL_MIN_DURATIONS
            }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "target_semantics_audit_v2_2_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
