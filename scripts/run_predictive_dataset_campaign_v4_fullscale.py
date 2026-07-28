"""Predictive Dataset V4 milestone, Phase 16/18 -- FULL-SCALE cross-
family structural-generalization campaign. 125 scenarios x 24
structural variants (3,000 scenarios total) -- the Phase 16 scale
decision, made from the pilot's evidence (600 scenarios / 770,358 rows
/ 88.3s / 8,725 rows/sec / 0 failures / every variant AND every family
contributed rows / no zero-walking-distance regressions / graph-context
distributions genuinely broadened, not identical to the old 4
families). 3,000 sits inside the milestone's own suggested 2,500-3,500
range, chosen for BALANCED coverage across all 24 variants (matching
V3's own 150/variant density closely) rather than row-count
maximization.

Reuses predictive_dataset.campaign_runner_v4.run_campaign_v4 verbatim
(same memory-safe streaming core the pilot already validated).

Usage: python scripts/run_predictive_dataset_campaign_v4_fullscale.py
"""

import csv as csv_module
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil

from predictive_dataset.campaign_config_v4 import (
    COVERAGE_TARGETS_V4,
    FULLSCALE_SCENARIOS_PER_VARIANT,
    build_campaign_config_v4,
)
from predictive_dataset.campaign_runner_v4 import run_campaign_v4
from predictive_dataset.quality_checks import zero_walking_distance_candidates
from predictive_dataset.topologies_v3 import with_scenario_count
from predictive_dataset.topologies_v4 import all_structural_variants_v4
from predictive_dataset.topology_diversity_v3 import structural_diversity_report

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v4"


def _verify_coverage(csv_path: Path, scenario_metadata: list) -> dict:
    """Streams the CSV once (never loads it whole) to mechanically check
    every one of COVERAGE_TARGETS_V4 against the ACTUAL generated data."""

    variant_scenarios = {}
    family_scenarios = {}
    for entry in scenario_metadata:
        variant_scenarios.setdefault(entry["structural_variant_id"], set()).add(entry["scenario_id"])
        family_scenarios.setdefault(entry["topology_family"], set()).add(entry["scenario_id"])

    high_occupancy_scenarios = {e["scenario_id"] for e in scenario_metadata if e["total_occupants"] >= 30}
    multi_floor_scenarios = {e["scenario_id"] for e in scenario_metadata if e["floor_count"] >= 2}
    total_lockout_scenarios_with_rows = {
        e["scenario_id"] for e in scenario_metadata
        if e["contributed_rows"] and e.get("blocked_exit_count", 0) >= e.get("exit_count", 1) and e.get("exit_count", 0) > 0
    }
    genuine_cycle_scenarios = {
        e["scenario_id"] for e in scenario_metadata
        if e.get("has_cycle") or e["topology_family"] == "ring_corridor" or e["structural_variant_id"] == "v1_fixed_dual_stair"
    }

    stair_rows_with_demand = 0
    multi_bottleneck_buckets: dict = {}
    zero_distance_seen: dict = {}
    non_bridge_rows = 0
    high_catchment_rows = 0
    high_betweenness_rows = 0
    mw_rows = 0
    rc_rows = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:

            family = row["topology_family"]
            if family == "multi_wing":
                mw_rows += 1
            elif family == "ring_corridor":
                rc_rows += 1

            if row["candidate_type"] == "Stair":
                qlen = float(row["candidate_queue_length"]) if row["candidate_queue_length"] not in ("", "None") else 0
                approaching = float(row["candidate_approaching_count"]) if row["candidate_approaching_count"] not in ("", "None") else 0
                if qlen > 0 or approaching > 0:
                    stair_rows_with_demand += 1

            if row["target_v2"] == "True":
                key = (row["scenario_id"], row["observation_time"])
                multi_bottleneck_buckets[key] = multi_bottleneck_buckets.get(key, 0) + 1

            if row["candidate_walking_distance"] in ("0.0", "0"):
                zero_distance_seen[row["candidate_id"]] = zero_distance_seen.get(row["candidate_id"], 0) + 1

            if row["candidate_is_bridge"] == "False":
                non_bridge_rows += 1
            if float(row["candidate_upstream_catchment_count"]) >= 5:
                high_catchment_rows += 1
            if float(row["candidate_betweenness_centrality"]) >= 0.5:
                high_betweenness_rows += 1

    multi_bottleneck_rows = sum(1 for c in multi_bottleneck_buckets.values() if c >= 2)

    actual = {
        "every_structural_variant_represented": len(variant_scenarios),
        "every_family_represented": len(family_scenarios),
        "high_occupancy_scenarios": len(high_occupancy_scenarios),
        "multiple_simultaneous_bottleneck_rows": multi_bottleneck_rows,
        "stair_candidate_rows_with_real_demand": stair_rows_with_demand,
        "total_lockout_scenarios_with_rows": len(total_lockout_scenarios_with_rows),
        "multi_floor_scenarios": len(multi_floor_scenarios),
        "multi_wing_family_rows": mw_rows,
        "ring_corridor_family_rows": rc_rows,
        "genuine_zone_cycle_scenarios": len(genuine_cycle_scenarios),
        "non_bridge_candidate_rows": non_bridge_rows,
        "high_catchment_candidate_rows": high_catchment_rows,
        "high_betweenness_candidate_rows": high_betweenness_rows,
    }

    results = {}
    for name, target in COVERAGE_TARGETS_V4.items():
        actual_value = actual.get(name, 0)
        results[name] = {
            "description": target.description, "minimum": target.minimum_count,
            "actual": actual_value, "passed": actual_value >= target.minimum_count,
        }

    zero_distance_check = zero_walking_distance_candidates([
        {"candidate_id": cid, "candidate_walking_distance": 0.0} for cid in zero_distance_seen
    ]) if zero_distance_seen else []

    return {"coverage_targets": results, "all_passed": all(r["passed"] for r in results.values()),
            "zero_walking_distance_candidates_with_rows": zero_distance_check}


def main() -> None:

    variants = all_structural_variants_v4()

    fullscale_variants = tuple(
        type(v)(v.family, v.variant_id, v.variant_label, with_scenario_count(v.topology, FULLSCALE_SCENARIOS_PER_VARIANT))
        for v in variants
    )

    config = build_campaign_config_v4(fullscale_variants)
    diversity = structural_diversity_report(variants)

    mem_before = psutil.virtual_memory()
    print(f"Memory before: available={mem_before.available/1e6:.0f}MB used={mem_before.percent}%", flush=True)

    result = run_campaign_v4(fullscale_variants, config, OUTPUT_DIR, log_every=200)

    mem_after = psutil.virtual_memory()
    print(f"Memory after: available={mem_after.available/1e6:.0f}MB used={mem_after.percent}%", flush=True)

    print("Verifying coverage targets (streaming)...", flush=True)
    coverage = _verify_coverage(Path(result["csv_path"]), result["scenario_metadata"])

    report = {
        "campaign_config": config.to_dict(),
        "structural_diversity": diversity,
        "requested_scenarios": sum(v.topology.scenario_count for v in fullscale_variants),
        "accepted_scenarios": result["accepted_scenarios"],
        "failed_scenarios": result["failed_scenarios"],
        "row_count": result["row_count"],
        "wall_seconds": result["wall_seconds"],
        "rows_per_second": result["rows_per_second"],
        "csv_columns": result["csv_columns"],
        "memory_before_mb_available": mem_before.available / 1e6,
        "memory_after_mb_available": mem_after.available / 1e6,
        "coverage_verification": coverage,
    }

    with open(OUTPUT_DIR / "campaign_v4_fullscale_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps({
        "accepted": result["accepted_scenarios"], "failed": result["failed_scenarios"],
        "rows": result["row_count"], "wall_seconds": result["wall_seconds"],
        "all_coverage_targets_passed": coverage["all_passed"],
        "zero_walking_distance_candidates_with_rows": coverage["zero_walking_distance_candidates_with_rows"],
    }, indent=2))
    print(f"Wrote {OUTPUT_DIR / 'campaign_v4_fullscale_report.json'}")


if __name__ == "__main__":
    main()
