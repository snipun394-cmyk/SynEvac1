"""Predictive Dataset V4 milestone, Phase 11 -- CONTROLLED PILOT
CAMPAIGN. ~25 scenarios x 24 structural variants (16 reused from V3 +
8 new: multi_wing x4, ring_corridor x4) to verify every structural
variant builds successfully under the NEW V4 extraction pipeline
(graph-context fields included), occupants can evacuate, Target V2
produces positives, and no known regressions (zero-walking-distance
Stair bug, total-lockout row loss, invalid references) reappear --
BEFORE committing to a multi-thousand-scenario full-scale run.

Usage: python scripts/run_predictive_dataset_campaign_v4_pilot.py
"""

import csv as csv_module
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictive_dataset.campaign_config_v4 import PILOT_SCENARIOS_PER_VARIANT, build_campaign_config_v4
from predictive_dataset.campaign_runner_v4 import run_campaign_v4
from predictive_dataset.quality_checks import zero_walking_distance_candidates
from predictive_dataset.topologies_v3 import with_scenario_count
from predictive_dataset.topologies_v4 import all_structural_variants_v4
from predictive_dataset.topology_diversity_v3 import structural_diversity_report

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v4_pilot"


def main() -> None:

    variants = all_structural_variants_v4()

    pilot_variants = tuple(
        type(v)(v.family, v.variant_id, v.variant_label, with_scenario_count(v.topology, PILOT_SCENARIOS_PER_VARIANT))
        for v in variants
    )

    config = build_campaign_config_v4(pilot_variants, campaign_version="predictive_dataset_campaign_v4_pilot")

    diversity = structural_diversity_report(variants)

    result = run_campaign_v4(pilot_variants, config, OUTPUT_DIR)

    rows_path = Path(result["csv_path"])

    candidate_type_counts = {}
    target_true = 0
    target_false = 0
    target_none = 0
    variant_row_counts = {}
    family_row_counts = {}
    stair_rows_with_demand = 0
    multi_bottleneck_buckets = {}
    zero_distance_seen = {}
    is_bridge_counts = {"True": 0, "False": 0}
    betweenness_values = []
    catchment_values = []

    with open(rows_path, newline="", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:

            ctype = row["candidate_type"]
            candidate_type_counts[ctype] = candidate_type_counts.get(ctype, 0) + 1
            variant_id = row["structural_variant_id"]
            variant_row_counts[variant_id] = variant_row_counts.get(variant_id, 0) + 1
            family = row["topology_family"]
            family_row_counts[family] = family_row_counts.get(family, 0) + 1

            target = row["target_v2"]
            if target == "True":
                target_true += 1
                key = (row["scenario_id"], row["observation_time"])
                multi_bottleneck_buckets[key] = multi_bottleneck_buckets.get(key, 0) + 1
            elif target == "False":
                target_false += 1
            else:
                target_none += 1

            if ctype == "Stair":
                qlen = float(row["candidate_queue_length"]) if row["candidate_queue_length"] not in ("", "None") else 0
                approaching = float(row["candidate_approaching_count"]) if row["candidate_approaching_count"] not in ("", "None") else 0
                if qlen > 0 or approaching > 0:
                    stair_rows_with_demand += 1

            wdist = row["candidate_walking_distance"]
            if wdist in ("0.0", "0"):
                zero_distance_seen[row["candidate_id"]] = zero_distance_seen.get(row["candidate_id"], 0) + 1

            is_bridge_counts[row["candidate_is_bridge"]] = is_bridge_counts.get(row["candidate_is_bridge"], 0) + 1
            betweenness_values.append(float(row["candidate_betweenness_centrality"]))
            catchment_values.append(float(row["candidate_upstream_catchment_count"]))

    multi_bottleneck_rows = sum(1 for c in multi_bottleneck_buckets.values() if c >= 2)

    zero_distance_check = zero_walking_distance_candidates([
        {"candidate_id": cid, "candidate_walking_distance": 0.0} for cid in zero_distance_seen
    ]) if zero_distance_seen else []

    def _stats(values):
        if not values:
            return {"min": None, "max": None, "mean": None}
        return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}

    report = {
        "campaign_config": config.to_dict(),
        "structural_diversity": diversity,
        "generation": {
            "requested_scenarios": sum(v.topology.scenario_count for v in pilot_variants),
            "accepted_scenarios": result["accepted_scenarios"],
            "failed_scenarios": result["failed_scenarios"],
            "row_count": result["row_count"],
            "wall_seconds": result["wall_seconds"],
            "rows_per_second": result["rows_per_second"],
        },
        "variant_row_counts": variant_row_counts,
        "family_row_counts": family_row_counts,
        "every_variant_contributed_rows": all(
            variant_row_counts.get(v.variant_id, 0) > 0 for v in pilot_variants
        ),
        "every_family_contributed_rows": all(
            family_row_counts.get(family, 0) > 0
            for family in set(v.family for v in pilot_variants)
        ),
        "candidate_type_row_counts": candidate_type_counts,
        "target_v2_distribution": {"True": target_true, "False": target_false, "None (currently congested)": target_none},
        "target_v2_positive_rate": target_true / (target_true + target_false) if (target_true + target_false) > 0 else None,
        "stair_rows_with_real_demand": stair_rows_with_demand,
        "multi_bottleneck_rows": multi_bottleneck_rows,
        "zero_walking_distance_candidates_with_rows": zero_distance_check,
        "graph_context_distribution": {
            "is_bridge_row_counts": is_bridge_counts,
            "betweenness_centrality": _stats(betweenness_values),
            "upstream_catchment_count": _stats(catchment_values),
        },
    }

    with open(OUTPUT_DIR / "pilot_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps({
        "accepted": result["accepted_scenarios"], "failed": result["failed_scenarios"],
        "rows": result["row_count"], "every_variant_contributed_rows": report["every_variant_contributed_rows"],
        "every_family_contributed_rows": report["every_family_contributed_rows"],
        "target_v2_positive_rate": report["target_v2_positive_rate"],
        "zero_walking_distance_candidates_with_rows": zero_distance_check,
        "structural_diversity_all_distinct": diversity["all_genuinely_distinct"],
        "graph_context_distribution": report["graph_context_distribution"],
    }, indent=2, default=str))
    print(f"Wrote {OUTPUT_DIR / 'pilot_report.json'}")


if __name__ == "__main__":
    main()
