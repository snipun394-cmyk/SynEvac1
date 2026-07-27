"""Predictive Dataset V3 milestone, Phase 9/13/17 -- coverage and
Target V2 quality report for a V3 campaign (pilot or full-scale).
Computes the COVERAGE_TARGETS_V3 mechanical pass/fail checks plus
Target V2 positive rates by family/variant/candidate_type, without
holding the whole CSV as a list-of-dicts (uses pandas, chunked for the
full-scale case).

Usage: python scripts/predictive_dataset_v3_coverage_report.py <campaign_dir>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from predictive_dataset.campaign_config_v3 import COVERAGE_TARGETS_V3

CHUNK_SIZE = 250_000

MULTI_STAIR_VARIANTS = {
    "twin_stair_highrise", "twin_stair_highrise_3stair", "twin_stair_chained_core",
    "v1_fixed_dual_stair", "v1_fixed_three_floor",
}
CHAINED_STAIR_VARIANTS = {"twin_stair_chained_core", "v1_fixed_three_floor"}
REDUCED_REDUNDANCY_VARIANTS = {"multi_exit_reduced_redundancy", "multi_exit_linear_chain"}
MULTI_FLOOR_VARIANT_MIN_FLOORS = 2


def main() -> None:

    campaign_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if campaign_dir is None:
        raise SystemExit("Usage: predictive_dataset_v3_coverage_report.py <campaign_dir>")

    csv_path = campaign_dir / "candidate_dataset_v3.csv"
    metadata_path = campaign_dir / "scenario_metadata.json"

    with open(metadata_path, encoding="utf-8") as f:
        scenario_metadata = json.load(f)

    meta_by_scenario = {m["scenario_id"]: m for m in scenario_metadata}

    single_exit_scenarios = sum(1 for m in scenario_metadata if m["topology_family"] == "single_exit_lowrise")
    multi_floor_scenarios = sum(1 for m in scenario_metadata if m["floor_count"] >= MULTI_FLOOR_VARIANT_MIN_FLOORS)
    high_occupancy_scenarios = sum(1 for m in scenario_metadata if m["total_occupants"] >= 30)
    multi_stair_scenarios = sum(1 for m in scenario_metadata if m["structural_variant_id"] in MULTI_STAIR_VARIANTS)
    chained_stair_scenarios = sum(1 for m in scenario_metadata if m["structural_variant_id"] in CHAINED_STAIR_VARIANTS)
    reduced_redundancy_scenarios = sum(1 for m in scenario_metadata if m["structural_variant_id"] in REDUCED_REDUNDANCY_VARIANTS)
    total_lockout_scenarios = [
        m for m in scenario_metadata
        if m["blocked_exit_count"] >= m["exit_count"] and m["exit_count"] > 0
    ]
    # Phase 9 definition (matching V2's own): a lockout scenario "with
    # rows" means it contributed ANY candidate-time row (the V1 bug this
    # guards against was ZERO rows for an unreachable-lockout scenario,
    # not the absence of a positively-labeled row).
    total_lockout_scenarios_with_rows = sum(1 for m in total_lockout_scenarios if m["contributed_rows"])

    variant_scenario_counts = {}
    for m in scenario_metadata:
        vid = m["structural_variant_id"]
        variant_scenario_counts[vid] = variant_scenario_counts.get(vid, 0) + (1 if m["contributed_rows"] else 0)
    variants_represented = sum(1 for c in variant_scenario_counts.values() if c > 0)

    candidate_type_counts = {"Door": 0, "Exit": 0, "Stair": 0}
    target_counts_by_type = {"Door": {"True": 0, "False": 0, "None": 0}, "Exit": {"True": 0, "False": 0, "None": 0}, "Stair": {"True": 0, "False": 0, "None": 0}}
    target_counts_by_family = {}
    target_counts_by_variant = {}
    stair_rows_with_demand = 0
    multi_bottleneck_bucket_counts = {}

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):

        for ctype in ("Door", "Exit", "Stair"):
            sub = chunk[chunk["candidate_type"] == ctype]
            candidate_type_counts[ctype] += len(sub)
            for label in ("True", "False"):
                target_counts_by_type[ctype][label] += (sub["target_v2"].astype(str) == label).sum()
            target_counts_by_type[ctype]["None"] += sub["target_v2"].isna().sum()

        for family, sub in chunk.groupby("topology_family"):
            bucket = target_counts_by_family.setdefault(family, {"True": 0, "False": 0})
            bucket["True"] += (sub["target_v2"].astype(str) == "True").sum()
            bucket["False"] += (sub["target_v2"].astype(str) == "False").sum()

        for variant, sub in chunk.groupby("structural_variant_id"):
            bucket = target_counts_by_variant.setdefault(variant, {"True": 0, "False": 0})
            bucket["True"] += (sub["target_v2"].astype(str) == "True").sum()
            bucket["False"] += (sub["target_v2"].astype(str) == "False").sum()

        stair_chunk = chunk[chunk["candidate_type"] == "Stair"]
        stair_rows_with_demand += (
            (stair_chunk["candidate_queue_length"].fillna(0) > 0) | (stair_chunk["candidate_approaching_count"].fillna(0) > 0)
        ).sum()

        positive_rows = chunk[chunk["target_v2"].astype(str) == "True"]
        for (scenario_id, obs_time), bucket in positive_rows.groupby(["scenario_id", "observation_time"]).size().items():
            key = (scenario_id, obs_time)
            multi_bottleneck_bucket_counts[key] = multi_bottleneck_bucket_counts.get(key, 0) + bucket

    multi_bottleneck_rows = sum(c for c in multi_bottleneck_bucket_counts.values() if c >= 2)

    actual_counts = {
        "every_structural_variant_represented": variants_represented,
        "single_exit_family_scenarios": single_exit_scenarios,
        "multi_floor_scenarios": multi_floor_scenarios,
        "stair_candidate_rows_with_real_demand": int(stair_rows_with_demand),
        "high_occupancy_scenarios": high_occupancy_scenarios,
        "multiple_simultaneous_bottleneck_rows": multi_bottleneck_rows,
        "total_lockout_scenarios_with_rows": total_lockout_scenarios_with_rows,
        "multi_stair_scenarios": multi_stair_scenarios,
        "chained_stair_connectivity_scenarios": chained_stair_scenarios,
        "reduced_redundancy_exit_scenarios": reduced_redundancy_scenarios,
    }

    coverage_verification = {
        name: {
            "description": target.description,
            "minimum_count": target.minimum_count,
            "actual_count": actual_counts.get(name),
            "passed": actual_counts.get(name, 0) >= target.minimum_count,
        }
        for name, target in COVERAGE_TARGETS_V3.items()
    }

    total_true = sum(b["True"] for b in target_counts_by_type.values())
    total_denom = sum(b["True"] + b["False"] for b in target_counts_by_type.values())

    report = {
        "campaign_dir": str(campaign_dir),
        "scenario_count": len(scenario_metadata),
        "coverage_verification": coverage_verification,
        "all_coverage_targets_passed": all(v["passed"] for v in coverage_verification.values()),
        "candidate_type_row_counts": candidate_type_counts,
        "target_v2_by_candidate_type": target_counts_by_type,
        "target_v2_positive_rate_overall": (total_true / total_denom) if total_denom else None,
        "target_v2_positive_rate_by_candidate_type": {
            ctype: (b["True"] / (b["True"] + b["False"])) if (b["True"] + b["False"]) > 0 else None
            for ctype, b in target_counts_by_type.items()
        },
        "target_v2_positive_rate_by_family": {
            fam: (b["True"] / (b["True"] + b["False"])) if (b["True"] + b["False"]) > 0 else None
            for fam, b in target_counts_by_family.items()
        },
        "target_v2_positive_rate_by_variant": {
            var: (b["True"] / (b["True"] + b["False"])) if (b["True"] + b["False"]) > 0 else None
            for var, b in target_counts_by_variant.items()
        },
    }

    out_path = campaign_dir / "coverage_and_target_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps({
        "all_coverage_targets_passed": report["all_coverage_targets_passed"],
        "coverage_verification": {k: v["passed"] for k, v in coverage_verification.items()},
        "target_v2_positive_rate_overall": report["target_v2_positive_rate_overall"],
        "target_v2_positive_rate_by_candidate_type": report["target_v2_positive_rate_by_candidate_type"],
    }, indent=2, default=str))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
