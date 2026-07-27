"""Predictive Dataset V3 milestone, Phases 14-16 -- structural
diversity comparison (V2 vs V3), feature-distribution overlap analysis,
and duplication/temporal-redundancy composition analysis. Chunked
throughout (Phase 12's memory-safe discipline extends to analysis, not
just generation) -- never loads the full ~2.7M-row CSV as one DataFrame
or one Python list of dicts.

Usage: python scripts/predictive_dataset_v3_post_campaign_analysis.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from predictive_dataset.topologies_v2 import all_topology_specs
from predictive_dataset.topologies_v3 import all_structural_variants_v3
from predictive_dataset.topology_diversity_v3 import compare_signature_sets
from predictive_dataset.topology_signature import compute_all_signatures, compute_structural_signature

REPO_ROOT = Path(__file__).resolve().parent.parent
V3_CAMPAIGN_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v3"
V2_CAMPAIGN_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v2"

CHUNK_SIZE = 250_000

FEATURE_COLUMNS_FOR_DUPLICATION = (
    "candidate_type", "total_active_occupant_count", "candidate_capacity", "candidate_walking_distance",
    "candidate_traversable", "candidate_adjacent_zone_occupancy", "candidate_queue_length",
    "candidate_approaching_count", "candidate_congestion_level", "candidate_recent_flow_rate",
    "candidate_congestion_trend", "candidate_alternative_route_count",
)

SHIFT_FEATURES = ("candidate_walking_distance", "candidate_alternative_route_count", "total_active_occupant_count")


def phase14_structural_comparison() -> dict:

    v2_signatures = compute_all_signatures(
        type("V2Wrap", (), {"family": spec.name, "variant_id": spec.name, "topology": spec})()
        for spec in all_topology_specs()
    )
    v3_signatures = compute_all_signatures(all_structural_variants_v3())

    return compare_signature_sets(v2_signatures, v3_signatures)


def phase15_and_16_row_level_analysis() -> dict:

    csv_path = V3_CAMPAIGN_DIR / "candidate_dataset_v3.csv"

    # Phase 15 -- per-(candidate_type, family) summary stats, accumulated
    # via running sum/sumsq/min/max/count (mathematically exact without
    # holding all values).
    shift_accumulators = {}  # (feature, family) -> {n, sum, sumsq, min, max}

    def _acc(feature, family):
        key = (feature, family)
        if key not in shift_accumulators:
            shift_accumulators[key] = {"n": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}
        return shift_accumulators[key]

    # Phase 16 -- exact feature-vector duplication (feature columns only,
    # never identity columns), running hash -> count + hash -> variant set.
    feature_hash_counts: dict = {}
    feature_hash_variants: dict = {}
    feature_hash_families: dict = {}
    feature_hash_scenarios: dict = {}

    total_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):

        total_rows += len(chunk)

        for feature in SHIFT_FEATURES:
            numeric = pd.to_numeric(chunk[feature], errors="coerce").dropna()
            for family, sub in numeric.groupby(chunk.loc[numeric.index, "topology_family"]):
                acc = _acc(feature, family)
                acc["n"] += len(sub)
                acc["sum"] += sub.sum()
                acc["sumsq"] += (sub ** 2).sum()
                acc["min"] = sub.min() if acc["min"] is None else min(acc["min"], sub.min())
                acc["max"] = sub.max() if acc["max"] is None else max(acc["max"], sub.max())

        feature_tuples = chunk[list(FEATURE_COLUMNS_FOR_DUPLICATION)].astype(str).agg(tuple, axis=1)
        hashes = feature_tuples.apply(hash)

        for h, variant, family, scenario_id in zip(
            hashes, chunk["structural_variant_id"], chunk["topology_family"], chunk["scenario_id"],
        ):
            feature_hash_counts[h] = feature_hash_counts.get(h, 0) + 1
            feature_hash_variants.setdefault(h, set()).add(variant)
            feature_hash_families.setdefault(h, set()).add(family)
            feature_hash_scenarios.setdefault(h, set()).add(scenario_id)

    # Phase 15 summary -- mean/std/min/max per (feature, family), plus an
    # explicit overlap check: do family ranges [min, max] intersect?
    feature_distribution_by_family = {}
    for (feature, family), acc in shift_accumulators.items():
        n = acc["n"]
        mean = acc["sum"] / n if n else None
        variance = (acc["sumsq"] / n - mean ** 2) if (n and mean is not None) else None
        feature_distribution_by_family.setdefault(feature, {})[family] = {
            "n": n, "mean": mean, "std": (variance ** 0.5) if (variance is not None and variance >= 0) else None,
            "min": acc["min"], "max": acc["max"],
        }

    overlap_summary = {}
    for feature, by_family in feature_distribution_by_family.items():
        ranges = {fam: (v["min"], v["max"]) for fam, v in by_family.items() if v["min"] is not None}
        families = list(ranges.keys())
        overlapping_pairs = 0
        total_pairs = 0
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                total_pairs += 1
                a_min, a_max = ranges[families[i]]
                b_min, b_max = ranges[families[j]]
                if a_min <= b_max and b_min <= a_max:
                    overlapping_pairs += 1
        overlap_summary[feature] = {
            "family_ranges": ranges,
            "overlapping_family_pairs": overlapping_pairs,
            "total_family_pairs": total_pairs,
            "fraction_overlapping": (overlapping_pairs / total_pairs) if total_pairs else None,
        }

    # Phase 16 -- duplication composition.
    distinct_feature_vectors = len(feature_hash_counts)
    duplicate_vector_row_count = sum(c for c in feature_hash_counts.values() if c > 1)
    cross_variant_hashes = sum(1 for variants in feature_hash_variants.values() if len(variants) > 1)
    cross_family_hashes = sum(1 for families in feature_hash_families.values() if len(families) > 1)
    within_scenario_only_hashes = sum(1 for scenarios in feature_hash_scenarios.values() if len(scenarios) == 1)
    cross_scenario_hashes = sum(1 for scenarios in feature_hash_scenarios.values() if len(scenarios) > 1)

    duplication_report = {
        "total_rows": total_rows,
        "distinct_feature_vector_count": distinct_feature_vectors,
        "duplicate_feature_vector_row_fraction": (duplicate_vector_row_count / total_rows) if total_rows else None,
        "cross_variant_duplicate_hash_count": cross_variant_hashes,
        "cross_family_duplicate_hash_count": cross_family_hashes,
        "within_scenario_only_hash_count": within_scenario_only_hashes,
        "cross_scenario_duplicate_hash_count": cross_scenario_hashes,
        "fraction_of_distinct_vectors_shared_across_variants": (
            cross_variant_hashes / distinct_feature_vectors
        ) if distinct_feature_vectors else None,
    }

    return {
        "feature_distribution_by_family": feature_distribution_by_family,
        "feature_range_overlap": overlap_summary,
        "duplication_and_temporal_redundancy": duplication_report,
    }


def main() -> None:

    print("Phase 14 -- structural diversity comparison (V2 vs V3)...", flush=True)
    structural_comparison = phase14_structural_comparison()

    print("Phase 15/16 -- chunked row-level analysis (this may take a few minutes)...", flush=True)
    row_level = phase15_and_16_row_level_analysis()

    report = {
        "phase14_structural_diversity_comparison": structural_comparison,
        "phase15_feature_distribution_analysis": row_level["feature_distribution_by_family"],
        "phase15_feature_range_overlap": row_level["feature_range_overlap"],
        "phase16_duplication_and_temporal_redundancy": row_level["duplication_and_temporal_redundancy"],
    }

    out_path = V3_CAMPAIGN_DIR / "post_campaign_analysis_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps({
        "phase14_new_distinct_signature_count": structural_comparison["new_distinct_signature_count"],
        "phase14_baseline_distinct_signature_count": structural_comparison["baseline_distinct_signature_count"],
        "phase15_walking_distance_overlap_fraction": row_level["feature_range_overlap"]["candidate_walking_distance"]["fraction_overlapping"],
        "phase15_alt_route_overlap_fraction": row_level["feature_range_overlap"]["candidate_alternative_route_count"]["fraction_overlapping"],
        "phase16_duplicate_feature_vector_row_fraction": row_level["duplication_and_temporal_redundancy"]["duplicate_feature_vector_row_fraction"],
        "phase16_cross_variant_duplicate_hash_count": row_level["duplication_and_temporal_redundancy"]["cross_variant_duplicate_hash_count"],
    }, indent=2, default=str))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
