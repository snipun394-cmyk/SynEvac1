from collections import Counter
from typing import Any, Dict, Sequence

from predictive_dataset.topology_signature import StructuralTopologySignature, compute_all_signatures


# =====================================================
# Predictive Dataset V3 milestone, Phase 4 -- mechanical verification
# that the 16 structural variants (predictive_dataset.topologies_v3)
# are GENUINELY distinct graphs, not merely renamed ids/shifted
# coordinates. Two variants are flagged as duplicates only if their
# StructuralTopologySignature.structural_key() (graph-shape fields,
# rounded) collide -- never by comparing names/ids.
# =====================================================


def structural_diversity_report(variants) -> Dict[str, Any]:

    signatures = compute_all_signatures(variants)

    key_to_variant_ids: Dict[Any, list] = {}
    for sig in signatures:
        key_to_variant_ids.setdefault(sig.structural_key(), []).append(sig.variant_id)

    duplicate_groups = {
        key: ids for key, ids in key_to_variant_ids.items() if len(ids) > 1
    }

    family_distribution = dict(Counter(sig.family for sig in signatures))
    floor_count_distribution = dict(Counter(sig.floor_count for sig in signatures))
    exit_count_distribution = dict(Counter(sig.exit_count for sig in signatures))
    stair_count_distribution = dict(Counter(sig.stair_count for sig in signatures))
    candidate_count_distribution = dict(Counter(sig.candidate_count for sig in signatures))

    route_redundancy_values = [sig.mean_alternative_route_count for sig in signatures]

    return {
        "requested_variant_count": len(signatures),
        "distinct_structural_signature_count": len(key_to_variant_ids),
        "duplicate_structural_signature_groups": [
            {"variant_ids": ids, "signature_key": list(key)} for key, ids in duplicate_groups.items()
        ],
        "family_distribution": family_distribution,
        "floor_count_distribution": floor_count_distribution,
        "exit_count_distribution": exit_count_distribution,
        "stair_count_distribution": stair_count_distribution,
        "candidate_count_distribution": candidate_count_distribution,
        "route_redundancy_distribution": {
            "min": min(route_redundancy_values) if route_redundancy_values else None,
            "max": max(route_redundancy_values) if route_redundancy_values else None,
            "mean": (sum(route_redundancy_values) / len(route_redundancy_values)) if route_redundancy_values else None,
            "values_by_variant": {sig.variant_id: sig.mean_alternative_route_count for sig in signatures},
        },
        "signatures": [sig.to_dict() for sig in signatures],
        "all_genuinely_distinct": len(duplicate_groups) == 0,
    }


def compare_signature_sets(
    baseline_signatures: Sequence[StructuralTopologySignature],
    new_signatures: Sequence[StructuralTopologySignature],
) -> Dict[str, Any]:
    """Phase 14 -- V2 (baseline, 4 fixed graphs) vs V3 (16 variants)
    structural-diversity comparison. Reports how much broader V3's
    ranges are across every signature dimension."""

    def _spread(values):
        values = list(values)
        if not values:
            return {"min": None, "max": None, "mean": None, "distinct_count": 0}
        return {
            "min": min(values), "max": max(values),
            "mean": sum(values) / len(values),
            "distinct_count": len(set(values)),
        }

    fields = (
        "floor_count", "zone_count", "door_count", "exit_count", "stair_count",
        "candidate_count", "mean_candidate_walking_distance", "mean_alternative_route_count",
    )

    return {
        "baseline_variant_count": len(baseline_signatures),
        "new_variant_count": len(new_signatures),
        "baseline_distinct_signature_count": len({s.structural_key() for s in baseline_signatures}),
        "new_distinct_signature_count": len({s.structural_key() for s in new_signatures}),
        "field_spreads": {
            field: {
                "baseline": _spread(getattr(s, field) for s in baseline_signatures),
                "new": _spread(getattr(s, field) for s in new_signatures),
            }
            for field in fields
        },
    }
