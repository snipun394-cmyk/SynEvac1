"""Predictive Dataset V4 milestone, Phase 19 -- V3 vs V4 structural
comparison. Quantifies whether V4 genuinely expanded structural
support beyond V3, across every dimension topology_signature_v4.py
measures -- not just "more rows".

Usage: python scripts/predictive_dataset_v4_structural_comparison.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictive_dataset.topologies_v3 import all_structural_variants_v3
from predictive_dataset.topologies_v4 import all_structural_variants_v4
from predictive_dataset.topology_signature_v4 import compute_all_signatures_v4

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v4"


def _range_and_distinct(values):
    return {
        "min": min(values), "max": max(values), "mean": sum(values) / len(values),
        "distinct_count": len(set(values)),
    }


def main() -> None:

    v3_sigs = compute_all_signatures_v4(all_structural_variants_v3())
    v4_sigs = compute_all_signatures_v4(all_structural_variants_v4())

    dimensions = {
        "floor_count": lambda s: s.v3_signature.floor_count,
        "zone_count": lambda s: s.v3_signature.zone_count,
        "door_count": lambda s: s.v3_signature.door_count,
        "exit_count": lambda s: s.v3_signature.exit_count,
        "stair_count": lambda s: s.v3_signature.stair_count,
        "candidate_count": lambda s: s.v3_signature.candidate_count,
        "mean_candidate_walking_distance": lambda s: s.v3_signature.mean_candidate_walking_distance,
        "mean_alternative_route_count": lambda s: s.v3_signature.mean_alternative_route_count,
        "bridge_edge_fraction": lambda s: s.bridge_edge_fraction,
        "mean_betweenness": lambda s: s.mean_betweenness,
        "max_betweenness": lambda s: s.max_betweenness,
        "mean_upstream_catchment": lambda s: s.mean_upstream_catchment,
        "max_upstream_catchment": lambda s: s.max_upstream_catchment,
        "mean_zone_degree": lambda s: s.mean_zone_degree,
        "max_zone_degree": lambda s: s.max_zone_degree,
        "cyclomatic_number": lambda s: s.cyclomatic_number,
        "corridor_depth_hops": lambda s: s.corridor_depth_hops,
        "exit_catchment_asymmetry": lambda s: s.exit_catchment_asymmetry,
    }

    comparison = {}
    for name, fn in dimensions.items():

        v3_values = [fn(s) for s in v3_sigs]
        v4_values = [fn(s) for s in v4_sigs]

        v3_stats = _range_and_distinct(v3_values)
        v4_stats = _range_and_distinct(v4_values)

        comparison[name] = {
            "v3": v3_stats, "v4": v4_stats,
            "range_broadened": bool(v4_stats["max"] > v3_stats["max"] or v4_stats["min"] < v3_stats["min"]),
            "distinct_count_increased": v4_stats["distinct_count"] > v3_stats["distinct_count"],
        }

    # Genuine zone-only cycle count (excludes the OUTSIDE-mediated
    # trivial-cycle artifact -- see Phase 5's own discovery).
    import networkx as nx
    from navigation.node import Node
    from predictive_dataset.graph_context_v4 import _build_undirected_graph

    def _zone_only_cyclomatic(variant) -> int:
        g, _ = _build_undirected_graph(variant.topology.building)
        zo = g.copy()
        if Node.OUTSIDE_NODE_ID in zo:
            zo.remove_node(Node.OUTSIDE_NODE_ID)
        nc = nx.number_connected_components(zo) if zo.number_of_nodes() else 0
        return zo.number_of_edges() - zo.number_of_nodes() + nc if zo.number_of_nodes() else 0

    v3_zone_cycles = sum(1 for v in all_structural_variants_v3() if _zone_only_cyclomatic(v) > 0)
    v4_zone_cycles = sum(1 for v in all_structural_variants_v4() if _zone_only_cyclomatic(v) > 0)

    report = {
        "v3_variant_count": len(v3_sigs),
        "v4_variant_count": len(v4_sigs),
        "v3_family_count": len(set(v.family for v in all_structural_variants_v3())),
        "v4_family_count": len(set(v.family for v in all_structural_variants_v4())),
        "dimension_comparison": comparison,
        "variants_with_genuine_zone_only_cycle": {"v3": v3_zone_cycles, "v3_of_total": f"{v3_zone_cycles}/16", "v4": v4_zone_cycles, "v4_of_total": f"{v4_zone_cycles}/24"},
        "every_dimension_broadened_or_equal": all(c["range_broadened"] or c["v4"]["max"] == c["v3"]["max"] for c in comparison.values()),
    }

    output_path = OUTPUT_DIR / "v3_vs_v4_structural_comparison.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    for name, c in comparison.items():
        print(f"{name}: v3[{c['v3']['min']}-{c['v3']['max']}, {c['v3']['distinct_count']} distinct] "
              f"-> v4[{c['v4']['min']}-{c['v4']['max']}, {c['v4']['distinct_count']} distinct] "
              f"broadened={c['range_broadened']}")

    print(f"\nGenuine zone-only cycles: V3 {v3_zone_cycles}/16 -> V4 {v4_zone_cycles}/24")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
