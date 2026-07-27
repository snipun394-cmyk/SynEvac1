"""Predictive Dataset V3 milestone, Phase 13 -- FULL-SCALE structural-
diversity campaign. 150 scenarios x 16 structural variants (2,400
scenarios total) -- the Phase 11 scale decision, made from Phase 8
pilot evidence (pilot: 400 scenarios / 443,622 rows / 46.5s / 9,544
rows/sec / 0 failures / every variant contributed rows / no zero-
walking-distance regressions / meaningful Target V2 positive rates in
every candidate type). Extrapolating the pilot's throughput, 2,400
scenarios is expected to produce ~2.7M rows in well under 10 minutes --
this scale was chosen for BALANCED structural coverage (every one of
COVERAGE_TARGETS_V3's structural-specific minimums, e.g. multi_stair
>=300 scenarios, chained_stair >=100, reduced_redundancy >=100, are
comfortably clearable at 150/variant), not row-count maximization
(Phase 11's own "aim for balanced structural coverage rather than
maximum rows" instruction) -- deliberately smaller than Dataset V2's
2,500 scenarios despite covering 4x as many structural templates.

Reuses predictive_dataset.campaign_runner_v3.run_campaign_v3 verbatim
(same memory-safe streaming core the pilot already validated) -- this
script only supplies the full-scale variant list and output directory.

Usage: python scripts/run_predictive_dataset_campaign_v3_fullscale.py
"""

import json
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil

from predictive_dataset.campaign_config_v3 import FULLSCALE_SCENARIOS_PER_VARIANT, build_campaign_config_v3
from predictive_dataset.campaign_runner_v3 import run_campaign_v3
from predictive_dataset.topologies_v3 import all_structural_variants_v3, with_scenario_count
from predictive_dataset.topology_diversity_v3 import structural_diversity_report

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v3"


def main() -> None:

    variants = all_structural_variants_v3()

    fullscale_variants = tuple(
        type(v)(v.family, v.variant_id, v.variant_label, with_scenario_count(v.topology, FULLSCALE_SCENARIOS_PER_VARIANT))
        for v in variants
    )

    config = build_campaign_config_v3(fullscale_variants)
    diversity = structural_diversity_report(variants)

    mem_before = psutil.virtual_memory()
    print(f"Memory before: available={mem_before.available/1e6:.0f}MB used={mem_before.percent}%", flush=True)

    result = run_campaign_v3(fullscale_variants, config, OUTPUT_DIR, log_every=100)

    mem_after = psutil.virtual_memory()
    print(f"Memory after: available={mem_after.available/1e6:.0f}MB used={mem_after.percent}%", flush=True)

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
    }

    with open(OUTPUT_DIR / "campaign_v3_fullscale_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps({
        "accepted": result["accepted_scenarios"], "failed": result["failed_scenarios"],
        "rows": result["row_count"], "wall_seconds": result["wall_seconds"],
    }, indent=2))
    print(f"Wrote {OUTPUT_DIR / 'campaign_v3_fullscale_report.json'}")


if __name__ == "__main__":
    main()
