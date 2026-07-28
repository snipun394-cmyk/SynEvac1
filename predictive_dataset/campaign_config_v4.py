from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from predictive_dataset.campaign_config_v2 import CoverageTarget
from predictive_dataset.schema_v4 import SCHEMA_VERSION_V4
from predictive_dataset.target_generator_v2 import TARGET_VERSION_V2


# =====================================================
# Predictive Dataset V4 milestone, Phase 15 -- campaign configuration
# for the cross-family structural-generalization campaign. Target V2
# remains the canonical, UNCHANGED target (Phase 10's freeze). Coverage
# targets defined BEFORE the full campaign runs, never tuned after
# seeing results (V3's own Phase 9 discipline, reused). The first 7
# mirror V3's own targets (scaled to V4's larger variant count); the
# final 6 are NEW, specific to what THIS milestone adds (2 new
# families, graph-context structural diversity).
# =====================================================

CAMPAIGN_VERSION_V4 = "predictive_dataset_campaign_v4"
MASTER_SEED_V4 = 20260729

MINIMUM_END_TIME_SECONDS = 30.0
HORIZON_SECONDS = 20.0

STRUCTURAL_VARIANT_VERSION_V4 = "v4-cross-family-24-variants"

PILOT_SCENARIOS_PER_VARIANT = 25
FULLSCALE_SCENARIOS_PER_VARIANT = 125  # 24 variants x 125 = 3,000 scenarios -- see Phase 16 decision


COVERAGE_TARGETS_V4: Dict[str, CoverageTarget] = {

    "every_structural_variant_represented": CoverageTarget(
        "Every one of the 24 structural variants contributes at least 1 scenario with candidate-time rows.", 24,
    ),
    "every_family_represented": CoverageTarget(
        "Every one of the 6 topology families (4 old + 2 new) contributes at least 1 scenario with rows.", 6,
    ),
    "high_occupancy_scenarios": CoverageTarget(
        "Scenarios with total_occupants >= 30.", 200,
    ),
    "multiple_simultaneous_bottleneck_rows": CoverageTarget(
        "Candidate-time rows where 2+ distinct candidates in the same scenario/tick have target=True.", 2000,
    ),
    "stair_candidate_rows_with_real_demand": CoverageTarget(
        "Stair candidate-time rows with candidate_queue_length > 0 OR candidate_approaching_count > 0.", 1000,
    ),
    "total_lockout_scenarios_with_rows": CoverageTarget(
        "Scenarios where every exit is simultaneously blocked/closed at t=0 that still contribute rows.", 3,
    ),
    "multi_floor_scenarios": CoverageTarget(
        "Scenarios drawn from a structural variant with 2+ floors.", 300,
    ),

    # --- NEW, V4-specific targets ---

    "multi_wing_family_rows": CoverageTarget(
        "Candidate-time rows drawn from the NEW multi_wing family (any of its 4 variants).", 20000,
    ),
    "ring_corridor_family_rows": CoverageTarget(
        "Candidate-time rows drawn from the NEW ring_corridor family (any of its 4 variants).", 20000,
    ),
    "genuine_zone_cycle_scenarios": CoverageTarget(
        "Scenarios drawn from a structural variant with a genuine zone-only cycle "
        "(ring_corridor's 4 variants, or v1_fixed_dual_stair) -- structurally near-absent from Dataset V3.", 300,
    ),
    "non_bridge_candidate_rows": CoverageTarget(
        "Candidate-time rows from a candidate whose graph_context.is_bridge is False -- "
        "route-redundant candidates, the structural region ring_corridor/multi_exit_wide target.", 50000,
    ),
    "high_catchment_candidate_rows": CoverageTarget(
        "Candidate-time rows from a candidate with graph_context.upstream_catchment_count >= 5 -- "
        "the high structural-load-bearing region multi_wing's funneling design targets.", 20000,
    ),
    "high_betweenness_candidate_rows": CoverageTarget(
        "Candidate-time rows from a candidate with graph_context.betweenness_centrality >= 0.5.", 50000,
    ),
}


@dataclass(frozen=True)
class CampaignConfigV4:

    campaign_version: str
    schema_version: str
    target_version: str
    structural_variant_version: str
    master_seed: int
    tick_dt_seconds: float
    horizon_seconds: float
    minimum_end_time_seconds: float
    variant_ids: Tuple[str, ...]
    scenario_counts_by_variant: Dict[str, int]
    coverage_targets: Dict[str, CoverageTarget] = field(default_factory=lambda: dict(COVERAGE_TARGETS_V4))

    def to_dict(self) -> Dict[str, Any]:

        return {
            "campaign_version": self.campaign_version,
            "schema_version": self.schema_version,
            "target_version": self.target_version,
            "structural_variant_version": self.structural_variant_version,
            "master_seed": self.master_seed,
            "tick_dt_seconds": self.tick_dt_seconds,
            "horizon_seconds": self.horizon_seconds,
            "minimum_end_time_seconds": self.minimum_end_time_seconds,
            "variant_ids": list(self.variant_ids),
            "scenario_counts_by_variant": self.scenario_counts_by_variant,
            "coverage_targets": {
                name: {"description": target.description, "minimum_count": target.minimum_count}
                for name, target in self.coverage_targets.items()
            },
        }


def build_campaign_config_v4(
    variants,
    *,
    campaign_version: str = CAMPAIGN_VERSION_V4,
    master_seed: int = MASTER_SEED_V4,
    tick_dt_seconds: float = 5.0,
    horizon_seconds: float = HORIZON_SECONDS,
    coverage_targets: Dict[str, CoverageTarget] = None,
) -> CampaignConfigV4:

    return CampaignConfigV4(
        campaign_version=campaign_version,
        schema_version=SCHEMA_VERSION_V4,
        target_version=TARGET_VERSION_V2,
        structural_variant_version=STRUCTURAL_VARIANT_VERSION_V4,
        master_seed=master_seed,
        tick_dt_seconds=tick_dt_seconds,
        horizon_seconds=horizon_seconds,
        minimum_end_time_seconds=MINIMUM_END_TIME_SECONDS,
        variant_ids=tuple(v.variant_id for v in variants),
        scenario_counts_by_variant={v.variant_id: v.topology.scenario_count for v in variants},
        coverage_targets=dict(coverage_targets) if coverage_targets is not None else dict(COVERAGE_TARGETS_V4),
    )
