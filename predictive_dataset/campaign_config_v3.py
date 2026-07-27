from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from predictive_dataset.campaign_config_v2 import CoverageTarget
from predictive_dataset.schema import SCHEMA_VERSION
from predictive_dataset.target_generator_v2 import TARGET_VERSION_V2


# =====================================================
# Predictive Dataset V3 milestone, Phase 2/9/11 -- campaign
# configuration for the structural-diversity campaign. Target V2
# (predictive_dataset.target_generator_v2, "v2-persistent-demand-
# service-imbalance") is the canonical, UNCHANGED target -- this module
# only versions the CAMPAIGN (which structural variants/scenario counts
# produced the rows), never the target semantics.
#
# COVERAGE_TARGETS_V3 are set BEFORE the full campaign runs (Phase 9's
# own "do not tune those thresholds after seeing results" rule) --
# checked mechanically against the actual generated data in Phase 13's
# analysis, reported pass/fail. The first six mirror
# campaign_config_v2.COVERAGE_TARGETS' intent (scaled to V3's smaller
# per-variant scenario counts); the final three are NEW, specific to
# structural diversity this campaign adds that V2 never had at all
# (multiple stairs, chained/serial stair connectivity, asymmetric/
# reduced-redundancy exit placement).
# =====================================================

CAMPAIGN_VERSION_V3 = "predictive_dataset_campaign_v3"
MASTER_SEED_V3 = 20260727

MINIMUM_END_TIME_SECONDS = 30.0
HORIZON_SECONDS = 20.0  # Primary prediction horizon (established Target V2 canonical horizon)

STRUCTURAL_VARIANT_VERSION = "v3-structural-diversity-16-variants"

PILOT_SCENARIOS_PER_VARIANT = 25
FULLSCALE_SCENARIOS_PER_VARIANT = 150  # see Phase 11 decision in docs/architecture/predictive_dataset_campaign_v3.md


COVERAGE_TARGETS_V3: Dict[str, CoverageTarget] = {
    "every_structural_variant_represented": CoverageTarget(
        "Every one of the 16 structural variants contributes at least 1 scenario with candidate-time rows.", 16,
    ),
    "single_exit_family_scenarios": CoverageTarget(
        "Scenarios drawn from the single_exit_lowrise family (any of its 4 structural variants).", 60,
    ),
    "multi_floor_scenarios": CoverageTarget(
        "Scenarios drawn from a structural variant with 2+ floors.", 300,
    ),
    "stair_candidate_rows_with_real_demand": CoverageTarget(
        "Stair candidate-time rows with candidate_queue_length > 0 OR candidate_approaching_count > 0.", 1000,
    ),
    "high_occupancy_scenarios": CoverageTarget(
        "Scenarios with total_occupants >= 30.", 200,
    ),
    "multiple_simultaneous_bottleneck_rows": CoverageTarget(
        "Candidate-time rows where 2+ distinct candidates in the same scenario/tick have target=True.", 2000,
    ),
    "total_lockout_scenarios_with_rows": CoverageTarget(
        "Scenarios where every exit is simultaneously blocked/closed at t=0 that still contribute rows.", 3,
    ),
    "multi_stair_scenarios": CoverageTarget(
        "Scenarios drawn from a structural variant with 2+ stairs "
        "(twin_stair_highrise/3stair/chained_core, v1_fixed_dual_stair/three_floor).", 300,
    ),
    "chained_stair_connectivity_scenarios": CoverageTarget(
        "Scenarios drawn from a serial/chained-stair structural variant "
        "(twin_stair_chained_core, v1_fixed_three_floor) -- connectivity pattern V2 never had.", 100,
    ),
    "reduced_redundancy_exit_scenarios": CoverageTarget(
        "Scenarios drawn from an asymmetric/reduced-route-redundancy exit-placement variant "
        "(multi_exit_reduced_redundancy, multi_exit_linear_chain) -- V2 never had asymmetric exit placement.", 100,
    ),
}


@dataclass(frozen=True)
class CampaignConfigV3:

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
    coverage_targets: Dict[str, CoverageTarget] = field(default_factory=lambda: dict(COVERAGE_TARGETS_V3))

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


def build_campaign_config_v3(
    variants,
    *,
    campaign_version: str = CAMPAIGN_VERSION_V3,
    master_seed: int = MASTER_SEED_V3,
    tick_dt_seconds: float = 5.0,
    horizon_seconds: float = HORIZON_SECONDS,
    coverage_targets: Dict[str, CoverageTarget] = None,
) -> CampaignConfigV3:

    return CampaignConfigV3(
        campaign_version=campaign_version,
        schema_version=SCHEMA_VERSION,
        target_version=TARGET_VERSION_V2,
        structural_variant_version=STRUCTURAL_VARIANT_VERSION,
        master_seed=master_seed,
        tick_dt_seconds=tick_dt_seconds,
        horizon_seconds=horizon_seconds,
        minimum_end_time_seconds=MINIMUM_END_TIME_SECONDS,
        variant_ids=tuple(v.variant_id for v in variants),
        scenario_counts_by_variant={v.variant_id: v.topology.scenario_count for v in variants},
        coverage_targets=dict(coverage_targets) if coverage_targets is not None else dict(COVERAGE_TARGETS_V3),
    )
