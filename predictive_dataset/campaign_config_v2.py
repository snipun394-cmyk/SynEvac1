from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from predictive_dataset.dataset_builder import DEFAULT_HORIZONS
from predictive_dataset.schema import SCHEMA_VERSION
from predictive_dataset.topologies_v2 import TopologySpec, all_topology_specs


# =====================================================
# Predictive Dataset V2 milestone, Phase 2 -- explicit, MEASURABLE
# coverage requirements defined BEFORE generation, per this milestone's
# own "do not rely on random generation alone to hopefully produce
# them" instruction. Every target below is checked mechanically against
# the actual generated campaign in Phase 13's analysis
# (predictive_dataset.diversity_v2/scripts/run_predictive_dataset_campaign_v2.py)
# and reported pass/fail, not asserted blindly.
#
# Phase 9's total-lockout decision (Option B -- see
# docs/architecture/predictive_dataset_campaign_v2.md Section 9): every
# V2 scenario's SimulationRuntime is given an explicit minimum end_time
# floor, so a scenario where nobody can ever move still produces at
# least a few real (non-fabricated) observation ticks instead of zero
# rows -- this constant is the floor value, applied uniformly (not a
# special case), documented here once.
# =====================================================

CAMPAIGN_VERSION_V2 = "predictive_dataset_campaign_v2"
MASTER_SEED_V2 = 20270115

MINIMUM_END_TIME_SECONDS = 30.0


@dataclass(frozen=True)
class CoverageTarget:

    description: str
    minimum_count: int


COVERAGE_TARGETS: Dict[str, CoverageTarget] = {
    "single_exit_scenarios": CoverageTarget(
        "Scenarios drawn from the single_exit_lowrise topology family (Phase 4).", 300,
    ),
    "multi_floor_scenarios": CoverageTarget(
        "Scenarios drawn from a topology family with 2+ floors (twin_stair_highrise or v1_topology_fixed) (Phase 5).", 900,
    ),
    "stair_candidate_rows_with_real_demand": CoverageTarget(
        "Stair candidate-time rows with candidate_queue_length > 0 OR candidate_approaching_count > 0 "
        "(V1 had ZERO of these across 501,696 stair rows) (Phase 6/14).", 1000,
    ),
    "high_occupancy_scenarios": CoverageTarget(
        "Scenarios with total_occupants >= 30 (Phase 8).", 300,
    ),
    "multiple_simultaneous_bottleneck_rows": CoverageTarget(
        "Candidate-time rows where 2+ distinct candidates in the same scenario/tick have target=True "
        "(Phase 7/15 -- V1's weakest operational case).", 5000,
    ),
    "total_lockout_scenarios_with_rows": CoverageTarget(
        "Scenarios where every exit is simultaneously blocked/closed at t=0 that STILL contribute "
        "candidate-time rows (Phase 9 -- V1 had ZERO of these, only zero-row scenarios).", 5,
    ),
}


@dataclass(frozen=True)
class CampaignConfigV2:

    campaign_version: str
    schema_version: str
    master_seed: int
    tick_dt_seconds: float
    horizons_seconds: Tuple[float, ...]
    minimum_end_time_seconds: float
    topology_family_names: Tuple[str, ...]
    scenario_counts_by_family: Dict[str, int]
    coverage_targets: Dict[str, CoverageTarget] = field(default_factory=lambda: dict(COVERAGE_TARGETS))

    def to_dict(self) -> Dict[str, Any]:

        return {
            "campaign_version": self.campaign_version,
            "schema_version": self.schema_version,
            "master_seed": self.master_seed,
            "tick_dt_seconds": self.tick_dt_seconds,
            "horizons_seconds": list(self.horizons_seconds),
            "minimum_end_time_seconds": self.minimum_end_time_seconds,
            "topology_family_names": list(self.topology_family_names),
            "scenario_counts_by_family": self.scenario_counts_by_family,
            "coverage_targets": {
                name: {"description": target.description, "minimum_count": target.minimum_count}
                for name, target in self.coverage_targets.items()
            },
        }


def build_campaign_config_v2(
    topology_specs: Tuple[TopologySpec, ...] = None,
    *,
    master_seed: int = MASTER_SEED_V2,
    tick_dt_seconds: float = 5.0,
    horizons_seconds: Tuple[float, ...] = DEFAULT_HORIZONS,
) -> CampaignConfigV2:

    specs = topology_specs if topology_specs is not None else all_topology_specs()

    return CampaignConfigV2(
        campaign_version=CAMPAIGN_VERSION_V2,
        schema_version=SCHEMA_VERSION,
        master_seed=master_seed,
        tick_dt_seconds=tick_dt_seconds,
        horizons_seconds=tuple(horizons_seconds),
        minimum_end_time_seconds=MINIMUM_END_TIME_SECONDS,
        topology_family_names=tuple(spec.name for spec in specs),
        scenario_counts_by_family={spec.name: spec.scenario_count for spec in specs},
    )
