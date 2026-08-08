"""
Human Behavior Calibration Campaign V1 -- Phase 5 Configuration
Comparison.

Compares three configurations of Adult_Default against the same four
NIST published-drill benchmarks already used throughout this
repository's validation work:

    1. SynEvac Baseline V1        -- DEFAULT_PROFILE_REGISTRY, unmodified
    2. Walking Speed only          -- Adult_Default.walking_speed = 1.6 m/s
                                       (the best-scoring ADOPT candidate found
                                       by the walking-speed automatic
                                       calibration campaign, every building)
    3. Walking Speed + Pre-Movement Delay -- (2) plus Adult_Default's own
                                       best median_delay found by
                                       run_human_behavior_calibration_campaign_v1_pre_movement_delay.py,
                                       per building

Reuses calibration_benchmark.run_calibration_benchmark() directly (the
same, unmodified paired-scenario harness every other campaign in this
repository already uses) for arms (1)<->(2) and (1)<->(3). The
(2)<->(3) comparison -- isolating pre-movement delay's OWN incremental
contribution on top of walking speed -- calls
research_framework.statistics.paired_comparison()/confidence_interval()/
effect_size_cohens_d() directly on the two candidates' own already-paired
per-scenario samples: the exact same statistical primitives
calibration_benchmark/harness.py itself calls internally (see its own
_compare() function), applied to a third pair of arms a single
run_calibration_benchmark() call does not natively produce. No new
statistical method is introduced anywhere in this script.

The ONE new class this script defines,
WalkingSpeedAndPreMovementDelayCombinedCandidate, is a script-local
composition of two already-existing calibration_benchmark candidates
for the SAME profile_id -- deliberately NOT routed through
automatic_calibration.grid_search's own _JointGridPointCandidate, which
explicitly refuses to compose two different customizations of one
profile_id (a documented limitation of its generic grid-search
composition, not a bug -- see its own module docstring). This is a
single, fixed, hand-built combination for this specific 3-configuration
comparison, following the exact dataclasses.replace() pattern
WalkingSpeedCandidate/PreMovementDelayCandidate each already use
individually. calibration_benchmark/candidates.py itself is not
modified.

Never applies any of the three configurations back to
DEFAULT_PROFILE_REGISTRY -- exactly the same non-mutation discipline
every existing candidate in this codebase already follows.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dataclasses import replace  # noqa: E402

from run_nist_10story_validation import (  # noqa: E402
    build_nist_10story_building, build_nist_10story_definition,
    DEFINITION_ID as DEFINITION_ID_10, MASTER_SEED as MASTER_SEED_10,
)
from run_nist_18story_validation import (  # noqa: E402
    build_nist_18story_building, build_nist_18story_definition,
    DEFINITION_ID as DEFINITION_ID_18, MASTER_SEED as MASTER_SEED_18,
)
from run_nist_24story_validation import (  # noqa: E402
    build_nist_24story_building, build_nist_24story_definition,
    DEFINITION_ID as DEFINITION_ID_24, MASTER_SEED as MASTER_SEED_24,
)
from run_nist_31story_validation import (  # noqa: E402
    build_nist_31story_building, build_nist_31story_definition,
    DEFINITION_ID as DEFINITION_ID_31, MASTER_SEED as MASTER_SEED_31,
)

from behavior_library.pre_movement_strategies import ProbabilisticPreMovementDelay  # noqa: E402
from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY  # noqa: E402

from calibration_benchmark import ParameterCandidate, WalkingSpeedCandidate, recommend, run_calibration_benchmark  # noqa: E402
from research_framework.statistics import confidence_interval, effect_size_cohens_d, paired_comparison  # noqa: E402


PUBLISHED_EVACUATION_TIME_S = {
    "10-story": 1022.0,
    "18-story": 1192.0,
    "24-story": 1090.0,
    "31-story": 1002.0,
}

BUILDINGS = {
    "10-story": (build_nist_10story_building, build_nist_10story_definition, DEFINITION_ID_10, MASTER_SEED_10),
    "18-story": (build_nist_18story_building, build_nist_18story_definition, DEFINITION_ID_18, MASTER_SEED_18),
    "24-story": (build_nist_24story_building, build_nist_24story_definition, DEFINITION_ID_24, MASTER_SEED_24),
    "31-story": (build_nist_31story_building, build_nist_31story_definition, DEFINITION_ID_31, MASTER_SEED_31),
}

# Best-scoring ADOPT candidate from the (already-executed)
# automatic-calibration walking-speed campaign -- every one of the four
# NIST buildings independently ranked 1.6 m/s as its best-scoring
# candidate (docs/architecture/automatic_calibration_walking_speed_campaign_raw_results.json).
WALKING_SPEED_CANDIDATE_VALUE = 1.6

PRE_MOVEMENT_DELAY_RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "docs", "architecture",
    "human_behavior_calibration_campaign_v1_pre_movement_delay_raw_results.json",
)


def _load_best_median_delay_by_building():

    with open(PRE_MOVEMENT_DELAY_RESULTS_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    return {
        building_name: result["best_median_delay"]
        for building_name, result in data["Adult_Default"].items()
    }


class WalkingSpeedAndPreMovementDelayCombinedCandidate(ParameterCandidate):

    def __init__(self, profile_id, candidate_speed, candidate_median_delay, candidate_spread, dataset_source, rationale):

        self.profile_id = profile_id
        self.candidate_speed = candidate_speed
        self.candidate_median_delay = candidate_median_delay
        self.candidate_spread = candidate_spread

        current_template = DEFAULT_PROFILE_REGISTRY[profile_id]

        super().__init__(
            name=f"{profile_id}.walking_speed+pre_movement_strategy.median_delay (combined)",
            subsystem="Walking Model + Pre-movement Model (combined)",
            calibration_tier="Tier 2",
            dataset_source=dataset_source,
            current_value={
                "walking_speed": current_template.walking_speed,
                "median_delay": current_template.pre_movement_strategy.median_delay,
            },
            candidate_value={"walking_speed": candidate_speed, "median_delay": candidate_median_delay},
            unit="m/s + seconds",
            rationale=rationale,
        )

    def baseline_registry(self):
        return DEFAULT_PROFILE_REGISTRY

    def candidate_registry(self):

        registry = dict(DEFAULT_PROFILE_REGISTRY)
        registry[self.profile_id] = replace(
            registry[self.profile_id],
            walking_speed=self.candidate_speed,
            pre_movement_strategy=ProbabilisticPreMovementDelay(
                median_delay=self.candidate_median_delay, spread=self.candidate_spread,
            ),
        )
        return registry


def _sample_field(samples, field_name):

    return [getattr(s, field_name) for s in samples]


def run_comparison_for_building(building_name, best_median_delay, n_scenarios=8, dt=1.0):

    build_building, build_definition, definition_id, master_seed = BUILDINGS[building_name]

    building = build_building()
    definition = build_definition()

    current_spread = DEFAULT_PROFILE_REGISTRY["Adult_Default"].pre_movement_strategy.spread

    walking_only_candidate = WalkingSpeedCandidate(
        "Adult_Default", WALKING_SPEED_CANDIDATE_VALUE,
        "automatic_calibration_walking_speed_campaign (already executed, ADOPT, best score every building)",
        "Human Behavior Calibration Campaign V1, Phase 5 -- configuration (2): walking speed only",
    )
    combined_candidate = WalkingSpeedAndPreMovementDelayCombinedCandidate(
        "Adult_Default", WALKING_SPEED_CANDIDATE_VALUE, best_median_delay, current_spread,
        "walking speed: automatic_calibration_walking_speed_campaign; "
        "median_delay: human_behavior_calibration_campaign_v1_pre_movement_delay (this campaign)",
        "Human Behavior Calibration Campaign V1, Phase 5 -- configuration (3): walking speed + pre-movement delay",
    )

    result_walking_only = run_calibration_benchmark(
        walking_only_candidate, building, definition, definition_id, master_seed, n_scenarios, dt=dt,
    )
    result_combined = run_calibration_benchmark(
        combined_candidate, building, definition, definition_id, master_seed, n_scenarios, dt=dt,
    )

    comparison_walking_only = result_walking_only.comparisons["evacuation_time"]
    comparison_combined = result_combined.comparisons["evacuation_time"]

    # (2) vs (3) -- isolates pre-movement delay's OWN incremental effect
    # on top of walking speed. Same master_seed / same run_batch_pipeline
    # call underlies both result_walking_only and result_combined (both
    # constructed from the identical (definition, definition_id, building,
    # master_seed, n_scenarios) tuple), so this remains a true paired
    # comparison -- scenario-for-scenario, not distribution-for-distribution.
    walking_only_times = _sample_field(result_walking_only.candidate_samples, "evacuation_time")
    combined_times = _sample_field(result_combined.candidate_samples, "evacuation_time")

    incremental_paired = paired_comparison(walking_only_times, combined_times)
    incremental_effect_size = effect_size_cohens_d(combined_times, walking_only_times)
    incremental_ci_walking_only = confidence_interval(walking_only_times)
    incremental_ci_combined = confidence_interval(combined_times)

    published = PUBLISHED_EVACUATION_TIME_S[building_name]
    baseline_mean = comparison_walking_only.baseline_mean  # identical arm in both results (same seed/definition)

    def _ratio(mean_value):
        return (mean_value / published) if mean_value is not None else None

    recommendation_walking_only = recommend(result_walking_only)
    recommendation_combined = recommend(result_combined)

    return {
        "building": building_name,
        "published_evacuation_time_s": published,
        "n_scenarios": n_scenarios,
        "best_median_delay_used": best_median_delay,
        "configurations": {
            "1_baseline": {
                "mean_evacuation_time": baseline_mean,
                "overprediction_ratio": _ratio(baseline_mean),
            },
            "2_walking_speed_only": {
                "mean_evacuation_time": comparison_walking_only.candidate_mean,
                "overprediction_ratio": _ratio(comparison_walking_only.candidate_mean),
                "vs_baseline_p_value": comparison_walking_only.paired.p_value,
                "vs_baseline_cohens_d": comparison_walking_only.effect_size.cohens_d,
                "vs_baseline_recommendation": recommendation_walking_only.overall_verdict.name,
            },
            "3_walking_speed_plus_pre_movement_delay": {
                "mean_evacuation_time": comparison_combined.candidate_mean,
                "overprediction_ratio": _ratio(comparison_combined.candidate_mean),
                "vs_baseline_p_value": comparison_combined.paired.p_value,
                "vs_baseline_cohens_d": comparison_combined.effect_size.cohens_d,
                "vs_baseline_recommendation": recommendation_combined.overall_verdict.name,
            },
        },
        "incremental_pre_movement_delay_effect": {
            "description": "configuration 2 (walking-speed-only) vs configuration 3 (walking-speed + pre-movement delay)",
            "walking_only_mean": incremental_ci_walking_only.mean,
            "combined_mean": incremental_ci_combined.mean,
            "mean_difference": incremental_paired.mean_difference,
            "p_value": incremental_paired.p_value,
            "cohens_d": incremental_effect_size.cohens_d,
            "n_pairs": incremental_paired.n,
        },
        "gap_closure": {
            "description": "fraction of the baseline-to-published gap closed by each configuration",
            "baseline_gap_s": (baseline_mean - published) if baseline_mean is not None else None,
            "walking_only_gap_s": (
                comparison_walking_only.candidate_mean - published
            ) if comparison_walking_only.candidate_mean is not None else None,
            "combined_gap_s": (
                comparison_combined.candidate_mean - published
            ) if comparison_combined.candidate_mean is not None else None,
            "walking_only_gap_closure_fraction": (
                1.0 - (comparison_walking_only.candidate_mean - published) / (baseline_mean - published)
            ) if (baseline_mean is not None and comparison_walking_only.candidate_mean is not None and baseline_mean != published) else None,
            "combined_gap_closure_fraction": (
                1.0 - (comparison_combined.candidate_mean - published) / (baseline_mean - published)
            ) if (baseline_mean is not None and comparison_combined.candidate_mean is not None and baseline_mean != published) else None,
        },
    }


def main():

    n_scenarios = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    dt = 1.0

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    best_median_delay_by_building = _load_best_median_delay_by_building()

    all_results = {}

    for building_name in BUILDINGS:

        print(
            f"Running 3-configuration comparison for {building_name} "
            f"(best_median_delay={best_median_delay_by_building[building_name]})...", flush=True,
        )
        result = run_comparison_for_building(
            building_name, best_median_delay_by_building[building_name], n_scenarios=n_scenarios, dt=dt,
        )
        all_results[building_name] = result
        print(json.dumps(result, indent=2, default=str))

    with open(
        os.path.join(output_dir, "human_behavior_calibration_campaign_v1_configuration_comparison_raw_results.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(all_results, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
