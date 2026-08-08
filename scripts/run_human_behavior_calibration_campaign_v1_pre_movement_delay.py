"""
Human Behavior Calibration Campaign V1 -- Pre-Movement Delay.

Mirrors scripts/run_automatic_calibration_walking_speed_campaign.py
exactly (same AutoCalibrationEngine + GridSearchStrategy + NIST
building/definition builders + PublishedValueObjective machinery),
substituting calibration_benchmark.PreMovementDelayCandidate for
WalkingSpeedCandidate. No production simulator/calibration_studio/
automatic_calibration/behavior_library/behaviour_profile_resolver source
file is modified -- this script is a pure CLIENT of the existing,
already-tested infrastructure, exactly the same discipline the
walking-speed campaign itself follows.

Search grid (Adult_Default only -- see below for why): see the
companion chat report (Human Behavior Calibration Campaign V1) for the
full literature citations. median_delay in {15, 20, 25, 30, 35, 45, 55}
seconds is grounded in NIST Technical Note 1664 ("Occupant Behavior in a
High-rise Office Building Fire", 19 observed occupants, individual
pre-evacuation delay 10-55s, mean 28s, SD 11s) -- the single most
directly comparable published figure available, since it is itself an
office/high-rise fire evacuation study, the same building class as the
four NIST validation buildings this campaign runs against. 30s
(Adult_Default's current production default) is deliberately included
as a grid point, exactly as 1.2 m/s was for the walking-speed campaign.

IMPORTANT, disclosed up front rather than discovered mid-run: all four
NIST validation buildings (run_nist_10story/18story/24story/31story_
validation.py) populate their occupant scenarios with
FixedValue("Adult_Default") for 100% of occupants -- verified by direct
source inspection (grep on FixedValue("...") across all four
build_*_definition() functions), not assumed. Calibrating
Child_Default/Elderly_Default/Wheelchair_Default/Visitor_Default's
pre-movement delay against these specific benchmarks is therefore a
STRUCTURAL no-op: zero occupants in any of these four scenarios ever
resolve to those profile ids, so no candidate value for them can change
any simulated outcome -- changing an unused profile's own template
produces a byte-for-byte identical simulation to baseline. Rather than
running a full multi-building grid sweep whose result is analytically
predictable (every candidate mean would equal the baseline mean, p-value
undefined/NaN, verdict INCONCLUSIVE, exactly like the walking-speed
campaign's own 1.2 m/s "no-op" grid point), this script runs ONE small
confirmatory diagnostic (Child_Default, 3-point grid, reduced
n_scenarios, 10-story building only) to empirically verify that
prediction rather than merely assert it, and reports the reasoning
(not a repeated empty sweep) for why Elderly_Default/Wheelchair_Default/
Visitor_Default are not separately executed here. See the companion
report's Phase 1 for each of those three profiles' own literature-
grounded grid proposals (not run against these benchmarks, for the same
disclosed reason).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

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

from calibration_benchmark import PreMovementDelayCandidate, recommend  # noqa: E402

from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, PublishedValue  # noqa: E402
from calibration_studio.studio import CalibrationStudio  # noqa: E402

from automatic_calibration.budget import AutoCalibrationBudget  # noqa: E402
from automatic_calibration.engine import AutoCalibrationEngine  # noqa: E402
from automatic_calibration.grid_search import GridSearchStrategy  # noqa: E402
from automatic_calibration.objectives import PublishedValueObjective  # noqa: E402
from automatic_calibration.search_space import ParameterDimension, SearchSpace  # noqa: E402


PUBLISHED_EVACUATION_TIME_S = {
    "10-story": 1022.0,
    "18-story": 1192.0,
    "24-story": 1090.0,
    "31-story": 1002.0,
}

BUILDINGS = {
    "10-story": (build_nist_10story_building, build_nist_10story_definition, DEFINITION_ID_10, MASTER_SEED_10,
                 "scripts.run_nist_10story_validation.build_nist_10story_building"),
    "18-story": (build_nist_18story_building, build_nist_18story_definition, DEFINITION_ID_18, MASTER_SEED_18,
                 "scripts.run_nist_18story_validation.build_nist_18story_building"),
    "24-story": (build_nist_24story_building, build_nist_24story_definition, DEFINITION_ID_24, MASTER_SEED_24,
                 "scripts.run_nist_24story_validation.build_nist_24story_building"),
    "31-story": (build_nist_31story_building, build_nist_31story_definition, DEFINITION_ID_31, MASTER_SEED_31,
                 "scripts.run_nist_31story_validation.build_nist_31story_building"),
}

# Literature-grounded -- NIST TN1664 (see module docstring). 30.0
# (today's production default) is deliberately a grid point, not just an
# external comparison value.
ADULT_MEDIAN_DELAY_GRID = (15.0, 20.0, 25.0, 30.0, 35.0, 45.0, 55.0)

# Exploratory-only confirmatory diagnostic (see module docstring) -- NOT
# claimed as literature-grounded for THIS building's population (no
# Child_Default occupants exist in any NIST benchmark scenario); values
# themselves are drawn from the companion report's own school-evacuation
# literature range for context only.
CHILD_MEDIAN_DELAY_DIAGNOSTIC_GRID = (45.0, 90.0, 120.0)


def _make_benchmark(building_name, geometry_ref):

    return PublishedBenchmark(
        title=f"NIST {building_name} Office Building Fire Drill Recreation",
        source_citation="NIST published evacuation drill recreation (existing SynEvac validation campaign)",
        dataset=f"nist_{building_name.replace('-', '_')}",
        benchmark_type=BenchmarkType.BUILDING_RECREATION,
        geometry_reference=GeometryVersion(version="v1", ref=geometry_ref),
        published_values={"evacuation_time_s": PublishedValue(value=PUBLISHED_EVACUATION_TIME_S[building_name], unit="s")},
    )


def run_campaign_for_profile_and_building(profile_id, grid, building_name, n_scenarios, dt=1.0):

    build_building, build_definition, definition_id, master_seed, geometry_ref = BUILDINGS[building_name]

    building = build_building()
    definition = build_definition()

    studio = CalibrationStudio()
    project = studio.create_project(
        name=f"Human Behavior Calibration Campaign V1 -- {profile_id} pre-movement delay -- {building_name}",
    )

    benchmark = _make_benchmark(building_name, geometry_ref)
    studio.benchmarks.register(benchmark)

    objective = PublishedValueObjective(
        benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
    )

    dimension_name = f"{profile_id}.pre_movement_strategy.median_delay"

    search_space = SearchSpace(dimensions=(
        ParameterDimension(
            name=dimension_name, bounds=(min(grid) - 5.0, max(grid) + 5.0),
            build=lambda v, profile_id=profile_id: PreMovementDelayCandidate(
                profile_id, v, "nist-tn1664-office-highrise-literature-grid",
                "Human Behavior Calibration Campaign V1 -- pre-movement delay",
            ),
        ),
    ))

    strategy = GridSearchStrategy(values={dimension_name: grid})

    engine = AutoCalibrationEngine(
        studio=studio, project=project, building=building, definition=definition,
        definition_id=definition_id, master_seed=master_seed, n_scenarios=n_scenarios, dt=dt,
        search_space=search_space, objective=objective, strategy=strategy,
        budget=AutoCalibrationBudget(max_evaluations=len(grid)),
    )

    run = engine.run()

    published = PUBLISHED_EVACUATION_TIME_S[building_name]
    per_value_results = []

    for session_id in run.session_ids:

        session = studio.get_session(session_id)
        comparison = session.result.comparisons["evacuation_time"] if session.result is not None else None
        adoption = recommend(session.result) if session.result is not None else None

        candidate_mean = comparison.candidate_mean if comparison else None

        per_value_results.append({
            "median_delay": session.candidate_snapshot["candidate_value"],
            "session_status": session.status.value,
            "candidate_mean_evacuation_time": candidate_mean,
            "baseline_mean_evacuation_time": comparison.baseline_mean if comparison else None,
            "candidate_ci": comparison.candidate_ci.to_dict() if comparison else None,
            "p_value_vs_production_default": comparison.paired.p_value if comparison else None,
            "cohens_d_vs_production_default": comparison.effect_size.cohens_d if comparison else None,
            "score_distance_from_published": objective.score(session),
            "overprediction_ratio": (candidate_mean / published) if candidate_mean is not None else None,
            "recommendation": adoption.overall_verdict.name if adoption else None,
            "recommendation_summary": adoption.summary if adoption else None,
        })

    best_session = studio.get_session(run.best_session_id) if run.best_session_id else None
    best_comparison = (
        best_session.result.comparisons["evacuation_time"]
        if best_session is not None and best_session.result is not None else None
    )
    best_evacuation_time = best_comparison.candidate_mean if best_comparison else None

    return {
        "profile_id": profile_id,
        "building": building_name,
        "published_evacuation_time_s": published,
        "n_scenarios": n_scenarios,
        "grid": list(grid),
        "run_status": run.status.value,
        "best_session_id": run.best_session_id,
        "best_median_delay": (
            best_session.candidate_snapshot["candidate_value"] if best_session is not None else None
        ),
        "best_score": run.best_score,
        "best_evacuation_time": best_evacuation_time,
        "best_overprediction_ratio": (best_evacuation_time / published) if best_evacuation_time is not None else None,
        "per_value_results": per_value_results,
    }


def main():

    n_scenarios_adult = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_scenarios_diagnostic = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dt = 1.0

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {"Adult_Default": {}, "Child_Default_diagnostic_only": {}}

    for building_name in BUILDINGS:

        print(
            f"[Adult_Default] Running pre-movement delay calibration for {building_name} "
            f"({len(ADULT_MEDIAN_DELAY_GRID)} grid points x {n_scenarios_adult} seeds)...", flush=True,
        )
        result = run_campaign_for_profile_and_building(
            "Adult_Default", ADULT_MEDIAN_DELAY_GRID, building_name, n_scenarios=n_scenarios_adult, dt=dt,
        )
        all_results["Adult_Default"][building_name] = result
        print(json.dumps(result, indent=2, default=str))

    # Confirmatory diagnostic ONLY -- see module docstring. Deliberately
    # one building, reduced grid, reduced n_scenarios: the predicted
    # result (byte-for-byte identical to baseline, since no Child_Default
    # occupant exists in this benchmark's population) does not require a
    # large sample to confirm, and running the full 4-building x 6-8-seed
    # sweep here would only spend compute reproducing an already-provable
    # null result four more times.
    print(
        f"[Child_Default DIAGNOSTIC] Running confirmatory no-op check for 10-story "
        f"({len(CHILD_MEDIAN_DELAY_DIAGNOSTIC_GRID)} grid points x {n_scenarios_diagnostic} seeds)...", flush=True,
    )
    diagnostic_result = run_campaign_for_profile_and_building(
        "Child_Default", CHILD_MEDIAN_DELAY_DIAGNOSTIC_GRID, "10-story", n_scenarios=n_scenarios_diagnostic, dt=dt,
    )
    all_results["Child_Default_diagnostic_only"]["10-story"] = diagnostic_result
    print(json.dumps(diagnostic_result, indent=2, default=str))

    with open(
        os.path.join(output_dir, "human_behavior_calibration_campaign_v1_pre_movement_delay_raw_results.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(all_results, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
