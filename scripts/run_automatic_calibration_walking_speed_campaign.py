"""
Automatic Calibration Engine, Phase 3 -- First Automatic Calibration
Campaign.

Uses the real, unmodified AutoCalibrationEngine + GridSearchStrategy
(Phases 1-2) to automatically calibrate Adult_Default.walking_speed
against the published NIST evacuation-time benchmark, independently for
each of the four NIST buildings already used throughout this repo's own
validation work (10/18/24/31-story). No new dataset acquired; no
production simulator code modified; no automatic_calibration/
calibration_studio/calibration_benchmark source file modified -- this
script is a pure CLIENT of the existing, already-tested infrastructure,
exactly the same discipline AutoCalibrationEngine itself follows one
layer down.

Search space: Adult_Default.walking_speed in {0.6, 0.8, 1.0, 1.2, 1.4,
1.6} m/s -- see the companion chat report for the literature grounding.
1.2 m/s (today's production default) is deliberately included as a grid
point, so the campaign can report objectively where the current default
ranks among the alternatives it evaluated, not just compare against it
as a fixed external reference.

One AutoCalibrationEngine run per building (Phase 1's own engine has no
multi-building concept -- extending it would be scope creep for an
"experimental evaluation only" milestone), each anchored to that
building's own PublishedBenchmark(published_values={"evacuation_time_s":
...}). Every evaluated grid point is a REAL CalibrationSession, produced
via CalibrationStudio.run_published_benchmark() exactly as any manual
calibration run would be.
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

from calibration_benchmark import WalkingSpeedCandidate, recommend  # noqa: E402

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

# Literature-grounded grid -- see the companion chat report for the
# full justification (Fruin/SFPE unimpeded adult walking speed range,
# NIST TN1675's own measured stairwell speeds, existing SynEvac profile
# values). 1.2 m/s (Adult_Default's current production default) is
# deliberately a grid point, not just an external comparison value.
WALKING_SPEED_GRID = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6)


def _make_benchmark(building_name, geometry_ref):

    return PublishedBenchmark(
        title=f"NIST {building_name} Office Building Fire Drill Recreation",
        source_citation="NIST published evacuation drill recreation (existing SynEvac validation campaign)",
        dataset=f"nist_{building_name.replace('-', '_')}",
        benchmark_type=BenchmarkType.BUILDING_RECREATION,
        geometry_reference=GeometryVersion(version="v1", ref=geometry_ref),
        published_values={"evacuation_time_s": PublishedValue(value=PUBLISHED_EVACUATION_TIME_S[building_name], unit="s")},
    )


def run_campaign_for_building(building_name, n_scenarios=8, dt=1.0):

    build_building, build_definition, definition_id, master_seed, geometry_ref = BUILDINGS[building_name]

    building = build_building()
    definition = build_definition()

    studio = CalibrationStudio()
    project = studio.create_project(name=f"Phase 3 Walking Speed Campaign -- {building_name}")

    benchmark = _make_benchmark(building_name, geometry_ref)
    studio.benchmarks.register(benchmark)

    objective = PublishedValueObjective(
        benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
    )

    search_space = SearchSpace(dimensions=(
        ParameterDimension(
            name="Adult_Default.walking_speed", bounds=(0.5, 1.7),
            build=lambda v: WalkingSpeedCandidate("Adult_Default", v, "test-fixture-literature-grid", "Phase 3 automatic calibration campaign"),
        ),
    ))

    strategy = GridSearchStrategy(values={"Adult_Default.walking_speed": WALKING_SPEED_GRID})

    engine = AutoCalibrationEngine(
        studio=studio, project=project, building=building, definition=definition,
        definition_id=definition_id, master_seed=master_seed, n_scenarios=n_scenarios, dt=dt,
        search_space=search_space, objective=objective, strategy=strategy,
        budget=AutoCalibrationBudget(max_evaluations=len(WALKING_SPEED_GRID)),
    )

    run = engine.run()

    per_value_results = []

    for session_id in run.session_ids:

        session = studio.get_session(session_id)
        comparison = session.result.comparisons["evacuation_time"] if session.result is not None else None
        adoption = recommend(session.result) if session.result is not None else None

        per_value_results.append({
            "walking_speed": session.candidate_snapshot["candidate_value"],
            "session_status": session.status.value,
            "candidate_mean_evacuation_time": comparison.candidate_mean if comparison else None,
            "baseline_mean_evacuation_time": comparison.baseline_mean if comparison else None,
            "candidate_ci": comparison.candidate_ci.to_dict() if comparison else None,
            "p_value_vs_production_default": comparison.paired.p_value if comparison else None,
            "cohens_d_vs_production_default": comparison.effect_size.cohens_d if comparison else None,
            "score_distance_from_published": objective.score(session),
            "recommendation": adoption.overall_verdict.name if adoption else None,
            "recommendation_summary": adoption.summary if adoption else None,
        })

    return {
        "building": building_name,
        "published_evacuation_time_s": PUBLISHED_EVACUATION_TIME_S[building_name],
        "n_scenarios": n_scenarios,
        "run_status": run.status.value,
        "best_session_id": run.best_session_id,
        "best_score": run.best_score,
        "per_value_results": per_value_results,
    }


def main():

    n_scenarios = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    dt = 1.0

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for building_name in BUILDINGS:

        print(f"Running automatic calibration campaign for {building_name} "
              f"({len(WALKING_SPEED_GRID)} grid points x {n_scenarios} seeds)...", flush=True)

        result = run_campaign_for_building(building_name, n_scenarios=n_scenarios, dt=dt)
        all_results[building_name] = result

        print(json.dumps(result, indent=2, default=str))

    with open(
        os.path.join(output_dir, "automatic_calibration_walking_speed_campaign_raw_results.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(all_results, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
