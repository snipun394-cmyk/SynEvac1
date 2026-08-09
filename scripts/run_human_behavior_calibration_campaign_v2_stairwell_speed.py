"""
Human Behavior Calibration Campaign V2 -- Direct Behavioral Calibration:
Walking Speed Against NIST TN 1675.

Unlike scripts/run_automatic_calibration_walking_speed_campaign.py (which
calibrates Adult_Default.walking_speed against AGGREGATE published
whole-building evacuation time), this campaign calibrates the SAME
parameter against a genuinely behavior-specific, local observable: NIST
Technical Note 1675 / Peacock, Hoskins & Kuligowski, "Overall and local
movement speeds during fire drill evacuations in buildings up to 31
stories" (Safety Science 50 (2012) 1655-1664; NIST TN 1675, Sept. 2010),
Table 3's per-building "Overall Movement Speed" -- the average speed each
building's real evacuee population moved down its own stairwell during
the actual fire drill, in m/s.

This is the SAME four buildings (10/18/24/31-story) and the SAME source
paper already used, since before this milestone, to build
scripts/run_nist_*_validation.py's own geometry/occupant counts (see
those scripts' own docstrings and Table 2 citations) and the aggregate
evacuation-time benchmark scripts/run_automatic_calibration_walking_speed_
campaign.py already validates against (1022/1192/1090/1002 s -- Table 2's
own "Evacuation time (s)" row, byte-for-byte). TN 1675's Table 3 values
were simply never operationalized as a calibration TARGET before now.

Mapping validity (see the companion chat report's own Phase 1 for the
full source-level derivation): BehaviorProfileTemplate.walking_speed is
NOT a stair-specific speed constant -- simulator/coordinator.py's own
_advance_occupant() applies it identically to every edge type via
`effective_speed = occupant.walking_speed * congestion_model.speed_factor(...)`,
`duration = edge.traversal_cost / effective_speed`. For a Stair edge,
`traversal_cost` (== `Staircase.travel_distance(building)`,
models/staircase.py) is the real along-incline path length
(`vertical_height / sin(35 deg)`), not a horizontal projection. So
walking_speed IS the free-flow along-incline stair descent speed --
directly comparable to TN 1675's own quantity -- but ONLY in the
congestion-free case; TN 1675's own Table 3 values are themselves
observed under real, escalating fire-drill crowd density, not free-flow.
Setting walking_speed := TN1675's raw mean would double-count crowding
(TN1675's mean already reflects real congestion; SynEvac's own
congestion_model would degrade it a second time). The correct comparison
is therefore SIMULATED (congestion-inclusive) local stair speed, measured
from real occupant timelines under a comparable simulated population,
against TN1675's own reported mean -- never walking_speed itself against
TN1675's mean directly.

Two distinct SynEvac-side speed metrics are computed per scenario (both
new AdditionalMetric implementations, defined here, script-local -- no
calibration_benchmark/ source file is modified, following the exact
discipline scripts/run_automatic_calibration_walking_speed_campaign.py
and docs/architecture/human_behavior_calibration_campaign_v1_pre_movement_delay.md
already established):

  - stairwell_descent_speed ("movement-only"): for each occupant, sum of
    (Stair edge walking_distance) / sum of (that edge's own end_time -
    start_time), averaged across occupants who used a stair. This
    isolates walking_speed x congestion_model.speed_factor -- the exact
    mechanism this campaign calibrates -- and EXCLUDES admission-control
    queue_wait_time (a separate, already-instrumented SynEvac concept;
    see simulator/multi_agent_result.py's own OccupantTimelineStep.
    queue_wait_time field). THIS is the campaign's calibration objective.

  - stairwell_span_speed ("wall-clock"): the same total distance divided
    by wall-clock elapsed time from first Stair-edge entry to last
    Stair-edge exit (i.e. INCLUDING any inter-segment admission-control
    queueing). Reported as a diagnostic only, never as the objective --
    an exploratory probe (see the companion chat report) found this
    value is dominated by Admission Control's own queueing behavior, not
    by walking_speed, at these buildings' real occupant densities: it is
    materially unaffected by which walking_speed candidate is under
    test. Conflating it with the movement-only metric would silently let
    a capacity/admission-control effect masquerade as a walking-speed
    calibration result -- exactly the kind of conflation Phase 6 of the
    companion report's own brief warns against for whole-building
    evacuation time.

Whole-building evacuation_time (METRIC_FIELDS' own built-in metric,
computed automatically for every session regardless of additional_metrics)
is read back from the SAME sessions this campaign already runs, for
Phase 6 out-of-target cross-validation against Table 2's published
evacuation times -- no second campaign, no second simulation, is run for
that purpose.

Extension used: automatic_calibration.objectives.PublishedValueObjective.
score() now falls back to session.result.additional_comparisons when a
result_metric_name isn't one of the five built-in METRIC_FIELDS (see that
module's own comment) -- the smallest change that let this campaign reuse
PublishedValueObjective/AutoCalibrationEngine/GridSearchStrategy exactly
as-is rather than building a second calibration engine or objective type.

No simulator/, behaviour_profile_resolver/, or calibration_benchmark/
source file is modified. No production default is changed by running
this script.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from navigation.edge import Edge  # noqa: E402

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
from calibration_benchmark.optional_metrics import AdditionalMetric  # noqa: E402

from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, PublishedValue  # noqa: E402
from calibration_studio.studio import CalibrationStudio  # noqa: E402

from automatic_calibration.budget import AutoCalibrationBudget  # noqa: E402
from automatic_calibration.engine import AutoCalibrationEngine  # noqa: E402
from automatic_calibration.grid_search import GridSearchStrategy  # noqa: E402
from automatic_calibration.objectives import PublishedValueObjective  # noqa: E402
from automatic_calibration.search_space import ParameterDimension, SearchSpace  # noqa: E402


# =====================================================
# TN 1675 Table 3 -- "Pre-evacuation time and stairwell movement speeds
# in three [four] fire drill evacuations". Directly-usable summary
# statistics (mean +/- SD) per building; TN 1675 provides no raw
# per-occupant data file accompanying this table (raw per-occupant CSVs
# are referenced as hosted at http://fire.nist.gov/egress/, NOT fetched
# or reproduced by this campaign -- see the companion chat report's
# Phase 2 DIRECTLY-USABLE vs SUMMARY-ONLY breakdown).
# =====================================================

STAIRWELL_SPEED_TARGET_MS = {
    "10-story": {"mean": 0.44, "sd": 0.19},
    "18-story": {"mean": 0.44, "sd": 0.15},
    "24-story": {"mean": 0.56, "sd": 0.12},
    "31-story": {"mean": 0.52, "sd": 0.10},
}

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

# Literature-grounded grid -- brackets Fruin's own "optimum" (0.48 m/s)
# to "maximum" (0.76 m/s) UNIMPEDED stair descent speed (TN1675's own
# Table 1, itself cited from Fruin/Pauls) -- deliberately NOT centered on
# TN1675's raw 0.44-0.56 m/s congested means (see this module's own
# docstring on double-counting). 1.2 m/s (today's Adult_Default
# production default, and the value used identically for horizontal
# movement) is retained as the campaign's own established diagnostic
# reference point.
WALKING_SPEED_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.2)


# =====================================================


class StairwellDescentSpeedMetric(AdditionalMetric):

    # PRIMARY calibration metric -- see this module's own docstring.
    # Movement-only: sums each Stair-edge step's own (distance, duration)
    # per occupant, EXCLUDING any inter-edge admission-control queue
    # time. None when a scenario produced no completed Stair traversal
    # at all (never a fabricated 0.0).

    name = "stairwell_descent_speed"

    def compute(self, ground_truth, movement_result, building, peak_occupancy_ratio, density_thresholds):

        speeds = []

        for timeline in movement_result.occupants.values():

            stair_steps = [step for step in timeline.steps if step.edge.edge_type == Edge.STAIR]

            if not stair_steps:
                continue

            total_distance = sum((step.edge.walking_distance or 0.0) for step in stair_steps)
            total_duration = sum((step.end_time - step.start_time) for step in stair_steps)

            if total_distance <= 0 or total_duration <= 0:
                continue

            speeds.append(total_distance / total_duration)

        return (sum(speeds) / len(speeds)) if speeds else None


class StairwellSpanSpeedMetric(AdditionalMetric):

    # DIAGNOSTIC ONLY -- see this module's own docstring. Same total
    # distance, but wall-clock span (last Stair-edge end_time minus
    # first Stair-edge start_time), so admission-control queueing between
    # Stair segments is INCLUDED. Never used as this campaign's objective.

    name = "stairwell_span_speed"

    def compute(self, ground_truth, movement_result, building, peak_occupancy_ratio, density_thresholds):

        speeds = []

        for timeline in movement_result.occupants.values():

            stair_steps = [step for step in timeline.steps if step.edge.edge_type == Edge.STAIR]

            if not stair_steps:
                continue

            total_distance = sum((step.edge.walking_distance or 0.0) for step in stair_steps)
            span = max(step.end_time for step in stair_steps) - min(step.start_time for step in stair_steps)

            if total_distance <= 0 or span <= 0:
                continue

            speeds.append(total_distance / span)

        return (sum(speeds) / len(speeds)) if speeds else None


ADDITIONAL_METRICS = [StairwellDescentSpeedMetric(), StairwellSpanSpeedMetric()]


# =====================================================


def _make_benchmark(building_name, geometry_ref):

    target = STAIRWELL_SPEED_TARGET_MS[building_name]

    return PublishedBenchmark(
        title=f"NIST {building_name} Office Building Fire Drill -- TN1675 Stairwell Speed",
        source_citation=(
            "Peacock, Hoskins, Kuligowski (2010/2012), NIST TN 1675 / Safety Science 50, "
            "Table 3 -- Overall Movement Speed"
        ),
        dataset=f"nist_tn1675_{building_name.replace('-', '_')}",
        benchmark_type=BenchmarkType.BUILDING_RECREATION,
        geometry_reference=GeometryVersion(version="v1", ref=geometry_ref),
        published_values={
            "stairwell_speed_ms": PublishedValue(value=target["mean"], unit="m/s"),
            "stairwell_speed_sd_ms": PublishedValue(value=target["sd"], unit="m/s"),
            "evacuation_time_s": PublishedValue(value=PUBLISHED_EVACUATION_TIME_S[building_name], unit="s"),
        },
    )


def run_campaign_for_building(building_name, n_scenarios=4, dt=1.0):

    build_building, build_definition, definition_id, master_seed, geometry_ref = BUILDINGS[building_name]

    building = build_building()
    definition = build_definition()

    studio = CalibrationStudio()
    project = studio.create_project(name=f"Human Behavior Campaign V2 -- Stairwell Speed -- {building_name}")

    benchmark = _make_benchmark(building_name, geometry_ref)
    studio.benchmarks.register(benchmark)

    objective = PublishedValueObjective(
        benchmark, published_metric_name="stairwell_speed_ms", result_metric_name="stairwell_descent_speed",
    )

    search_space = SearchSpace(dimensions=(
        ParameterDimension(
            name="Adult_Default.walking_speed", bounds=(0.4, 1.3),
            build=lambda v: WalkingSpeedCandidate(
                "Adult_Default", v, "NIST TN1675 Table 1 (Fruin optimum-maximum stair range)",
                "Human Behavior Calibration Campaign V2 -- direct TN1675 stairwell-speed calibration",
            ),
        ),
    ))

    strategy = GridSearchStrategy(values={"Adult_Default.walking_speed": WALKING_SPEED_GRID})

    engine = AutoCalibrationEngine(
        studio=studio, project=project, building=building, definition=definition,
        definition_id=definition_id, master_seed=master_seed, n_scenarios=n_scenarios, dt=dt,
        search_space=search_space, objective=objective, strategy=strategy,
        budget=AutoCalibrationBudget(max_evaluations=len(WALKING_SPEED_GRID)),
        additional_metrics=ADDITIONAL_METRICS,
    )

    run = engine.run()

    per_value_results = []

    for session_id in run.session_ids:

        session = studio.get_session(session_id)
        result = session.result

        movement_comparison = result.additional_comparisons.get("stairwell_descent_speed") if result else None
        span_comparison = result.additional_comparisons.get("stairwell_span_speed") if result else None
        evac_comparison = result.comparisons.get("evacuation_time") if result else None

        adoption = recommend(result) if result is not None else None

        per_value_results.append({
            "walking_speed": session.candidate_snapshot["candidate_value"],
            "session_status": session.status.value,

            "movement_only_stairwell_speed_candidate_mean": movement_comparison.candidate_mean if movement_comparison else None,
            "movement_only_stairwell_speed_baseline_mean": movement_comparison.baseline_mean if movement_comparison else None,
            "movement_only_stairwell_speed_candidate_ci": movement_comparison.candidate_ci.to_dict() if movement_comparison else None,
            "movement_only_p_value_vs_production_default": movement_comparison.paired.p_value if movement_comparison else None,
            "movement_only_cohens_d_vs_production_default": movement_comparison.effect_size.cohens_d if movement_comparison else None,

            "wallclock_span_stairwell_speed_candidate_mean": span_comparison.candidate_mean if span_comparison else None,
            "wallclock_span_stairwell_speed_baseline_mean": span_comparison.baseline_mean if span_comparison else None,

            "whole_building_evacuation_time_candidate_mean": evac_comparison.candidate_mean if evac_comparison else None,
            "whole_building_evacuation_time_baseline_mean": evac_comparison.baseline_mean if evac_comparison else None,
            "whole_building_evac_p_value_vs_production_default": evac_comparison.paired.p_value if evac_comparison else None,
            "whole_building_evac_cohens_d_vs_production_default": evac_comparison.effect_size.cohens_d if evac_comparison else None,

            "score_distance_from_tn1675_mean": objective.score(session),
            "recommendation": adoption.overall_verdict.name if adoption else None,
            "recommendation_summary": adoption.summary if adoption else None,
        })

    return {
        "building": building_name,
        "published_stairwell_speed_mean_ms": STAIRWELL_SPEED_TARGET_MS[building_name]["mean"],
        "published_stairwell_speed_sd_ms": STAIRWELL_SPEED_TARGET_MS[building_name]["sd"],
        "published_evacuation_time_s": PUBLISHED_EVACUATION_TIME_S[building_name],
        "n_scenarios": n_scenarios,
        "run_status": run.status.value,
        "best_session_id": run.best_session_id,
        "best_score": run.best_score,
        "per_value_results": per_value_results,
    }


def main():

    n_scenarios = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dt = 1.0

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for building_name in BUILDINGS:

        print(f"Running TN1675 stairwell-speed calibration campaign for {building_name} "
              f"({len(WALKING_SPEED_GRID)} grid points x {n_scenarios} seeds)...", flush=True)

        result = run_campaign_for_building(building_name, n_scenarios=n_scenarios, dt=dt)
        all_results[building_name] = result

        print(json.dumps(result, indent=2, default=str))

        with open(
            os.path.join(output_dir, "human_behavior_calibration_campaign_v2_stairwell_speed_raw_results.json"),
            "w", encoding="utf-8",
        ) as handle:
            json.dump(all_results, handle, indent=2, default=str)

    print("Campaign complete.", flush=True)


if __name__ == "__main__":
    main()
