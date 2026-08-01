"""
Calibration Benchmark Framework V1 -- demonstration run.

Benchmarks the Calibration Mapping Report's own worked example: Adult
walking speed. Current SynEvac default (behaviour_profile_resolver/
registry.py::DEFAULT_PROFILE_REGISTRY["Adult_Default"].walking_speed =
1.2 m/s) versus a candidate value derived from the already-acquired
Julich Pedestrian Dynamics Data Archive stair-egress trajectories
(tu11-tu14.txt, free-flow-condition instantaneous speed, mean 0.649 m/s,
rounded to 0.65 m/s -- see this run's own report for the exact
derivation and its scoping caveat: Julich's trajectories are STAIR
descent under a calm, instructed experiment, not level-ground walking
under a real emergency, so this candidate should be read as "how would
SynEvac's own evacuation model react to a materially slower per-person
speed," not as a final, ready-to-adopt replacement value).

Does not modify any SynEvac default. Writes a Markdown report to
docs/architecture/calibration_benchmark_v1_demo_report.md.
"""

import os

from calibration_benchmark.candidates import WalkingSpeedCandidate
from calibration_benchmark.harness import run_calibration_benchmark
from calibration_benchmark.optional_metrics import PredictionAccuracyMetric, RecommendationEffectivenessMetric
from calibration_benchmark.recommendation import recommend
from calibration_benchmark.report import render_markdown_report

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


def main():

    building = make_building()
    definition = make_definition(occupant_count=25)

    candidate = WalkingSpeedCandidate(
        profile_id="Adult_Default",
        candidate_speed=0.65,
        dataset_source=(
            "Julich Pedestrian Dynamics Data Archive, stair-egress experiment "
            "2009.04.29_Duesseldorf_Arena_Hermes (DOI 10.34735/ped.2009.5), files "
            "tu11.txt-tu14.txt -- free-flow-condition instantaneous speed (bottom "
            "quartile of concurrent occupancy per run), mean 0.649 m/s, n=1818 samples"
        ),
        rationale=(
            "SynEvac Calibration Mapping Report, Part 3.1 (Walking Model, Tier 2): "
            "DEFAULT_PROFILE_REGISTRY's own walking_speed constants are self-disclosed "
            "as illustrative, not validated. This candidate is the free-flow speed "
            "derived directly from real, physical stair-egress trajectories -- disclosed "
            "here as a STAIR-descent value, materially slower than level-ground walking, "
            "so a large discrepancy against SynEvac's current flat (level-and-stair) "
            "constant is an expected, informative finding, not a data error."
        ),
    )

    result = run_calibration_benchmark(
        candidate, building, definition, DEFINITION_ID, MASTER_SEED,
        n_scenarios=30, dt=1.0,
        additional_metrics=[RecommendationEffectivenessMetric(), PredictionAccuracyMetric()],
    )

    recommendation = recommend(result)
    markdown = render_markdown_report(result, recommendation=recommendation)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "calibration_benchmark_v1_demo_report.md")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)

    print(f"n_completed_pairs: {result.n_completed_pairs}")
    print(f"overall verdict: {recommendation.overall_verdict.name}")
    print(f"report written to: {output_path}")


if __name__ == "__main__":
    main()
