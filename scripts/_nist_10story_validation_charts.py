"""
One-off chart generation for the NIST 10-story validation report. Reads
docs/architecture/nist_10story_validation_v1_raw_results.json (already
produced by scripts/run_nist_10story_validation.py) and produces two
PNGs in the NFSC research folder. Not part of any SynEvac package; not
imported by any test or production code.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.run_nist_10story_validation import (
    build_nist_10story_building, build_nist_10story_definition, DEFINITION_ID, MASTER_SEED,
)
from calibration_benchmark.simulation_seam import run_with_overrides
from scenario_pipeline import run_batch_pipeline

NIST_PUBLISHED_EVACUATION_TIME_S = 1022.0

OUTPUT_DIR = r"C:\Users\riddh\Desktop\NFSC\mini project\charts"


def chart_evacuation_time_comparison(results_path):

    with open(results_path, encoding="utf-8") as handle:
        data = json.load(handle)

    official = data["official_summary"]
    diagnostic = data["diagnostic_summary"]

    labels = ["NIST published\n(single drill)", "SynEvac\n(current defaults)", "SynEvac\n(diagnostic: stair\ncapacity 1\u219210)"]
    means = [NIST_PUBLISHED_EVACUATION_TIME_S, official["evacuation_time_mean"], diagnostic["evacuation_time_mean"]]
    lower_err = [
        0,
        official["evacuation_time_mean"] - official["evacuation_time_ci_lower"],
        diagnostic["evacuation_time_mean"] - diagnostic["evacuation_time_ci_lower"],
    ]
    upper_err = [
        0,
        official["evacuation_time_ci_upper"] - official["evacuation_time_mean"],
        diagnostic["evacuation_time_ci_upper"] - diagnostic["evacuation_time_mean"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#DD8452", "#DD8452"]
    bars = ax.bar(labels, means, yerr=[lower_err, upper_err], capsize=6, color=colors, alpha=0.85)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 60, f"{mean:.0f}s", ha="center", fontweight="bold")

    ax.set_ylabel("Total evacuation time (seconds)")
    ax.set_title("NIST 10-Story Building: Published vs. SynEvac Simulated Evacuation Time\n(95% CI shown for SynEvac runs; NIST is a single real drill, no CI)")
    ax.axhline(NIST_PUBLISHED_EVACUATION_TIME_S, color="#4C72B0", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "evacuation_time_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def chart_per_segment_queue_wait():

    building = build_nist_10story_building()
    definition = build_nist_10story_definition()
    batch = run_batch_pipeline(definition, DEFINITION_ID, building, MASTER_SEED, 1)
    scenario = batch.scenarios[0]
    movement_result, ground_truth, _ = run_with_overrides(scenario, building, dt=1.0)

    arrived = [t for t in movement_result.occupants.values() if t.state.name == "ARRIVED"]
    worst = max(arrived, key=lambda t: t.steps[-1].end_time if t.steps else 0)

    # step.start_time/end_time bracket ONLY the active traversal of this
    # edge; queue_wait_time is the (larger) time already spent waiting
    # BEFORE start_time, not a component of (end_time - start_time) --
    # confirmed empirically (a single step's own queue_wait_time value
    # was observed to exceed its own end_time-start_time duration,
    # which would be impossible if the two overlapped). The two are
    # therefore sequential, not overlapping, and are stacked as such.
    labels = [step.edge.id for step in worst.steps]
    queue_waits = [getattr(step, "queue_wait_time", 0.0) or 0.0 for step in worst.steps]
    travel_times = [step.end_time - step.start_time for step in worst.steps]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(labels))
    ax.bar(x, queue_waits, label="Queue wait time", color="#C44E52")
    ax.bar(x, travel_times, bottom=queue_waits, label="Actual travel time", color="#55A868")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title(
        "One occupant's full descent (floor 10 \u2192 exit), SynEvac defaults\n"
        f"Total: {worst.steps[-1].end_time:.0f}s -- queue waiting dominates every individual stair flight"
    )
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "per_segment_queue_wait.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":

    results_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "architecture", "nist_10story_validation_v1_raw_results.json",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path1 = chart_evacuation_time_comparison(results_path)
    path2 = chart_per_segment_queue_wait()

    print(path1)
    print(path2)
