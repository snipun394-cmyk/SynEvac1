"""
One-off chart generation for the NIST 18-story validation report (V2).
Reads docs/architecture/nist_18story_validation_v1_raw_results.json and
the V1 10-story results for direct comparison. Not part of any SynEvac
package; not imported by any test or production code.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = r"C:\Users\riddh\Desktop\NFSC\mini project\charts"

NIST_10STORY_PUBLISHED_S = 1022.0
NIST_18STORY_PUBLISHED_S = 1192.0


def chart_evacuation_time_comparison(results_10, results_18):

    labels = [
        "NIST 10-story\npublished", "SynEvac 10-story\n(defaults)",
        "NIST 18-story\npublished", "SynEvac 18-story\n(defaults)",
    ]
    off10 = results_10["official_summary"]
    off18 = results_18["official_summary"]

    means = [NIST_10STORY_PUBLISHED_S, off10["evacuation_time_mean"], NIST_18STORY_PUBLISHED_S, off18["evacuation_time_mean"]]
    lower = [0, off10["evacuation_time_mean"] - off10["evacuation_time_ci_lower"], 0, off18["evacuation_time_mean"] - off18["evacuation_time_ci_lower"]]
    upper = [0, off10["evacuation_time_ci_upper"] - off10["evacuation_time_mean"], 0, off18["evacuation_time_ci_upper"] - off18["evacuation_time_mean"]]
    colors = ["#4C72B0", "#DD8452", "#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=[lower, upper], capsize=6, color=colors, alpha=0.85)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean * 1.02, f"{mean:.0f}s", ha="center", fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("Total evacuation time (seconds, log scale)")
    ax.set_title("NIST 10-Story vs. 18-Story: Published vs. SynEvac Simulated Evacuation Time")

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "evacuation_time_comparison_10_vs_18.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def chart_stair1_vs_stair7(results_18):

    stair1 = results_18["stair1_trace"]["trace"]
    stair7 = results_18["stair7_trace"]["trace"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax, trace, title in [
        (axes[0], stair1, f"Stair 1 (direct exit)\nTotal: {results_18['stair1_trace']['final_arrival_time']:.0f}s"),
        (axes[1], stair7, f"Stair 7 (lobby-merge)\nTotal: {results_18['stair7_trace']['final_arrival_time']:.0f}s"),
    ]:
        labels = [s["edge_id"] for s in trace]
        queue_waits = [s["queue_wait_time"] for s in trace]
        travel_times = [s["end_time"] - s["start_time"] for s in trace]

        x = range(len(labels))
        ax.bar(x, queue_waits, label="Queue wait time", color="#C44E52")
        ax.bar(x, travel_times, bottom=queue_waits, label="Actual travel time", color="#55A868")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylabel("Seconds")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Same building, two topologies, one occupant's full descent each (SynEvac defaults)\n"
        "Stair 7's final 0.91m DOOR (not a Staircase) is the single largest bottleneck of either trace",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(OUTPUT_DIR, "stair1_vs_stair7_queue_wait.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":

    base = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")

    with open(os.path.join(base, "nist_10story_validation_v1_raw_results.json"), encoding="utf-8") as f:
        results_10 = json.load(f)
    with open(os.path.join(base, "nist_18story_validation_v1_raw_results.json"), encoding="utf-8") as f:
        results_18 = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(chart_evacuation_time_comparison(results_10, results_18))
    print(chart_stair1_vs_stair7(results_18))
