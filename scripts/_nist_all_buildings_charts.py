"""
Combined chart generation for the four-building NIST Published Scenario
Validation Campaign (V1-V3). Reads all four buildings' raw results JSON
files and produces comparison charts in the NFSC research folder. Not
part of any SynEvac package; not imported by any test or production code.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = r"C:\Users\riddh\Desktop\NFSC\mini project\charts"

PUBLISHED_S = {"10-story": 1022.0, "18-story": 1192.0, "24-story": 1090.0, "31-story": 1002.0}
FLOOR_COUNTS = {"10-story": 10, "18-story": 18, "24-story": 24, "31-story": 31}


def load_all(base_dir):

    files = {
        "10-story": "nist_10story_validation_v1_raw_results.json",
        "18-story": "nist_18story_validation_v1_raw_results.json",
        "24-story": "nist_24story_validation_v1_raw_results.json",
        "31-story": "nist_31story_validation_v1_raw_results.json",
    }
    data = {}
    for label, filename in files.items():
        with open(os.path.join(base_dir, filename), encoding="utf-8") as f:
            data[label] = json.load(f)
    return data


def chart_overprediction_ratio(data):

    labels = list(PUBLISHED_S.keys())
    ratios = [data[label]["official_summary"]["evacuation_time_mean"] / PUBLISHED_S[label] for label in labels]
    floor_counts = [FLOOR_COUNTS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, ratios, color="#DD8452", alpha=0.85)
    for bar, ratio, floors in zip(bars, ratios, floor_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, ratio + 0.3, f"{ratio:.1f}x\n({floors} floors)", ha="center", fontweight="bold", fontsize=9)

    ax.set_ylabel("SynEvac / Published evacuation time (overprediction ratio)")
    ax.set_title("Overprediction Ratio Across All Four NIST Buildings\n(SynEvac defaults vs. published drill time)")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect agreement")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "overprediction_ratio_all_buildings.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def chart_queue_wait_percentage(data):

    # For each building, the percentage of the worst-traced occupant's
    # own total time spent waiting, split by stair-topology type
    # (direct-exit vs lobby/merge-transition), read directly from each
    # building's own saved per-occupant trace.

    trace_keys = {
        "10-story": [("direct", "stair1_trace")] if "stair1_trace" in data["10-story"] else [],
        "18-story": [("direct", "stair1_trace"), ("merge", "stair7_trace")],
        "24-story": [("merge", "stairA_trace"), ("direct", "stairB_trace")],
        "31-story": [("direct-ish", "north_trace"), ("direct", "south_trace")],
    }

    labels, percentages, colors = [], [], []
    color_map = {"direct": "#55A868", "merge": "#C44E52", "direct-ish": "#8172B2"}

    for building, entries in trace_keys.items():
        for kind, key in entries:
            if key not in data[building]:
                continue
            trace = data[building][key]
            pct = 100.0 * trace["total_queue_wait"] / trace["final_arrival_time"]
            labels.append(f"{building}\n{key.replace('_trace', '')}")
            percentages.append(pct)
            colors.append(color_map[kind])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, percentages, color=colors)
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 1, f"{pct:.1f}%", ha="center", fontweight="bold", fontsize=9)

    ax.set_ylabel("% of occupant's total evacuation time spent queue-waiting")
    ax.set_title("Percentage of Evacuation Time Spent Waiting, Every Traced Stair, All Four Buildings\n(green=direct-exit topology, red=lobby/door-merge topology)")
    ax.set_ylim(0, 105)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "queue_wait_percentage_all_buildings.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def chart_evacuation_time_by_building(data):

    labels = list(PUBLISHED_S.keys())
    published = [PUBLISHED_S[l] for l in labels]
    synevac = [data[l]["official_summary"]["evacuation_time_mean"] for l in labels]
    synevac_lower = [synevac[i] - data[labels[i]]["official_summary"]["evacuation_time_ci_lower"] for i in range(len(labels))]
    synevac_upper = [data[labels[i]]["official_summary"]["evacuation_time_ci_upper"] - synevac[i] for i in range(len(labels))]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar([i - width / 2 for i in x], published, width, label="Published (real drill)", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], synevac, width, yerr=[synevac_lower, synevac_upper], capsize=4, label="SynEvac (defaults)", color="#DD8452")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Total evacuation time (seconds, log scale)")
    ax.set_title("All Four NIST Buildings: Published vs. SynEvac Evacuation Time")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "evacuation_time_all_four_buildings.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":

    base = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    data = load_all(base)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(chart_overprediction_ratio(data))
    print(chart_queue_wait_percentage(data))
    print(chart_evacuation_time_by_building(data))
