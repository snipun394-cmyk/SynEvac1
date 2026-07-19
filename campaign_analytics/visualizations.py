import os

import matplotlib

# Agg is the non-interactive, headless raster backend -- must be
# selected before pyplot is ever imported. Without this, matplotlib
# picks whatever GUI backend is importable (this project depends on
# PyQt6, so it would otherwise pick a Qt backend), which requires a
# running event loop/display and is not deterministic or safe to call
# from a test suite or a background campaign-analysis run.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use())

from typing import Any, Dict, Optional


# =====================================================
# Phase 6 -- Visualizations. Every chart here is drawn only from
# numbers CampaignAnalysis (analyzer.analyze_campaign()) already
# computed -- no artifact is re-read, no simulation is touched.
# =====================================================

CHART_DPI = 100
CHART_FACECOLOR = "#3b82f6"
DEFAULT_REPORTS_DIRNAME = "reports"

EVACUATION_TIME_HISTOGRAM_FILENAME = "evacuation_time_histogram.png"
FIRE_ORIGIN_DISTRIBUTION_FILENAME = "fire_origin_distribution.png"
BOTTLENECK_FREQUENCY_FILENAME = "bottleneck_frequency.png"
EXIT_USAGE_FILENAME = "exit_usage.png"
STAIR_USAGE_FILENAME = "stair_usage.png"
RECOMMENDATION_FREQUENCY_FILENAME = "recommendation_frequency.png"
DETECTOR_FAILURES_FILENAME = "detector_failures.png"
CAMERA_FAILURES_FILENAME = "camera_failures.png"


def _bar_chart(
    data: Dict[Any, float], *, title: str, xlabel: str, ylabel: str, file_path: str,
) -> str:

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    labels = [str(key) for key in data.keys()]
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=CHART_DPI)

    if labels:
        ax.bar(range(len(labels)), values, color=CHART_FACECOLOR)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)

    return file_path


def _histogram_chart(
    histogram: Dict[float, int], *, title: str, xlabel: str, ylabel: str, file_path: str,
) -> str:

    ordered_keys = sorted(histogram.keys())
    data = {str(key): histogram[key] for key in ordered_keys}

    return _bar_chart(data, title=title, xlabel=xlabel, ylabel=ylabel, file_path=file_path)


# =====================================================
# Individual charts -- each takes the specific already-computed slice
# of CampaignAnalysis it needs, so it can also be called/tested in
# isolation without building a full CampaignAnalysis.
# =====================================================


def save_evacuation_time_histogram(distributions: Dict[str, Any], file_path: str) -> str:

    return _histogram_chart(
        distributions["evacuation_time"]["histogram"],
        title="Evacuation Time Histogram",
        xlabel="Evacuation time (s, bucketed)",
        ylabel="Scenario count",
        file_path=file_path,
    )


def save_fire_origin_distribution(overview, file_path: str) -> str:

    return _bar_chart(
        overview.ignition_zone_frequencies,
        title="Fire Origin Distribution",
        xlabel="Ignition zone",
        ylabel="Scenario count",
        file_path=file_path,
    )


def save_bottleneck_frequency_chart(distributions: Dict[str, Any], file_path: str) -> str:

    return _bar_chart(
        distributions["bottleneck_locations"],
        title="Bottleneck Frequency by Door",
        xlabel="Door",
        ylabel="Bottleneck occurrences",
        file_path=file_path,
    )


def save_exit_usage_chart(distributions: Dict[str, Any], file_path: str) -> str:

    return _bar_chart(
        distributions["exit_utilization"],
        title="Exit Usage (most-congested-exit frequency)",
        xlabel="Exit",
        ylabel="Scenario count",
        file_path=file_path,
    )


def save_stair_usage_chart(distributions: Dict[str, Any], file_path: str) -> str:

    return _bar_chart(
        distributions["stair_utilization"],
        title="Stair Usage (most-congested-stair frequency)",
        xlabel="Stair",
        ylabel="Scenario count",
        file_path=file_path,
    )


def save_recommendation_frequency_chart(distributions: Dict[str, Any], file_path: str) -> str:

    return _bar_chart(
        distributions["recommendation_frequencies"],
        title="Ground Truth Recommendation Frequency",
        xlabel="Recommendation target type",
        ylabel="Recommendation count",
        file_path=file_path,
    )


def save_detector_failures_chart(overview, file_path: str) -> str:

    return _bar_chart(
        overview.detector_failure_frequencies,
        title="Detector State Frequency",
        xlabel="State",
        ylabel="Count",
        file_path=file_path,
    )


def save_camera_failures_chart(overview, file_path: str) -> str:

    return _bar_chart(
        overview.camera_failure_frequencies,
        title="Camera State Frequency",
        xlabel="State",
        ylabel="Count",
        file_path=file_path,
    )


def generate_all_charts(analysis, output_dir: Optional[str] = None) -> Dict[str, str]:

    output_dir = output_dir or DEFAULT_REPORTS_DIRNAME
    os.makedirs(output_dir, exist_ok=True)

    return {
        "evacuation_time_histogram": save_evacuation_time_histogram(
            analysis.distributions, os.path.join(output_dir, EVACUATION_TIME_HISTOGRAM_FILENAME),
        ),
        "fire_origin_distribution": save_fire_origin_distribution(
            analysis.overview, os.path.join(output_dir, FIRE_ORIGIN_DISTRIBUTION_FILENAME),
        ),
        "bottleneck_frequency": save_bottleneck_frequency_chart(
            analysis.distributions, os.path.join(output_dir, BOTTLENECK_FREQUENCY_FILENAME),
        ),
        "exit_usage": save_exit_usage_chart(
            analysis.distributions, os.path.join(output_dir, EXIT_USAGE_FILENAME),
        ),
        "stair_usage": save_stair_usage_chart(
            analysis.distributions, os.path.join(output_dir, STAIR_USAGE_FILENAME),
        ),
        "recommendation_frequency": save_recommendation_frequency_chart(
            analysis.distributions, os.path.join(output_dir, RECOMMENDATION_FREQUENCY_FILENAME),
        ),
        "detector_failures": save_detector_failures_chart(
            analysis.overview, os.path.join(output_dir, DETECTOR_FAILURES_FILENAME),
        ),
        "camera_failures": save_camera_failures_chart(
            analysis.overview, os.path.join(output_dir, CAMERA_FAILURES_FILENAME),
        ),
    }
