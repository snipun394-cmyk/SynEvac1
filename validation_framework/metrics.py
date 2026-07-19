from typing import Any, Dict, Sequence

from ai_training.metrics import (
    classification_metrics,
    multioutput_regression_metrics,
    regression_metrics,
)
from rl_training.evaluator import EpisodeMetrics


# =====================================================
# Phase 2 -- AI accuracy metrics are entirely ai_training.metrics'
# own MAE/RMSE/R2/Accuracy/Precision/Recall/F1/ROC-AUC (re-exported
# below, never recomputed). The one genuine gap is Phase 3's RL-arm
# metrics (Evacuation Time, Occupants Saved, Congestion Reduction,
# Smoke Exposure, Travel Distance): rl_training.evaluator.EvaluationReport
# only exposes a few pre-aggregated averages, not the full named set,
# so rl_policy_metrics() re-aggregates from the underlying per-episode
# EpisodeMetrics tuple. Smoke Exposure/Travel Distance are honestly
# None -- EpisodeMetrics carries no such field, and inventing one here
# would mean guessing at simulation internals this package must not
# touch.
# =====================================================

__all__ = [
    "classification_metrics",
    "multioutput_regression_metrics",
    "regression_metrics",
    "rl_policy_metrics",
]


def rl_policy_metrics(episodes: Sequence[EpisodeMetrics]) -> Dict[str, Any]:

    episodes = list(episodes)

    if not episodes:

        return {
            "episode_count": 0,
            "reward_mean": None,
            "evacuation_time_mean": None,
            "occupants_saved_mean": None,
            "people_trapped_mean": None,
            "peak_congestion_mean": None,
            "smoke_exposure_mean": None,
            "travel_distance_mean": None,
        }

    evacuation_times = [e.total_evacuation_time for e in episodes if e.total_evacuation_time is not None]
    congestion_values = [e.peak_congestion_value for e in episodes if e.peak_congestion_value is not None]

    return {
        "episode_count": len(episodes),
        "reward_mean": sum(e.total_reward for e in episodes) / len(episodes),
        "evacuation_time_mean": (sum(evacuation_times) / len(evacuation_times)) if evacuation_times else None,
        "occupants_saved_mean": sum(e.people_evacuated for e in episodes) / len(episodes),
        "people_trapped_mean": sum(e.people_trapped for e in episodes) / len(episodes),
        "peak_congestion_mean": (sum(congestion_values) / len(congestion_values)) if congestion_values else None,
        # Not exposed by rl_training.evaluator.EpisodeMetrics -- honestly
        # absent rather than fabricated (see module docstring above).
        "smoke_exposure_mean": None,
        "travel_distance_mean": None,
    }
