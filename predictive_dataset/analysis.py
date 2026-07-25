from collections import defaultdict
from typing import Any, Dict, List, Sequence


# =====================================================
# Per-Candidate Predictive AI Data Foundation milestone, Phase 15/17 --
# offline reporting over an already-built candidate dataset (a list of
# row dicts from predictive_dataset.dataset_builder). Pure aggregation,
# no simulation, no model training -- consumed by
# scripts/generate_predictive_dataset_campaign.py and by tests.
# =====================================================


def class_balance_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    trainable = [row for row in rows if row["target"] is not None]
    excluded_currently_congested = [row for row in rows if row["target"] is None]

    positive = sum(1 for row in trainable if row["target"] is True)
    negative = sum(1 for row in trainable if row["target"] is False)

    by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
    by_horizon: Dict[float, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
    by_observation_time_bucket: Dict[str, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})

    for row in trainable:

        key = "positive" if row["target"] is True else "negative"

        by_type[row["candidate_type"]][key] += 1
        by_horizon[row["prediction_horizon"]][key] += 1
        by_observation_time_bucket[_time_bucket(row["observation_time"])][key] += 1

    scenario_ids = {row["scenario_id"] for row in rows}

    return {
        "total_rows": len(rows),
        "trainable_rows": len(trainable),
        "excluded_currently_congested_rows": len(excluded_currently_congested),
        "positive_count": positive,
        "negative_count": negative,
        "positive_rate": (positive / len(trainable)) if trainable else None,
        "distinct_scenario_count": len(scenario_ids),
        "by_candidate_type": dict(by_type),
        "by_horizon": dict(by_horizon),
        "by_observation_time_bucket": dict(by_observation_time_bucket),
    }


def _time_bucket(observation_time: float) -> str:

    bucket_start = int(observation_time // 60) * 60

    return f"{bucket_start}-{bucket_start + 60}s"


# =====================================================


def horizon_analysis(rows: Sequence[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:

    horizons = sorted({row["prediction_horizon"] for row in rows})

    report = {}

    for horizon in horizons:

        horizon_rows = [row for row in rows if row["prediction_horizon"] == horizon]
        trainable = [row for row in horizon_rows if row["target"] is not None]
        already_congested = [row for row in horizon_rows if row["target"] is None]

        positive = sum(1 for row in trainable if row["target"] is True)

        type_distribution: Dict[str, int] = defaultdict(int)
        for row in horizon_rows:
            type_distribution[row["candidate_type"]] += 1

        report[horizon] = {
            "total_rows": len(horizon_rows),
            "trainable_rows": len(trainable),
            "positive_count": positive,
            "positive_rate": (positive / len(trainable)) if trainable else None,
            "already_congested_at_observation_rows": len(already_congested),
            "already_congested_fraction": (
                len(already_congested) / len(horizon_rows) if horizon_rows else None
            ),
            "candidate_type_distribution": dict(type_distribution),
        }

    return report


def recommend_first_horizon(horizon_report: Dict[float, Dict[str, Any]]) -> float:

    # Phase 17's own explicit rule: do not select a horizon merely
    # because a prior document proposed one -- but "most statistically
    # usable" alone is not sufficient either (a very short horizon is
    # trivially "usable" precisely because it is closest to already-
    # observed current state, which docs/architecture/ai_operational_
    # role.md §10 already identifies as adding no predictive value: a
    # live deployment's Crowd Intelligence/Evacuation Progress packages
    # already report CURRENT congestion/queue/throughput instantly and
    # deterministically). Two independent, disclosed floors must BOTH
    # clear, not just the statistical one:
    #
    #   1. GENUINE ADVANCE WARNING -- horizon >= MIN_ADVANCE_WARNING_
    #      SECONDS (20s, citing the same §10 reasoning: LiveOrchestrator's
    #      ~1Hz cycle means anything shorter is barely distinguishable
    #      from "report current state", not a genuinely predictive claim).
    #   2. STATISTICAL USABILITY -- at least MIN_POSITIVE_ROWS positive
    #      examples AND at least MIN_POSITIVE_RATE positive rate among
    #      trainable rows (non-fabricated floors: low enough to accept
    #      real class imbalance, high enough that a horizon with near-
    #      zero positives is not silently preferred just for having more
    #      raw rows).
    #
    # Among horizons clearing both, the SHORTEST wins (more actionable,
    # more learnable) -- this is empirical selection from the actual
    # campaign numbers, not a re-assertion of the prior document's guess.

    MIN_ADVANCE_WARNING_SECONDS = 20.0
    MIN_POSITIVE_ROWS = 20
    MIN_POSITIVE_RATE = 0.02

    usable = [
        horizon for horizon, stats in sorted(horizon_report.items())
        if horizon >= MIN_ADVANCE_WARNING_SECONDS
        and stats["positive_count"] >= MIN_POSITIVE_ROWS
        and (stats["positive_rate"] or 0.0) >= MIN_POSITIVE_RATE
    ]

    if usable:
        return usable[0]

    # No horizon cleared both bars -- fall back to the one with the most
    # positive examples in absolute terms, an honest "least bad" choice,
    # never silently defaulting to a fixed number.
    return max(horizon_report, key=lambda horizon: horizon_report[horizon]["positive_count"])
