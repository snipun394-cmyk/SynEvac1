import gc
import random
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import pandas as pd

from predictive_model.feature_prep import PreparedFeatures, build_feature_matrix
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.metrics import compute_metrics


# =====================================================
# Localized Predictive Model V2 milestone, Phase 15 -- does the leading
# model actually need all ~9.6M/4 = ~2.4M rows (of which ~1.7M are
# train-split trainable rows at the 20s horizon)? Subsamples TRAIN
# SCENARIOS ONLY (never rows -- the same leakage-avoidance discipline
# predictive_model.scenario_split already established) at increasing
# fractions, retrains the same model architecture at each fraction, and
# evaluates every fraction against the SAME, FULL, UNCHANGED validation
# split -- isolating "how much train data is needed" from "how much
# eval data is used".
# =====================================================


def training_size_study(
    train_trainable: pd.DataFrame,
    val_feat: PreparedFeatures,
    model_factory: Callable[[], Any],
    *,
    fractions: Sequence[float] = (0.10, 0.25, 0.50, 1.00),
    seed: int = 20260726,
    feature_builder: Callable[[pd.DataFrame], PreparedFeatures] = build_feature_matrix,
    progress_fn: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """feature_builder defaults to the frozen V1/V2 9-field
    build_feature_matrix (V2's own call sites are completely unaffected)
    -- exposed so a caller using a different, still-schema-consistent
    feature matrix (e.g. predictive_model.feature_prep_v2_1.
    build_experimental_feature_matrix, whatever `val_feat` itself was
    already built with) can pass it through instead of silently
    building val/train feature matrices with two DIFFERENT schemas."""

    train_scenario_ids = sorted(train_trainable["scenario_id"].astype(str).unique())

    rng = random.Random(seed)
    shuffled = list(train_scenario_ids)
    rng.shuffle(shuffled)

    total_scenarios = len(shuffled)
    results = {}

    for fraction in fractions:

        n_scenarios = max(1, round(total_scenarios * fraction))
        subset_ids = set(shuffled[:n_scenarios])

        subset_frame = train_trainable[train_trainable["scenario_id"].astype(str).isin(subset_ids)]
        subset_feat = feature_builder(subset_frame)

        class_weight_map = compute_class_weight_map(subset_feat.y)
        sample_weight = sample_weights_from_class_weight(subset_feat.y, class_weight_map)

        model = model_factory()
        model.fit(subset_feat.X, subset_feat.y, sample_weight=sample_weight)

        val_prob = model.predict_proba(val_feat.X)
        val_metrics = compute_metrics(val_feat.y, val_prob, threshold=0.5)

        results[str(fraction)] = {
            "fraction": fraction,
            "train_scenario_count": n_scenarios,
            "train_scenario_count_available": total_scenarios,
            "train_row_count": int(len(subset_feat.y)),
            "train_positive_rate": float(subset_feat.y.mean()),
            "val_roc_auc": val_metrics.roc_auc,
            "val_pr_auc": val_metrics.pr_auc,
        }

        if progress_fn is not None:
            progress_fn(str(fraction), results[str(fraction)])

        del subset_frame, subset_feat, model, val_prob, sample_weight
        gc.collect()

    return results
