from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


# =====================================================
# First Localized Predictive Congestion Model milestone, Phase 11 --
# scientific sanity checks. Three checks, each answering a different
# "is this result real" question:
#   1. feature-family ablation: removing a real signal family should
#      hurt performance by a REASONABLE amount (not zero -- that would
#      mean the family is dead weight; not catastrophic either, unless
#      it really is the dominant signal).
#   2. leakage correlation re-check: no single feature should be
#      suspiciously close to a 1:1 relationship with the target (reuses
#      predictive_dataset.correlation's own 0.9 review threshold).
#   3. label-shuffle test: a model trained on RANDOMLY SHUFFLED labels
#      must perform at chance (ROC-AUC ~ 0.5) on real labels -- if it
#      doesn't, something is leaking through a channel that survives
#      label shuffling (e.g. row ordering, a leaked identifier).
# =====================================================


LEAKAGE_REVIEW_THRESHOLD = 0.9  # predictive_dataset.correlation.LEAKAGE_REVIEW_THRESHOLD, reused verbatim


def feature_family_ablation_report(
    train_fn: Callable[[np.ndarray, np.ndarray], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Sequence[str],
    families: Dict[str, List[str]],
) -> Dict[str, Any]:
    """train_fn(X, y) -> a fitted model exposing predict_proba(X). Each
    family's columns are ZEROED OUT (not dropped -- keeps the matrix
    shape/model architecture identical to the full-feature run, isolating
    the effect of removing that family's information) in both train and
    val, a fresh model is retrained, and the ROC-AUC/PR-AUC drop vs. the
    full-feature model is reported."""

    from sklearn.metrics import average_precision_score

    name_to_index = {name: index for index, name in enumerate(feature_names)}

    full_model = train_fn(X_train, y_train)
    full_val_prob = full_model.predict_proba(X_val)
    full_roc_auc = float(roc_auc_score(y_val, full_val_prob))
    full_pr_auc = float(average_precision_score(y_val, full_val_prob))

    family_results = {}

    for family_name, family_features in families.items():

        indices = [name_to_index[name] for name in family_features if name in name_to_index]
        if not indices:
            continue

        X_train_ablated = X_train.copy()
        X_val_ablated = X_val.copy()
        X_train_ablated[:, indices] = 0.0
        X_val_ablated[:, indices] = 0.0

        ablated_model = train_fn(X_train_ablated, y_train)
        ablated_val_prob = ablated_model.predict_proba(X_val_ablated)

        ablated_roc_auc = float(roc_auc_score(y_val, ablated_val_prob))
        ablated_pr_auc = float(average_precision_score(y_val, ablated_val_prob))

        family_results[family_name] = {
            "features_removed": family_features,
            "roc_auc_with_family": full_roc_auc,
            "roc_auc_without_family": ablated_roc_auc,
            "roc_auc_drop": full_roc_auc - ablated_roc_auc,
            "pr_auc_with_family": full_pr_auc,
            "pr_auc_without_family": ablated_pr_auc,
            "pr_auc_drop": full_pr_auc - ablated_pr_auc,
        }

    return {
        "full_feature_roc_auc": full_roc_auc,
        "full_feature_pr_auc": full_pr_auc,
        "by_family": family_results,
    }


def leakage_correlation_recheck(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Pearson's r between every model-input column (after this package's
    own encoding -- see feature_prep.py) and the binary target, flagging
    anything |r| >= 0.9 for manual leakage review -- the SAME threshold
    predictive_dataset.correlation.LEAKAGE_REVIEW_THRESHOLD already
    applied to the raw feature schema. This re-check runs it again
    post-encoding, over whichever split is passed in (train, by
    convention), since one-hot/imputation columns are new artifacts of
    this package, not the raw schema correlation already checked."""

    y_float = y.astype(float)
    correlations = {}

    y_std = y_float.std()

    for index, name in enumerate(feature_names):

        column = X[:, index].astype(float)
        col_std = column.std()

        if col_std == 0.0 or y_std == 0.0:
            correlations[name] = None
            continue

        r = float(np.corrcoef(column, y_float)[0, 1])
        correlations[name] = r

    flagged = sorted(
        name for name, r in correlations.items() if r is not None and abs(r) >= LEAKAGE_REVIEW_THRESHOLD
    )

    return {
        "correlation_with_target": correlations,
        "leakage_review_threshold": LEAKAGE_REVIEW_THRESHOLD,
        "flagged_for_leakage_review": flagged,
    }


def label_shuffle_test(
    train_fn: Callable[[np.ndarray, np.ndarray], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 20260726,
) -> Dict[str, Any]:
    """Train the SAME model type on train features paired with RANDOMLY
    SHUFFLED train labels, then evaluate against the REAL val labels.
    Expected result: ROC-AUC collapses to ~0.5 (chance) -- if it doesn't,
    the model is finding signal through some channel that survives
    breaking the true X/y correspondence (e.g. a leaked row-order
    artifact), which would be a serious, disqualifying finding."""

    rng = np.random.default_rng(seed)
    y_shuffled = y_train.copy()
    rng.shuffle(y_shuffled)

    shuffled_model = train_fn(X_train, y_shuffled)
    shuffled_val_prob = shuffled_model.predict_proba(X_val)

    shuffled_roc_auc = float(roc_auc_score(y_val, shuffled_val_prob))

    return {
        "seed": seed,
        "shuffled_label_roc_auc_on_real_val_labels": shuffled_roc_auc,
        "near_chance": abs(shuffled_roc_auc - 0.5) < 0.05,
    }
