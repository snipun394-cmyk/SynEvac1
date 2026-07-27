from typing import Optional

import numpy as np


# =====================================================
# Localized Predictive Model V3.1 milestone -- shared, tested
# implementations of the exact-duplicate-feature-vector grouping and
# within-group label shuffling used throughout this milestone's
# investigation scripts (scripts/model_v3_1_*.py). Previously each
# script reimplemented an identical byte-void-view trick ad-hoc; this
# module consolidates it into one tested, reusable place per the
# milestone's own explicit "add tests for duplicate-group calculations
# / shuffled-label controls" requirement.
# =====================================================


def feature_vector_group_ids(X: np.ndarray) -> np.ndarray:
    """Assigns every row of X an integer group id such that two rows
    get the SAME id if and only if they are byte-identical (exact
    duplicate feature vectors). Deterministic given the same X --
    group id ASSIGNMENT (which integer a given distinct vector gets) is
    stable only in the sense that np.unique's own lexicographic-sort
    order determines it, but which ROWS share a group is always
    correct and reproducible."""

    if X.ndim != 2:
        raise ValueError(f"expected a 2D feature matrix, got shape {X.shape}")

    contiguous = np.ascontiguousarray(X)
    view = contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1])))
    _, inverse = np.unique(view, return_inverse=True)
    return inverse.reshape(-1)


def shuffle_within_groups(y: np.ndarray, group_keys: np.ndarray, seed: int) -> np.ndarray:
    """Shuffles y independently within each distinct value of
    group_keys -- cross-group structure (how many positives each group
    has) is preserved EXACTLY; only the within-group row assignment is
    randomized. group_keys may be any hashable/sortable dtype (string
    scenario ids, integer duplicate-group ids, etc.) -- boundaries are
    found via inequality comparison (not np.diff), so this works for
    string arrays too, unlike a naive np.diff-based approach."""

    if len(y) != len(group_keys):
        raise ValueError(f"y and group_keys must be the same length ({len(y)} vs {len(group_keys)})")

    rng = np.random.default_rng(seed)
    y2 = y.copy()
    order = np.argsort(group_keys, kind="stable")
    sorted_keys = group_keys[order]

    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(sorted_keys)]))

    y_sorted = y2[order]
    for start, end in zip(starts, ends):
        segment = y_sorted[start:end].copy()
        rng.shuffle(segment)
        y_sorted[start:end] = segment

    result = np.empty_like(y2)
    result[order] = y_sorted
    return result
