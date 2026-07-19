from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from sklearn.model_selection import GroupShuffleSplit, train_test_split


# =====================================================
# Phase 1 -- deterministic train/validation/test splitting. Every
# split below is a pure function of (n_rows, groups, test_size,
# val_size, random_state): the same inputs always produce the same
# three index sets, since sklearn's own splitters are themselves
# deterministic for a fixed random_state.
# =====================================================


@dataclass(frozen=True)
class DatasetSplit:

    train_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]
    val_indices: Tuple[int, ...] = ()

    @property
    def has_validation(self) -> bool:

        return len(self.val_indices) > 0


# =====================================================


def make_split(
    n_rows: int,
    *,
    groups: Optional[Sequence[Any]] = None,
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 0,
) -> DatasetSplit:

    # groups (e.g. a scenario_id per row) keeps every row sharing one
    # group value entirely on one side of every split -- required for
    # per-timestep rows (SmokePredictionModel), where consecutive
    # timesteps of the same scenario are highly correlated and must
    # never straddle train/test/val.

    indices = np.arange(n_rows)
    groups_array = np.asarray(groups) if groups is not None else None

    train_val_idx, test_idx = _split_indices(
        indices, groups_array, test_size=test_size, random_state=random_state,
    )

    if val_size <= 0:

        return DatasetSplit(
            train_indices=tuple(sorted(train_val_idx.tolist())),
            test_indices=tuple(sorted(test_idx.tolist())),
        )

    relative_val_size = val_size / (1.0 - test_size)

    train_val_groups = groups_array[train_val_idx] if groups_array is not None else None

    train_idx, val_idx = _split_indices(
        train_val_idx, train_val_groups, test_size=relative_val_size, random_state=random_state,
    )

    return DatasetSplit(
        train_indices=tuple(sorted(train_idx.tolist())),
        test_indices=tuple(sorted(test_idx.tolist())),
        val_indices=tuple(sorted(val_idx.tolist())),
    )


# =====================================================


def _split_indices(
    indices: np.ndarray, groups: Optional[np.ndarray], *, test_size: float, random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:

    if groups is not None:

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_positions, test_positions = next(splitter.split(indices, groups=groups))

        return indices[train_positions], indices[test_positions]

    train_part, test_part = train_test_split(indices, test_size=test_size, random_state=random_state)

    return train_part, test_part


# =====================================================


def apply_split(
    rows: Sequence[Any], split: DatasetSplit,
) -> Tuple[List[Any], List[Any], List[Any]]:

    train_rows = [rows[index] for index in split.train_indices]
    val_rows = [rows[index] for index in split.val_indices]
    test_rows = [rows[index] for index in split.test_indices]

    return train_rows, val_rows, test_rows
