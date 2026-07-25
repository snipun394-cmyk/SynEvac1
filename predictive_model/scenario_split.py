import random
from dataclasses import dataclass
from typing import Sequence, Tuple

import pandas as pd


# =====================================================
# First Localized Predictive Congestion Model milestone, Phase 2 --
# scenario-level train/val/test splitting ONLY. Row-level random
# splitting would leak: rows from the same scenario share the same
# building, fire, and occupant draw, so a row-level split would let a
# model see near-duplicate context from the same scenario in both
# train and test, inflating every metric. Every split boundary here is
# a SET OF SCENARIO IDS, applied to the frame only via
# frame["scenario_id"].isin(...) -- never a row index.
# =====================================================


@dataclass(frozen=True)
class ScenarioSplit:

    train_scenario_ids: Tuple[str, ...]
    val_scenario_ids: Tuple[str, ...]
    test_scenario_ids: Tuple[str, ...]
    seed: int
    ratios: Tuple[float, float, float]

    def to_dict(self):
        return {
            "train_scenario_count": len(self.train_scenario_ids),
            "val_scenario_count": len(self.val_scenario_ids),
            "test_scenario_count": len(self.test_scenario_ids),
            "seed": self.seed,
            "ratios": list(self.ratios),
        }


def split_scenarios(
    scenario_ids: Sequence[str],
    *,
    seed: int,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> ScenarioSplit:

    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")

    # sorted() first so shuffling is deterministic regardless of the
    # order scenario_ids happened to be encountered in the source frame
    unique_ids = sorted(set(scenario_ids))

    rng = random.Random(seed)
    shuffled = list(unique_ids)
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = round(total * ratios[0])
    n_val = round(total * ratios[1])

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    return ScenarioSplit(
        train_scenario_ids=tuple(train_ids),
        val_scenario_ids=tuple(val_ids),
        test_scenario_ids=tuple(test_ids),
        seed=seed,
        ratios=tuple(ratios),
    )


def apply_split(frame: pd.DataFrame, split: ScenarioSplit) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_ids = set(split.train_scenario_ids)
    val_ids = set(split.val_scenario_ids)
    test_ids = set(split.test_scenario_ids)

    train_df = frame[frame["scenario_id"].isin(train_ids)]
    val_df = frame[frame["scenario_id"].isin(val_ids)]
    test_df = frame[frame["scenario_id"].isin(test_ids)]

    return train_df, val_df, test_df


def assert_no_scenario_overlap(
    split: ScenarioSplit,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Mechanical proof (not a spot check): no scenario_id present in any
    split's ID set is also present in another split's ID set, AND no
    scenario_id present in any split's actual output rows is also
    present in another split's actual output rows. Raises AssertionError
    naming the offending scenario ids if this is ever violated."""

    train_id_set = set(split.train_scenario_ids)
    val_id_set = set(split.val_scenario_ids)
    test_id_set = set(split.test_scenario_ids)

    id_overlaps = {
        "train_val": train_id_set & val_id_set,
        "train_test": train_id_set & test_id_set,
        "val_test": val_id_set & test_id_set,
    }
    offending = {name: ids for name, ids in id_overlaps.items() if ids}
    if offending:
        raise AssertionError(f"Scenario ID sets overlap between splits: {offending}")

    row_train_ids = set(train_df["scenario_id"].unique())
    row_val_ids = set(val_df["scenario_id"].unique())
    row_test_ids = set(test_df["scenario_id"].unique())

    row_overlaps = {
        "train_val": row_train_ids & row_val_ids,
        "train_test": row_train_ids & row_test_ids,
        "val_test": row_val_ids & row_test_ids,
    }
    offending_rows = {name: ids for name, ids in row_overlaps.items() if ids}
    if offending_rows:
        raise AssertionError(f"Rows from the same scenario appear in multiple splits: {offending_rows}")
