from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import pandas as pd


# =====================================================
# Localized Predictive Model V2 milestone, Phase 4 -- leave-one-
# topology-family-out evaluation. A normal scenario-level split (Phase 3,
# predictive_model.scenario_split) proves rows from one scenario never
# leak across splits, but every split still draws from the SAME four
# topology families -- it cannot tell us whether the model learned
# transferable congestion relationships or memorized topology-specific
# patterns. This module builds a DIFFERENT kind of split: train on every
# scenario from three families, test on every scenario from the fourth,
# held-out family. Grouping key is `topology_family` (from
# predictive_dataset_campaign_v2's scenario_metadata.json), never a row.
# =====================================================


@dataclass(frozen=True)
class TopologyHoldoutSplit:

    held_out_family: str
    train_scenario_ids: Tuple[str, ...]
    test_scenario_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "held_out_family": self.held_out_family,
            "train_scenario_count": len(self.train_scenario_ids),
            "test_scenario_count": len(self.test_scenario_ids),
        }


def topology_families(scenario_metadata: Sequence[Dict[str, Any]]) -> Tuple[str, ...]:
    """Distinct topology_family values present in scenario_metadata, sorted
    for deterministic iteration order."""

    return tuple(sorted({entry["topology_family"] for entry in scenario_metadata}))


def build_topology_holdout_splits(scenario_metadata: Sequence[Dict[str, Any]]) -> Tuple[TopologyHoldoutSplit, ...]:
    """One TopologyHoldoutSplit per distinct topology_family: that
    family's scenarios become the test set, every other family's
    scenarios become the train set. No scenario ever appears in both --
    each scenario belongs to exactly one topology_family by construction
    (predictive_dataset.topologies_v2's own per-family scenario
    generation), so this partition is guaranteed disjoint, but
    assert_no_holdout_overlap() below re-proves it mechanically rather
    than trusting that guarantee blindly."""

    by_family: Dict[str, list] = {}
    for entry in scenario_metadata:
        by_family.setdefault(entry["topology_family"], []).append(entry["scenario_id"])

    families = sorted(by_family.keys())
    splits = []

    for held_out in families:

        test_ids = tuple(sorted(by_family[held_out]))
        train_ids = tuple(sorted(
            scenario_id
            for family in families if family != held_out
            for scenario_id in by_family[family]
        ))

        splits.append(TopologyHoldoutSplit(
            held_out_family=held_out, train_scenario_ids=train_ids, test_scenario_ids=test_ids,
        ))

    return tuple(splits)


def apply_topology_holdout(frame: pd.DataFrame, split: TopologyHoldoutSplit) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_ids = set(split.train_scenario_ids)
    test_ids = set(split.test_scenario_ids)

    train_df = frame[frame["scenario_id"].isin(train_ids)]
    test_df = frame[frame["scenario_id"].isin(test_ids)]

    return train_df, test_df


def assert_no_holdout_overlap(split: TopologyHoldoutSplit, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Mechanical proof: no scenario id is in both the train and test id
    sets, AND no scenario id present in one split's actual rows is also
    present in the other's."""

    train_id_set = set(split.train_scenario_ids)
    test_id_set = set(split.test_scenario_ids)

    id_overlap = train_id_set & test_id_set
    if id_overlap:
        raise AssertionError(
            f"Topology holdout split for {split.held_out_family!r} has overlapping scenario ids: {id_overlap}"
        )

    row_train_ids = set(train_df["scenario_id"].unique())
    row_test_ids = set(test_df["scenario_id"].unique())

    row_overlap = row_train_ids & row_test_ids
    if row_overlap:
        raise AssertionError(
            f"Rows from the same scenario appear in both train and held-out test rows "
            f"for {split.held_out_family!r}: {row_overlap}"
        )
