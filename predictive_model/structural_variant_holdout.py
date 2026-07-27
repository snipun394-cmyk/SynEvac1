from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import pandas as pd


# =====================================================
# Predictive Dataset V3 milestone, Phase 18A -- leave-one-STRUCTURAL-
# VARIANT-out evaluation. Exactly the same partition discipline as
# predictive_model.topology_holdout's leave-one-topology-FAMILY-out
# split (train on scenarios from every other group, test on the held-
# out group, grouped by scenario_metadata's own key, never a row) --
# mirrored here at the finer structural_variant_id grain rather than
# extending topology_holdout.py itself, since the two questions are
# genuinely different ("can the model generalize to an unseen BUILDING
# FAMILY" vs "can it generalize to an unseen STRUCTURAL SHAPE within a
# family it has otherwise seen").
# =====================================================


@dataclass(frozen=True)
class StructuralVariantHoldoutSplit:

    held_out_variant: str
    held_out_family: str
    train_scenario_ids: Tuple[str, ...]
    test_scenario_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "held_out_variant": self.held_out_variant,
            "held_out_family": self.held_out_family,
            "train_scenario_count": len(self.train_scenario_ids),
            "test_scenario_count": len(self.test_scenario_ids),
        }


def structural_variants(scenario_metadata: Sequence[Dict[str, Any]]) -> Tuple[str, ...]:

    return tuple(sorted({entry["structural_variant_id"] for entry in scenario_metadata}))


def build_structural_variant_holdout_splits(
    scenario_metadata: Sequence[Dict[str, Any]],
) -> Tuple[StructuralVariantHoldoutSplit, ...]:
    """One split per distinct structural_variant_id: that variant's
    scenarios become the test set, every OTHER variant's scenarios
    (including other variants of the SAME family) become the train set
    -- so the model has seen the family's base shape but never this
    exact structural graph."""

    by_variant: Dict[str, list] = {}
    family_by_variant: Dict[str, str] = {}
    for entry in scenario_metadata:
        by_variant.setdefault(entry["structural_variant_id"], []).append(entry["scenario_id"])
        family_by_variant[entry["structural_variant_id"]] = entry["topology_family"]

    variants = sorted(by_variant.keys())
    splits = []

    for held_out in variants:

        test_ids = tuple(sorted(by_variant[held_out]))
        train_ids = tuple(sorted(
            scenario_id
            for variant in variants if variant != held_out
            for scenario_id in by_variant[variant]
        ))

        splits.append(StructuralVariantHoldoutSplit(
            held_out_variant=held_out, held_out_family=family_by_variant[held_out],
            train_scenario_ids=train_ids, test_scenario_ids=test_ids,
        ))

    return tuple(splits)


def apply_structural_variant_holdout(
    frame: pd.DataFrame, split: StructuralVariantHoldoutSplit,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_ids = set(split.train_scenario_ids)
    test_ids = set(split.test_scenario_ids)

    train_df = frame[frame["scenario_id"].isin(train_ids)]
    test_df = frame[frame["scenario_id"].isin(test_ids)]

    return train_df, test_df


def assert_no_variant_holdout_overlap(
    split: StructuralVariantHoldoutSplit, train_df: pd.DataFrame, test_df: pd.DataFrame,
) -> None:

    train_id_set = set(split.train_scenario_ids)
    test_id_set = set(split.test_scenario_ids)

    id_overlap = train_id_set & test_id_set
    if id_overlap:
        raise AssertionError(
            f"Structural-variant holdout split for {split.held_out_variant!r} has overlapping scenario ids: {id_overlap}"
        )

    row_train_ids = set(train_df["scenario_id"].unique())
    row_test_ids = set(test_df["scenario_id"].unique())

    row_overlap = row_train_ids & row_test_ids
    if row_overlap:
        raise AssertionError(
            f"Rows from the same scenario appear in both train and held-out test rows "
            f"for variant {split.held_out_variant!r}: {row_overlap}"
        )
