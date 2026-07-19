import json

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


# =====================================================
# Manifest (Phase 5) -- the one JSON document describing an exported
# dataset well enough that a future ML pipeline can load it without
# guessing: what campaign it came from, how big it is, how it was
# split, and which columns are features vs. labels.
# =====================================================

SCHEMA_VERSION = "1.0"

# scenario_id/seed identify a row, definition_id names the generating
# ScenarioDefinition -- none of the three is itself a trainable
# feature.
_NON_FEATURE_COLUMNS = ("scenario_id", "definition_id", "seed")

# The engineering outcome columns dataset_builder.schema.
# SIMULATION_OUTCOME_COLUMNS already fixes -- read here only as the
# label list for the manifest, never imported, so this package stays
# fully decoupled from dataset_builder's own schema module.
_DEFAULT_LABEL_COLUMNS = (
    "total_evacuation_time",
    "average_evacuation_time",
    "last_occupant_exit_time",
    "people_evacuated",
    "people_trapped",
    "reachable_occupants",
    "unreachable_occupants",
    "building_cleared",
)


def build_manifest(
    dataset: Iterable,
    split,
    *,
    campaign_name: Optional[str] = None,
    master_seed: Optional[int] = None,
    feature_list: Optional[List[str]] = None,
    label_list: Optional[List[str]] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:

    samples = list(dataset)

    if feature_list is None:
        feature_list = _infer_feature_list(samples)

    if label_list is None:
        label_list = _infer_label_list(samples)

    campaign_dir = getattr(dataset, "campaign_dir", None)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "campaign": {
            "name": campaign_name,
            "directory": campaign_dir,
            "master_seed": master_seed,
        },
        "num_samples": len(samples),
        "split_sizes": (
            {
                "train": len(split.train_ids),
                "validation": len(split.validation_ids),
                "test": len(split.test_ids),
            }
            if split is not None
            else None
        ),
        "feature_list": feature_list,
        "label_list": label_list,
    }


def _infer_feature_list(samples) -> List[str]:

    columns: List[str] = []
    seen = set()

    for sample in samples:

        for key in sample.scenario_features:

            if key in _NON_FEATURE_COLUMNS or key in seen:
                continue

            seen.add(key)
            columns.append(key)

    return columns


def _infer_label_list(samples) -> List[str]:

    if not samples:
        return list(_DEFAULT_LABEL_COLUMNS)

    columns: List[str] = []
    seen = set()

    for sample in samples:

        for key in sample.simulation_outcome:

            if key == "scenario_id" or key in seen:
                continue

            seen.add(key)
            columns.append(key)

    return columns


def write_manifest(manifest: Dict[str, Any], file_path: str) -> None:

    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
