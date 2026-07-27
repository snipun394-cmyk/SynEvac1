import json
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import pandas as pd

from predictive_dataset.campaign_config import CAMPAIGN_VERSION
from predictive_dataset.dataset_builder import CSV_COLUMNS, DEFAULT_HORIZONS
from predictive_dataset.schema import CANDIDATE_FEATURE_NAMES, SCHEMA_VERSION
from predictive_dataset.versioning import FEATURE_VERSION, TARGET_VERSION


# =====================================================
# First Localized Predictive Congestion Model milestone, Phase 1 --
# a clean, versioned dataset loader. This is a NEW package
# (predictive_model/), deliberately separate from predictive_dataset/
# (which only builds and validates the raw data, never trains
# anything). This module's only job: read the campaign's own
# dataset_version manifest (predictive_dataset.versioning.dataset_
# version(), already embedded in campaign_v1_report.json's
# "dataset_version" key) and REFUSE to load a CSV whose version
# doesn't match what this training code was written against, rather
# than silently training on a schema it was never validated for.
# =====================================================


class IncompatibleDatasetVersionError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetRequirement:
    """What this training code expects. `None` on any field means "accept any"."""

    schema_version: Optional[str] = SCHEMA_VERSION
    campaign_version: Optional[str] = CAMPAIGN_VERSION
    feature_version: Optional[str] = FEATURE_VERSION
    target_version: Optional[str] = TARGET_VERSION


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    campaign_version: str
    feature_version: str
    target_version: str
    recommended_horizon_seconds: float
    report_path: str


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    manifest: DatasetManifest
    csv_path: str
    available_horizons: Tuple[float, ...]


def load_dataset_manifest(report_path: str) -> DatasetManifest:

    with open(report_path, "r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    if "dataset_version" not in report:
        raise IncompatibleDatasetVersionError(
            f"{report_path!r} has no 'dataset_version' key -- cannot verify compatibility."
        )

    dataset_version = report["dataset_version"]

    required_keys = (
        "schema_version", "campaign_version", "feature_version",
        "target_version", "prediction_horizon_seconds",
    )
    missing = [key for key in required_keys if key not in dataset_version]
    if missing:
        raise IncompatibleDatasetVersionError(
            f"{report_path!r}'s dataset_version is missing required keys: {missing}."
        )

    return DatasetManifest(
        schema_version=dataset_version["schema_version"],
        campaign_version=dataset_version["campaign_version"],
        feature_version=dataset_version["feature_version"],
        target_version=dataset_version["target_version"],
        recommended_horizon_seconds=dataset_version["prediction_horizon_seconds"],
        report_path=report_path,
    )


def assert_compatible(manifest: DatasetManifest, requirement: DatasetRequirement = DatasetRequirement()) -> None:

    mismatches = []

    for field_name in ("schema_version", "campaign_version", "feature_version", "target_version"):

        required_value = getattr(requirement, field_name)
        if required_value is None:
            continue

        actual_value = getattr(manifest, field_name)
        if actual_value != required_value:
            mismatches.append(f"{field_name}: required {required_value!r}, found {actual_value!r}")

    if mismatches:
        raise IncompatibleDatasetVersionError(
            f"Dataset at {manifest.report_path!r} is incompatible with this training code: "
            + "; ".join(mismatches)
        )


def load_dataset(
    csv_path: str,
    report_path: str,
    *,
    requirement: DatasetRequirement = DatasetRequirement(),
) -> LoadedDataset:
    """Load and validate a predictive_dataset campaign CSV.

    Rejects (raises IncompatibleDatasetVersionError) any dataset whose
    schema/campaign/feature/target version doesn't match `requirement`,
    or whose CSV columns don't match this schema version's expected
    columns -- this training code must never silently train against a
    dataset shape it wasn't written for.
    """

    manifest = load_dataset_manifest(report_path)
    assert_compatible(manifest, requirement)

    frame = pd.read_csv(csv_path)

    expected_columns = set(CSV_COLUMNS)
    actual_columns = set(frame.columns)
    if expected_columns != actual_columns:
        raise IncompatibleDatasetVersionError(
            f"{csv_path!r} columns do not match predictive_dataset.dataset_builder.CSV_COLUMNS "
            f"for schema_version {manifest.schema_version!r}. "
            f"Missing: {sorted(expected_columns - actual_columns)}, "
            f"Unexpected: {sorted(actual_columns - expected_columns)}"
        )

    available_horizons = tuple(sorted(frame["prediction_horizon"].unique().tolist()))

    return LoadedDataset(
        frame=frame,
        manifest=manifest,
        csv_path=csv_path,
        available_horizons=available_horizons,
    )


# =====================================================
# Localized Predictive Model V2 milestone, Phase 2 -- memory-efficient
# loading for campaigns too large to hold every horizon in memory at
# once (V2: 9,620,196 rows across 4 horizons; this development machine
# has ~7.3GB total RAM). load_dataset() above is UNCHANGED and remains
# exactly what V1 used (a single pd.read_csv of the whole file) -- V1's
# own dataset (2,508,480 rows) never needed this. This function is a
# separate, additive entry point: read the CSV in chunks, keep ONLY
# rows matching the one horizon this training run cares about, and cast
# numeric columns to compact dtypes (float32/int32) as each chunk
# arrives -- peak memory is one raw chunk plus the accumulating
# ~1/len(horizons)-sized filtered frame, never the full multi-horizon
# CSV. Callers must NOT call select_horizon() again on the result (the
# returned frame already IS the single-horizon slice).
# =====================================================

_COMPACT_DTYPES: Dict[str, str] = {
    "observation_time": "float32",
    "prediction_horizon": "float32",
    "total_active_occupant_count": "int32",
    "candidate_capacity": "float32",
    "candidate_walking_distance": "float32",
    "candidate_traversable": "bool",
    "candidate_adjacent_zone_occupancy": "float32",
    "candidate_queue_length": "float32",
    "candidate_approaching_count": "float32",
    "currently_congested": "bool",
    "had_any_activity_in_window": "bool",
    # scenario_id/candidate_id/candidate_type/candidate_congestion_level/
    # target are deliberately NOT dtype-forced at chunk-read time:
    # target has real per-row missingness (~3.2% of rows, see
    # docs/architecture/predictive_dataset_campaign_v2.md Section 4) and
    # pandas infers it as an object column of True/False/NaN -- forcing
    # a dtype here risks a different inferred type per chunk (some
    # chunks may contain zero NaNs) which would break pd.concat. They
    # are categorized/typed AFTER concatenation instead, once, over the
    # whole filtered frame.
}


def load_dataset_single_horizon_chunked(
    csv_path: str,
    report_path: str,
    horizon_seconds: float,
    *,
    requirement: DatasetRequirement = DatasetRequirement(),
    chunksize: int = 250_000,
) -> LoadedDataset:
    """Load exactly one prediction_horizon slice of a large campaign CSV
    without ever materializing the full multi-horizon file in memory.

    Equivalent in result to select_horizon(load_dataset(csv_path, ...),
    horizon_seconds), but reads csv_path in `chunksize`-row chunks,
    filters each chunk to `horizon_seconds` before accumulating it, and
    downcasts numeric columns to float32/int32 -- so peak memory during
    loading is bounded by one raw chunk, not the whole file.
    """

    manifest = load_dataset_manifest(report_path)
    assert_compatible(manifest, requirement)

    expected_columns = set(CSV_COLUMNS)

    filtered_chunks = []
    rows_seen = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):

        rows_seen += len(chunk)

        actual_columns = set(chunk.columns)
        if expected_columns != actual_columns:
            raise IncompatibleDatasetVersionError(
                f"{csv_path!r} columns do not match predictive_dataset.dataset_builder.CSV_COLUMNS "
                f"for schema_version {manifest.schema_version!r}. "
                f"Missing: {sorted(expected_columns - actual_columns)}, "
                f"Unexpected: {sorted(actual_columns - expected_columns)}"
            )

        matching = chunk[chunk["prediction_horizon"] == horizon_seconds]
        if len(matching) == 0:
            continue

        matching = matching.astype({k: v for k, v in _COMPACT_DTYPES.items() if k in matching.columns})
        filtered_chunks.append(matching.copy())

    if not filtered_chunks:
        raise IncompatibleDatasetVersionError(
            f"Requested horizon {horizon_seconds}s produced zero rows from {csv_path!r} "
            f"({rows_seen} total rows scanned)."
        )

    frame = pd.concat(filtered_chunks, ignore_index=True)
    del filtered_chunks

    for column in ("scenario_id", "candidate_id", "candidate_type", "candidate_congestion_level"):
        if column in frame.columns:
            frame[column] = frame[column].astype("category")

    return LoadedDataset(
        frame=frame,
        manifest=manifest,
        csv_path=csv_path,
        available_horizons=(horizon_seconds,),
    )


def select_horizon(dataset: LoadedDataset, horizon_seconds: float) -> pd.DataFrame:
    """Return only the rows for one prediction horizon. Raises if that
    horizon isn't present in the loaded dataset (Phase 1's own
    "reject incompatible" -- a horizon this training code asks for but
    the campaign never generated is a version mismatch, not a silent
    empty result)."""

    if horizon_seconds not in dataset.available_horizons:
        raise IncompatibleDatasetVersionError(
            f"Requested horizon {horizon_seconds}s not in dataset's available horizons "
            f"{dataset.available_horizons} (expected one of predictive_dataset.dataset_builder."
            f"DEFAULT_HORIZONS={DEFAULT_HORIZONS})."
        )

    return dataset.frame[dataset.frame["prediction_horizon"] == horizon_seconds].copy()


FEATURE_COLUMNS: Tuple[str, ...] = CANDIDATE_FEATURE_NAMES
