import csv
import os

from typing import Any, Dict, Iterable, List, Optional

from training_dataset.loader import SimulationSample
from training_dataset.manifest import build_manifest, write_manifest


# =====================================================
# Export (Phase 5) -- turns already-loaded/split samples into files an
# ML pipeline can consume directly: one combined CSV (always), an
# optional combined Parquet (only when pandas + a parquet engine are
# actually installed), and the manifest JSON.
# =====================================================

COMBINED_CSV_FILENAME = "combined.csv"
COMBINED_PARQUET_FILENAME = "combined.parquet"
MANIFEST_FILENAME = "manifest.json"


class ParquetUnavailableError(RuntimeError):

    # Raised only when Parquet export is explicitly requested
    # (include_parquet=True / a direct export_combined_parquet() call)
    # and the optional dependency isn't installed -- never raised
    # silently, since a caller who didn't ask for Parquet should never
    # be blocked by its absence (requirements.txt carries no pandas/
    # pyarrow pin in this project today).

    pass


def combined_row(sample: SimulationSample, split_label: Optional[str] = None) -> Dict[str, Any]:

    row: Dict[str, Any] = dict(sample.scenario_features)

    for key, value in sample.simulation_outcome.items():

        if key == "scenario_id":
            continue

        row[f"outcome_{key}"] = value

    if split_label is not None:
        row["split"] = split_label

    return row


def _union_columns(rows: List[Dict[str, Any]]) -> List[str]:

    columns: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    return columns


def _combined_rows(dataset: Iterable, split) -> List[Dict[str, Any]]:

    return [
        combined_row(sample, split.split_for(sample.scenario_id) if split is not None else None)
        for sample in dataset
    ]


def export_combined_csv(dataset: Iterable, split, file_path: str) -> str:

    rows = _combined_rows(dataset, split)
    fieldnames = _union_columns(rows)

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, "w", newline="", encoding="utf-8") as handle:

        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})

    return file_path


def export_combined_parquet(dataset: Iterable, split, file_path: str) -> str:

    try:
        import pandas as pd
    except ImportError as exc:
        raise ParquetUnavailableError(
            "Parquet export requires the optional 'pandas' dependency (plus a parquet "
            "engine such as 'pyarrow'), which is not installed in this environment."
        ) from exc

    frame = pd.DataFrame(_combined_rows(dataset, split))

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        frame.to_parquet(file_path)
    except ImportError as exc:
        raise ParquetUnavailableError(
            "Parquet export requires a parquet engine ('pyarrow' or 'fastparquet'), "
            "which is not installed in this environment."
        ) from exc

    return file_path


def export_manifest(dataset: Iterable, split, file_path: str, **manifest_kwargs) -> str:

    manifest = build_manifest(dataset, split, **manifest_kwargs)

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    write_manifest(manifest, file_path)

    return file_path


def export_dataset(
    dataset: Iterable,
    split,
    output_dir: str,
    *,
    campaign_name: Optional[str] = None,
    master_seed: Optional[int] = None,
    include_parquet: bool = False,
) -> Dict[str, str]:

    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "combined_csv": export_combined_csv(
            dataset, split, os.path.join(output_dir, COMBINED_CSV_FILENAME),
        ),
        "manifest": export_manifest(
            dataset, split, os.path.join(output_dir, MANIFEST_FILENAME),
            campaign_name=campaign_name, master_seed=master_seed,
        ),
    }

    if include_parquet:
        # Deliberately not caught here -- a caller who explicitly asked
        # for Parquet should see ParquetUnavailableError, not a silent
        # gap in the returned paths dict.
        paths["combined_parquet"] = export_combined_parquet(
            dataset, split, os.path.join(output_dir, COMBINED_PARQUET_FILENAME),
        )

    return paths
