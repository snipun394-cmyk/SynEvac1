from pathlib import Path


# =====================================================
# Automatic Calibration Engine, Phase 1 -- Core Architecture.
#
# Mirrors calibration_studio/paths.py's own shape exactly: one JSON
# file per record + one running catalog, no sharding (this package's
# own scale -- a handful of search runs per project -- is smaller still
# than Calibration Studio's own already-documented reasoning for
# skipping scenario_storage's sharding convention).
# =====================================================

RUNS_SUBDIRECTORY = "automatic_calibration_runs"
RUNS_CATALOG_FILENAME = "automatic_calibration_runs_catalog.csv"


def run_json_filename(run_id: str) -> str:

    return f"{run_id}.json"


def run_json_path(storage_root, run_id: str) -> Path:

    return Path(storage_root) / RUNS_SUBDIRECTORY / run_json_filename(run_id)


def runs_catalog_path(storage_root) -> Path:

    return Path(storage_root) / RUNS_CATALOG_FILENAME
