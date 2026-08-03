import csv

from automatic_calibration.paths import run_json_filename, runs_catalog_path


# =====================================================
# Automatic Calibration Engine, Phase 1 -- Core Architecture.
#
# Mirrors calibration_studio/catalog.py's own shape exactly: append-
# only CSV, an identity index only ("this run_id exists, here is its
# filename"), never a history of every save -- an AutoCalibrationRun is
# a living record (RUNNING -> COMPLETED/FAILED) that gets saved again
# as it changes, so a fresh row every save would make list_runs() see
# the same run N times.
# =====================================================


RUN_CATALOG_COLUMNS = ("run_id", "project_id", "json_filename", "created_at")


def read_run_catalog_rows(storage_root) -> list:

    path = runs_catalog_path(storage_root)

    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_run_catalog_row_if_new(storage_root, run) -> None:

    path = runs_catalog_path(storage_root)
    existing_ids = {row.get("run_id") for row in read_run_catalog_rows(storage_root)}

    if run.run_id in existing_ids:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as handle:

        writer = csv.DictWriter(handle, fieldnames=RUN_CATALOG_COLUMNS)

        if write_header:
            writer.writeheader()

        writer.writerow({
            "run_id": run.run_id,
            "project_id": run.project_id,
            "json_filename": run_json_filename(run.run_id),
            "created_at": run.created_at,
        })
