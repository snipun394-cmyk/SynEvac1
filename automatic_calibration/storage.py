import json
from pathlib import Path
from typing import Optional, Tuple

from serialization.json_reader import JsonReader
from serialization.json_writer import JsonWriter

from automatic_calibration.catalog import append_run_catalog_row_if_new, read_run_catalog_rows
from automatic_calibration.paths import run_json_path
from automatic_calibration.run import SUPPORTED_SCHEMA_VERSIONS, AutoCalibrationRun


# =====================================================
# Automatic Calibration Engine, Phase 1 -- Core Architecture.
#
# Mirrors calibration_studio/storage.py's own shape exactly: built
# entirely on serialization/json_writer.py + json_reader.py (the same
# primitives Calibration Studio's own persistence layer is built on) --
# no second persistence framework. save_run() always overwrites in
# place (an AutoCalibrationRun is a living record, same reasoning as
# CalibrationProject/CalibrationSession); catalog rows are only ever
# appended once per run_id.
#
# Its own IncompatibleSchemaVersionError/CorruptedRecordFileError are
# NOT imported from calibration_studio.storage -- deliberately. Every
# persistence module in this codebase (scenario_storage, calibration_
# studio) defines its own pair with the same shape rather than sharing
# one global exception type across packages; this module follows that
# same established convention rather than introducing a new,
# inconsistent one.
# =====================================================


class IncompatibleSchemaVersionError(Exception):
    pass


class CorruptedRecordFileError(Exception):
    pass


def _read_record_json(path: Path) -> dict:

    try:
        return JsonReader.read(str(path))
    except json.JSONDecodeError as exc:
        raise CorruptedRecordFileError(f"Corrupted record file at {path}: {exc}") from exc


def _check_schema_version(data: dict, record_id) -> None:

    found = data.get("schema_version")

    if found not in SUPPORTED_SCHEMA_VERSIONS:
        raise IncompatibleSchemaVersionError(
            f"Cannot load run {record_id!r}: schema_version {found!r} is not one of the "
            f"versions this code supports ({sorted(SUPPORTED_SCHEMA_VERSIONS)!r}). A newer "
            f"Automatic Calibration Engine wrote this file with a schema this version has "
            f"never heard of, or the file is otherwise not a genuine run record.",
        )


def save_run(run: AutoCalibrationRun, storage_root) -> Path:

    storage_root = Path(storage_root)
    json_path = run_json_path(storage_root, run.run_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    JsonWriter.write(str(json_path), run.to_dict())
    append_run_catalog_row_if_new(storage_root, run)

    return json_path


def load_run(run_id: str, storage_root) -> AutoCalibrationRun:

    storage_root = Path(storage_root)
    json_path = run_json_path(storage_root, run_id)

    data = _read_record_json(json_path)
    _check_schema_version(data, run_id)

    return AutoCalibrationRun.from_dict(data)


def list_runs(storage_root, *, project_id: Optional[str] = None) -> Tuple[AutoCalibrationRun, ...]:

    storage_root = Path(storage_root)
    runs = []

    for row in read_run_catalog_rows(storage_root):

        run_id = row.get("run_id")

        if not run_id:
            continue

        if project_id is not None and row.get("project_id") != project_id:
            continue

        try:
            runs.append(load_run(run_id, storage_root))
        except FileNotFoundError:
            continue

    return tuple(runs)
