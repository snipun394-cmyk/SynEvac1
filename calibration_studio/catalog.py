import csv

from calibration_studio.paths import project_json_filename, projects_catalog_path, session_json_filename, sessions_catalog_path


# =====================================================
# Calibration Studio Phase 2 -- Persistence Layer.
#
# Mirrors scenario_storage/catalog.py's own shape (append-only CSV,
# CATALOG_COLUMNS is the one place the schema is defined, new columns
# may only ever be appended at the end, never reordered/removed) with
# one deliberate difference in WHEN a row is appended: scenario_storage
# appends exactly once per accepted Scenario, because a Scenario is
# write-once (scenario_storage.save_scenario() refuses to overwrite).
# CalibrationProject/CalibrationSession are living records that get
# saved again every time they change -- appending a fresh row on every
# save would make list_projects()/list_sessions() see the same project
# N times. These catalogs are therefore purely an IDENTITY index ("this
# id exists, here is its filename") -- append-if-new, never rewritten
# once a row exists -- not a history of every save. Current
# name/status/tags/etc. are never stored here; a caller wanting current
# state always gets it from the record's own JSON file (the single
# source of truth for mutable fields), which storage.py's
# list_projects()/list_sessions() already do on every call.
# =====================================================


PROJECT_CATALOG_COLUMNS = ("project_id", "json_filename", "created_at")
SESSION_CATALOG_COLUMNS = ("session_id", "project_id", "json_filename", "created_at")


def read_project_catalog_rows(storage_root) -> list:

    path = projects_catalog_path(storage_root)

    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_project_catalog_row_if_new(storage_root, project) -> None:

    path = projects_catalog_path(storage_root)
    existing_ids = {row.get("project_id") for row in read_project_catalog_rows(storage_root)}

    if project.project_id in existing_ids:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as handle:

        writer = csv.DictWriter(handle, fieldnames=PROJECT_CATALOG_COLUMNS)

        if write_header:
            writer.writeheader()

        writer.writerow({
            "project_id": project.project_id,
            "json_filename": project_json_filename(project.project_id),
            "created_at": project.created_at,
        })


def read_session_catalog_rows(storage_root) -> list:

    path = sessions_catalog_path(storage_root)

    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_session_catalog_row_if_new(storage_root, session) -> None:

    path = sessions_catalog_path(storage_root)
    existing_ids = {row.get("session_id") for row in read_session_catalog_rows(storage_root)}

    if session.session_id in existing_ids:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as handle:

        writer = csv.DictWriter(handle, fieldnames=SESSION_CATALOG_COLUMNS)

        if write_header:
            writer.writeheader()

        writer.writerow({
            "session_id": session.session_id,
            "project_id": session.project_id or "",
            "json_filename": session_json_filename(session.session_id),
            "created_at": session.created_at,
        })
