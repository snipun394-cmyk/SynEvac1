import csv

from scenario_storage.paths import catalog_path


# The append-only CSV catalog -- architecture doc §11: "written by
# appending one row per newly accepted scenario, never by rewriting the
# file in full." Every column here is a denormalized view of a field
# already resolved onto Scenario/ScenarioMetadata (§7/§11's own "never
# a second source of truth" rule) -- this module computes a row once
# from an already-built Scenario, it never derives or stores anything
# new.
#
# This implementation phase's own column list (a deliberately narrower
# subset of §11's full frozen list -- open_exit_ids/closed_door_ids/
# blocked_stair_ids/obstacle_summary/behaviour_profile_summary/
# rejected_attempt_count are not included here, only what this phase's
# brief explicitly names). CATALOG_COLUMNS is the one place the schema
# is defined; "future metadata can be appended without breaking
# compatibility" means exactly this, and no more, given CSV's own
# structural limits: new columns may be appended to the *end* of this
# tuple in a future phase, and every row already written under the
# shorter, older header remains readable (csv.DictReader exposes a
# missing trailing column as None for old rows) -- but changing an
# *already-written* file's header to retroactively add columns to
# already-written rows is not something CSV supports, and is not
# attempted here or implied as a promise this module keeps. Columns
# must never be reordered or removed, only appended.

CATALOG_COLUMNS = (
    "scenario_id",
    "json_filename",
    "definition_id",
    "definition_content_hash",
    "generation_version",
    "seed",
    "fire_profile",
    "fire_origin",
    "occupant_count",
    "difficulty",
    "created_at",
)


def catalog_row_for(scenario, json_filename: str) -> dict:

    metadata = scenario.metadata
    fire = scenario.fire

    return {
        "scenario_id": metadata.scenario_id,
        "json_filename": json_filename,
        "definition_id": metadata.definition_id,
        "definition_content_hash": metadata.definition_content_hash,
        "generation_version": metadata.generation_version,
        "seed": metadata.seed,
        "fire_profile": fire.fire_profile if fire is not None else "",
        "fire_origin": fire.ignition_zone_id if fire is not None else "",
        "occupant_count": len(scenario.occupants),
        # Reserved (this phase's own brief) -- absent (None) until a
        # future post-validation scoring phase populates
        # Scenario.difficulty (§10, unchanged).
        "difficulty": scenario.difficulty if scenario.difficulty is not None else "",
        "created_at": metadata.created_at,
    }


def append_catalog_row(storage_root, scenario, json_filename: str) -> None:

    path = catalog_path(storage_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as handle:

        writer = csv.DictWriter(handle, fieldnames=CATALOG_COLUMNS)

        if write_header:
            writer.writeheader()

        writer.writerow(catalog_row_for(scenario, json_filename))


def read_catalog_rows(storage_root) -> list:

    path = catalog_path(storage_root)

    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as handle:

        reader = csv.DictReader(handle)

        return list(reader)
