from pathlib import Path
from typing import FrozenSet

from scenario import Scenario
from scenario_validator import compute_candidate_content_hash

from serialization.json_reader import JsonReader
from serialization.json_writer import JsonWriter

from scenario_storage.catalog import append_catalog_row, read_catalog_rows
from scenario_storage.paths import scenario_json_path


# JSON persistence for accepted Scenarios -- architecture doc §11: "one
# serialized JSON file per accepted Scenario," "built on
# serialization/json_writer.py/json_reader.py, not Serializer." Reuses
# Scenario.to_dict()/from_dict() as-is (§7, frozen) -- this module
# never re-derives or re-shapes a Scenario's own serialized form, only
# decides *where on disk* that already-defined shape gets written and
# read back from.
#
# This package performs no generation, no validation, no simulation --
# save_scenario() accepts whatever Scenario it is handed and persists
# it verbatim. Confirming a candidate was actually accepted is
# entirely the caller's responsibility (scenario_pipeline's
# PipelineResult.accepted), the same boundary §11 already draws:
# "the catalog and per-scenario files only ever describe accepted
# scenarios" is a statement about what callers are expected to persist,
# not a check this module performs.


def save_scenario(scenario: Scenario, storage_root) -> Path:

    storage_root = Path(storage_root)
    json_path = scenario_json_path(storage_root, scenario.metadata.scenario_id)

    if json_path.exists():

        # scenario_id is a stable identity (§4.3/§7) -- silently
        # overwriting an existing file at the same id would either be a
        # no-op (identical content re-saved) or silent data corruption
        # (different content claiming the same id, which should never
        # happen but is exactly the kind of thing a persistence layer
        # should refuse rather than paper over). Not a validation
        # check on Scenario *content* -- a structural guard on this
        # package's own identity invariant.
        raise FileExistsError(
            f"A scenario is already stored at scenario_id "
            f"{scenario.metadata.scenario_id!r} ({json_path}) -- refusing to "
            f"overwrite it.",
        )

    json_path.parent.mkdir(parents=True, exist_ok=True)

    JsonWriter.write(str(json_path), scenario.to_dict())

    append_catalog_row(storage_root, scenario, json_path.name)

    return json_path


def load_scenario_by_filename(filename: str, storage_root) -> Scenario:

    scenario_id = filename[: -len(".json")] if filename.endswith(".json") else filename

    return load_scenario_by_id(scenario_id, storage_root)


def load_scenario_by_id(scenario_id: str, storage_root) -> Scenario:

    json_path = scenario_json_path(Path(storage_root), scenario_id)

    data = JsonReader.read(str(json_path))

    return Scenario.from_dict(data)


def load_accepted_hashes(storage_root) -> FrozenSet[str]:

    # The persistence-side half of §4.9's "accepted_hashes must be
    # seeded by reading the existing CSV catalog... not assumed
    # empty" -- deliberately just a function returning a FrozenSet[str]
    # a caller may pass into scenario_pipeline.run_pipeline()/
    # run_batch_pipeline()'s own accepted_hashes parameter; this
    # package never calls into scenario_pipeline itself ("do not
    # redesign the pipeline," this phase's own brief).
    #
    # This phase's catalog schema (catalog.py) has no content_hash
    # column of its own -- computing the hash scenario_validator.
    # compute_candidate_content_hash() actually produces requires the
    # full resolved content, not just the catalog row, so each
    # referenced JSON file is loaded once here. A future phase could
    # add a content_hash column to make this cheaper without changing
    # this function's signature or behavior.

    hashes = set()

    for row in read_catalog_rows(storage_root):

        filename = row.get("json_filename")

        if not filename:
            continue

        scenario = load_scenario_by_filename(filename, storage_root)
        hashes.add(compute_candidate_content_hash(scenario))

    return frozenset(hashes)
