from pathlib import Path


# Sharded directory structure -- architecture doc
# docs/architecture/scenario_engine.md §11: "Scenario files are sharded
# by a fixed prefix of scenario_id (e.g., a subdirectory per first two
# hex characters) rather than written flat into one directory."
# §13 leaves the exact prefix length/depth as an undecided
# implementation detail -- this module fixes it at 2, matching the
# doc's own worked example precisely, single-level (no deeper nesting
# needed until file counts are enormously larger than this project's
# near-term scale).
#
# scenario_id itself is "scn-<16 hex chars>" (scenario_generator.
# metadata_builder.derive_scenario_id) -- every id shares the same
# literal "scn-" label, so sharding on the string's raw first two
# characters would put every single scenario in one "sc" bucket,
# defeating sharding entirely. shard_key() strips that known label
# prefix first and shards on the actual (high-entropy, sha256-derived)
# digest that follows it.

SHARD_KEY_LENGTH = 2
SCENARIO_ID_LABEL_PREFIX = "scn-"

SCENARIOS_SUBDIRECTORY = "scenarios"
CATALOG_FILENAME = "catalog.csv"


def shard_key(scenario_id: str) -> str:

    digest_part = (
        scenario_id[len(SCENARIO_ID_LABEL_PREFIX):]
        if scenario_id.startswith(SCENARIO_ID_LABEL_PREFIX)
        else scenario_id
    )

    # Defensive padding only -- guards a pathologically short/empty id
    # from producing an unusably short (or empty) shard directory name.
    # Real scenario_ids are always well over SHARD_KEY_LENGTH long.
    padded = digest_part.ljust(SHARD_KEY_LENGTH, "0")

    return padded[:SHARD_KEY_LENGTH]


def scenario_json_filename(scenario_id: str) -> str:

    return f"{scenario_id}.json"


def scenario_json_path(storage_root, scenario_id: str) -> Path:

    return (
        Path(storage_root)
        / SCENARIOS_SUBDIRECTORY
        / shard_key(scenario_id)
        / scenario_json_filename(scenario_id)
    )


def catalog_path(storage_root) -> Path:

    # One running catalog file for the whole storage root, not one per
    # batch -- matches §11's "append-only" framing (a single,
    # continuously-growing index), and keeps load_accepted_hashes()
    # (storage.py) a single-file read regardless of how many separate
    # save_scenario() calls produced the rows in it.

    return Path(storage_root) / CATALOG_FILENAME
