import hashlib
import random


# The frozen seed hierarchy -- architecture doc
# docs/architecture/scenario_engine.md §4.6:
#
#   Master Seed -> Scenario Seed -> Attempt Seed -> Category child streams
#
# Every derivation step below is *name/index-keyed*, never sequential
# ("spawn the next stream from a running index/order"). This is the one
# property §4.6 fixes precisely: a category's child stream is a
# deterministic function of (attempt_seed, category_key) only -- never
# of call order, stage index, or which other categories exist in a
# given run. The same index-keyed (not stream-position-keyed) property
# applies one level up for Scenario Seed derivation from a batch's
# Master Seed (§4.8) -- scenario i's seed never depends on scenarios
# 0..i-1 having been generated first.
#
# Deliberately NOT using Python's builtin hash() for strings/ints --
# str hashing is salted per-process by default (PYTHONHASHSEED),
# so hash("fire") differs across separate runs/machines and would
# silently break exactly the reproducibility guarantee this hierarchy
# exists to provide. hashlib.sha256 is stable across processes,
# machines, and Python versions, which is the actual requirement.
#
# This module fixes the *property* the derivation function must
# satisfy; the concrete function (sha256-based, here) is a
# construction-module implementation detail the architecture doc
# deliberately leaves undesigned (§4.6) -- a later revision could swap
# it for another stable derivation without changing anything else in
# this package's public shape.


CATEGORY_KEYS = (
    "fire",
    "occupant",
    "behaviour_profile",
    "door",
    "exit",
    "stair",
    "obstacle",
    "camera",
    "detector",
    "event",
    # Human Population & Assisted Evacuation Modeling Phase 5 --
    # additive, same index-keyed-not-order-keyed guarantee as every
    # category above. "assistance" affects only ScenarioOccupant.
    # assisting_occupant_id/assistance_type; "firefighter" is its own,
    # fully independent population (Phase 5's own "generate firefighter
    # deployments independently from occupant populations" rule).
    "assistance",
    "firefighter",
)

# Reserved, not implemented -- architecture doc §3.3/§4.4. Named here
# so a future Environmental generation stage keys its stream exactly
# like every other category, the moment it exists.
RESERVED_CATEGORY_KEYS = ("environmental",)


def derive_seed(*parts) -> int:

    # The one shared primitive every derivation step below builds on:
    # a stable, deterministic 64-bit integer keyed by an arbitrary
    # tuple of (seed, label, index, ...) parts.

    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big")


def derive_scenario_seed(master_seed: int, index: int) -> int:

    # Index-keyed, not stream-position-keyed (§4.8) -- generating
    # scenario 1000 in a second, later run produces the identical seed
    # to generating it as part of one big [0, 2000) run, because it
    # never depended on scenarios 0..999 having been generated first.

    return derive_seed(master_seed, "scenario", index)


def derive_attempt_seed(scenario_seed: int, attempt_index: int) -> int:

    # One per Generator/Validator retry attempt (§4.7). Attempt 0 is
    # the only attempt this phase ever produces -- no
    # scenario_validator/ exists yet to reject a candidate and trigger
    # a retry -- but the derivation itself is already correct for when
    # that orchestration is added later, without revisiting this
    # module.

    return derive_seed(scenario_seed, "attempt", attempt_index)


def category_rng(attempt_seed: int, category_key: str) -> random.Random:

    # The property this function exists to guarantee: two calls with
    # the same (attempt_seed, category_key) always produce an
    # identically-seeded random.Random, and two calls with different
    # category_key values (for the same attempt_seed) are independent
    # of each other regardless of what order a caller happens to
    # invoke them in, or whether other category_key values are ever
    # requested at all.

    if category_key not in CATEGORY_KEYS and category_key not in RESERVED_CATEGORY_KEYS:
        raise ValueError(f"Unknown category key: {category_key!r}")

    return random.Random(derive_seed(attempt_seed, "category", category_key))
