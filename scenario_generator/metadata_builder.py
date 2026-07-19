import hashlib
import json
from datetime import datetime, timezone

from scenario import ScenarioMetadata

from scenario_generator.seed_manager import derive_seed


# Identifies the Generator release (sampling interpreter +
# seed-derivation function) that produced a Scenario -- architecture
# doc §4.3: reproducibility is a claim about (definition_content_hash,
# seed, generation_version) jointly, never (definition_id, seed) alone,
# precisely so a later change to either the sampling interpreter or
# seed_manager.derive_seed's algorithm is visible/auditable rather than
# silently assumed compatible with scenarios generated under an earlier
# version. Bump this whenever either changes in a way that would alter
# sampled output for the same (definition_content_hash, seed).
GENERATION_VERSION = "scenario_generator/1"


def compute_definition_content_hash(definition) -> str:

    # Computed the same way the (not-yet-implemented) Scenario
    # Validator's own uniqueness check will hash candidates (§4.3/§5.7)
    # -- a canonical (sort_keys=True, so key order never depends on
    # incidental dict-construction order), in-memory serialization,
    # hashed with sha256. json.dumps here produces a string, never
    # touches a file -- not the file I/O this package must avoid.

    canonical = json.dumps(definition.to_dict(), sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def derive_scenario_id(definition_content_hash: str, seed: int, attempt_index: int) -> str:

    # Deterministic, not uuid4() -- this package owns Scenario Seed's
    # randomness, but an *identity* string must still be a pure
    # function of (definition_content_hash, seed, attempt_index) so
    # that "identical seeds produce identical scenarios" (this
    # implementation phase's own testing requirement) holds for the id
    # too, not just for the sampled content.

    digest = derive_seed(definition_content_hash, "scenario_id", seed, attempt_index)

    return f"scn-{digest:016x}"


def build_metadata(
    definition,
    definition_id: str,
    seed: int,
    attempt_index: int = 0,
    rejected_attempt_count: int = 0,
) -> ScenarioMetadata:

    # rejected_attempt_count defaults to 0: with no scenario_validator/
    # yet to reject a candidate, no attempt is ever discarded by this
    # phase's own code. The field is populated correctly the moment
    # orchestration wiring against a real Validator exists, without any
    # change needed here.

    definition_content_hash = compute_definition_content_hash(definition)

    return ScenarioMetadata(
        scenario_id=derive_scenario_id(definition_content_hash, seed, attempt_index),
        definition_id=definition_id,
        definition_content_hash=definition_content_hash,
        generation_version=GENERATION_VERSION,
        seed=seed,
        created_at=datetime.now(timezone.utc).isoformat(),
        rejected_attempt_count=rejected_attempt_count,
    )
