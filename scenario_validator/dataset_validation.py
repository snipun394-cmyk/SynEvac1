import hashlib
import json

from scenario import Scenario

from scenario_validator.issue import FailureCategory, ScenarioValidationIssue
from scenario_validator.report import ScenarioValidationReport


# Dataset Validation -- architecture doc §5.3, module 7, tagged
# DATASET. This implementation phase's own brief: duplicate-content is
# a *hook* only (accepted_hashes is externally supplied and never
# persisted by this package, §5.2/§4.9/§11 -- persistence is a later
# phase's job entirely).
#
# Deliberately does NOT import scenario_generator.metadata_builder's
# compute_definition_content_hash/GENERATION_VERSION, even though the
# hashing approach here is the same one (§5.7: "computed the same way
# the Validator's uniqueness check already hashes candidates") --
# scenario_validator/ must never import scenario_generator (§5.9/§12).
# The few lines of hashing logic are duplicated locally rather than
# shared, and KNOWN_GENERATION_VERSIONS is this package's own,
# independent allow-list, kept in sync with the Generator's
# GENERATION_VERSION constant by convention/documentation, never by
# import.

KNOWN_GENERATION_VERSIONS = frozenset({"scenario_generator/1"})

REQUIRED_METADATA_FIELDS = (
    "scenario_id", "definition_id", "definition_content_hash", "seed", "generation_version",
)


def _content_hash_from_data(data: dict) -> str:

    # The actual hashing work, split out from compute_candidate_content_hash()
    # so validate_dataset() below can reuse a to_dict() it already had to
    # compute for the serialization round-trip check, instead of paying
    # for candidate.to_dict() -- a full recursive dataclass-to-dict walk
    # of the whole Scenario -- a second time for the exact same
    # candidate. compute_candidate_content_hash()'s own public signature
    # and behavior are unchanged; every existing caller (scenario_pipeline,
    # scenario_storage, campaign_worker, ...) still just passes a
    # candidate and gets a hash back.

    content_only = {key: value for key, value in data.items() if key != "metadata"}

    canonical = json.dumps(content_only, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_candidate_content_hash(candidate) -> str:

    # Genuine bug fixed here (discovered integrating this function into
    # scenario_pipeline/, which exercises it end to end for the first
    # time): hashing candidate.to_dict() wholesale includes `metadata`
    # -- and metadata.created_at is a wall-clock timestamp,
    # metadata.scenario_id is derived from `seed` (§4.3). Two
    # candidates with byte-identical *sampled* content (fire, occupant
    # placement, every engineering state, events) generated a moment
    # apart, or from two different seeds that happen to land on the
    # same content -- exactly the case §4.9's "finite legal-and-unique
    # space" discusses content-hash dedup existing to catch -- would
    # never hash equal, silently defeating duplicate detection
    # entirely. The hash must cover only the resolved *content* a
    # Scenario carries, never its provenance/identity metadata.

    return _content_hash_from_data(candidate.to_dict())


def validate_dataset(candidate, accepted_hashes=frozenset()) -> ScenarioValidationReport:

    report = ScenarioValidationReport()

    metadata = candidate.metadata

    for field_name in REQUIRED_METADATA_FIELDS:

        value = getattr(metadata, field_name)

        if value is None or value == "":

            report.add(
                FailureCategory.DATASET, ScenarioValidationReport.ERROR,
                "MISSING_METADATA_FIELD",
                f"Scenario metadata.{field_name} is required and must not be empty.",
            )

    if metadata.generation_version and metadata.generation_version not in KNOWN_GENERATION_VERSIONS:

        report.add(
            FailureCategory.DATASET, ScenarioValidationReport.ERROR,
            "UNSUPPORTED_GENERATION_VERSION",
            f"generation_version {metadata.generation_version!r} is not one this "
            f"Validator release knows how to interpret.",
        )

    # Computed once and reused below for the hash, rather than calling
    # candidate.to_dict() a second time for the same candidate.
    candidate_data = candidate.to_dict()

    try:

        round_tripped = Scenario.from_dict(candidate_data)

        if round_tripped != candidate:

            report.add(
                FailureCategory.DATASET, ScenarioValidationReport.ERROR,
                "SERIALIZATION_ROUND_TRIP_MISMATCH",
                "Scenario.from_dict(candidate.to_dict()) does not equal the "
                "original candidate -- a serialization bug would otherwise only "
                "surface on read-back, after the candidate has already been "
                "written to disk.",
            )

    except Exception as error:  # noqa: BLE001 -- deliberately broad: any

        # serialization failure here is itself the finding, not a bug
        # in this check.
        report.add(
            FailureCategory.DATASET, ScenarioValidationReport.ERROR,
            "SERIALIZATION_ROUND_TRIP_FAILED",
            f"Scenario.to_dict()/from_dict() round-trip raised {error!r}.",
        )

    content_hash = _content_hash_from_data(candidate_data)

    if content_hash in accepted_hashes:

        report.add(
            FailureCategory.DATASET, ScenarioValidationReport.ERROR,
            "DUPLICATE",
            "This candidate's canonical serialized form matches an already-"
            "accepted scenario's content hash.",
        )

    return report
