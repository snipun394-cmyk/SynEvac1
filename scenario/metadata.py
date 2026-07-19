from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from scenario.serialization import mapping_to_dict


@dataclass(frozen=True)
class ScenarioMetadata:

    # Immutable provenance/reproducibility record for one generated
    # Scenario -- architecture doc docs/architecture/scenario_engine.md
    # §4.3/§4.10/§7. Every field here is either required for
    # reproducibility (scenario_id, definition_id,
    # definition_content_hash, seed, generation_version -- §4.3: a
    # Scenario is reproducible only as a joint claim over
    # (definition_content_hash, seed, generation_version), never
    # definition_id/seed alone) or provenance-only (created_at,
    # rejected_attempt_count -- describe the candidate but play no role
    # in regenerating it, §4.4's reproducibility-vs-provenance split).
    #
    # No field here has a default_factory that calls uuid4()/
    # datetime.now() -- unlike models/base_object.py's BaseObject or
    # hazard/occupancy's snapshot types. This package is plain
    # immutable data only (no generation, no randomness, no
    # computation); minting a fresh scenario_id or timestamp is the
    # future Generator's job (§4), not this model's. Every field here
    # must be supplied explicitly by the caller.

    scenario_id: str
    definition_id: str
    definition_content_hash: str
    generation_version: str
    seed: int

    created_at: str

    rejected_attempt_count: int = 0

    # Open-ended bag for future metrics -- same "safe because this
    # type has no rebuild-from-source-of-truth lifecycle" reasoning
    # behavior/profile.py's BehaviorProfile.traits already documents.
    # Lets a future metric be added without a schema change, and
    # without breaking to_dict()/from_dict() round-tripping for
    # already-serialized Scenarios that predate it (an absent key
    # simply means "no value for that metric").
    extra: Mapping[str, Any] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "scenario_id": self.scenario_id,
            "definition_id": self.definition_id,
            "definition_content_hash": self.definition_content_hash,
            "generation_version": self.generation_version,
            "seed": self.seed,
            "created_at": self.created_at,
            "rejected_attempt_count": self.rejected_attempt_count,
            "extra": mapping_to_dict(self.extra),
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioMetadata":

        return cls(
            scenario_id=data["scenario_id"],
            definition_id=data["definition_id"],
            definition_content_hash=data["definition_content_hash"],
            generation_version=data["generation_version"],
            seed=data["seed"],
            created_at=data["created_at"],
            rejected_attempt_count=data.get("rejected_attempt_count", 0),
            extra=data.get("extra", {}),
        )
