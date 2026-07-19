from dataclasses import dataclass
from typing import Optional

from models.building import Building

from scenario_definition import ScenarioDefinition


@dataclass(frozen=True)
class GenerationRequest:

    # The minimal envelope the Generator's input contract requires
    # (§4.2 point 4): the Definition, the Building its ids resolve
    # against, and an already-resolved seed -- "resolved per §4.6,
    # either caller-supplied (single-scenario request) or derived from
    # a batch's master seed and this scenario's index (§4.8)." Seed
    # resolution itself happens *before* a GenerationRequest is built
    # (see batch_generator.py for the batch case); this class only
    # carries the already-resolved value.
    #
    # definition_id has no counterpart field anywhere in the frozen
    # scenario_definition.ScenarioDefinition schema -- a Definition
    # value object carries no id of its own (§3.2's field table has no
    # such field). Reproducibility (§4.3) depends on
    # definition_content_hash *and* a stable definition_id that
    # survives the Definition being edited -- which only a
    # Definition-storage/catalog layer above this package can supply.
    # This is why definition_id is a required field here rather than
    # something derived from `definition` itself.
    #
    # batch_index is the "mode flag (single vs batch-member)" §4.2
    # point 4 asks for, carrying the actual index rather than a bare
    # bool: None means a standalone, single-scenario request;
    # otherwise it is this scenario's index within a batch (purely
    # informational/provenance -- generation behavior never branches on
    # it, since the seed was already resolved before this request was
    # built).

    definition: ScenarioDefinition
    definition_id: str
    building: Building

    seed: int

    attempt_index: int = 0
    batch_index: Optional[int] = None


@dataclass(frozen=True)
class BatchGenerationRequest:

    # "Generate scenarios at indices [start_index, start_index + count)"
    # (§4.8) -- not "generate `count` scenarios, however that happens
    # to unfold." start_index defaults to 0 (a full, from-scratch
    # batch); passing a non-zero start_index is what makes a later run
    # resumable/appendable: generating [1000, 2000) in a second run
    # produces byte-identical scenarios to having generated [0, 2000)
    # in one run, because scenario 1000's seed never depended on
    # scenarios 0..999 having been generated first (seed_manager.
    # derive_scenario_seed is index-keyed, not stream-position-keyed).

    definition: ScenarioDefinition
    definition_id: str
    building: Building

    master_seed: int
    count: int

    start_index: int = 0

    # Meaningful once orchestration against a real scenario_validator/
    # exists (§4.7); every attempt in this phase is attempt 0, always
    # accepted, since there is no Validator yet to reject a candidate.
    max_attempts_per_scenario: int = 1
