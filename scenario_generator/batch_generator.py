from typing import Iterator, List

from scenario import Scenario

from scenario_generator.generator import generate_scenario
from scenario_generator.request import BatchGenerationRequest, GenerationRequest
from scenario_generator.seed_manager import derive_scenario_seed


# Batch generation -- architecture doc §4.8: "generate scenarios at
# indices [start_index, start_index + count)", never "generate `count`
# scenarios, however that happens to unfold." Generating scenario i is
# fully independent of generating scenario j (i != j) -- every input is
# already fully determined by (master_seed, i) before either starts
# (§4.8: "embarrassingly parallel, with exactly one coupling point" --
# that one coupling point, the Validator's accepted_hashes uniqueness
# set, does not exist in this phase at all, since no scenario_validator/
# has been built yet).


def iter_batch(request: BatchGenerationRequest) -> Iterator[Scenario]:

    # Lazy, one Scenario at a time -- "large batch generation" (this
    # implementation phase's own brief) should not require holding
    # millions of Scenario objects in memory simultaneously just to
    # produce them; generate_batch() below is the convenience list
    # form for the common, smaller case.

    for index in range(request.start_index, request.start_index + request.count):

        scenario_seed = derive_scenario_seed(request.master_seed, index)

        generation_request = GenerationRequest(
            definition=request.definition,
            definition_id=request.definition_id,
            building=request.building,
            seed=scenario_seed,
            attempt_index=0,
            batch_index=index,
        )

        yield generate_scenario(generation_request)


def generate_batch(request: BatchGenerationRequest) -> List[Scenario]:

    return list(iter_batch(request))
