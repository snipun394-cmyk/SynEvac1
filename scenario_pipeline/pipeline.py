import time
from dataclasses import replace
from typing import FrozenSet

from scenario_generator import GenerationRequest, derive_scenario_seed, generate_scenario
from scenario_validator import compute_candidate_content_hash, validate

from scenario_pipeline.result import BatchPipelineResult, PipelineResult
from scenario_pipeline.statistics import aggregate_statistics, build_statistics


# The orchestration layer -- architecture doc §1's frozen pipeline
# diagram (Definition -> Generator -> Candidate Scenario -> Validator
# -> Accepted -> [stored] / Rejected -> Generate Again), §4.6's Attempt
# Seed layer, and §5.8's "every aggregate quantity is computed entirely
# by the orchestration module... never by the Validator reasoning about
# sequence." Coordinates scenario_generator and scenario_validator; it
# contains no generation logic of its own (every candidate comes from
# generate_scenario()), no validation logic of its own (every verdict
# comes from validate()), and never repairs a rejected candidate --
# on rejection it simply discards it and asks the Generator for a
# fresh one at the next attempt index, exactly as §1/§4.7 already
# freeze.
#
# Physically a new top-level package (scenario_pipeline/) rather than
# scenario_generator/pipeline.py as §12 named it while scenario_
# validator/ didn't yet exist -- the substantive rule §12 actually
# states (the *construction* module must never import
# scenario_validator; only the orchestration code may depend on both)
# is unaffected by which package that orchestration code physically
# lives in, and this phase's own brief explicitly asks for a dedicated
# package. See this package's final report for the full reasoning.

DEFAULT_MAX_ATTEMPTS = 10


def run_pipeline(
    definition,
    definition_id,
    building,
    seed,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    accepted_hashes: FrozenSet[str] = frozenset(),
) -> PipelineResult:

    # One Scenario Seed's worth of retries -- §4.6's Attempt Seed layer
    # exists exactly for this loop: attempt_index increments by exactly
    # one per rejection, `seed` (the Scenario Seed) never changes, and
    # generate_scenario()/seed_manager derive a fresh, independent
    # Attempt Seed from (seed, attempt_index) every time, so retry N
    # never reuses retry N-1's random draws. Never enters an infinite
    # loop: attempt_index is bounded by max_attempts, checked up front.

    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}.")

    start_time = time.perf_counter()

    total_generation_time = 0.0
    total_validation_time = 0.0
    rejection_categories = {}
    last_report = None

    for attempt_index in range(max_attempts):

        request = GenerationRequest(
            definition=definition, definition_id=definition_id, building=building,
            seed=seed, attempt_index=attempt_index,
        )

        generation_start = time.perf_counter()
        candidate = generate_scenario(request)
        total_generation_time += time.perf_counter() - generation_start

        validation_start = time.perf_counter()
        report = validate(candidate, definition, building, accepted_hashes=accepted_hashes)
        total_validation_time += time.perf_counter() - validation_start

        last_report = report

        if report.accepted:

            statistics = build_statistics(
                attempts=attempt_index + 1,
                accepted_count=1,
                rejection_categories=rejection_categories,
                generation_time_seconds=total_generation_time,
                validation_time_seconds=total_validation_time,
                total_time_seconds=time.perf_counter() - start_time,
            )

            return PipelineResult(
                accepted=True, scenario=candidate, statistics=statistics, last_report=report,
            )

        # Rejected -- the candidate is discarded outright (never
        # mutated, never patched, never resubmitted); the next loop
        # iteration asks the Generator for a completely fresh candidate
        # at attempt_index + 1.
        for category in report.error_categories():
            rejection_categories[category] = rejection_categories.get(category, 0) + 1

    statistics = build_statistics(
        attempts=max_attempts,
        accepted_count=0,
        rejection_categories=rejection_categories,
        generation_time_seconds=total_generation_time,
        validation_time_seconds=total_validation_time,
        total_time_seconds=time.perf_counter() - start_time,
    )

    return PipelineResult(
        accepted=False, scenario=None, statistics=statistics, last_report=last_report,
        failure_reason="MAX_ATTEMPTS_EXCEEDED",
    )


def run_batch_pipeline(
    definition,
    definition_id,
    building,
    master_seed,
    count,
    start_index=0,
    max_attempts_per_scenario=DEFAULT_MAX_ATTEMPTS,
    accepted_hashes: FrozenSet[str] = frozenset(),
) -> BatchPipelineResult:

    # "Generate scenarios at indices [start_index, start_index +
    # count)" (§4.8) -- each scenario's own Scenario Seed is derived
    # via seed_manager.derive_scenario_seed(master_seed, index), the
    # exact same index-keyed (not stream-position-keyed) primitive
    # scenario_generator.batch_generator already uses for the
    # no-retry case, reused here rather than reimplemented so a batch
    # generated through this pipeline and one generated (without
    # validation) through scenario_generator.generate_batch() agree on
    # which seed belongs to which index.
    #
    # accepted_hashes is the *one* cross-scenario coupling point
    # (§4.8/§4.9): it starts from whatever the caller passed in
    # (forward-compatible with a future persistence layer seeding it
    # from a prior run's catalog, §4.9 -- not implemented by this
    # phase) and grows by one entry per scenario *this* batch run
    # accepts, so scenario i+1 never duplicates scenario i within the
    # same run.

    if count < 0:
        raise ValueError(f"count must be >= 0, got {count!r}.")

    start_time = time.perf_counter()

    running_hashes = set(accepted_hashes)
    results = []

    for index in range(start_index, start_index + count):

        scenario_seed = derive_scenario_seed(master_seed, index)

        result = run_pipeline(
            definition, definition_id, building, scenario_seed,
            max_attempts=max_attempts_per_scenario,
            accepted_hashes=frozenset(running_hashes),
        )

        results.append(result)

        if result.accepted:
            running_hashes.add(compute_candidate_content_hash(result.scenario))

    scenarios = tuple(result.scenario for result in results if result.accepted)

    aggregated = aggregate_statistics([result.statistics for result in results])

    # aggregate_statistics() sums each per-scenario total_time_seconds,
    # which double-counts wall-clock time that actually overlapped
    # sequentially -- override with the batch's own single, real
    # elapsed duration instead.
    aggregated = replace(aggregated, total_time_seconds=time.perf_counter() - start_time)

    return BatchPipelineResult(scenarios=scenarios, results=tuple(results), statistics=aggregated)
