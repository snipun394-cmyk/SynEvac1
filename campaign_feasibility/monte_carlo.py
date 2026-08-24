import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from scipy import stats as scipy_stats

from scenario_generator import GenerationRequest, derive_seed, generate_scenario
from scenario_validator import validate_navigation

from campaign_feasibility.exact_validity import (
    DEFAULT_MAX_ENUMERATED_STATES,
    CandidateValidityResult,
    compute_exact_candidate_validity,
)


# Scenario Campaign Feasibility Preflight -- Phase 2C: Monte Carlo
# Fallback for Candidate Validity (and, via campaign_feasibility.
# yield_analysis, Campaign Yield).
# docs/architecture/scenario_campaign_feasibility_phase2_investigation.txt
# docs/architecture/scenario_campaign_feasibility_phase2a_implementation_report.txt
# docs/architecture/scenario_campaign_feasibility_phase2b_implementation_report.txt
# docs/architecture/scenario_campaign_feasibility_min_open_exits_occupancy_implementation_report.txt
# docs/architecture/scenario_campaign_yield_analysis_implementation_report.txt
# docs/architecture/scenario_campaign_feasibility_phase2c_monte_carlo_implementation_report.txt
#
# WHY THIS MODULE, UNLIKE exact_validity.py, IMPORTS scenario_generator/
# scenario_validator: exact_validity.py's own discipline ("never import
# Generator/Validator, only MIRROR their logic") exists so that module
# can PROVE things analytically, without sampling, without RNG. This
# module's entire purpose is the opposite: when that analytical proof
# is not tractable (`CandidateValidityResult.state_space_too_large`),
# estimate the SAME quantity by actually sampling the REAL candidate-
# generation path -- reusing `scenario_generator.generate_scenario()`
# and `scenario_validator.validate_navigation()` VERBATIM is what makes
# this an honest estimate of the real distribution, not a second,
# independently-authored approximation of Generator/Validator
# semantics (the task's own explicit "do not duplicate" instruction).
# Importing these packages to CALL their existing public functions is
# not the same thing as MODIFYING them -- neither is edited by this
# module, confirmed in the accompanying implementation report.
#
# THE CANDIDATE-GENERATION SEAM (traced fresh this task, not assumed):
# `scenario_generator.generate_scenario(request: GenerationRequest) ->
# Scenario` (scenario_generator/generator.py) is a pure, deterministic
# function of `(definition, building, seed, attempt_index)` -- "always
# returns a candidate Scenario... whether that candidate is acceptable
# is entirely the Scenario Validator's concern" (its own docstring). It
# performs NO accept/reject branching, NO retry loop, NO duplicate-hash
# bookkeeping, and NO filesystem I/O of any kind (re-verified this
# task: `metadata_builder.py`'s own hashing is in-memory `json.dumps`,
# never a file write) -- exactly the "one freshly generated candidate
# before acceptance/rejection" primitive this phase needs, with zero
# campaign side effects, because `scenario_pipeline.run_pipeline()`
# itself is built from exactly these same two calls
# (`scenario_pipeline/pipeline.py:64-76`: `generate_scenario(request)`
# then `validate(...)`) plus the RETRY LOOP and DUPLICATE-HASH state
# this phase must specifically avoid reusing (see below).
#
# RNG INDEPENDENCE: every category's RNG stream
# (`scenario_generator/seed_manager.py::category_rng()`) is a fresh
# `random.Random` instance seeded via `derive_seed(attempt_seed,
# "category", category_key)` -- a SHA256-derived, index/name-keyed hash
# -- and `attempt_seed = derive_attempt_seed(request.seed, request.
# attempt_index)`, itself SHA256-derived from `(request.seed,
# request.attempt_index)`. `generate_scenario()` therefore NEVER
# touches the global `random` module's state (re-verified this task,
# grep-confirmed: this whole call chain only ever constructs local
# `random.Random(...)` instances) -- satisfying the task's own "do not
# interfere with global RNG state" requirement for free, simply by
# reusing this seam rather than reimplementing it.
#
# THIS MODULE'S OWN SEED NAMESPACE: `MonteCarloConfig.seed` is an
# analysis-only integer, deliberately never fed into
# `derive_scenario_seed()` (the REAL campaign master-seed-to-scenario-
# seed derivation) -- instead, `base_seed = derive_seed(config.seed,
# "campaign_feasibility_monte_carlo_candidate_validity")` (a distinct,
# fixed label no real campaign scenario/attempt seed derivation ever
# uses) produces this analysis's own private "scenario seed" analogue,
# and each of the `n` independent samples reuses the SAME mechanism a
# real campaign already uses for RETRY attempts within one slot:
# `attempt_index = sample_index` (0, 1, 2, ...), passed straight into
# `GenerationRequest`/`derive_attempt_seed()` exactly as `scenario_
# pipeline.run_pipeline()`'s own attempt loop already does
# (`pipeline.py:64-69`). This is not a new independence mechanism --
# it is the SAME one the real system already relies on for "retry N+1
# never reuses retry N's random draws" (re-confirmed, Phase 2's own
# investigation, unchanged), simply pointed at this analysis's own
# private seed rather than any real campaign's master seed -- so it
# can never collide with, or be confused for, a real generated
# scenario's own provenance.
#
# WHY validate_navigation() DIRECTLY, NOT THE FULL validate(): Phase
# 2A/2B's own exact analysis is explicitly, deliberately scoped to
# NAVIGATION-category validity only (fire origin / door / exit / stair
# reachability / min_open_exits / occupancy) -- "P(candidate valid)"
# throughout this whole thread's own architecture means P(passes
# NAVIGATION), never the full six-category Scenario Validator
# acceptance. Calling the full `scenario_validator.validate()` would
# (a) measure a DIFFERENT quantity than what this fallback is a
# fallback FOR, and (b) require an `accepted_hashes` argument for its
# own DATASET/duplicate-content check -- a campaign-run-history-
# dependent concept with no meaning for "one freshly generated
# candidate in isolation" (this task's own explicit scope). Calling
# `scenario_validator.navigation_validation.validate_navigation(
# candidate, definition, building)` directly measures EXACTLY the same
# quantity `compute_exact_candidate_validity()` computes analytically,
# with no duplicate-rejection state, no retry loop, and no campaign
# mutation of any kind -- confirmed the smallest, most honest seam for
# this specific fallback.


_ANALYSIS_DEFINITION_ID = "campaign-feasibility-monte-carlo-analysis"


@dataclass(frozen=True)
class MonteCarloConfig:

    # `seed` is this analysis's own, private reproducibility key (see
    # module docstring) -- never a real campaign's master seed.
    seed: int = 0

    minimum_samples: int = 200
    maximum_samples: int = 5000

    target_interval_half_width: float = 0.02
    confidence_level: float = 0.95

    # =====================================================

    def __post_init__(self):

        # Matches this repository's own established config-validation
        # convention for plain call-parameter objects (e.g. `scenario_
        # pipeline.run_pipeline()`'s own `if max_attempts < 1: raise
        # ValueError(...)`) -- a direct `ValueError`, not a report
        # object (that convention is reserved for user-authored
        # ScenarioDefinition content, `scenario_definition.validation`,
        # which this config is not).

        if self.minimum_samples <= 0:
            raise ValueError(f"minimum_samples must be > 0, got {self.minimum_samples!r}.")

        if self.maximum_samples < self.minimum_samples:
            raise ValueError(
                f"maximum_samples ({self.maximum_samples!r}) must be >= minimum_samples "
                f"({self.minimum_samples!r})."
            )

        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError(f"confidence_level must be strictly between 0 and 1, got {self.confidence_level!r}.")

        if self.target_interval_half_width <= 0.0:
            raise ValueError(
                f"target_interval_half_width must be > 0, got {self.target_interval_half_width!r}."
            )


@dataclass(frozen=True)
class MonteCarloCandidateValidityResult:

    # The Phase 2C fallback result -- a deliberately SEPARATE type from
    # `CandidateValidityResult` (never claims `exact=True`, never
    # reuses that field name for a different meaning) so a caller can
    # never mistake an ESTIMATE for an exact figure merely by duck-
    # typing on a shared boolean. See `CandidateValidityAnalysis` below
    # for the unified read surface across both.

    estimated_p_valid: float

    confidence_lower: float
    confidence_upper: float
    confidence_level: float

    samples_run: int
    valid_samples: int
    invalid_samples: int

    # True iff the Wilson interval's own half-width reached `target_
    # interval_half_width` before `maximum_samples` was exhausted --
    # kept as its own explicit field (never inferred from `samples_run
    # < maximum_samples`, which would be ambiguous whenever the target
    # happens to be met on the very last allowed sample) so "precision
    # target reached" and "maximum samples reached first" are never
    # collapsed into one ambiguous signal.
    precision_target_met: bool
    target_interval_half_width: float

    minimum_samples: int
    maximum_samples: int

    # This analysis's own reproducibility key (see module docstring) --
    # NOT a real campaign seed.
    seed: int

    warnings: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CandidateValidityAnalysis:

    # The orchestrated "exact, falling back to Monte Carlo only when
    # tractability requires it" result -- the task's own diagram made
    # concrete. `exact_result` is ALWAYS present (Layer 1 is always
    # attempted first); `monte_carlo_result` is present if and only if
    # `exact_result.state_space_too_large` triggered the fallback.
    # Never collapses the two into one ambiguous shape -- both remain
    # separately inspectable.

    exact_result: CandidateValidityResult
    monte_carlo_result: Optional[MonteCarloCandidateValidityResult] = None

    # =====================================================

    @property
    def used_monte_carlo(self) -> bool:

        return self.monte_carlo_result is not None

    # =====================================================

    @property
    def p_valid(self) -> Optional[float]:

        if self.monte_carlo_result is not None:
            return self.monte_carlo_result.estimated_p_valid

        return self.exact_result.p_valid

    # =====================================================

    @property
    def exact(self) -> bool:

        # Never True merely because a NUMBER is available -- an
        # estimate is never exact, regardless of how tight its
        # interval is or whether its own precision target was met.
        return self.monte_carlo_result is None and self.exact_result.exact


# =====================================================
# Wilson score interval -- chosen (over a normal/Wald approximation
# interval) specifically because it stays well-behaved at k=0 and
# k=n, exactly the task's own named requirement. Standard closed form
# (Wilson 1927; re-derived and hand-verified against known reference
# values in the accompanying implementation report/tests, not merely
# assumed correct from a citation):
#
#   p_hat = k / n
#   z     = two-sided standard normal quantile for `confidence_level`
#   denom = 1 + z^2/n
#   center     = (p_hat + z^2/(2n)) / denom
#   half_width = (z/denom) * sqrt(p_hat(1-p_hat)/n + z^2/(4n^2))
#   [lower, upper] = [center - half_width, center + half_width]
#
# `z` is obtained via `scipy.stats.norm.ppf()` (scipy is already a
# repository dependency -- `validation_framework/statistics.py` and
# `research_framework/statistics.py` already import `scipy.stats` the
# same way) rather than a hand-rolled inverse-erf routine, so this
# supports an arbitrary CONFIGURABLE `confidence_level`, not just a
# hardcoded 95% table entry.
# =====================================================


def _wilson_interval(k: int, n: int, confidence_level: float) -> Tuple[float, float]:

    if n <= 0:
        return (0.0, 1.0)

    z = float(scipy_stats.norm.ppf((1.0 + confidence_level) / 2.0))
    z2 = z * z

    p_hat = k / n
    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denominator
    half_width = (z / denominator) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))

    lower = max(0.0, min(1.0, center - half_width))
    upper = max(0.0, min(1.0, center + half_width))

    return lower, upper


# =====================================================
# The Monte Carlo sampling loop.
# =====================================================


def estimate_candidate_validity_monte_carlo(
    building, definition, config: Optional[MonteCarloConfig] = None,
) -> MonteCarloCandidateValidityResult:

    config = config or MonteCarloConfig()

    if building is None:

        return MonteCarloCandidateValidityResult(
            estimated_p_valid=0.0, confidence_lower=0.0, confidence_upper=1.0,
            confidence_level=config.confidence_level,
            samples_run=0, valid_samples=0, invalid_samples=0,
            precision_target_met=False,
            target_interval_half_width=config.target_interval_half_width,
            minimum_samples=config.minimum_samples, maximum_samples=config.maximum_samples,
            seed=config.seed,
            warnings=("No Building was supplied -- nothing to sample.",),
        )

    # This analysis's own private "scenario seed" analogue -- see the
    # module docstring's "THIS MODULE'S OWN SEED NAMESPACE" section for
    # why this can never collide with, or be mistaken for, a real
    # campaign's own seed derivation.
    base_seed = derive_seed(config.seed, "campaign_feasibility_monte_carlo_candidate_validity")

    valid_count = 0
    samples_run = 0
    precision_target_met = False

    for sample_index in range(config.maximum_samples):

        # Each `sample_index` is its own, independent Attempt Seed
        # (`base_seed` never changes) -- the SAME index-keyed
        # independence mechanism `scenario_pipeline.run_pipeline()`'s
        # own retry loop already relies on (module docstring), so
        # sample N+1 never reuses sample N's random draws, and no
        # sample is ever reused/replayed as if it were a fresh
        # independent draw.
        request = GenerationRequest(
            definition=definition, definition_id=_ANALYSIS_DEFINITION_ID, building=building,
            seed=base_seed, attempt_index=sample_index,
        )

        candidate = generate_scenario(request)
        report = validate_navigation(candidate, definition, building)

        samples_run += 1

        if report.accepted:
            valid_count += 1

        if samples_run >= config.minimum_samples:

            lower, upper = _wilson_interval(valid_count, samples_run, config.confidence_level)

            if (upper - lower) / 2.0 <= config.target_interval_half_width:
                precision_target_met = True
                break

    lower, upper = _wilson_interval(valid_count, samples_run, config.confidence_level)
    estimated_p_valid = (valid_count / samples_run) if samples_run > 0 else 0.0

    warnings = [
        "Estimated via Monte Carlo sampling of the real scenario_generator.generate_"
        "scenario() + scenario_validator.navigation_validation.validate_navigation() -- "
        "scoped to the SAME NAVIGATION-only quantity the exact analysis computes; "
        "duplicate-content rejection is deliberately not applied (no meaning for one "
        "freshly generated candidate in isolation -- see the accompanying implementation "
        "report).",
    ]

    if not precision_target_met:
        warnings.append(
            f"Reached maximum_samples ({config.maximum_samples}) before the target "
            f"confidence-interval half-width ({config.target_interval_half_width}) was "
            f"achieved -- the reported interval is wider than requested.",
        )

    return MonteCarloCandidateValidityResult(
        estimated_p_valid=estimated_p_valid,
        confidence_lower=lower, confidence_upper=upper,
        confidence_level=config.confidence_level,
        samples_run=samples_run, valid_samples=valid_count, invalid_samples=samples_run - valid_count,
        precision_target_met=precision_target_met,
        target_interval_half_width=config.target_interval_half_width,
        minimum_samples=config.minimum_samples, maximum_samples=config.maximum_samples,
        seed=config.seed,
        warnings=tuple(warnings),
    )


# =====================================================
# Fallback routing -- "attempt exact, fall back to Monte Carlo only
# when tractability requires it."
# =====================================================


def compute_candidate_validity(
    building, definition, max_states: int = DEFAULT_MAX_ENUMERATED_STATES,
    monte_carlo_config: Optional[MonteCarloConfig] = None,
) -> CandidateValidityAnalysis:

    exact_result = compute_exact_candidate_validity(building, definition, max_states=max_states)

    if not exact_result.state_space_too_large:

        # Covers BOTH "exact analysis succeeded" AND "a deterministic
        # impossibility/degenerate case Phase 1/2A already proved
        # (e.g. every guaranteed-occupied zone unreachable, every
        # fire-eligible zone LETHAL)" -- neither ever sets `state_
        # space_too_large`, so Monte Carlo is never invoked for either,
        # exactly matching the task's own explicit routing rule. A
        # user wanting an approximate analysis regardless can always
        # call `estimate_candidate_validity_monte_carlo()` directly
        # (a public function) -- no separate "force" flag is needed on
        # this orchestration entry point for that capability to exist.
        return CandidateValidityAnalysis(exact_result=exact_result, monte_carlo_result=None)

    monte_carlo_result = estimate_candidate_validity_monte_carlo(
        building, definition, monte_carlo_config or MonteCarloConfig(),
    )

    return CandidateValidityAnalysis(exact_result=exact_result, monte_carlo_result=monte_carlo_result)
