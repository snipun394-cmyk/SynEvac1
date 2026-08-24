import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from scenario_definition.distributions import UniformRange

from campaign_feasibility.analysis import _can_be_occupied
from campaign_feasibility.exact_validity import CandidateValidityResult
from campaign_feasibility.monte_carlo import CandidateValidityAnalysis, MonteCarloCandidateValidityResult


# Scenario Campaign Feasibility Preflight -- Campaign-Level Yield
# Analysis (the layer this thread's own acceptance/uniqueness
# investigation calls "Layer 2").
# docs/architecture/scenario_campaign_feasibility_phase2_investigation.txt
# docs/architecture/scenario_campaign_acceptance_and_uniqueness_investigation.txt
# docs/architecture/scenario_campaign_feasibility_phase2a_implementation_report.txt
# docs/architecture/scenario_campaign_feasibility_phase2b_implementation_report.txt
# docs/architecture/scenario_campaign_feasibility_min_open_exits_occupancy_implementation_report.txt
# docs/architecture/scenario_campaign_yield_analysis_implementation_report.txt
#
# Translates `CandidateValidityResult.p_valid` (Layer 1, exact_validity.py
# -- read here, never recomputed) into predictions about the REAL
# campaign generation loop (`designer/campaign/campaign_worker.py`'s
# `CampaignWorker.execute()`): `for index in range(config.count):`
# (one "slot" per index) each running `for attempt_index in range(
# config.max_attempts):` (traced fresh this task, not assumed) until
# either a candidate is accepted or the attempt budget is exhausted,
# after which the campaign moves on to the next slot regardless (no
# cross-slot retry pooling, no whole-campaign abort on a single slot's
# failure -- confirmed against the real code).
#
# Per-attempt independence: every (slot index, attempt index) pair
# derives its own RNG seed via `derive_attempt_seed(derive_scenario_
# seed(master_seed, index), attempt_index)`, itself a SHA256-derived,
# index-keyed (never stream-position-keyed) hash
# (scenario_generator/seed_manager.py) -- re-confirmed this task,
# unchanged from the Phase 2 investigation's own independence proof.
# This is what makes "P(candidate valid)" the SAME fixed probability
# `p` on every attempt of every slot, and what makes attempts within
# one slot -- and slots themselves -- genuinely independent Bernoulli
# trials, PROVIDED duplicate-content rejection never applies (see
# `_duplicate_rejection_is_negligible()` below, and the accompanying
# implementation report's Section 9/10) -- duplicate rejection is the
# ONE mechanism that can break this independence, because it makes a
# slot's success probability depend on which specific values earlier
# slots happened to accept (`accepted_hashes`, campaign_worker.py),
# not merely on `p` alone.
#
# This module itself never imports scenario_generator or scenario_
# validator, never samples, never consumes RNG -- every probability
# here is computed analytically from `candidate_validity.p_valid`
# (whether that came from Layer 1's own exact analysis or, as of Phase
# 2C, its Monte Carlo fallback -- see docs/architecture/scenario_
# campaign_feasibility_phase2c_monte_carlo_implementation_report.txt)
# and from the Definition's own declared distributions (a cheap,
# structural presence/absence check, never a second enumeration engine
# -- see `_duplicate_rejection_is_negligible()`). Phase 2C's own
# `campaign_feasibility.monte_carlo` module (imported below, for its
# RESULT TYPES only -- never for sampling logic) is the one place in
# this package that DOES import scenario_generator/scenario_validator,
# by necessity (see that module's own docstring for why).


def _is_continuous(distribution) -> bool:

    # A distribution contributes an effectively-uncountable value space
    # only as a continuous UniformRange (discrete=False, the default) --
    # FixedValue and WeightedOptions are always finite by construction,
    # and a discrete UniformRange is finite too (re-derived, matching
    # the acceptance/uniqueness investigation's own Part 4 classification).
    return isinstance(distribution, UniformRange) and not distribution.discrete


def _duplicate_rejection_is_negligible(definition) -> bool:

    # Mirrors, exactly, the four conditions the acceptance/uniqueness
    # investigation proved jointly necessary for duplicate-content
    # collision risk to be non-negligible (docs/architecture/scenario_
    # campaign_acceptance_and_uniqueness_investigation.txt, Part 4) --
    # returns True (negligible, the common/normal case) the instant ANY
    # ONE of them is violated, exactly matching that report's own "if
    # ANY of these does not hold, K is effectively unbounded" framing.
    # A cheap, purely structural read of the Definition -- no Building,
    # no NavigationGraph, no enumeration of any kind, matching the
    # investigation's own Part 5 "Layer 3 requires NO enumeration at
    # all" finding for this (the common) branch.

    # (1) Zero occupants ever placed?
    if any(
        _can_be_occupied(distribution)
        for distribution in definition.occupant.occupancy_distribution.values()
    ):
        return True

    # (2) Zero firefighters ever deployed?
    if definition.firefighter.entry_zone_ids and _can_be_occupied(
        definition.firefighter.team_count_distribution,
    ):
        return True

    # (3) Fire growth continuously parameterized?
    if _is_continuous(definition.fire.growth_parameter_distribution):
        return True

    # (4) Any event template's own time continuously parameterized?
    if any(_is_continuous(template.time) for template in definition.event_templates):
        return True

    return False


# =====================================================
# Numerically stable Binomial building blocks -- `accepted ~
# Binomial(requested_count, p_slot_success)` whenever slots are i.i.d.
# (see `CampaignYieldResult.slots_independent`). Computed in LOG SPACE
# via `math.lgamma` (never `math.comb(n, k)` multiplied directly into a
# float) specifically because `math.comb(n, k)` is an exact Python
# bigint that, for large n with k near n/2, has far more digits than a
# float64 can represent -- converting it directly would raise
# `OverflowError` long before the actual PROBABILITY becomes too small
# to represent, which is the failure mode this task's own "avoid
# factorial overflow" instruction warns against. Log-space evaluation
# only ever exponentiates the final, already-bounded log-probability,
# so it never forms an intermediate value larger than a probability
# itself could ever be.
# =====================================================


def _binomial_pmf(n: int, k: int, s: float) -> float:

    if n < 0 or k < 0 or k > n:
        return 0.0

    if s <= 0.0:
        return 1.0 if k == 0 else 0.0

    if s >= 1.0:
        return 1.0 if k == n else 0.0

    log_coefficient = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    log_pmf = log_coefficient + k * math.log(s) + (n - k) * math.log(1.0 - s)

    if log_pmf < -700.0:
        # Underflows float64's smallest representable positive value
        # (~5e-324, ln of which is ~-744) -- the TRUE probability is
        # genuinely, correctly indistinguishable from 0.0 at this
        # scale; this is not an error condition.
        return 0.0

    return max(0.0, min(1.0, math.exp(log_pmf)))


def _binomial_survival(n: int, target: int, s: float) -> float:

    # P(X >= target), X ~ Binomial(n, s). Sums whichever of the two
    # complementary tails (k = target..n, or k = 0..target-1) is
    # SHORTER, via `_binomial_pmf()` -- cost is O(min(target, n -
    # target + 1)) point evaluations, each O(1) (a handful of `lgamma`/
    # `log`/`exp` calls) -- never O(2^n), never a single unbounded
    # intermediate value, and never a full-range PMF array materialized
    # for its own sake.

    if target <= 0:
        return 1.0

    if target > n:
        return 0.0

    upper_tail_size = n - target + 1

    if upper_tail_size <= target:
        total = sum(_binomial_pmf(n, k, s) for k in range(target, n + 1))
        return max(0.0, min(1.0, total))

    total = sum(_binomial_pmf(n, k, s) for k in range(0, target))
    return max(0.0, min(1.0, 1.0 - total))


# =====================================================
# The campaign-level result model.
# =====================================================


@dataclass(frozen=True)
class CampaignYieldResult:

    # A NEW, separate result type from `CandidateValidityResult` (Layer
    # 1) -- this answers a CAMPAIGN-wide "how many accepted scenarios
    # should we expect from N requested slots" question, built ON TOP
    # OF a `CandidateValidityResult` (read, never recomputed) plus the
    # REAL retry semantics (`requested_count`/`max_attempts`, the same
    # two numbers `CampaignConfig.count`/`CampaignConfig.max_attempts`
    # already carry -- this module deliberately does not import
    # `designer.campaign.campaign_worker.CampaignConfig` itself, to
    # keep `campaign_feasibility/` decoupled from `designer/`, exactly
    # like `compute_exact_candidate_validity()`'s own plain int
    # parameters already establish as this package's convention).

    requested_count: int
    max_attempts: int

    # Layer 1's own result, exposed verbatim (never recalculated) --
    # `None` whenever Layer 1 itself could not produce an exact
    # probability (state-space too large, or no Building supplied).
    p_candidate_valid: Optional[float] = None

    p_slot_success: Optional[float] = None
    p_slot_failure: Optional[float] = None
    expected_attempts_per_slot: Optional[float] = None

    expected_accepted: Optional[float] = None
    p_complete_success: Optional[float] = None
    p_underproduction: Optional[float] = None

    # True iff campaign slots are genuinely independent, identically
    # distributed Bernoulli(p_slot_success) trials under the REAL
    # generation architecture -- i.e. iff `_duplicate_rejection_is_
    # negligible()` held for the analyzed Definition. Every probability
    # on this result (including `p_accepted_equals()`/`p_accepted_at_
    # least()` below) is computed AS IF this were true; when it is
    # False, those numbers are the acceptance/uniqueness investigation's
    # own "excellent approximation, not exact" figures (Part 9, Case 1
    # framing) at best, and potentially materially wrong (Part 9, Case
    # 2's "at least count-K slots are GUARANTEED to fail" finding) at
    # worst -- ALWAYS accompanied by `exact=False`, `duplicate_
    # rejection_risk=True`, and an explicit warning in that case; never
    # silently presented as exact.
    slots_independent: bool = False

    # True iff the analyzed Definition falls into the narrow, provable
    # degenerate configuration (docs/architecture/scenario_campaign_
    # acceptance_and_uniqueness_investigation.txt, Part 4) where
    # duplicate-content rejection is NOT structurally negligible --
    # i.e. `not slots_independent`, kept as its own explicitly-named
    # field (rather than forcing callers to infer it from `not slots_
    # independent`) specifically so a caller can distinguish WHY a
    # result is not exact (this reason, vs. Layer 1's own state-space
    # limit, vs. both) without string-matching `warnings`.
    duplicate_rejection_risk: bool = False

    # Layer 1's own `exact` flag, exposed verbatim for transparency --
    # this campaign-level result can never be MORE exact than Layer 1
    # was.
    candidate_validity_exact: bool = False

    # True only when EVERY source of randomness this result's own
    # numbers could depend on was included: Layer 1 was itself exact
    # (`candidate_validity_exact`) AND slots are genuinely i.i.d.
    # (`slots_independent`, i.e. duplicate rejection is negligible).
    exact: bool = False

    # "exact" or "monte_carlo" -- which Layer 1 result this campaign-
    # level result was derived from. Kept as an explicit string rather
    # than forcing a caller to infer it from `monte_carlo_source is not
    # None` (equivalent, but this is self-documenting in isolation).
    candidate_validity_source: str = "exact"

    # The full Phase 2C Monte Carlo result this analysis was derived
    # from, when `candidate_validity_source == "monte_carlo"` -- `None`
    # otherwise. Carried verbatim (never re-derived into separate,
    # same-meaning fields on THIS class -- see the accompanying
    # implementation report's "avoid redundant metrics" discipline) so
    # a caller can inspect `samples_run`/`confidence_level`/
    # `precision_target_met`/etc. directly from the source.
    monte_carlo_source: Optional["MonteCarloCandidateValidityResult"] = None

    # Interval-propagated bounds -- populated ONLY when this result was
    # derived from a Monte Carlo estimate AND `slots_independent` is
    # True (propagating an interval through a model that is already
    # not exactly correct, Case 2's duplicate-rejection scenario, would
    # compound two different kinds of error into one number -- omitted
    # entirely in that case, per the task's own "clearly omit interval
    # bounds" guidance, rather than presented as if equally trustworthy).
    # Each pair is the monotonic image of [confidence_lower, confidence_
    # upper] through the correspondingly-named point formula -- see
    # `_compute_yield_point()` and the accompanying implementation
    # report's monotonicity proofs for `p_slot_success`/`expected_
    # accepted`/`p_complete_success` (increasing in p) and `p_under
    # production` (DECREASING in p -- its lower/upper bounds are the
    # correspondingly SWAPPED endpoints, not a copy-paste of the other
    # three).
    p_slot_success_lower: Optional[float] = None
    p_slot_success_upper: Optional[float] = None
    expected_accepted_lower: Optional[float] = None
    expected_accepted_upper: Optional[float] = None
    p_complete_success_lower: Optional[float] = None
    p_complete_success_upper: Optional[float] = None
    p_underproduction_lower: Optional[float] = None
    p_underproduction_upper: Optional[float] = None

    warnings: Tuple[str, ...] = field(default_factory=tuple)

    # =====================================================

    def p_accepted_at_least_interval(self, target: int) -> Optional[Tuple[float, float]]:

        # A bonus capability beyond the three metrics the task
        # explicitly requires interval propagation for (slot success,
        # expected accepted, complete success): P(accepted >= target)
        # is ALSO monotonic non-decreasing in `p_slot_success` for any
        # fixed `target` -- a standard stochastic-dominance fact about
        # Binomial(n, s) as a function of s (a higher per-trial success
        # probability can only shift probability mass toward larger
        # counts, pathwise, never the reverse) -- so propagating
        # through `p_slot_success_lower`/`_upper` is mathematically
        # valid here too, unlike `p_accepted_equals()` (Section 7 of
        # the accompanying implementation report explains why an exact-
        # k pmf query is NOT monotonic in p and is therefore never
        # given interval bounds).
        if self.p_slot_success_lower is None or self.p_slot_success_upper is None:
            return None

        return (
            _binomial_survival(self.requested_count, target, self.p_slot_success_lower),
            _binomial_survival(self.requested_count, target, self.p_slot_success_upper),
        )

    # =====================================================

    def p_accepted_equals(self, k: int) -> Optional[float]:

        # Exact (given `slots_independent`) P(accepted == k),
        # `accepted ~ Binomial(requested_count, p_slot_success)`.
        if self.p_slot_success is None:
            return None

        return _binomial_pmf(self.requested_count, k, self.p_slot_success)

    # =====================================================

    def p_accepted_at_least(self, target: int) -> Optional[float]:

        if self.p_slot_success is None:
            return None

        return _binomial_survival(self.requested_count, target, self.p_slot_success)

    # =====================================================

    def p_accepted_at_most(self, target: int) -> Optional[float]:

        at_least = self.p_accepted_at_least(target + 1)

        if at_least is None:
            return None

        return max(0.0, min(1.0, 1.0 - at_least))


# =====================================================
# The point-estimate formulas -- extracted into one shared helper so
# EXACT analysis (called once, at `p_hat`) and Monte Carlo analysis
# (called three times, at `p_hat`/`confidence_lower`/`confidence_upper`,
# for the point estimate and its propagated bounds respectively) never
# duplicate this arithmetic. See the original Phase 2 (exact-only)
# implementation report for the full derivation of each formula --
# unchanged here, only relocated.
# =====================================================


@dataclass(frozen=True)
class _YieldPoint:

    p_slot_success: float
    p_slot_failure: float
    expected_attempts_per_slot: float
    expected_accepted: float
    p_complete_success: float
    p_underproduction: float


def _compute_yield_point(p: float, count: int, max_attempts: int) -> _YieldPoint:

    p = max(0.0, min(1.0, p))

    # `slot success` = at least one of up to `max_attempts` i.i.d.
    # Bernoulli(p) attempts succeeds -- a truncated geometric process.
    # `E[attempts actually made] = (1 - (1-p)^A) / p` for p > 0 (the
    # standard truncated-geometric expectation, NOT the unbounded
    # `1/p`); `= A` when p == 0 (every attempt is always consumed,
    # handled as an explicit special case purely to avoid a 0/0
    # division).
    if max_attempts <= 0:
        p_slot_success = 0.0
        expected_attempts = 0.0
    elif p <= 0.0:
        p_slot_success = 0.0
        expected_attempts = float(max_attempts)
    elif p >= 1.0:
        p_slot_success = 1.0
        expected_attempts = 1.0
    else:
        q = 1.0 - p
        q_pow_a = q ** max_attempts
        p_slot_success = max(0.0, min(1.0, 1.0 - q_pow_a))
        expected_attempts = (1.0 - q_pow_a) / p

    p_slot_failure = max(0.0, min(1.0, 1.0 - p_slot_success))
    expected_accepted = count * p_slot_success
    p_complete_success = _binomial_pmf(count, count, p_slot_success)
    p_underproduction = max(0.0, min(1.0, 1.0 - p_complete_success))

    return _YieldPoint(
        p_slot_success=p_slot_success, p_slot_failure=p_slot_failure,
        expected_attempts_per_slot=expected_attempts, expected_accepted=expected_accepted,
        p_complete_success=p_complete_success, p_underproduction=p_underproduction,
    )


_DUPLICATE_RISK_WARNING = (
    "This Definition matches the narrow degenerate configuration (no continuous "
    "occupant/firefighter/fire-growth/event-time entropy) in which duplicate-"
    "content rejection can materially deplete the pool of distinct valid content "
    "across slots -- campaign slots are then NOT independent, identically "
    "distributed trials, and the reported yield figures (computed AS IF they "
    "were, the standard Binomial model) may overstate actual yield, especially "
    "for a requested count approaching or exceeding the number of distinct valid "
    "outcomes. Duplicate-aware modeling is not implemented here."
)


# =====================================================
# The main entry point.
# =====================================================


def analyze_campaign_yield(
    candidate_validity, definition, count: int, max_attempts: int,
) -> CampaignYieldResult:

    # Accepts a bare `CandidateValidityResult` (Layer 1, exact --
    # unchanged behavior, byte-identical to every prior phase), a bare
    # `MonteCarloCandidateValidityResult` (Phase 2C, estimated), or the
    # orchestrated `CandidateValidityAnalysis` wrapper (unwrapped to
    # whichever of the two it actually carried) -- see the accompanying
    # Phase 2C implementation report for why this is additive, not a
    # breaking signature change: every existing caller passing a bare
    # `CandidateValidityResult` sees IDENTICAL behavior to before this
    # phase.
    if isinstance(candidate_validity, CandidateValidityAnalysis):
        candidate_validity = (
            candidate_validity.monte_carlo_result
            if candidate_validity.monte_carlo_result is not None
            else candidate_validity.exact_result
        )

    if isinstance(candidate_validity, MonteCarloCandidateValidityResult):
        return _analyze_campaign_yield_monte_carlo(candidate_validity, definition, count, max_attempts)

    return _analyze_campaign_yield_exact(candidate_validity, definition, count, max_attempts)


def _analyze_campaign_yield_exact(
    candidate_validity: CandidateValidityResult, definition, count: int, max_attempts: int,
) -> CampaignYieldResult:

    p = candidate_validity.p_valid

    if p is None:

        return CampaignYieldResult(
            requested_count=count, max_attempts=max_attempts,
            p_candidate_valid=None,
            candidate_validity_exact=candidate_validity.exact,
            exact=False,
            candidate_validity_source="exact",
            warnings=(
                "Candidate validity analysis did not produce an exact probability (state-"
                "space too large, or no Building was supplied) -- campaign-level yield "
                "cannot be computed from it.",
            ),
        )

    point = _compute_yield_point(p, count, max_attempts)

    slots_independent = _duplicate_rejection_is_negligible(definition)

    warnings = [] if slots_independent else [_DUPLICATE_RISK_WARNING]

    exact = bool(candidate_validity.exact) and slots_independent

    return CampaignYieldResult(
        requested_count=count, max_attempts=max_attempts,
        p_candidate_valid=max(0.0, min(1.0, p)),
        p_slot_success=point.p_slot_success, p_slot_failure=point.p_slot_failure,
        expected_attempts_per_slot=point.expected_attempts_per_slot,
        expected_accepted=point.expected_accepted,
        p_complete_success=point.p_complete_success,
        p_underproduction=point.p_underproduction,
        slots_independent=slots_independent,
        duplicate_rejection_risk=not slots_independent,
        candidate_validity_exact=bool(candidate_validity.exact),
        exact=exact,
        candidate_validity_source="exact",
        warnings=tuple(warnings),
    )


def _analyze_campaign_yield_monte_carlo(
    monte_carlo: MonteCarloCandidateValidityResult, definition, count: int, max_attempts: int,
) -> CampaignYieldResult:

    point = _compute_yield_point(monte_carlo.estimated_p_valid, count, max_attempts)

    slots_independent = _duplicate_rejection_is_negligible(definition)

    warnings = [
        f"Candidate validity is an ESTIMATE from Monte Carlo sampling "
        f"({monte_carlo.samples_run} samples, {monte_carlo.confidence_level:.0%} Wilson CI "
        f"[{monte_carlo.confidence_lower:.4f}, {monte_carlo.confidence_upper:.4f}]), not an "
        f"exact probability -- every campaign-level figure below is itself an estimate, "
        f"never exact.",
    ]

    if not monte_carlo.precision_target_met:
        warnings.append(
            "The Monte Carlo analysis reached its maximum_samples limit before its own "
            "target confidence-interval precision was achieved -- this estimate's "
            "uncertainty may be wider than requested.",
        )

    bounds = {}

    if slots_independent:

        # p_slot_success / expected_accepted / p_complete_success are
        # each monotonic INCREASING in p (proven in the accompanying
        # implementation report), so the interval [confidence_lower,
        # confidence_upper] maps directly to [lower_point.X,
        # upper_point.X] for each. p_underproduction = 1 -
        # p_complete_success is monotonic DECREASING in p -- its
        # bounds are the two points' values with lower/upper SWAPPED,
        # not a copy of the other three's own pattern.
        lower_point = _compute_yield_point(monte_carlo.confidence_lower, count, max_attempts)
        upper_point = _compute_yield_point(monte_carlo.confidence_upper, count, max_attempts)

        bounds = dict(
            p_slot_success_lower=lower_point.p_slot_success,
            p_slot_success_upper=upper_point.p_slot_success,
            expected_accepted_lower=lower_point.expected_accepted,
            expected_accepted_upper=upper_point.expected_accepted,
            p_complete_success_lower=lower_point.p_complete_success,
            p_complete_success_upper=upper_point.p_complete_success,
            p_underproduction_lower=upper_point.p_underproduction,
            p_underproduction_upper=lower_point.p_underproduction,
        )

    else:
        warnings.append(_DUPLICATE_RISK_WARNING)

    return CampaignYieldResult(
        requested_count=count, max_attempts=max_attempts,
        p_candidate_valid=monte_carlo.estimated_p_valid,
        p_slot_success=point.p_slot_success, p_slot_failure=point.p_slot_failure,
        expected_attempts_per_slot=point.expected_attempts_per_slot,
        expected_accepted=point.expected_accepted,
        p_complete_success=point.p_complete_success,
        p_underproduction=point.p_underproduction,
        slots_independent=slots_independent,
        duplicate_rejection_risk=not slots_independent,
        candidate_validity_exact=False,
        exact=False,
        candidate_validity_source="monte_carlo",
        monte_carlo_source=monte_carlo,
        warnings=tuple(warnings),
        **bounds,
    )
