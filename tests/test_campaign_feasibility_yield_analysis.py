import time
import unittest

from scenario_definition import (
    EventTemplate,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
    UniformRange,
    WeightedOptions,
)
from scenario_definition.firefighter_definition import FirefighterDeploymentDefinition

from campaign_feasibility import CandidateValidityResult, CampaignYieldResult, analyze_campaign_yield


# =====================================================
# Campaign-Level Yield Analysis (Layer 2) -- translates Layer 1's
# `CandidateValidityResult.p_valid` into predictions about the REAL
# campaign generation/retry loop (`designer/campaign/campaign_worker.py`).
# See docs/architecture/scenario_campaign_yield_analysis_implementation_
# report.txt for the full derivation each test below is hand-checked
# against.
# =====================================================


def _candidate(p_valid, exact=True, exact_for_analyzed_dimensions=True, **overrides):

    defaults = dict(
        exact=exact, exact_for_analyzed_dimensions=exact_for_analyzed_dimensions, p_valid=p_valid,
    )
    defaults.update(overrides)
    return CandidateValidityResult(**defaults)


def _definition_with_occupants():

    # A NORMAL (non-degenerate) Definition -- at least one zone that
    # can be occupied, so continuous occupant POSITION sampling makes
    # duplicate-content collision negligible (the common case).
    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-1": FixedValue(1)},
            behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
        ),
    )


def _fully_degenerate_definition():

    # Matches ALL FOUR conditions the acceptance/uniqueness
    # investigation proved jointly necessary for duplicate-content
    # collision risk to be non-negligible: zero occupants, zero
    # firefighters, a FixedValue fire growth parameter, and no
    # continuously-timed event templates.
    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
        occupant=OccupantDefinition(),
        firefighter=FirefighterDeploymentDefinition(),
    )


# =====================================================
# 1 -- p = 1
# =====================================================


class CandidateAlwaysValidTests(unittest.TestCase):

    def test_p_equals_one(self):

        result = analyze_campaign_yield(
            _candidate(1.0), _definition_with_occupants(), count=5, max_attempts=10,
        )

        self.assertAlmostEqual(result.p_slot_success, 1.0, places=9)
        self.assertAlmostEqual(result.p_slot_failure, 0.0, places=9)
        self.assertAlmostEqual(result.expected_attempts_per_slot, 1.0, places=9)
        self.assertAlmostEqual(result.p_complete_success, 1.0, places=9)
        self.assertAlmostEqual(result.expected_accepted, 5.0, places=9)
        self.assertAlmostEqual(result.p_underproduction, 0.0, places=9)


# =====================================================
# 2 -- p = 0
# =====================================================


class CandidateNeverValidTests(unittest.TestCase):

    def test_p_equals_zero(self):

        result = analyze_campaign_yield(
            _candidate(0.0), _definition_with_occupants(), count=5, max_attempts=10,
        )

        self.assertAlmostEqual(result.p_slot_success, 0.0, places=9)
        self.assertAlmostEqual(result.p_slot_failure, 1.0, places=9)
        self.assertAlmostEqual(result.expected_accepted, 0.0, places=9)
        self.assertAlmostEqual(result.p_complete_success, 0.0, places=9)
        self.assertAlmostEqual(result.p_underproduction, 1.0, places=9)


# =====================================================
# 3 -- max_attempts = 1
# =====================================================


class SingleAttemptTests(unittest.TestCase):

    def test_slot_success_equals_p_when_only_one_attempt_allowed(self):

        result = analyze_campaign_yield(
            _candidate(0.37), _definition_with_occupants(), count=1, max_attempts=1,
        )

        self.assertAlmostEqual(result.p_slot_success, 0.37, places=9)
        self.assertAlmostEqual(result.expected_attempts_per_slot, 1.0, places=9)


# =====================================================
# 4 -- hand-derived retry case: p=0.5, max_attempts=3
# =====================================================


class HandDerivedRetryTests(unittest.TestCase):

    def test_p_half_three_attempts(self):

        result = analyze_campaign_yield(
            _candidate(0.5), _definition_with_occupants(), count=1, max_attempts=3,
        )

        # P(success) = 1 - 0.5^3 = 0.875.
        self.assertAlmostEqual(result.p_slot_success, 0.875, places=9)
        self.assertAlmostEqual(result.p_slot_failure, 0.125, places=9)

        # E[attempts] = 1*0.5 + 2*0.25 + 3*0.25 = 1.75 (hand-enumerated:
        # T=1 w.p. 0.5, T=2 w.p. 0.25, T=3 w.p. 0.25).
        self.assertAlmostEqual(result.expected_attempts_per_slot, 1.75, places=9)

    def test_slot_success_plus_failure_is_one(self):

        for p in (0.0, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0):

            result = analyze_campaign_yield(
                _candidate(p), _definition_with_occupants(), count=1, max_attempts=7,
            )
            self.assertAlmostEqual(result.p_slot_success + result.p_slot_failure, 1.0, places=9)


# =====================================================
# 5 -- multiple campaign slots: N=2, s=0.75 (max_attempts=1 so
# p_slot_success == p_candidate_valid exactly, isolating this to the
# multi-slot distribution math).
# =====================================================


class MultipleSlotsTests(unittest.TestCase):

    def test_all_succeed_probability(self):

        result = analyze_campaign_yield(
            _candidate(0.75), _definition_with_occupants(), count=2, max_attempts=1,
        )

        self.assertAlmostEqual(result.p_slot_success, 0.75, places=9)
        self.assertAlmostEqual(result.p_complete_success, 0.5625, places=9)

    def test_full_yield_distribution_matches_hand_derivation(self):

        result = analyze_campaign_yield(
            _candidate(0.75), _definition_with_occupants(), count=2, max_attempts=1,
        )

        # Binomial(2, 0.75): P(0)=0.25^2=0.0625, P(1)=2*0.75*0.25=0.375,
        # P(2)=0.75^2=0.5625.
        self.assertAlmostEqual(result.p_accepted_equals(0), 0.0625, places=9)
        self.assertAlmostEqual(result.p_accepted_equals(1), 0.375, places=9)
        self.assertAlmostEqual(result.p_accepted_equals(2), 0.5625, places=9)

        total = sum(result.p_accepted_equals(k) for k in range(3))
        self.assertAlmostEqual(total, 1.0, places=9)


# =====================================================
# 6 -- expected accepted count
# =====================================================


class ExpectedAcceptedTests(unittest.TestCase):

    def test_expected_accepted_matches_hand_derivation(self):

        # max_attempts=1 isolates p_slot_success == p_candidate_valid,
        # so E[accepted] = N * p exactly: 10 * 0.6 = 6.0.
        result = analyze_campaign_yield(
            _candidate(0.6), _definition_with_occupants(), count=10, max_attempts=1,
        )

        self.assertAlmostEqual(result.expected_accepted, 6.0, places=9)


# =====================================================
# 7 -- underproduction probability, and a partial target
# =====================================================


class UnderproductionTests(unittest.TestCase):

    def test_underproduction_and_partial_target(self):

        # N=3, s=0.5 (max_attempts=1 so s == p exactly).
        result = analyze_campaign_yield(
            _candidate(0.5), _definition_with_occupants(), count=3, max_attempts=1,
        )

        # P(accepted < 3) = 1 - 0.5^3 = 0.875.
        self.assertAlmostEqual(result.p_underproduction, 0.875, places=9)

        # P(accepted >= 2) = P(2) + P(3) = 3*0.5^2*0.5 + 0.5^3
        #                  = 0.375 + 0.125 = 0.5.
        self.assertAlmostEqual(result.p_accepted_at_least(2), 0.5, places=9)

        # Consistency: p_accepted_at_least(N) must equal
        # p_complete_success (same quantity, two access paths).
        self.assertAlmostEqual(
            result.p_accepted_at_least(3), result.p_complete_success, places=9,
        )
        self.assertAlmostEqual(
            result.p_accepted_at_most(2), result.p_underproduction, places=9,
        )


# =====================================================
# 8 -- large retry count / numerical stability
# =====================================================


class LargeRetryCountStabilityTests(unittest.TestCase):

    def test_tiny_p_large_max_attempts_stays_bounded(self):

        # p is small enough that naive repeated multiplication could
        # accumulate error, and max_attempts is large enough that
        # (1-p)^A could underflow -- both must resolve to sensible,
        # in-bounds values, never NaN/inf/out-of-range.
        result = analyze_campaign_yield(
            _candidate(1e-6), _definition_with_occupants(), count=1, max_attempts=2_000_000,
        )

        self.assertGreaterEqual(result.p_slot_success, 0.0)
        self.assertLessEqual(result.p_slot_success, 1.0)
        self.assertGreaterEqual(result.expected_attempts_per_slot, 0.0)
        self.assertLessEqual(result.expected_attempts_per_slot, 2_000_000.0)
        # 1 - (1-1e-6)^2e6 ~= 1 - e^-2 ~= 0.8647 (a well-known limit).
        self.assertAlmostEqual(result.p_slot_success, 1.0 - pow(2.718281828459045, -2.0), places=3)

    def test_p_very_close_to_one_stays_bounded(self):

        result = analyze_campaign_yield(
            _candidate(1.0 - 1e-15), _definition_with_occupants(), count=1, max_attempts=1000,
        )

        self.assertGreaterEqual(result.p_slot_success, 0.0)
        self.assertLessEqual(result.p_slot_success, 1.0)
        self.assertGreaterEqual(result.expected_attempts_per_slot, 1.0)


# =====================================================
# 9 -- large campaign size: must not enumerate 2^N slot combinations
# =====================================================


class LargeCampaignSizeTests(unittest.TestCase):

    def test_large_n_completes_quickly_and_stays_bounded(self):

        result = analyze_campaign_yield(
            _candidate(0.3), _definition_with_occupants(), count=200_000, max_attempts=1,
        )

        start = time.time()
        # A target near the middle of the distribution is the worst
        # case for the tail-summation loop -- still must be fast (O(N)
        # point evaluations, never O(2^N) outcome enumeration).
        at_least = result.p_accepted_at_least(60_000)
        elapsed = time.time() - start

        self.assertGreaterEqual(at_least, 0.0)
        self.assertLessEqual(at_least, 1.0)
        self.assertLess(elapsed, 10.0)
        self.assertAlmostEqual(result.expected_accepted, 60_000.0, places=3)


# =====================================================
# 10 -- candidate exactness unavailable
# =====================================================


class CandidateExactnessUnavailableTests(unittest.TestCase):

    def test_state_space_too_large_does_not_invent_an_exact_answer(self):

        unavailable = CandidateValidityResult(
            exact=False, exact_for_analyzed_dimensions=False,
            p_valid=None, state_space_too_large=True,
        )

        result = analyze_campaign_yield(
            unavailable, _definition_with_occupants(), count=5, max_attempts=10,
        )

        self.assertFalse(result.exact)
        self.assertIsNone(result.p_candidate_valid)
        self.assertIsNone(result.p_slot_success)
        self.assertIsNone(result.expected_accepted)
        self.assertIsNone(result.p_accepted_equals(0))
        self.assertTrue(any("candidate validity" in w.lower() for w in result.warnings))

    def test_analyzed_but_not_fully_exact_candidate_propagates(self):

        partially_exact = _candidate(0.6, exact=False, exact_for_analyzed_dimensions=True)

        result = analyze_campaign_yield(
            partially_exact, _definition_with_occupants(), count=5, max_attempts=10,
        )

        # Numbers ARE still computed (candidate p_valid was available),
        # but the campaign-level result must not claim full exactness.
        self.assertIsNotNone(result.p_slot_success)
        self.assertFalse(result.candidate_validity_exact)
        self.assertFalse(result.exact)


# =====================================================
# 11 -- duplicate-sensitive degenerate case
# =====================================================


class DuplicateDegenerateCaseTests(unittest.TestCase):

    def test_fully_degenerate_definition_is_flagged_not_exact(self):

        result = analyze_campaign_yield(
            _candidate(0.5, exact=True), _fully_degenerate_definition(), count=5, max_attempts=3,
        )

        self.assertFalse(result.slots_independent)
        self.assertTrue(result.duplicate_rejection_risk)
        self.assertFalse(result.exact)
        self.assertTrue(result.candidate_validity_exact)
        self.assertTrue(any("duplicate" in w.lower() for w in result.warnings))
        # Numbers are still populated (a disclosed approximation), not
        # withheld.
        self.assertIsNotNone(result.p_slot_success)
        self.assertIsNotNone(result.expected_accepted)

    def test_any_one_broken_condition_restores_full_exactness(self):

        with_occupants = _definition_with_occupants()
        with_firefighters = ScenarioDefinition(
            fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
            occupant=OccupantDefinition(),
            firefighter=FirefighterDeploymentDefinition(
                team_count_distribution=FixedValue(1),
                entry_zone_ids=("zone-1",),
            ),
        )
        with_continuous_fire = ScenarioDefinition(
            fire=FireDefinition(growth_parameter_distribution=UniformRange(100.0, 300.0)),
            occupant=OccupantDefinition(),
        )
        with_continuous_event = ScenarioDefinition(
            fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
            occupant=OccupantDefinition(),
            event_templates=(
                EventTemplate(
                    target_type="door", target_id="door-1", event_type="close",
                    occurs=FixedValue(True), time=UniformRange(60.0, 120.0),
                ),
            ),
        )

        for definition in (with_occupants, with_firefighters, with_continuous_fire, with_continuous_event):

            result = analyze_campaign_yield(_candidate(0.5), definition, count=5, max_attempts=3)
            self.assertTrue(result.slots_independent, definition)
            self.assertFalse(result.duplicate_rejection_risk, definition)
            self.assertTrue(result.exact, definition)

    def test_weighted_options_occupancy_and_firefighter_counts_also_negligible(self):

        # WeightedOptions with any positive-count option that could
        # round to >= 1 must be recognized the same way FixedValue is.
        definition = ScenarioDefinition(
            fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": WeightedOptions({0: 0.5, 2: 0.5})},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
        )

        result = analyze_campaign_yield(_candidate(0.5), definition, count=5, max_attempts=3)
        self.assertTrue(result.slots_independent)


# =====================================================
# Regression: DEFAULT_MAX_ENUMERATED_STATES / CandidateValidityResult
# imports still work unchanged (import-shape regression, the full
# standing suite is run separately).
# =====================================================


class ImportShapeRegressionTests(unittest.TestCase):

    def test_public_exports_unchanged_and_extended(self):

        import campaign_feasibility

        self.assertTrue(hasattr(campaign_feasibility, "compute_exact_candidate_validity"))
        self.assertTrue(hasattr(campaign_feasibility, "CandidateValidityResult"))
        self.assertTrue(hasattr(campaign_feasibility, "analyze_campaign_yield"))
        self.assertTrue(hasattr(campaign_feasibility, "CampaignYieldResult"))
        self.assertIsInstance(CampaignYieldResult, type)


if __name__ == "__main__":
    unittest.main()
