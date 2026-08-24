import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario_definition import (
    EngineeringConstraints,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
    WeightedOptions,
)

from campaign_feasibility import (
    MonteCarloCandidateValidityResult,
    MonteCarloConfig,
    analyze_campaign_yield,
    compute_candidate_validity,
    compute_exact_candidate_validity,
    estimate_candidate_validity_monte_carlo,
)
from campaign_feasibility.monte_carlo import _wilson_interval


# =====================================================
# Phase 2C -- Monte Carlo Fallback for Candidate Validity and Campaign
# Yield. See docs/architecture/scenario_campaign_feasibility_phase2c_
# monte_carlo_implementation_report.txt for the full derivation each
# test below is hand-checked against.
# =====================================================


def _fire(**overrides):

    defaults = dict(growth_parameter_distribution=FixedValue(200.0))
    defaults.update(overrides)
    return FireDefinition(**defaults)


def _two_zone_building():

    door = Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")
    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="R1", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-2", name="R2", x=10.0, y=0.0, width=8.0, height=8.0),
        ],
        doors=[door],
        exits=[Exit(id="exit-1", zone_id="zone-2")],
    )
    return Building(name="Two Zone", id="b-1", floors=[floor])


def _definition_with_door_distribution(distribution):

    return ScenarioDefinition(
        fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-1": FixedValue(1)},
            behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
        ),
        engineering=EngineeringConstraints(door_state_distribution={"door-1": distribution}),
    )


def _oversized_chain_building_and_definition(door_count=14):

    # 14 independent uncertain doors -> 2^14 = 16384 engineering-state
    # combinations, exceeding DEFAULT_MAX_ENUMERATED_STATES (4096) --
    # forces `compute_exact_candidate_validity()` to report `state_
    # space_too_large=True`, the ONLY signal `compute_candidate_
    # validity()` uses to trigger the Monte Carlo fallback.
    zones = [Zone(id="zone-0", name="Z0", x=0.0, y=0.0, width=8.0, height=8.0)]
    doors = []
    door_state_distribution = {}

    for i in range(1, door_count + 1):

        zones.append(Zone(id=f"zone-{i}", name=f"Z{i}", x=float(i * 10), y=0.0, width=8.0, height=8.0))
        doors.append(Door(id=f"door-{i}", normally_open=True, zone_a_id=f"zone-{i - 1}", zone_b_id=f"zone-{i}"))
        door_state_distribution[f"door-{i}"] = WeightedOptions({"OPEN": 0.6, "LOCKED": 0.4})

    exits = [Exit(id="exit-1", zone_id=f"zone-{door_count}")]
    floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)
    building = Building(name="Chain", id="b-chain", floors=[floor])

    definition = ScenarioDefinition(
        fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-0"})),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-0": FixedValue(1)},
            behaviour_profile_distribution={"zone-0": FixedValue("Staff_Default")},
        ),
        engineering=EngineeringConstraints(door_state_distribution=door_state_distribution),
    )

    return building, definition


# =====================================================
# 1 -- Wilson interval known cases (hand-calculated, not merely
# "lower <= estimate <= upper").
# =====================================================


class WilsonIntervalKnownCasesTests(unittest.TestCase):

    def test_k_equals_zero(self):

        # Standard reference value for n=10, k=0, 95% Wilson CI.
        lower, upper = _wilson_interval(0, 10, 0.95)

        self.assertAlmostEqual(lower, 0.0, places=9)
        self.assertAlmostEqual(upper, 0.2775327998628892, places=9)

    def test_k_equals_n(self):

        # The exact mirror image of the k=0 case (Wilson's own
        # symmetry: p_hat=1 reflects p_hat=0).
        lower, upper = _wilson_interval(10, 10, 0.95)

        self.assertAlmostEqual(lower, 0.7224672001371107, places=9)
        self.assertAlmostEqual(upper, 1.0, places=9)

    def test_intermediate_case(self):

        # n=20, k=10 (p_hat=0.5) -- a well-known textbook Wilson score
        # example.
        lower, upper = _wilson_interval(10, 20, 0.95)

        self.assertAlmostEqual(lower, 0.2992980081982123, places=9)
        self.assertAlmostEqual(upper, 0.7007019918017877, places=9)

    def test_interval_widens_at_lower_confidence_and_narrows_at_higher(self):

        lower_90, upper_90 = _wilson_interval(50, 100, 0.90)
        lower_95, upper_95 = _wilson_interval(50, 100, 0.95)
        lower_99, upper_99 = _wilson_interval(50, 100, 0.99)

        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99

        self.assertLess(width_90, width_95)
        self.assertLess(width_95, width_99)


# =====================================================
# 2 -- deterministic valid configuration
# =====================================================


class DeterministicValidConfigurationTests(unittest.TestCase):

    def test_always_open_door_yields_all_valid_samples(self):

        building = _two_zone_building()
        definition = _definition_with_door_distribution(FixedValue("OPEN"))

        result = estimate_candidate_validity_monte_carlo(
            building, definition,
            MonteCarloConfig(seed=7, minimum_samples=50, maximum_samples=200, target_interval_half_width=0.01),
        )

        self.assertEqual(result.valid_samples, result.samples_run)
        self.assertEqual(result.invalid_samples, 0)
        self.assertAlmostEqual(result.estimated_p_valid, 1.0, places=9)

        # A finite-sample Wilson interval at k=n is NEVER [1.0, 1.0] --
        # the lower bound must be strictly below 1.0.
        self.assertLess(result.confidence_lower, 1.0)
        self.assertAlmostEqual(result.confidence_upper, 1.0, places=9)


# =====================================================
# 3 -- deterministic invalid configuration
# =====================================================


class DeterministicInvalidConfigurationTests(unittest.TestCase):

    def test_always_locked_door_yields_zero_valid_samples(self):

        building = _two_zone_building()
        definition = _definition_with_door_distribution(FixedValue("LOCKED"))

        result = estimate_candidate_validity_monte_carlo(
            building, definition,
            MonteCarloConfig(seed=7, minimum_samples=50, maximum_samples=200, target_interval_half_width=0.01),
        )

        self.assertEqual(result.valid_samples, 0)
        self.assertAlmostEqual(result.estimated_p_valid, 0.0, places=9)

        # A finite-sample Wilson interval at k=0 is NEVER [0.0, 0.0] --
        # the upper bound must be strictly above 0.0.
        self.assertAlmostEqual(result.confidence_lower, 0.0, places=9)
        self.assertGreater(result.confidence_upper, 0.0)


# =====================================================
# 4 -- reproducibility
# =====================================================


class ReproducibilityTests(unittest.TestCase):

    def test_identical_seed_produces_identical_result(self):

        building = _two_zone_building()
        definition = _definition_with_door_distribution(WeightedOptions({"OPEN": 0.7, "LOCKED": 0.3}))
        config = MonteCarloConfig(seed=42, minimum_samples=100, maximum_samples=1000, target_interval_half_width=0.05)

        first = estimate_candidate_validity_monte_carlo(building, definition, config)
        second = estimate_candidate_validity_monte_carlo(building, definition, config)

        self.assertEqual(first, second)

    def test_different_seed_produces_a_different_sample_sequence(self):

        building = _two_zone_building()
        definition = _definition_with_door_distribution(WeightedOptions({"OPEN": 0.7, "LOCKED": 0.3}))

        result_a = estimate_candidate_validity_monte_carlo(
            building, definition,
            MonteCarloConfig(seed=42, minimum_samples=100, maximum_samples=1000, target_interval_half_width=0.05),
        )
        result_b = estimate_candidate_validity_monte_carlo(
            building, definition,
            MonteCarloConfig(seed=43, minimum_samples=100, maximum_samples=1000, target_interval_half_width=0.05),
        )

        # Not artificially fixed to the same sample sequence: at least
        # one of the two runs' own accounting must differ (samples_run
        # is itself a function of the actual sampled sequence, via the
        # adaptive stop, so two different seeds virtually never produce
        # byte-identical results).
        self.assertNotEqual(
            (result_a.samples_run, result_a.valid_samples),
            (result_b.samples_run, result_b.valid_samples),
        )


# =====================================================
# 5 -- early precision stop
# =====================================================


class EarlyPrecisionStopTests(unittest.TestCase):

    def test_stops_before_maximum_samples_once_precision_reached(self):

        building = _two_zone_building()
        definition = _definition_with_door_distribution(FixedValue("OPEN"))

        result = estimate_candidate_validity_monte_carlo(
            building, definition,
            MonteCarloConfig(seed=7, minimum_samples=50, maximum_samples=5000, target_interval_half_width=0.05),
        )

        self.assertTrue(result.precision_target_met)
        self.assertGreaterEqual(result.samples_run, result.minimum_samples)
        self.assertLess(result.samples_run, result.maximum_samples)


# =====================================================
# 6 -- maximum sample stop
# =====================================================


class MaximumSampleStopTests(unittest.TestCase):

    def test_stops_at_maximum_samples_when_precision_unreachable(self):

        building = _two_zone_building()
        definition = _definition_with_door_distribution(FixedValue("OPEN"))

        # An essentially unreachable target half-width within a tiny
        # maximum_samples budget.
        result = estimate_candidate_validity_monte_carlo(
            building, definition,
            MonteCarloConfig(seed=7, minimum_samples=20, maximum_samples=25, target_interval_half_width=0.0001),
        )

        self.assertEqual(result.samples_run, 25)
        self.assertFalse(result.precision_target_met)
        self.assertTrue(any("maximum_samples" in w for w in result.warnings))


# =====================================================
# Invalid configuration (edge case 6)
# =====================================================


class InvalidConfigurationTests(unittest.TestCase):

    def test_non_positive_minimum_samples_rejected(self):

        with self.assertRaises(ValueError):
            MonteCarloConfig(minimum_samples=0)

    def test_maximum_below_minimum_rejected(self):

        with self.assertRaises(ValueError):
            MonteCarloConfig(minimum_samples=100, maximum_samples=50)

    def test_confidence_level_boundaries_rejected(self):

        for level in (0.0, 1.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                MonteCarloConfig(confidence_level=level)

    def test_non_positive_precision_target_rejected(self):

        for width in (0.0, -0.01):
            with self.assertRaises(ValueError):
                MonteCarloConfig(target_interval_half_width=width)


# =====================================================
# 7 -- exact-analysis fallback routing
# =====================================================


class FallbackRoutingTests(unittest.TestCase):

    def test_oversized_state_space_uses_monte_carlo(self):

        building, definition = _oversized_chain_building_and_definition()

        exact = compute_exact_candidate_validity(building, definition)
        self.assertTrue(exact.state_space_too_large)

        analysis = compute_candidate_validity(
            building, definition,
            monte_carlo_config=MonteCarloConfig(
                seed=1, minimum_samples=80, maximum_samples=400, target_interval_half_width=0.05,
            ),
        )

        self.assertTrue(analysis.used_monte_carlo)
        self.assertIsNotNone(analysis.monte_carlo_result)
        self.assertFalse(analysis.exact)
        self.assertIsNotNone(analysis.p_valid)

    def test_small_configuration_does_not_invoke_monte_carlo(self):

        building = _two_zone_building()
        definition = _definition_with_door_distribution(WeightedOptions({"OPEN": 0.6, "LOCKED": 0.4}))

        analysis = compute_candidate_validity(building, definition)

        self.assertFalse(analysis.used_monte_carlo)
        self.assertIsNone(analysis.monte_carlo_result)
        self.assertTrue(analysis.exact)
        self.assertAlmostEqual(analysis.p_valid, 0.6, places=9)


# =====================================================
# 8 -- candidate validity correctness against a known probability
# =====================================================


class CandidateValidityCorrectnessTests(unittest.TestCase):

    def test_true_probability_lies_within_the_reported_interval(self):

        building = _two_zone_building()
        definition = _definition_with_door_distribution(WeightedOptions({"OPEN": 0.7, "LOCKED": 0.3}))

        exact = compute_exact_candidate_validity(building, definition)
        self.assertAlmostEqual(exact.p_valid, 0.7, places=9)

        # A fixed, deterministic seed and a generously-sized sample
        # budget -- deterministic, not flaky: the SAME seed always
        # produces the SAME sample sequence, so this either always
        # passes or always fails on this codebase, never intermittently.
        result = estimate_candidate_validity_monte_carlo(
            building, definition,
            MonteCarloConfig(seed=42, minimum_samples=300, maximum_samples=3000, target_interval_half_width=0.03),
        )

        self.assertLessEqual(result.confidence_lower, exact.p_valid)
        self.assertGreaterEqual(result.confidence_upper, exact.p_valid)
        self.assertAlmostEqual(result.estimated_p_valid, exact.p_valid, delta=0.05)


# =====================================================
# 9/10/11 -- campaign-yield interval propagation
# =====================================================


class CampaignYieldIntervalPropagationTests(unittest.TestCase):

    def _yield_result(self):

        definition = _definition_with_door_distribution(WeightedOptions({"OPEN": 0.7, "LOCKED": 0.3}))

        monte_carlo = MonteCarloCandidateValidityResult(
            estimated_p_valid=0.7, confidence_lower=0.6, confidence_upper=0.8, confidence_level=0.95,
            samples_run=1000, valid_samples=700, invalid_samples=300,
            precision_target_met=True, target_interval_half_width=0.05,
            minimum_samples=200, maximum_samples=5000, seed=1,
        )

        return analyze_campaign_yield(monte_carlo, definition, count=5, max_attempts=3)

    def test_slot_success_interval_propagation(self):

        result = self._yield_result()

        # s(p) = 1 - (1-p)^3: s(0.6)=0.936, s(0.7)=0.973, s(0.8)=0.992.
        self.assertAlmostEqual(result.p_slot_success_lower, 0.936, places=9)
        self.assertAlmostEqual(result.p_slot_success, 0.973, places=9)
        self.assertAlmostEqual(result.p_slot_success_upper, 0.992, places=9)
        self.assertLessEqual(result.p_slot_success_lower, result.p_slot_success)
        self.assertLessEqual(result.p_slot_success, result.p_slot_success_upper)

    def test_expected_accepted_interval_propagation(self):

        result = self._yield_result()

        # E(p) = 5 * s(p): E(0.6)=4.68, E(0.7)=4.865, E(0.8)=4.96.
        self.assertAlmostEqual(result.expected_accepted_lower, 4.68, places=9)
        self.assertAlmostEqual(result.expected_accepted, 4.865, places=9)
        self.assertAlmostEqual(result.expected_accepted_upper, 4.96, places=9)

    def test_complete_campaign_success_interval_propagation(self):

        result = self._yield_result()

        # C(p) = s(p)^5: C(0.6)=0.71842..., C(0.7)=0.87210..., C(0.8)=0.96063...
        self.assertAlmostEqual(result.p_complete_success_lower, 0.7184213723381757, places=6)
        self.assertAlmostEqual(result.p_complete_success, 0.872095812856093, places=6)
        self.assertAlmostEqual(result.p_complete_success_upper, 0.960634900447232, places=6)

    def test_underproduction_bounds_are_correctly_swapped(self):

        # p_underproduction = 1 - C(p) is DECREASING in p, so its lower
        # bound comes from confidence_UPPER and its upper bound from
        # confidence_LOWER -- the opposite pairing from the other three
        # metrics. This is the single most error-prone line in the
        # whole propagation and gets its own dedicated test.
        result = self._yield_result()

        self.assertAlmostEqual(result.p_underproduction_lower, 1.0 - 0.960634900447232, places=6)
        self.assertAlmostEqual(result.p_underproduction_upper, 1.0 - 0.7184213723381757, places=6)
        self.assertLessEqual(result.p_underproduction_lower, result.p_underproduction)
        self.assertLessEqual(result.p_underproduction, result.p_underproduction_upper)

    def test_result_is_never_labeled_exact(self):

        result = self._yield_result()

        self.assertEqual(result.candidate_validity_source, "monte_carlo")
        self.assertFalse(result.exact)
        self.assertFalse(result.candidate_validity_exact)
        self.assertIsNotNone(result.monte_carlo_source)
        self.assertTrue(any("ESTIMATE" in w for w in result.warnings))

    def test_bounds_omitted_when_duplicate_rejection_risk_present(self):

        # A fully degenerate Definition (no occupants at all) -- the
        # SAME narrow duplicate-risk condition the Phase 2 yield
        # analysis already detects -- must suppress interval bounds
        # entirely, not merely leave `exact=False`.
        degenerate_definition = ScenarioDefinition(fire=_fire(), occupant=OccupantDefinition())

        monte_carlo = MonteCarloCandidateValidityResult(
            estimated_p_valid=0.7, confidence_lower=0.6, confidence_upper=0.8, confidence_level=0.95,
            samples_run=1000, valid_samples=700, invalid_samples=300,
            precision_target_met=True, target_interval_half_width=0.05,
            minimum_samples=200, maximum_samples=5000, seed=1,
        )

        result = analyze_campaign_yield(monte_carlo, degenerate_definition, count=5, max_attempts=3)

        self.assertFalse(result.slots_independent)
        self.assertIsNone(result.p_slot_success_lower)
        self.assertIsNone(result.p_slot_success_upper)
        self.assertIsNotNone(result.p_slot_success)


# =====================================================
# Bonus: p_accepted_at_least_interval() (stochastic-dominance-
# justified propagation beyond the three explicitly required metrics).
# =====================================================


class AcceptedAtLeastIntervalTests(unittest.TestCase):

    def test_matches_binomial_survival_at_each_bound(self):

        definition = _definition_with_door_distribution(WeightedOptions({"OPEN": 0.7, "LOCKED": 0.3}))

        monte_carlo = MonteCarloCandidateValidityResult(
            estimated_p_valid=0.7, confidence_lower=0.6, confidence_upper=0.8, confidence_level=0.95,
            samples_run=1000, valid_samples=700, invalid_samples=300,
            precision_target_met=True, target_interval_half_width=0.05,
            minimum_samples=200, maximum_samples=5000, seed=1,
        )

        result = analyze_campaign_yield(monte_carlo, definition, count=5, max_attempts=3)

        interval = result.p_accepted_at_least_interval(3)

        self.assertIsNotNone(interval)
        lower, upper = interval
        self.assertLessEqual(lower, result.p_accepted_at_least(3))
        self.assertLessEqual(result.p_accepted_at_least(3), upper)


if __name__ == "__main__":
    unittest.main()
