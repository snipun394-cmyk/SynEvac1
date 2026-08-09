import unittest

from behavior.route_choice import ShortestRouteChoiceStrategy
from behavior_library.route_choice_strategies import StaticHerdingRouteChoiceStrategy
from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY
from crowd_intelligence.models import DensityThresholds
from simulator.capacity import DefaultCapacityModel, StairCapacityModel
from simulator.congestion import DefaultCongestionModel, StairAwareCongestionModel
from simulator.flow_region_capacity import FlowRegionCapacityModel
from simulator.flow_region_congestion import FlowRegionCongestionModel

from navigation.edge import Edge

from calibration_benchmark.candidates import (
    CapacityWidthCandidate,
    ComplianceLevelCandidate,
    CongestionMinimumSpeedFactorCandidate,
    DensityThresholdCandidate,
    FlowRegionCapacityCandidate,
    HerdingFollowProbabilityCandidate,
    PreMovementDelayCandidate,
    StairCounterflowPenaltyCandidate,
    StairSpeedCandidate,
    WalkingSpeedCandidate,
)


class WalkingSpeedCandidateTests(unittest.TestCase):

    def test_baseline_registry_is_the_untouched_production_default(self):

        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "source", "rationale")

        self.assertIs(candidate.baseline_registry(), DEFAULT_PROFILE_REGISTRY)

    def test_candidate_registry_overrides_only_the_named_profile(self):

        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "source", "rationale")
        registry = candidate.candidate_registry()

        self.assertEqual(registry["Adult_Default"].walking_speed, 0.65)

        for profile_id, template in DEFAULT_PROFILE_REGISTRY.items():
            if profile_id != "Adult_Default":
                self.assertEqual(registry[profile_id], template)

    def test_original_registry_is_never_mutated(self):

        original_speed = DEFAULT_PROFILE_REGISTRY["Adult_Default"].walking_speed

        WalkingSpeedCandidate("Adult_Default", 0.65, "source", "rationale").candidate_registry()

        self.assertEqual(DEFAULT_PROFILE_REGISTRY["Adult_Default"].walking_speed, original_speed)

    def test_current_value_is_read_from_the_real_registry(self):

        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "source", "rationale")

        self.assertEqual(candidate.current_value, DEFAULT_PROFILE_REGISTRY["Adult_Default"].walking_speed)

    def test_unknown_profile_id_raises(self):

        with self.assertRaises(KeyError):
            WalkingSpeedCandidate("Not_A_Real_Profile", 1.0, "source", "rationale")


class StairSpeedCandidateTests(unittest.TestCase):

    # Edge-Type-Specific Movement Speed (Experimental Branch V1) --
    # mirrors WalkingSpeedCandidateTests exactly, plus an explicit
    # isolation check (walking_speed on the same profile is untouched).

    def test_baseline_registry_is_the_untouched_production_default(self):

        candidate = StairSpeedCandidate("Adult_Default", 0.55, "source", "rationale")

        self.assertIs(candidate.baseline_registry(), DEFAULT_PROFILE_REGISTRY)

    def test_candidate_registry_overrides_only_the_named_profiles_stair_speed(self):

        candidate = StairSpeedCandidate("Adult_Default", 0.55, "source", "rationale")
        registry = candidate.candidate_registry()

        self.assertEqual(registry["Adult_Default"].stair_speed, 0.55)

        for profile_id, template in DEFAULT_PROFILE_REGISTRY.items():
            if profile_id != "Adult_Default":
                self.assertEqual(registry[profile_id], template)

    def test_walking_speed_on_the_same_profile_is_unchanged(self):

        candidate = StairSpeedCandidate("Adult_Default", 0.55, "source", "rationale")
        registry = candidate.candidate_registry()

        self.assertEqual(
            registry["Adult_Default"].walking_speed,
            DEFAULT_PROFILE_REGISTRY["Adult_Default"].walking_speed,
        )

    def test_original_registry_is_never_mutated(self):

        original_stair_speed = DEFAULT_PROFILE_REGISTRY["Adult_Default"].stair_speed

        StairSpeedCandidate("Adult_Default", 0.55, "source", "rationale").candidate_registry()

        self.assertEqual(DEFAULT_PROFILE_REGISTRY["Adult_Default"].stair_speed, original_stair_speed)

    def test_current_value_is_read_from_the_real_registry(self):

        candidate = StairSpeedCandidate("Adult_Default", 0.55, "source", "rationale")

        self.assertEqual(candidate.current_value, DEFAULT_PROFILE_REGISTRY["Adult_Default"].stair_speed)

    def test_unknown_profile_id_raises(self):

        with self.assertRaises(KeyError):
            StairSpeedCandidate("Not_A_Real_Profile", 0.55, "source", "rationale")


class PreMovementDelayCandidateTests(unittest.TestCase):

    def test_candidate_registry_overrides_median_delay_and_preserves_spread_by_default(self):

        current_strategy = DEFAULT_PROFILE_REGISTRY["Adult_Default"].pre_movement_strategy

        candidate = PreMovementDelayCandidate("Adult_Default", 12.0, "source", "rationale")
        registry = candidate.candidate_registry()

        new_strategy = registry["Adult_Default"].pre_movement_strategy
        self.assertEqual(new_strategy.median_delay, 12.0)
        self.assertEqual(new_strategy.spread, current_strategy.spread)

    def test_rejects_a_profile_without_a_probabilistic_strategy(self):

        with self.assertRaises(TypeError):
            PreMovementDelayCandidate("Staff_Default", 12.0, "source", "rationale")


class CapacityWidthCandidateTests(unittest.TestCase):

    def test_default_capacity_candidate_changes_door_exit_capacity_not_stair(self):

        candidate = CapacityWidthCandidate(3.0, "source", "rationale", stair_specific=False)
        model = candidate.candidate_capacity_model()

        # A non-stair-specific candidate is still wrapped by
        # StairCapacityModel (only Door/Exit capacity changes; Stair
        # keeps its own, unmodified narrower default) -- see
        # candidates.py::CapacityWidthCandidate.candidate_capacity_model().
        self.assertEqual(model.base_model.PEOPLE_PER_METER_OF_WIDTH, 3.0)
        self.assertEqual(model.PEOPLE_PER_METER_OF_WIDTH, StairCapacityModel.PEOPLE_PER_METER_OF_WIDTH)

    def test_stair_specific_capacity_candidate_changes_only_stair_constant(self):

        candidate = CapacityWidthCandidate(0.5, "source", "rationale", stair_specific=True)
        model = candidate.candidate_capacity_model()

        self.assertEqual(model.PEOPLE_PER_METER_OF_WIDTH, 0.5)

    def test_production_classes_are_never_mutated(self):

        CapacityWidthCandidate(9.9, "source", "rationale").candidate_capacity_model()
        CapacityWidthCandidate(9.9, "source", "rationale", stair_specific=True).candidate_capacity_model()

        self.assertEqual(DefaultCapacityModel.PEOPLE_PER_METER_OF_WIDTH, 1.5)
        self.assertEqual(StairCapacityModel.PEOPLE_PER_METER_OF_WIDTH, 1.2)


class CongestionMinimumSpeedFactorCandidateTests(unittest.TestCase):

    def test_candidate_congestion_model_uses_the_new_minimum(self):

        candidate = CongestionMinimumSpeedFactorCandidate(0.1, "source", "rationale")
        model = candidate.candidate_congestion_model()

        self.assertEqual(model.base_model.MINIMUM_SPEED_FACTOR, 0.1)

    def test_production_class_is_never_mutated(self):

        CongestionMinimumSpeedFactorCandidate(0.1, "source", "rationale").candidate_congestion_model()

        self.assertEqual(DefaultCongestionModel.MINIMUM_SPEED_FACTOR, 0.3)


class DensityThresholdCandidateTests(unittest.TestCase):

    def test_baseline_thresholds_are_the_untouched_production_defaults(self):

        candidate = DensityThresholdCandidate(DensityThresholds(moderate_at=0.3), "source", "rationale")
        baseline = candidate.baseline_density_thresholds()

        self.assertEqual(baseline.moderate_at, 1.0)

    def test_candidate_thresholds_are_returned_as_given(self):

        thresholds = DensityThresholds(moderate_at=0.3, high_at=0.6, very_high_at=0.9, critical_at=1.2)
        candidate = DensityThresholdCandidate(thresholds, "source", "rationale")

        self.assertEqual(candidate.candidate_density_thresholds(), thresholds)


class ComplianceLevelCandidateTests(unittest.TestCase):

    def test_baseline_registry_is_the_untouched_production_default(self):

        candidate = ComplianceLevelCandidate("Adult_Default", 0.5, "source", "rationale")

        self.assertIs(candidate.baseline_registry(), DEFAULT_PROFILE_REGISTRY)

    def test_candidate_registry_overrides_only_the_named_profiles_compliance_level(self):

        candidate = ComplianceLevelCandidate("Child_Default", 0.3, "source", "rationale")
        registry = candidate.candidate_registry()

        self.assertEqual(registry["Child_Default"].compliance_level, 0.3)

        for profile_id, template in DEFAULT_PROFILE_REGISTRY.items():
            if profile_id != "Child_Default":
                self.assertEqual(registry[profile_id], template)

    def test_original_registry_is_never_mutated(self):

        original_level = DEFAULT_PROFILE_REGISTRY["Child_Default"].compliance_level

        ComplianceLevelCandidate("Child_Default", 0.3, "source", "rationale").candidate_registry()

        self.assertEqual(DEFAULT_PROFILE_REGISTRY["Child_Default"].compliance_level, original_level)

    def test_current_value_is_read_from_the_real_registry(self):

        candidate = ComplianceLevelCandidate("Adult_Default", 0.5, "source", "rationale")

        self.assertEqual(candidate.current_value, DEFAULT_PROFILE_REGISTRY["Adult_Default"].compliance_level)

    def test_unknown_profile_id_raises(self):

        with self.assertRaises(KeyError):
            ComplianceLevelCandidate("Not_A_Real_Profile", 0.5, "source", "rationale")


class HerdingFollowProbabilityCandidateTests(unittest.TestCase):

    def test_current_value_is_zero_when_the_profile_uses_no_herding_strategy_today(self):

        # Every DEFAULT_PROFILE_REGISTRY profile uses ShortestRouteChoiceStrategy
        # today -- current_value must document that honestly as zero
        # herding influence, not raise or fabricate a number.
        candidate = HerdingFollowProbabilityCandidate("Adult_Default", 0.8, "source", "rationale")

        self.assertEqual(candidate.current_value, 0.0)

    def test_candidate_registry_installs_a_herding_strategy_with_the_given_follow_probability(self):

        candidate = HerdingFollowProbabilityCandidate("Adult_Default", 0.8, "source", "rationale")
        registry = candidate.candidate_registry()

        new_strategy = registry["Adult_Default"].route_choice_strategy
        self.assertIsInstance(new_strategy, StaticHerdingRouteChoiceStrategy)
        self.assertEqual(new_strategy.follow_probability, 0.8)

    def test_other_profiles_are_untouched(self):

        candidate = HerdingFollowProbabilityCandidate("Adult_Default", 0.8, "source", "rationale")
        registry = candidate.candidate_registry()

        for profile_id, template in DEFAULT_PROFILE_REGISTRY.items():
            if profile_id != "Adult_Default":
                self.assertEqual(registry[profile_id], template)

    def test_original_registry_is_never_mutated(self):

        HerdingFollowProbabilityCandidate("Adult_Default", 0.8, "source", "rationale").candidate_registry()

        self.assertIsInstance(
            DEFAULT_PROFILE_REGISTRY["Adult_Default"].route_choice_strategy, ShortestRouteChoiceStrategy,
        )

    def test_unknown_profile_id_raises(self):

        with self.assertRaises(KeyError):
            HerdingFollowProbabilityCandidate("Not_A_Real_Profile", 0.8, "source", "rationale")


class StairCounterflowPenaltyCandidateTests(unittest.TestCase):

    def test_candidate_congestion_model_uses_the_new_penalty(self):

        candidate = StairCounterflowPenaltyCandidate(0.4, "source", "rationale")
        model = candidate.candidate_congestion_model()

        self.assertEqual(model.COUNTERFLOW_PENALTY_PER_OPPOSING, 0.4)

    def test_production_class_is_never_mutated(self):

        StairCounterflowPenaltyCandidate(0.4, "source", "rationale").candidate_congestion_model()

        self.assertEqual(StairAwareCongestionModel.COUNTERFLOW_PENALTY_PER_OPPOSING, 0.15)

    def test_the_new_penalty_actually_changes_speed_factor_for_opposing_stair_traffic(self):

        # Unlike StairCapacityModel's stair-only constants (see
        # candidates.py's own Phase 8 header comment),
        # StairAwareCongestionModel.speed_factor() reads
        # self.COUNTERFLOW_PENALTY_PER_OPPOSING polymorphically -- this
        # test proves the override actually changes simulated behaviour,
        # not just the constructed model object's own class attribute.
        stair_edge = Edge(id="s1", edge_type=Edge.STAIR, from_node="a", to_node="b")

        baseline_model = StairCounterflowPenaltyCandidate(0.4, "source", "rationale").baseline_congestion_model()
        candidate_model = StairCounterflowPenaltyCandidate(0.4, "source", "rationale").candidate_congestion_model()

        baseline_factor = baseline_model.speed_factor(stair_edge, other_occupants=1, capacity=5, opposing_occupants=1)
        candidate_factor = candidate_model.speed_factor(stair_edge, other_occupants=1, capacity=5, opposing_occupants=1)

        self.assertLess(candidate_factor, baseline_factor)


class BaseParameterCandidateFlowRegionDefaultsTests(unittest.TestCase):

    # None of the candidates above override these -- every one of them
    # must inherit the same "off" default, unchanged by this milestone.

    def test_every_existing_candidate_defaults_flow_regions_off_on_both_arms(self):

        candidates = (
            WalkingSpeedCandidate("Adult_Default", 0.6, "source", "rationale"),
            PreMovementDelayCandidate("Adult_Default", 12.0, "source", "rationale"),
            CapacityWidthCandidate(3.0, "source", "rationale"),
            CongestionMinimumSpeedFactorCandidate(0.1, "source", "rationale"),
            DensityThresholdCandidate(DensityThresholds(), "source", "rationale"),
            ComplianceLevelCandidate("Adult_Default", 0.5, "source", "rationale"),
            HerdingFollowProbabilityCandidate("Adult_Default", 0.8, "source", "rationale"),
            StairCounterflowPenaltyCandidate(0.4, "source", "rationale"),
        )

        for candidate in candidates:
            self.assertFalse(candidate.baseline_use_flow_regions())
            self.assertFalse(candidate.candidate_use_flow_regions())


class FlowRegionCapacityCandidateTests(unittest.TestCase):

    def test_baseline_arm_is_the_untouched_production_default(self):

        candidate = FlowRegionCapacityCandidate("source", "rationale")

        self.assertIsInstance(candidate.baseline_capacity_model(), StairCapacityModel)
        self.assertIsInstance(candidate.baseline_congestion_model(), StairAwareCongestionModel)
        self.assertFalse(candidate.baseline_use_flow_regions())

    def test_candidate_arm_uses_flow_region_aware_models_and_enables_the_map(self):

        candidate = FlowRegionCapacityCandidate("source", "rationale")

        self.assertIsInstance(candidate.candidate_capacity_model(), FlowRegionCapacityModel)
        self.assertIsInstance(candidate.candidate_congestion_model(), FlowRegionCongestionModel)
        self.assertTrue(candidate.candidate_use_flow_regions())

    def test_describe_reports_an_architecture_comparison_not_a_scalar(self):

        candidate = FlowRegionCapacityCandidate("source", "rationale")
        description = candidate.describe()

        self.assertIsInstance(description["current_value"], str)
        self.assertIsInstance(description["candidate_value"], str)
        self.assertNotEqual(description["current_value"], description["candidate_value"])


if __name__ == "__main__":
    unittest.main()
