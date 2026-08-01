import unittest

from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY
from crowd_intelligence.models import DensityThresholds
from simulator.capacity import DefaultCapacityModel, StairCapacityModel
from simulator.congestion import DefaultCongestionModel

from calibration_benchmark.candidates import (
    CapacityWidthCandidate,
    CongestionMinimumSpeedFactorCandidate,
    DensityThresholdCandidate,
    PreMovementDelayCandidate,
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


if __name__ == "__main__":
    unittest.main()
