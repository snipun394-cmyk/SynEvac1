import math
import unittest

from calibration_benchmark.candidates import WalkingSpeedCandidate
from calibration_benchmark.harness import run_calibration_benchmark

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


def _nan_safe_equal(a, b) -> bool:

    # Plain dict equality treats NaN != NaN (Python's own float
    # semantics), which would misreport a genuinely deterministic,
    # identical "no variance" paired-comparison result (scipy's own
    # documented NaN-on-zero-variance behaviour, see
    # calibration_benchmark/report.py's own handling of the same case)
    # as a reproducibility failure. Recurses through the same
    # dict/list/float shapes to_dict() produces.

    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_nan_safe_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_nan_safe_equal(x, y) for x, y in zip(a, b))
    return a == b


# =====================================================
# Calibration Studio Phase 0 -- Deterministic Experiment Infrastructure.
# Verifies, through calibration_benchmark's own public API (not just at
# the lower behaviour_profile_resolver layer
# tests/test_reproducibility_fix.py already covers), that a calibration
# experiment is genuinely reproducible: identical master_seed produces
# identical output, a different master_seed produces different output,
# and the new evidence fields (master_seed, baseline_randomness,
# candidate_randomness, reproducible) reflect real, checked state.
#
# make_definition()'s occupant_count=25, all Adult_Default (behaviour_
# profile_distribution fixed) -- ComplianceDecisionStrategy and
# ProbabilisticPreMovementDelay both draw randomness for every one of
# them, every scenario, so a determinism failure anywhere in the chain
# has ample opportunity to surface here.
# =====================================================


class CalibrationBenchmarkEndToEndDeterminismTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.definition = make_definition()
        self.candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")

    def _run(self, master_seed):

        return run_calibration_benchmark(
            self.candidate, self.building, self.definition, DEFINITION_ID, master_seed, n_scenarios=3, dt=1.0,
        )

    def test_identical_master_seed_produces_identical_baseline_samples(self):

        result_a = self._run(MASTER_SEED)
        result_b = self._run(MASTER_SEED)

        self.assertEqual(result_a.baseline_samples, result_b.baseline_samples)

    def test_identical_master_seed_produces_identical_candidate_samples(self):

        result_a = self._run(MASTER_SEED)
        result_b = self._run(MASTER_SEED)

        self.assertEqual(result_a.candidate_samples, result_b.candidate_samples)

    def test_identical_master_seed_produces_identical_comparisons(self):

        result_a = self._run(MASTER_SEED)
        result_b = self._run(MASTER_SEED)

        for metric_name in result_a.comparisons:
            dict_a = result_a.comparisons[metric_name].to_dict()
            dict_b = result_b.comparisons[metric_name].to_dict()
            self.assertTrue(
                _nan_safe_equal(dict_a, dict_b),
                msg=f"{metric_name}: {dict_a!r} != {dict_b!r}",
            )

    def test_different_master_seed_produces_different_output(self):

        result_a = self._run(MASTER_SEED)
        result_b = self._run(MASTER_SEED + 1)

        # A behavioural difference (pre-movement delay draws, compliance
        # rolls) must show up somewhere in evacuation_time -- the one
        # metric this fixture's building (a single capacity-2 door/exit)
        # cannot floor to an identical value regardless of behaviour, the
        # way queue_length/peak_occupancy_ratio/exit_utilization_balance
        # already do for every candidate this framework's own demo report
        # observed (docs/architecture/calibration_benchmark_v1_demo_report.md).
        baseline_times_a = [s.evacuation_time for s in result_a.baseline_samples]
        baseline_times_b = [s.evacuation_time for s in result_b.baseline_samples]

        self.assertNotEqual(baseline_times_a, baseline_times_b)


class CalibrationBenchmarkReproducibilityEvidenceTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.definition = make_definition()

    def test_master_seed_is_recorded_on_the_result(self):

        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")
        result = run_calibration_benchmark(
            candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=1, dt=1.0,
        )

        self.assertEqual(result.master_seed, MASTER_SEED)

    def test_baseline_and_candidate_randomness_are_fully_controlled_for_a_standard_candidate(self):

        # WalkingSpeedCandidate's candidate_registry() only replaces
        # walking_speed via dataclasses.replace() -- decision_strategy/
        # pre_movement_strategy/route_choice_strategy are untouched, so
        # both arms' registries should audit identically to
        # DEFAULT_PROFILE_REGISTRY itself: fully controlled.
        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")
        result = run_calibration_benchmark(
            candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=1, dt=1.0,
        )

        self.assertTrue(result.baseline_randomness.fully_controlled)
        self.assertTrue(result.candidate_randomness.fully_controlled)

    def test_reproducible_is_true_using_real_evidence_not_a_default(self):

        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")
        result = run_calibration_benchmark(
            candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=1, dt=1.0,
        )

        self.assertIs(result.reproducible, True)

    def test_reproducible_is_none_when_result_was_constructed_without_randomness_evidence(self):

        # Backward-compatibility path: a CalibrationBenchmarkResult built
        # the old way (as tests/test_calibration_benchmark_recommendation.py
        # still does) has no randomness evidence at all -- reproducible
        # must honestly report None (unknown), never silently claim True.
        from calibration_benchmark.harness import CalibrationBenchmarkResult

        candidate = WalkingSpeedCandidate("Adult_Default", 0.6, "test", "test")
        result = CalibrationBenchmarkResult(
            candidate=candidate, n_scenarios_requested=1, n_completed_pairs=1,
            baseline_samples=(), candidate_samples=(),
        )

        self.assertIsNone(result.reproducible)


if __name__ == "__main__":
    unittest.main()
