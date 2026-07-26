import unittest

from predictive_dataset.quality_checks import duplicate_scenario_ids, run_quality_checks, zero_walking_distance_candidates


def make_row(**overrides):

    base = {
        "scenario_id": "scn-1", "observation_time": 0.0, "candidate_id": "exit-1",
        "prediction_horizon": 20.0, "candidate_type": "Exit", "candidate_capacity": 2,
        "candidate_walking_distance": 8.0, "candidate_traversable": True,
        "candidate_adjacent_zone_occupancy": 0, "candidate_queue_length": 0,
        "candidate_approaching_count": 0, "candidate_congestion_level": "LOW",
        "total_active_occupant_count": 5, "currently_congested": False,
        "had_any_activity_in_window": False, "target": False,
    }
    base.update(overrides)
    return base


class DuplicateDetectionTests(unittest.TestCase):

    def test_exact_duplicate_rows_are_counted(self):

        rows = [make_row(), make_row()]
        report = run_quality_checks(rows, known_candidate_ids={"exit-1"})

        self.assertEqual(report["duplicate_exact_rows"], 1)

    def test_duplicate_identity_key_with_differing_values_still_flagged(self):

        rows = [make_row(candidate_queue_length=0), make_row(candidate_queue_length=3)]
        report = run_quality_checks(rows, known_candidate_ids={"exit-1"})

        self.assertEqual(report["duplicate_exact_rows"], 0)
        self.assertEqual(report["duplicate_identity_keys"], 1)

    def test_no_duplicates_in_clean_data(self):

        rows = [make_row(observation_time=t) for t in (0.0, 5.0, 10.0)]
        report = run_quality_checks(rows, known_candidate_ids={"exit-1"})

        self.assertEqual(report["duplicate_exact_rows"], 0)
        self.assertEqual(report["duplicate_identity_keys"], 0)


class InvalidValueTests(unittest.TestCase):

    def test_unknown_candidate_id_flagged(self):

        rows = [make_row(candidate_id="not-a-real-candidate")]
        report = run_quality_checks(rows, known_candidate_ids={"exit-1"})

        self.assertIn("not-a-real-candidate", report["invalid_candidate_ids"])

    def test_negative_queue_length_flagged(self):

        rows = [make_row(candidate_queue_length=-1)]
        report = run_quality_checks(rows, known_candidate_ids={"exit-1"})

        self.assertEqual(report["invalid_ranges"]["candidate_queue_length_negative"], 1)

    def test_invalid_congestion_level_flagged(self):

        rows = [make_row(candidate_congestion_level="NOT_A_REAL_LEVEL")]
        report = run_quality_checks(rows, known_candidate_ids={"exit-1"})

        self.assertEqual(report["invalid_congestion_level_count"], 1)

    def test_valid_none_congestion_level_not_flagged(self):

        rows = [make_row(candidate_congestion_level=None)]
        report = run_quality_checks(rows, known_candidate_ids={"exit-1"})

        self.assertEqual(report["invalid_congestion_level_count"], 0)

    def test_currently_congested_true_with_a_target_is_inconsistent(self):

        rows = [make_row(currently_congested=True, target=True)]
        report = run_quality_checks(rows, known_candidate_ids={"exit-1"})

        self.assertEqual(report["currently_congested_target_inconsistencies"], 1)

    def test_currently_congested_true_with_none_target_is_consistent(self):

        rows = [make_row(currently_congested=True, target=None)]
        report = run_quality_checks(rows, known_candidate_ids={"exit-1"})

        self.assertEqual(report["currently_congested_target_inconsistencies"], 0)


class DuplicateScenarioIdTests(unittest.TestCase):

    def test_duplicate_scenario_ids_detected(self):

        metadata = [{"scenario_id": "scn-1"}, {"scenario_id": "scn-1"}, {"scenario_id": "scn-2"}]

        self.assertEqual(duplicate_scenario_ids(metadata), 1)

    def test_no_duplicates(self):

        metadata = [{"scenario_id": "scn-1"}, {"scenario_id": "scn-2"}]

        self.assertEqual(duplicate_scenario_ids(metadata), 0)


class ZeroWalkingDistanceCandidateTests(unittest.TestCase):
    """Predictive Dataset V2 milestone, Phase 1/16 -- the mechanical
    guard that would have caught V1's Stair from_floor_id bug."""

    def test_candidate_with_all_zero_distance_rows_is_flagged(self):

        rows = [
            make_row(candidate_id="stair-1", candidate_walking_distance=0.0),
            make_row(candidate_id="stair-1", candidate_walking_distance=0.0),
        ]

        report = zero_walking_distance_candidates(rows)

        self.assertEqual(report["flagged_zero_distance_candidate_ids"], ["stair-1"])

    def test_candidate_with_positive_distance_is_not_flagged(self):

        rows = [make_row(candidate_id="exit-1", candidate_walking_distance=8.0)]

        report = zero_walking_distance_candidates(rows)

        self.assertEqual(report["flagged_zero_distance_candidate_ids"], [])
        self.assertIn("exit-1", report["candidates_checked"])

    def test_candidate_with_mixed_zero_and_nonzero_is_not_flagged(self):
        # Should never actually happen for a real structural distance
        # (same building, same candidate -> same geometry every row) --
        # but the check must not falsely flag it if it somehow did.

        rows = [
            make_row(candidate_id="door-1", candidate_walking_distance=0.0),
            make_row(candidate_id="door-1", candidate_walking_distance=5.0),
        ]

        report = zero_walking_distance_candidates(rows)

        self.assertEqual(report["flagged_zero_distance_candidate_ids"], [])


if __name__ == "__main__":
    unittest.main()
