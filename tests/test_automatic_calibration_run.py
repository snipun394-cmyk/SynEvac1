import unittest

from calibration_benchmark import WalkingSpeedCandidate

from automatic_calibration.budget import AutoCalibrationBudget
from automatic_calibration.run import (
    AutoCalibrationRun,
    AutoCalibrationRunStatus,
    CorruptedRunRecordError,
    InvalidRunTransitionError,
)
from automatic_calibration.search_space import ParameterDimension, SearchSpace


def _search_space():

    return SearchSpace(dimensions=(
        ParameterDimension(
            name="Adult_Default.walking_speed", bounds=(0.8, 1.6),
            build=lambda v: WalkingSpeedCandidate("Adult_Default", v, "test", "test"),
        ),
    ))


def _make_run(**overrides):

    defaults = dict(
        project_id="proj-1",
        search_space=_search_space(),
        objective_description="test objective",
        objective_direction="minimize",
        strategy_description="test strategy",
        budget=AutoCalibrationBudget(max_evaluations=5),
        search_seed=123,
    )
    defaults.update(overrides)
    return AutoCalibrationRun(**defaults)


class AutoCalibrationRunIdentityTests(unittest.TestCase):

    def test_run_id_is_generated_and_nonempty(self):

        self.assertTrue(_make_run().run_id)

    def test_two_runs_get_distinct_ids(self):

        self.assertNotEqual(_make_run().run_id, _make_run().run_id)

    def test_project_id_is_required_and_stored(self):

        self.assertEqual(_make_run().project_id, "proj-1")

    def test_search_space_is_snapshotted_at_construction_never_the_live_object(self):

        run = _make_run()

        self.assertEqual(run.search_space_description, ({"name": "Adult_Default.walking_speed", "bounds": [0.8, 1.6]},))
        self.assertFalse(hasattr(run, "search_space"))

    def test_starts_pending(self):

        self.assertEqual(_make_run().status, AutoCalibrationRunStatus.PENDING)

    def test_session_ids_start_empty(self):

        self.assertEqual(_make_run().session_ids, ())
        self.assertEqual(_make_run().n_evaluations, 0)


class AutoCalibrationRunTransitionTests(unittest.TestCase):

    def test_pending_to_running_to_completed(self):

        run = _make_run()
        run.mark_running()
        self.assertEqual(run.status, AutoCalibrationRunStatus.RUNNING)
        self.assertIsNotNone(run.started_at)

        run.mark_completed()
        self.assertEqual(run.status, AutoCalibrationRunStatus.COMPLETED)
        self.assertIsNotNone(run.completed_at)

    def test_pending_to_running_to_failed_records_reason(self):

        run = _make_run()
        run.mark_running()
        run.mark_failed("RuntimeError: boom")

        self.assertEqual(run.status, AutoCalibrationRunStatus.FAILED)
        self.assertEqual(run.failure_reason, "RuntimeError: boom")

    def test_pending_can_be_cancelled_directly(self):

        run = _make_run()
        run.mark_cancelled()

        self.assertEqual(run.status, AutoCalibrationRunStatus.CANCELLED)

    def test_completed_run_cannot_transition_again(self):

        run = _make_run()
        run.mark_running()
        run.mark_completed()

        with self.assertRaises(InvalidRunTransitionError):
            run.mark_failed("too late")

    def test_pending_run_cannot_be_marked_completed_directly(self):

        with self.assertRaises(InvalidRunTransitionError):
            _make_run().mark_completed()

    def test_record_evaluation_requires_running_status(self):

        with self.assertRaises(InvalidRunTransitionError):
            _make_run().record_evaluation("session-1", 10.0)


class AutoCalibrationRunEvaluationTrackingTests(unittest.TestCase):

    def test_record_evaluation_appends_session_id(self):

        run = _make_run()
        run.mark_running()

        run.record_evaluation("session-1", 5.0)
        run.record_evaluation("session-2", 3.0)

        self.assertEqual(run.session_ids, ("session-1", "session-2"))
        self.assertEqual(run.n_evaluations, 2)

    def test_a_none_score_is_still_recorded_as_an_evaluation_but_never_becomes_best(self):

        run = _make_run()
        run.mark_running()

        run.record_evaluation("session-1", None)

        self.assertEqual(run.session_ids, ("session-1",))
        self.assertIsNone(run.best_session_id)
        self.assertIsNone(run.best_score)

    def test_minimize_direction_prefers_the_lower_score(self):

        run = _make_run(objective_direction="minimize")
        run.mark_running()

        run.record_evaluation("session-1", 10.0)
        run.record_evaluation("session-2", 3.0)
        run.record_evaluation("session-3", 7.0)

        self.assertEqual(run.best_session_id, "session-2")
        self.assertEqual(run.best_score, 3.0)

    def test_maximize_direction_prefers_the_higher_score(self):

        run = _make_run(objective_direction="maximize")
        run.mark_running()

        run.record_evaluation("session-1", 10.0)
        run.record_evaluation("session-2", 3.0)
        run.record_evaluation("session-3", 25.0)

        self.assertEqual(run.best_session_id, "session-3")
        self.assertEqual(run.best_score, 25.0)


class AutoCalibrationRunSerializationTests(unittest.TestCase):

    # Unlike CalibrationSession (candidate/result are None after a
    # reload), EVERY field on AutoCalibrationRun is already a plain
    # snapshot -- to_dict()/from_dict() should be a genuine, full
    # round trip, including a real (not None) AutoCalibrationBudget
    # object.

    def test_round_trip_preserves_every_field(self):

        run = _make_run()
        run.mark_running()
        run.record_evaluation("session-1", 12.5)
        run.record_evaluation("session-2", 4.0)
        run.mark_completed()

        restored = AutoCalibrationRun.from_dict(run.to_dict())

        self.assertEqual(restored.run_id, run.run_id)
        self.assertEqual(restored.project_id, run.project_id)
        self.assertEqual(restored.search_space_description, run.search_space_description)
        self.assertEqual(restored.objective_description, run.objective_description)
        self.assertEqual(restored.objective_direction, run.objective_direction)
        self.assertEqual(restored.strategy_description, run.strategy_description)
        self.assertEqual(restored.search_seed, run.search_seed)
        self.assertEqual(restored.status, run.status)
        self.assertEqual(restored.session_ids, run.session_ids)
        self.assertEqual(restored.best_session_id, run.best_session_id)
        self.assertEqual(restored.best_score, run.best_score)

    def test_budget_survives_the_round_trip_as_a_real_reconstructed_object(self):

        # The one genuine departure from CalibrationSession's own
        # persistence pattern -- see automatic_calibration/run.py's own
        # module docstring.
        run = _make_run()
        restored = AutoCalibrationRun.from_dict(run.to_dict())

        self.assertIsInstance(restored.budget, AutoCalibrationBudget)
        self.assertEqual(restored.budget, run.budget)

    def test_unrecognised_status_raises_corrupted_run_record_error(self):

        data = _make_run().to_dict()
        data["status"] = "NOT_A_REAL_STATUS"

        with self.assertRaises(CorruptedRunRecordError):
            AutoCalibrationRun.from_dict(data)

    def test_unknown_top_level_keys_are_folded_into_extra_not_discarded(self):

        data = _make_run().to_dict()
        data["a_field_a_future_schema_version_added"] = "some value"

        restored = AutoCalibrationRun.from_dict(data)

        self.assertEqual(restored.extra["a_field_a_future_schema_version_added"], "some value")


if __name__ == "__main__":
    unittest.main()
