import unittest

from calibration_benchmark import WalkingSpeedCandidate
from calibration_benchmark.harness import CalibrationBenchmarkResult

from calibration_studio.git_provenance import GitProvenance
from calibration_studio.session import (
    SCHEMA_VERSION,
    CalibrationSession,
    CorruptedSessionRecordError,
    SessionStatus,
)


def _candidate():
    return WalkingSpeedCandidate("Adult_Default", 0.65, "Julich test", "test rationale")


def _fake_result():
    return CalibrationBenchmarkResult(
        candidate=_candidate(), n_scenarios_requested=1, n_completed_pairs=1,
        baseline_samples=(), candidate_samples=(),
    )


class CalibrationSessionRoundTripTests(unittest.TestCase):

    def test_pending_session_round_trips(self):

        session = CalibrationSession(
            project_id="proj-1", benchmark_id="nist-10story",
            candidate=_candidate(), master_seed=90210,
            git_provenance=GitProvenance(commit_hash="a" * 40, dirty=True),
        )

        restored = CalibrationSession.from_dict(session.to_dict())

        self.assertEqual(restored.session_id, session.session_id)
        self.assertEqual(restored.project_id, "proj-1")
        self.assertEqual(restored.benchmark_id, "nist-10story")
        self.assertEqual(restored.master_seed, 90210)
        self.assertEqual(restored.created_at, session.created_at)
        self.assertEqual(restored.status, SessionStatus.PENDING)

    def test_git_provenance_round_trips_exactly_not_recaptured(self):

        session = CalibrationSession(git_provenance=GitProvenance(commit_hash="b" * 40, dirty=False))

        restored = CalibrationSession.from_dict(session.to_dict())

        self.assertEqual(restored.git_commit_hash, "b" * 40)
        self.assertIs(restored.git_dirty, False)

    def test_candidate_snapshot_round_trips_but_live_candidate_does_not(self):

        session = CalibrationSession(candidate=_candidate())

        restored = CalibrationSession.from_dict(session.to_dict())

        self.assertIsNone(restored.candidate)
        self.assertEqual(restored.candidate_snapshot, _candidate().describe())

    def test_running_session_with_progress_round_trips(self):

        session = CalibrationSession()
        session.mark_running(n_scenarios_total=10)
        session.update_progress(3)

        restored = CalibrationSession.from_dict(session.to_dict())

        self.assertEqual(restored.status, SessionStatus.RUNNING)
        self.assertEqual(restored.n_scenarios_completed, 3)
        self.assertEqual(restored.n_scenarios_total, 10)
        self.assertAlmostEqual(restored.progress, 0.3)
        self.assertEqual(restored.started_at, session.started_at)

    def test_completed_session_result_snapshot_and_reproducible_round_trip(self):

        session = CalibrationSession()
        session.mark_running()
        session.mark_completed(_fake_result())

        restored = CalibrationSession.from_dict(session.to_dict())

        self.assertEqual(restored.status, SessionStatus.COMPLETED)
        self.assertIsNone(restored.result)
        self.assertIsNotNone(restored.result_snapshot)
        self.assertEqual(restored.reproducible, session.reproducible)
        self.assertEqual(restored.completed_at, session.completed_at)

    def test_failed_session_round_trips_failure_reason(self):

        session = CalibrationSession()
        session.mark_running()
        session.mark_failed("simulation crashed")

        restored = CalibrationSession.from_dict(session.to_dict())

        self.assertEqual(restored.status, SessionStatus.FAILED)
        self.assertEqual(restored.failure_reason, "simulation crashed")

    def test_extra_round_trips(self):

        session = CalibrationSession(extra={"note": "manual override"})

        restored = CalibrationSession.from_dict(session.to_dict())

        self.assertEqual(restored.extra, {"note": "manual override"})


class CalibrationSessionForwardCompatibilityTests(unittest.TestCase):

    def test_missing_optional_fields_use_sensible_defaults(self):

        minimal = {"schema_version": SCHEMA_VERSION, "session_id": "sess-minimal"}

        restored = CalibrationSession.from_dict(minimal)

        self.assertEqual(restored.session_id, "sess-minimal")
        self.assertEqual(restored.status, SessionStatus.PENDING)
        self.assertEqual(restored.n_scenarios_completed, 0)
        self.assertIsNone(restored.n_scenarios_total)
        self.assertIsNone(restored.master_seed)
        self.assertEqual(restored.simulator_id, "synevac")
        self.assertIsNone(restored.candidate_snapshot)
        self.assertEqual(restored.extra, {})

    def test_unknown_top_level_field_is_preserved_in_extra_not_dropped(self):

        session = CalibrationSession()
        data = session.to_dict()
        data["a_future_field_this_version_has_never_heard_of"] = {"nested": 42}

        restored = CalibrationSession.from_dict(data)

        self.assertEqual(
            restored.extra["a_future_field_this_version_has_never_heard_of"], {"nested": 42},
        )

    def test_unknown_field_does_not_collide_with_a_known_field(self):

        session = CalibrationSession(master_seed=1)
        data = session.to_dict()
        data["some_new_field"] = "value"

        restored = CalibrationSession.from_dict(data)

        self.assertEqual(restored.master_seed, 1)
        self.assertEqual(restored.extra["some_new_field"], "value")

    def test_unrecognised_status_raises_corrupted_session_record_error(self):

        session = CalibrationSession()
        data = session.to_dict()
        data["status"] = "NOT_A_REAL_STATUS"

        with self.assertRaises(CorruptedSessionRecordError):
            CalibrationSession.from_dict(data)

    def test_missing_git_provenance_block_degrades_to_none_none(self):

        minimal = {"schema_version": SCHEMA_VERSION, "session_id": "sess-2"}

        restored = CalibrationSession.from_dict(minimal)

        self.assertIsNone(restored.git_commit_hash)
        self.assertIsNone(restored.git_dirty)


if __name__ == "__main__":
    unittest.main()
