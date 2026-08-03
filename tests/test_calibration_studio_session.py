import unittest

from calibration_benchmark import ParameterCandidate, WalkingSpeedCandidate
from calibration_benchmark.harness import CalibrationBenchmarkResult

from calibration_studio.session import CalibrationSession, InvalidSessionTransitionError, SessionStatus


def _candidate():

    return WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")


def _fake_result(reproducible=True):

    result = CalibrationBenchmarkResult(
        candidate=_candidate(), n_scenarios_requested=1, n_completed_pairs=1,
        baseline_samples=(), candidate_samples=(),
    )
    # reproducible is a derived property (baseline/candidate_randomness-
    # based); this fixture only ever needs a real CalibrationBenchmarkResult
    # instance to prove mark_completed()/reproducible delegate to it, not
    # a specific reproducible value -- constructed via the real type, not
    # a mock, so a change to CalibrationBenchmarkResult's own shape would
    # break this test rather than silently drift from it.
    return result


class CalibrationSessionIdentityTests(unittest.TestCase):

    def test_session_id_is_generated_and_nonempty(self):

        session = CalibrationSession()
        self.assertTrue(session.session_id)

    def test_two_sessions_get_distinct_ids(self):

        self.assertNotEqual(CalibrationSession().session_id, CalibrationSession().session_id)

    def test_created_at_is_set(self):

        self.assertTrue(CalibrationSession().created_at)

    def test_project_id_and_benchmark_id_default_to_none(self):

        session = CalibrationSession()
        self.assertIsNone(session.project_id)
        self.assertIsNone(session.benchmark_id)

    def test_project_id_and_benchmark_id_are_stored_verbatim(self):

        session = CalibrationSession(project_id="proj-1", benchmark_id="nist-10story")
        self.assertEqual(session.project_id, "proj-1")
        self.assertEqual(session.benchmark_id, "nist-10story")


class CalibrationSessionCandidateCompositionTests(unittest.TestCase):

    def test_candidate_defaults_to_none(self):

        self.assertIsNone(CalibrationSession().candidate)

    def test_candidate_holds_the_real_calibration_benchmark_type_by_reference(self):

        # Composition, not duplication: the exact object passed in is
        # what comes back out -- CalibrationSession never copies,
        # wraps, or reimplements ParameterCandidate.
        candidate = _candidate()
        session = CalibrationSession(candidate=candidate)

        self.assertIsInstance(session.candidate, ParameterCandidate)
        self.assertIs(session.candidate, candidate)


class CalibrationSessionReproducibilityMetadataTests(unittest.TestCase):

    def test_master_seed_round_trips(self):

        self.assertEqual(CalibrationSession(master_seed=90210).master_seed, 90210)

    def test_simulator_id_defaults_to_synevac(self):

        self.assertEqual(CalibrationSession().simulator_id, "synevac")

    def test_git_provenance_is_captured_on_construction(self):

        # This repository IS a git repo (the whole test suite runs
        # inside it), so a real commit hash must be present.
        session = CalibrationSession()
        self.assertIsNotNone(session.git_commit_hash)
        self.assertEqual(len(session.git_commit_hash), 40)

    def test_reproducible_is_none_before_completion(self):

        self.assertIsNone(CalibrationSession().reproducible)


class CalibrationSessionProgressTests(unittest.TestCase):

    def test_progress_is_none_when_total_is_unknown(self):

        session = CalibrationSession()
        session.mark_running()

        self.assertIsNone(session.progress)

    def test_progress_reflects_completed_over_total(self):

        session = CalibrationSession()
        session.mark_running(n_scenarios_total=10)
        session.update_progress(4)

        self.assertAlmostEqual(session.progress, 0.4)

    def test_progress_is_capped_at_one(self):

        session = CalibrationSession()
        session.mark_running(n_scenarios_total=10)
        session.update_progress(15)

        self.assertEqual(session.progress, 1.0)

    def test_update_progress_before_running_raises(self):

        session = CalibrationSession()

        with self.assertRaises(InvalidSessionTransitionError):
            session.update_progress(1)


class CalibrationSessionStatusMachineTests(unittest.TestCase):

    def test_starts_pending(self):

        self.assertEqual(CalibrationSession().status, SessionStatus.PENDING)

    def test_mark_running_transitions_and_sets_started_at(self):

        session = CalibrationSession()
        session.mark_running(n_scenarios_total=5)

        self.assertEqual(session.status, SessionStatus.RUNNING)
        self.assertIsNotNone(session.started_at)
        self.assertEqual(session.n_scenarios_total, 5)

    def test_mark_completed_stores_result_and_completed_at(self):

        session = CalibrationSession()
        session.mark_running()

        result = _fake_result()
        session.mark_completed(result)

        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertIs(session.result, result)
        self.assertIsNotNone(session.completed_at)

    def test_reproducible_delegates_to_the_attached_result(self):

        session = CalibrationSession()
        session.mark_running()

        result = _fake_result()
        session.mark_completed(result)

        self.assertEqual(session.reproducible, result.reproducible)

    def test_mark_failed_stores_reason(self):

        session = CalibrationSession()
        session.mark_running()
        session.mark_failed("simulation crashed")

        self.assertEqual(session.status, SessionStatus.FAILED)
        self.assertEqual(session.failure_reason, "simulation crashed")

    def test_mark_cancelled_from_pending(self):

        session = CalibrationSession()
        session.mark_cancelled()

        self.assertEqual(session.status, SessionStatus.CANCELLED)

    def test_mark_cancelled_from_running(self):

        session = CalibrationSession()
        session.mark_running()
        session.mark_cancelled()

        self.assertEqual(session.status, SessionStatus.CANCELLED)

    def test_pending_to_completed_directly_is_invalid(self):

        session = CalibrationSession()

        with self.assertRaises(InvalidSessionTransitionError):
            session.mark_completed(_fake_result())

    def test_completed_is_terminal(self):

        session = CalibrationSession()
        session.mark_running()
        session.mark_completed(_fake_result())

        with self.assertRaises(InvalidSessionTransitionError):
            session.mark_cancelled()

    def test_failed_is_terminal(self):

        session = CalibrationSession()
        session.mark_running()
        session.mark_failed("boom")

        with self.assertRaises(InvalidSessionTransitionError):
            session.mark_running()

    def test_cancelled_is_terminal(self):

        session = CalibrationSession()
        session.mark_cancelled()

        with self.assertRaises(InvalidSessionTransitionError):
            session.mark_running()


class CalibrationSessionToDictTests(unittest.TestCase):

    def test_to_dict_reflects_current_state(self):

        session = CalibrationSession(
            project_id="proj-1", benchmark_id="nist-10story",
            candidate=_candidate(), master_seed=90210,
        )
        session.mark_running(n_scenarios_total=2)
        session.update_progress(1)

        as_dict = session.to_dict()

        self.assertEqual(as_dict["session_id"], session.session_id)
        self.assertEqual(as_dict["project_id"], "proj-1")
        self.assertEqual(as_dict["benchmark_id"], "nist-10story")
        self.assertEqual(as_dict["master_seed"], 90210)
        self.assertEqual(as_dict["status"], "RUNNING")
        self.assertEqual(as_dict["n_scenarios_completed"], 1)
        self.assertEqual(as_dict["n_scenarios_total"], 2)
        self.assertAlmostEqual(as_dict["progress"], 0.5)
        self.assertIsInstance(as_dict["candidate"], dict)


if __name__ == "__main__":
    unittest.main()
