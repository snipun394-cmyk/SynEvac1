import json
import unittest

from calibration_benchmark import CapacityWidthCandidate, ComplianceLevelCandidate, StairCounterflowPenaltyCandidate, WalkingSpeedCandidate

from calibration_studio.session import CalibrationSession

from automatic_calibration.budget import AutoCalibrationBudget
from automatic_calibration.engine import AutoCalibrationEngine
from automatic_calibration.grid_search import GridSearchStrategy, _JointGridPointCandidate
from automatic_calibration.objectives import CalibrationObjective, ObjectiveDirection
from automatic_calibration.run import AutoCalibrationRunStatus
from automatic_calibration.search_space import ParameterDimension, SearchSpace

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


def _walking_speed_dimension(profile_id="Adult_Default", bounds=(0.6, 1.6)):

    return ParameterDimension(
        name=f"{profile_id}.walking_speed", bounds=bounds,
        build=lambda v: WalkingSpeedCandidate(profile_id, v, "test", "test"),
    )


def _compliance_dimension(profile_id="Adult_Default", bounds=(0.0, 1.0)):

    return ParameterDimension(
        name=f"{profile_id}.compliance_level", bounds=bounds,
        build=lambda v: ComplianceLevelCandidate(profile_id, v, "test", "test"),
    )


def _counterflow_dimension(bounds=(0.0, 1.0)):

    return ParameterDimension(
        name="StairAwareCongestionModel.COUNTERFLOW_PENALTY_PER_OPPOSING", bounds=bounds,
        build=lambda v: StairCounterflowPenaltyCandidate(v, "test", "test"),
    )


def _capacity_width_dimension(stair_specific=False, bounds=(0.5, 5.0)):

    return ParameterDimension(
        name=f"CapacityWidth.stair_specific={stair_specific}", bounds=bounds,
        build=lambda v: CapacityWidthCandidate(v, "test", "test", stair_specific=stair_specific),
    )


def _session_with_candidate_value(value):

    # A LIVE session, as if still in-process.
    class _FakeCandidate:
        candidate_value = value

    session = CalibrationSession()
    session._candidate = _FakeCandidate()  # noqa: SLF001 -- test-only, exercising the "live" branch directly
    return session


def _reloaded_session_with_candidate_value(value):

    # Simulates a session RELOADED from persistence: candidate is None,
    # only candidate_snapshot (a plain dict, exactly what
    # CalibrationSession.from_dict() itself constructs) survives.
    return CalibrationSession(candidate=None, candidate_snapshot={"candidate_value": value})


class OneDimensionalGridTests(unittest.TestCase):

    def test_proposes_the_raw_sub_candidate_not_a_composite(self):

        space = SearchSpace(dimensions=(_walking_speed_dimension(),))
        strategy = GridSearchStrategy(values={"Adult_Default.walking_speed": [0.8, 1.0, 1.2]})

        candidate = strategy.propose(search_space=space, objective=None, history=())

        self.assertIsInstance(candidate, WalkingSpeedCandidate)
        self.assertNotIsInstance(candidate, _JointGridPointCandidate)
        self.assertEqual(candidate.candidate_value, 0.8)

    def test_visits_every_grid_value_exactly_once_then_exhausts(self):

        space = SearchSpace(dimensions=(_walking_speed_dimension(),))
        strategy = GridSearchStrategy(values={"Adult_Default.walking_speed": [0.8, 1.0, 1.2]})

        history = []
        seen = []

        for _ in range(4):

            candidate = strategy.propose(search_space=space, objective=None, history=tuple(history))

            if candidate is None:
                break

            seen.append(candidate.candidate_value)
            history.append(_session_with_candidate_value(candidate.candidate_value))

        self.assertEqual(seen, [0.8, 1.0, 1.2])

        # A fifth call, grid already exhausted -- must return None, not
        # repeat.
        self.assertIsNone(strategy.propose(search_space=space, objective=None, history=tuple(history)))


class TwoDimensionalGridTests(unittest.TestCase):

    def test_visits_the_full_cartesian_product_as_joint_candidates(self):

        space = SearchSpace(dimensions=(_walking_speed_dimension(), _counterflow_dimension()))
        strategy = GridSearchStrategy(values={
            "Adult_Default.walking_speed": [0.8, 1.2],
            "StairAwareCongestionModel.COUNTERFLOW_PENALTY_PER_OPPOSING": [0.1, 0.2, 0.3],
        })

        history = []
        seen = []

        for _ in range(10):

            candidate = strategy.propose(search_space=space, objective=None, history=tuple(history))

            if candidate is None:
                break

            self.assertIsInstance(candidate, _JointGridPointCandidate)
            seen.append(candidate.candidate_value)
            history.append(_session_with_candidate_value(candidate.candidate_value))

        # 2 x 3 = 6 total grid points, every one distinct.
        self.assertEqual(len(seen), 6)
        self.assertEqual(len(set(seen)), 6)

    def test_deterministic_order_matches_itertools_product_rightmost_fastest(self):

        space = SearchSpace(dimensions=(_walking_speed_dimension(), _counterflow_dimension()))
        strategy = GridSearchStrategy(values={
            "Adult_Default.walking_speed": [0.8, 1.2],
            "StairAwareCongestionModel.COUNTERFLOW_PENALTY_PER_OPPOSING": [0.1, 0.2],
        })

        expected_order = [(0.8, 0.1), (0.8, 0.2), (1.2, 0.1), (1.2, 0.2)]

        history = []
        actual_order = []

        for _ in range(4):

            candidate = strategy.propose(search_space=space, objective=None, history=tuple(history))
            actual_order.append(candidate.candidate_value)
            history.append(_session_with_candidate_value(candidate.candidate_value))

        self.assertEqual(actual_order, expected_order)

    def test_composite_candidate_actually_applies_both_dimensions(self):

        space = SearchSpace(dimensions=(_walking_speed_dimension(), _counterflow_dimension()))
        strategy = GridSearchStrategy(values={
            "Adult_Default.walking_speed": [0.9],
            "StairAwareCongestionModel.COUNTERFLOW_PENALTY_PER_OPPOSING": [0.42],
        })

        candidate = strategy.propose(search_space=space, objective=None, history=())

        registry = candidate.candidate_registry()
        self.assertEqual(registry["Adult_Default"].walking_speed, 0.9)

        congestion_model = candidate.candidate_congestion_model()
        self.assertEqual(congestion_model.COUNTERFLOW_PENALTY_PER_OPPOSING, 0.42)


class ArbitraryDimensionOrderingTests(unittest.TestCase):

    def test_swapping_dimension_order_changes_which_axis_varies_fastest(self):

        values = {
            "Adult_Default.walking_speed": [0.8, 1.2],
            "StairAwareCongestionModel.COUNTERFLOW_PENALTY_PER_OPPOSING": [0.1, 0.2],
        }

        forward = SearchSpace(dimensions=(_walking_speed_dimension(), _counterflow_dimension()))
        reversed_space = SearchSpace(dimensions=(_counterflow_dimension(), _walking_speed_dimension()))

        forward_strategy = GridSearchStrategy(values=dict(values))
        reversed_strategy = GridSearchStrategy(values=dict(values))

        forward_first = forward_strategy.propose(search_space=forward, objective=None, history=())
        reversed_first = reversed_strategy.propose(search_space=reversed_space, objective=None, history=())

        # Same underlying values, but the FIRST proposed combination's
        # own tuple order reflects each SearchSpace's own dimension
        # order -- (speed, penalty) vs (penalty, speed) -- proving the
        # strategy honors search_space.dimensions verbatim rather than
        # imposing its own fixed convention.
        self.assertEqual(forward_first.candidate_value, (0.8, 0.1))
        self.assertEqual(reversed_first.candidate_value, (0.1, 0.8))


class ThreeDimensionalGridTests(unittest.TestCase):

    def test_three_non_conflicting_dimensions_compose_and_enumerate_the_full_product(self):

        space = SearchSpace(dimensions=(
            _walking_speed_dimension("Adult_Default"),
            _walking_speed_dimension("Child_Default", bounds=(0.5, 1.3)),
            _counterflow_dimension(),
        ))
        strategy = GridSearchStrategy(values={
            "Adult_Default.walking_speed": [0.8, 1.2],
            "Child_Default.walking_speed": [0.6, 0.9],
            "StairAwareCongestionModel.COUNTERFLOW_PENALTY_PER_OPPOSING": [0.1, 0.2],
        })

        history = []
        seen = []

        for _ in range(20):

            candidate = strategy.propose(search_space=space, objective=None, history=tuple(history))

            if candidate is None:
                break

            seen.append(candidate.candidate_value)
            history.append(_session_with_candidate_value(candidate.candidate_value))

        # 2 x 2 x 2 = 8 total grid points.
        self.assertEqual(len(seen), 8)
        self.assertEqual(len(set(seen)), 8)

    def test_three_dimension_composite_applies_all_three_profiles_and_hooks(self):

        space = SearchSpace(dimensions=(
            _walking_speed_dimension("Adult_Default"),
            _walking_speed_dimension("Child_Default", bounds=(0.5, 1.3)),
            _counterflow_dimension(),
        ))
        strategy = GridSearchStrategy(values={
            "Adult_Default.walking_speed": [1.1],
            "Child_Default.walking_speed": [0.7],
            "StairAwareCongestionModel.COUNTERFLOW_PENALTY_PER_OPPOSING": [0.33],
        })

        candidate = strategy.propose(search_space=space, objective=None, history=())
        registry = candidate.candidate_registry()

        self.assertEqual(registry["Adult_Default"].walking_speed, 1.1)
        self.assertEqual(registry["Child_Default"].walking_speed, 0.7)
        self.assertEqual(candidate.candidate_congestion_model().COUNTERFLOW_PENALTY_PER_OPPOSING, 0.33)


class ConflictDetectionTests(unittest.TestCase):

    def test_two_dimensions_customizing_the_same_profile_raise(self):

        space = SearchSpace(dimensions=(
            _walking_speed_dimension("Adult_Default"),
            _compliance_dimension("Adult_Default"),
        ))
        strategy = GridSearchStrategy(values={
            "Adult_Default.walking_speed": [1.1],
            "Adult_Default.compliance_level": [0.5],
        })

        with self.assertRaises(ValueError):
            strategy.propose(search_space=space, objective=None, history=())

    def test_two_dimensions_overriding_the_same_capacity_model_hook_raise(self):

        space = SearchSpace(dimensions=(
            _capacity_width_dimension(stair_specific=False),
            _capacity_width_dimension(stair_specific=True),
        ))
        strategy = GridSearchStrategy(values={
            "CapacityWidth.stair_specific=False": [2.0],
            "CapacityWidth.stair_specific=True": [1.0],
        })

        with self.assertRaises(ValueError):
            strategy.propose(search_space=space, objective=None, history=())


class ConfigurationValidationTests(unittest.TestCase):

    def test_missing_grid_values_for_a_dimension_raises(self):

        space = SearchSpace(dimensions=(_walking_speed_dimension(),))
        strategy = GridSearchStrategy(values={})

        with self.assertRaises(ValueError):
            strategy.propose(search_space=space, objective=None, history=())

    def test_grid_value_outside_declared_bounds_raises(self):

        space = SearchSpace(dimensions=(_walking_speed_dimension(bounds=(0.6, 1.6)),))
        strategy = GridSearchStrategy(values={"Adult_Default.walking_speed": [5.0]})

        with self.assertRaises(ValueError):
            strategy.propose(search_space=space, objective=None, history=())

    def test_empty_grid_values_list_raises_at_construction(self):

        with self.assertRaises(ValueError):
            GridSearchStrategy(values={"x": []})


class DuplicatePreventionTests(unittest.TestCase):

    def test_no_grid_point_is_ever_proposed_twice_across_a_full_walk(self):

        space = SearchSpace(dimensions=(_walking_speed_dimension(), _counterflow_dimension()))
        strategy = GridSearchStrategy(values={
            "Adult_Default.walking_speed": [0.8, 1.0, 1.2],
            "StairAwareCongestionModel.COUNTERFLOW_PENALTY_PER_OPPOSING": [0.1, 0.2],
        })

        history = []
        all_proposed = []

        for _ in range(6):

            candidate = strategy.propose(search_space=space, objective=None, history=tuple(history))
            all_proposed.append(candidate.candidate_value)
            history.append(_session_with_candidate_value(candidate.candidate_value))

        self.assertEqual(len(all_proposed), len(set(all_proposed)))
        self.assertEqual(len(all_proposed), 6)


class ResumeSafeIterationTests(unittest.TestCase):

    def test_skips_grid_points_already_present_as_live_sessions(self):

        space = SearchSpace(dimensions=(_walking_speed_dimension(),))
        strategy = GridSearchStrategy(values={"Adult_Default.walking_speed": [0.8, 1.0, 1.2]})

        history = (_session_with_candidate_value(0.8),)

        candidate = strategy.propose(search_space=space, objective=None, history=history)

        self.assertEqual(candidate.candidate_value, 1.0)

    def test_skips_grid_points_already_present_as_reloaded_snapshot_only_sessions(self):

        # The genuine cross-process resume case: candidate is None,
        # only candidate_snapshot survives -- exactly what
        # CalibrationSession.from_dict() produces after a real
        # save/load round trip.
        space = SearchSpace(dimensions=(_walking_speed_dimension(),))
        strategy = GridSearchStrategy(values={"Adult_Default.walking_speed": [0.8, 1.0, 1.2]})

        history = (_reloaded_session_with_candidate_value(0.8), _reloaded_session_with_candidate_value(1.0))

        candidate = strategy.propose(search_space=space, objective=None, history=history)

        self.assertEqual(candidate.candidate_value, 1.2)

    def test_skips_a_composite_grid_point_after_a_real_json_round_trip(self):

        # Full round trip through CalibrationSession.to_dict()/from_dict()
        # (not just a hand-built snapshot dict) -- proves resume works
        # against genuinely persisted-and-reloaded data for the
        # multi-dimensional (composite) case too, where candidate_value
        # is a tuple that JSON round-trips as a list.
        space = SearchSpace(dimensions=(_walking_speed_dimension(), _counterflow_dimension()))
        strategy = GridSearchStrategy(values={
            "Adult_Default.walking_speed": [0.8, 1.2],
            "StairAwareCongestionModel.COUNTERFLOW_PENALTY_PER_OPPOSING": [0.1, 0.2],
        })

        first_candidate = strategy.propose(search_space=space, objective=None, history=())
        self.assertEqual(first_candidate.candidate_value, (0.8, 0.1))

        # Genuine JSON round trip (not just to_dict()/from_dict() in
        # memory, which would leave the tuple untouched since Python
        # dicts freely hold tuples) -- this is what actually happens
        # after a real save_session()/load_session() disk round trip,
        # and is the only way candidate_value's tuple->list conversion
        # is exercised for real.
        live_session = CalibrationSession(candidate=first_candidate)
        json_text = json.dumps(live_session.to_dict(), default=str)
        reloaded_session = CalibrationSession.from_dict(json.loads(json_text))

        self.assertIsNone(reloaded_session.candidate)
        self.assertIsInstance(reloaded_session.candidate_snapshot["candidate_value"], list)

        second_candidate = strategy.propose(search_space=space, objective=None, history=(reloaded_session,))

        self.assertEqual(second_candidate.candidate_value, (0.8, 0.2))


class AutoCalibrationStrategyInterfaceComplianceTests(unittest.TestCase):

    def test_grid_search_strategy_is_an_auto_calibration_strategy(self):

        from automatic_calibration.strategy import AutoCalibrationStrategy

        self.assertIsInstance(GridSearchStrategy(values={"x": [1.0]}), AutoCalibrationStrategy)

    def test_describe_lists_every_dimensions_own_grid_values(self):

        strategy = GridSearchStrategy(values={"a": [1.0, 2.0], "b": [3.0]})
        description = strategy.describe()

        self.assertIn("a", description)
        self.assertIn("b", description)
        self.assertIn("1.0", description)


class _DummyMaximizeObjective(CalibrationObjective):

    direction = ObjectiveDirection.MAXIMIZE

    def score(self, session):

        if session.result is None:
            return None

        comparison = session.result.comparisons.get("evacuation_time")

        return comparison.candidate_mean if comparison else None

    def describe(self):

        return "_DummyMaximizeObjective -- test fixture only"


class EngineIntegrationTests(unittest.TestCase):

    # End-to-end through the real AutoCalibrationEngine (Phase 1,
    # unmodified) -- proves GridSearchStrategy genuinely works as a
    # first-class AutoCalibrationStrategy, not just in isolation.

    def setUp(self):

        from calibration_studio.studio import CalibrationStudio

        self.studio = CalibrationStudio()
        self.project = self.studio.create_project(name="Grid Search Engine Tests")
        self.building = make_building()
        self.definition = make_definition()

    def _engine(self, values, max_evaluations):

        space = SearchSpace(dimensions=(_walking_speed_dimension(bounds=(0.6, 1.6)),))
        strategy = GridSearchStrategy(values=values)

        return AutoCalibrationEngine(
            studio=self.studio, project=self.project, building=self.building, definition=self.definition,
            definition_id=DEFINITION_ID, master_seed=MASTER_SEED, n_scenarios=2, dt=1.0,
            search_space=space, objective=_DummyMaximizeObjective(), strategy=strategy,
            budget=AutoCalibrationBudget(max_evaluations=max_evaluations),
        )

    def test_budget_smaller_than_grid_size_stops_after_the_budget_not_the_whole_grid(self):

        engine = self._engine({"Adult_Default.walking_speed": [0.8, 1.0, 1.2, 1.4, 1.6]}, max_evaluations=3)
        run = engine.run()

        self.assertEqual(run.status, AutoCalibrationRunStatus.COMPLETED)
        self.assertEqual(run.n_evaluations, 3)

    def test_the_budget_limited_run_evaluates_the_first_n_grid_points_in_order(self):

        engine = self._engine({"Adult_Default.walking_speed": [0.8, 1.0, 1.2, 1.4, 1.6]}, max_evaluations=3)
        run = engine.run()

        evaluated_values = [
            self.studio.get_session(sid).candidate_snapshot["candidate_value"] for sid in run.session_ids
        ]

        self.assertEqual(evaluated_values, [0.8, 1.0, 1.2])

    def test_grid_smaller_than_budget_stops_when_the_grid_is_exhausted_not_the_budget(self):

        engine = self._engine({"Adult_Default.walking_speed": [0.8, 1.0]}, max_evaluations=10)
        run = engine.run()

        self.assertEqual(run.n_evaluations, 2)

    def test_simulated_resume_continues_from_a_prior_partial_runs_own_sessions(self):

        # First "run": only 2 of 4 grid points, as if the process was
        # interrupted after that budget.
        first_engine = self._engine({"Adult_Default.walking_speed": [0.8, 1.0, 1.2, 1.4]}, max_evaluations=2)
        first_run = first_engine.run()

        prior_sessions = tuple(self.studio.get_session(sid) for sid in first_run.session_ids)

        # A FRESH GridSearchStrategy instance (as a new process would
        # construct), given the prior run's own sessions as history,
        # must propose the next unvisited grid point -- never repeat
        # 0.8 or 1.0.
        resumed_strategy = GridSearchStrategy(values={"Adult_Default.walking_speed": [0.8, 1.0, 1.2, 1.4]})
        space = SearchSpace(dimensions=(_walking_speed_dimension(bounds=(0.6, 1.6)),))

        next_candidate = resumed_strategy.propose(search_space=space, objective=None, history=prior_sessions)

        self.assertEqual(next_candidate.candidate_value, 1.2)


if __name__ == "__main__":
    unittest.main()
