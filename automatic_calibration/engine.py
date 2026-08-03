from typing import Any, Optional, Sequence

from calibration_studio.project import CalibrationProject
from calibration_studio.session import CalibrationSession
from calibration_studio.studio import CalibrationStudio

from automatic_calibration.budget import AutoCalibrationBudget
from automatic_calibration.objectives import CalibrationObjective
from automatic_calibration.run import AutoCalibrationRun
from automatic_calibration.search_space import SearchSpace
from automatic_calibration.strategy import AutoCalibrationStrategy


# =====================================================
# Automatic Calibration Engine, Phase 1 -- Core Architecture.
#
# AutoCalibrationEngine is a pure CLIENT of CalibrationStudio, never a
# peer or a replacement -- this milestone's own hard requirement. The
# only two calls this class ever makes into Calibration Studio are
# CalibrationStudio.run_published_benchmark() and
# CalibrationStudio.run_parameter_sweep() (see _evaluate() below) --
# both already produce a real, normal CalibrationSession, attached to
# the same CalibrationProject a human would use, with its own full
# lifecycle (PENDING -> RUNNING -> COMPLETED/FAILED), reproducibility
# auditing, and reporting/replay/dashboard compatibility already built
# in. Nothing in this class calls calibration_benchmark.
# run_calibration_benchmark() or run_with_overrides() directly, and
# nothing here reimplements paired comparison, statistics, or
# recommendation logic -- every evaluation's real judgment still comes
# from calibration_benchmark.recommend(), read through the session's
# own already-real result, never recomputed.
#
# A human still adopts, nothing here ever does -- AutoCalibrationEngine
# has no method that writes to any SynEvac production default; run()
# returns an AutoCalibrationRun for a human to review (via the exact
# same generate_session_report()/generate_validation_dashboard() every
# manually-run session already supports), the same hard boundary
# calibration_benchmark and calibration_studio both already enforce.
# =====================================================


class AutoCalibrationEngine:

    def __init__(
        self,
        *,
        studio: CalibrationStudio,
        project: CalibrationProject,
        building,
        definition,
        definition_id: str,
        master_seed: int,
        n_scenarios: int,
        search_space: SearchSpace,
        objective: CalibrationObjective,
        strategy: AutoCalibrationStrategy,
        budget: AutoCalibrationBudget,
        dt: float = 5.0,
        search_seed: Optional[int] = None,
        additional_metrics: Sequence[Any] = (),
    ):

        self.studio = studio
        self.project = project
        self.building = building
        self.definition = definition
        self.definition_id = definition_id
        self.master_seed = master_seed
        self.n_scenarios = n_scenarios
        self.dt = dt
        self.search_space = search_space
        self.objective = objective
        self.strategy = strategy
        self.budget = budget
        self.search_seed = search_seed
        self.additional_metrics = additional_metrics

        # Derived, never a separate constructor parameter that could
        # drift out of sync with the objective's own benchmark --
        # PublishedValueObjective already knows exactly which benchmark
        # it was built against. An objective with no notion of a
        # benchmark (a future non-benchmark-anchored objective type)
        # simply leaves this None, and every evaluation runs through
        # run_parameter_sweep() instead of run_published_benchmark().
        self.benchmark_id = getattr(objective, "benchmark_id", None)

    def run(self) -> AutoCalibrationRun:

        run = AutoCalibrationRun(
            project_id=self.project.project_id,
            benchmark_id=self.benchmark_id,
            search_space=self.search_space,
            objective_description=self.objective.describe(),
            objective_direction=self.objective.direction.value,
            strategy_description=self.strategy.describe(),
            budget=self.budget,
            search_seed=self.search_seed,
        )
        run.mark_running()

        history = []

        try:

            while len(history) < self.budget.max_evaluations:

                candidate = self.strategy.propose(
                    search_space=self.search_space, objective=self.objective, history=tuple(history),
                )

                if candidate is None:
                    # The strategy's own generic exhaustion signal (see
                    # AutoCalibrationStrategy.propose()'s own docstring)
                    # -- a normal, expected way for a run to end early,
                    # not an error.
                    break

                session = self._evaluate(candidate)

                history.append(session)

                score = self.objective.score(session)
                run.record_evaluation(session.session_id, score)

        except Exception as exc:  # noqa: BLE001 -- a run-level failure is a normal terminal outcome, not re-raised
            run.mark_failed(f"{type(exc).__name__}: {exc}")
            return run

        run.mark_completed()

        return run

    def _evaluate(self, candidate) -> CalibrationSession:

        # The ONLY two calls this class ever makes into Calibration
        # Studio -- see this module's own header comment.

        if self.benchmark_id is not None:

            return self.studio.run_published_benchmark(
                project=self.project,
                benchmark_id=self.benchmark_id,
                candidate=candidate,
                definition=self.definition,
                definition_id=self.definition_id,
                master_seed=self.master_seed,
                n_scenarios=self.n_scenarios,
                dt=self.dt,
                building=self.building,
                additional_metrics=self.additional_metrics,
            )

        sessions = self.studio.run_parameter_sweep(
            project=self.project,
            candidates=[candidate],
            building=self.building,
            definition=self.definition,
            definition_id=self.definition_id,
            master_seed=self.master_seed,
            n_scenarios=self.n_scenarios,
            dt=self.dt,
            additional_metrics=self.additional_metrics,
        )

        return sessions[0]

    def __repr__(self) -> str:

        return f"AutoCalibrationEngine(project={self.project.project_id!r}, max_evaluations={self.budget.max_evaluations})"
