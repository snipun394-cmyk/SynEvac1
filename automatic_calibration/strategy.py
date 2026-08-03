from typing import Optional, Sequence

from calibration_benchmark import ParameterCandidate

from calibration_studio.session import CalibrationSession

from automatic_calibration.objectives import CalibrationObjective
from automatic_calibration.search_space import SearchSpace


# =====================================================
# Automatic Calibration Engine, Phase 1 -- Core Architecture.
#
# Abstract interface ONLY -- this milestone's own explicit scope. No
# concrete strategy (Grid Search, Random Search, Bayesian Optimization,
# Evolutionary Algorithms, Pareto/Multi-Objective Search) is implemented
# anywhere in this package yet; AutoCalibrationEngine is built and
# tested entirely against this interface, never against a specific
# concrete strategy, so a later phase can add each one without any
# change to the engine itself.
# =====================================================


class AutoCalibrationStrategy:

    def propose(
        self, *, search_space: SearchSpace, objective: CalibrationObjective,
        history: Sequence[CalibrationSession],
    ) -> Optional[ParameterCandidate]:

        # Called once per evaluation, given the FULL evaluation history
        # so far (every real CalibrationSession this run has already
        # produced, in order) -- an adaptive strategy uses it to decide
        # its next proposal; a non-adaptive one may ignore it entirely.
        #
        # Returning None signals "no further proposals" (search-space
        # exhaustion, a convergence criterion being met, ...) -- one
        # single, generic exhaustion signal every future concrete
        # strategy can use, so AutoCalibrationEngine never needs a
        # second, strategy-specific "is this strategy done" method.

        raise NotImplementedError

    def describe(self) -> str:

        raise NotImplementedError
