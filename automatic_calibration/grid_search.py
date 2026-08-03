import itertools
from typing import Dict, Optional, Sequence, Tuple

from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY

from calibration_benchmark import ParameterCandidate

from calibration_studio.session import CalibrationSession

from automatic_calibration.search_space import SearchSpace
from automatic_calibration.strategy import AutoCalibrationStrategy


# =====================================================
# Automatic Calibration Engine, Phase 2 -- Grid Search Strategy.
#
# GridSearchStrategy is the first concrete AutoCalibrationStrategy.
# Exhaustive only -- no sampling, no adaptation, no convergence
# criterion. Every architectural constraint from Phase 1 is honored
# unchanged: this strategy NEVER executes an experiment itself, it only
# ever returns the next ParameterCandidate (or None); AutoCalibrationEngine
# remains the only thing that calls into CalibrationStudio.
#
# THE ONE-CANDIDATE-PER-EVALUATION CONSTRAINT, and how this module
# resolves it for genuinely joint multi-dimensional search: calibration_
# benchmark.run_calibration_benchmark() (and therefore
# CalibrationStudio.run_published_benchmark()/run_parameter_sweep())
# always compares baseline against exactly ONE ParameterCandidate --
# there is no mechanism anywhere in this codebase for combining two
# DIFFERENT ParameterCandidate instances into one simulated run. Phase
# 1's own AutoCalibrationStrategy.propose() interface (unchanged here)
# reflects that: it returns Optional[ParameterCandidate], singular.
#
# For a true Cartesian-product grid (not just N independent one-factor
# sweeps -- see this module's own tests for why that distinction
# matters for "duplicate prevention"), each grid POINT in a
# multi-dimensional space must therefore become ONE candidate. This
# module builds that candidate by COMPOSING each dimension's own
# already-valid ParameterCandidate (built via ParameterDimension.
# build_candidate()) into a single _JointGridPointCandidate --
# never by modifying calibration_benchmark, never by reaching into any
# frozen production class. Composition only succeeds when it is
# genuinely unambiguous:
#
# - candidate_registry(): sub-candidates that touch DIFFERENT profile_ids
#   (e.g. Adult_Default's walking speed + Child_Default's walking speed)
#   compose cleanly, since each only ever replaces its own profile's
#   entry. Two sub-candidates that customize the SAME profile_id (e.g.
#   Adult_Default's walking speed + Adult_Default's compliance level)
#   are a genuine conflict this module does not attempt to field-merge
#   -- it raises a clear ValueError instead of silently picking one or
#   producing a wrong composite.
# - candidate_capacity_model()/candidate_congestion_model()/
#   candidate_density_thresholds()/candidate_use_flow_regions(): at
#   most ONE dimension in a grid point may override any single one of
#   these hooks (composing two different capacity_model overrides, for
#   instance, would require actually merging two model OBJECTS, which
#   this module does not attempt) -- more than one raises the same
#   kind of clear, immediate ValueError.
#
# For a ONE-dimensional SearchSpace, no composite is built at all --
# propose() returns the dimension's own natural ParameterCandidate
# directly (e.g. a plain WalkingSpeedCandidate), preserving its own
# meaningful name/subsystem/unit for reporting -- exactly matching this
# milestone's own "one-dimensional search" requirement as a first-class
# case, not a degenerate wrapper.
# =====================================================


def _overrides(candidate: ParameterCandidate, method_name: str) -> bool:

    # Detects whether `candidate`'s own CLASS customizes this hook, at
    # all, without needing to compare arbitrary model objects for
    # equality (StairCapacityModel/StairAwareCongestionModel/... define
    # no __eq__, so `!=` would always be True even for two functionally
    # identical instances -- this is the correct, unambiguous check).

    return getattr(type(candidate), method_name) is not getattr(ParameterCandidate, method_name)


class _JointGridPointCandidate(ParameterCandidate):

    # Every conflict check happens ONCE, eagerly, at construction --
    # propose() therefore fails immediately on an invalid grid
    # combination, never deep inside a wasted simulation run.

    def __init__(self, sub_candidates: Tuple[ParameterCandidate, ...]):

        self.sub_candidates = sub_candidates

        joined_name = " & ".join(c.name for c in sub_candidates)

        super().__init__(
            name=f"GridSearch[{joined_name}]",
            subsystem="Grid Search (joint)",
            calibration_tier="Tier 2",
            dataset_source="automatic_calibration.grid_search",
            current_value=tuple(c.current_value for c in sub_candidates),
            candidate_value=tuple(c.candidate_value for c in sub_candidates),
            unit="composite",
            rationale=f"Joint grid point combining: {joined_name}",
        )

        self._resolved_registry = self._resolve_registry()
        self._resolved_capacity_model = self._resolve_single("candidate_capacity_model")
        self._resolved_congestion_model = self._resolve_single("candidate_congestion_model")
        self._resolved_density_thresholds = self._resolve_single("candidate_density_thresholds")
        self._resolved_use_flow_regions = self._resolve_single("candidate_use_flow_regions")

    def _resolve_registry(self):

        overriding = [c for c in self.sub_candidates if _overrides(c, "candidate_registry")]

        if not overriding:
            return super().candidate_registry()

        merged = dict(DEFAULT_PROFILE_REGISTRY)

        for c in overriding:

            sub_registry = c.candidate_registry()

            for profile_id, template in sub_registry.items():

                if template == DEFAULT_PROFILE_REGISTRY.get(profile_id):
                    continue

                already_customized = (
                    profile_id in merged
                    and merged[profile_id] != DEFAULT_PROFILE_REGISTRY.get(profile_id)
                    and merged[profile_id] != template
                )

                if already_customized:
                    raise ValueError(
                        f"GridSearchStrategy: two dimensions in this grid point both customize "
                        f"profile {profile_id!r}'s registry entry -- cannot compose them jointly "
                        f"(field-level merging of two customizations of the same profile is not "
                        f"supported). Combine them into a single ParameterDimension instead.",
                    )

                merged[profile_id] = template

        return merged

    def _resolve_single(self, method_name: str):

        overriding = [c for c in self.sub_candidates if _overrides(c, method_name)]

        if not overriding:
            return getattr(super(), method_name)()

        if len(overriding) > 1:
            raise ValueError(
                f"GridSearchStrategy: more than one dimension in this grid point overrides "
                f"{method_name} -- cannot compose them jointly (merging two customizations of "
                f"the same model hook is not supported).",
            )

        return getattr(overriding[0], method_name)()

    def candidate_registry(self):
        return self._resolved_registry

    def candidate_capacity_model(self):
        return self._resolved_capacity_model

    def candidate_congestion_model(self):
        return self._resolved_congestion_model

    def candidate_density_thresholds(self):
        return self._resolved_density_thresholds

    def candidate_use_flow_regions(self):
        return self._resolved_use_flow_regions


class GridSearchStrategy(AutoCalibrationStrategy):

    def __init__(self, values: Dict[str, Sequence[float]]):

        # `values` maps EACH ParameterDimension.name this strategy will
        # be used against to its own explicit, ordered list of grid
        # values -- deliberately not derived from ParameterDimension.
        # bounds (a continuous range, not a set of discrete points this
        # strategy could invent on its own without a caller-chosen
        # resolution/spacing).

        self._values: Dict[str, Tuple[float, ...]] = {}

        for name, raw_values in values.items():

            resolved = tuple(raw_values)

            if not resolved:
                raise ValueError(f"GridSearchStrategy: no grid values supplied for dimension {name!r}.")

            self._values[name] = resolved

    def _grid_points(self, search_space: SearchSpace) -> Sequence[tuple]:

        for dimension in search_space.dimensions:

            if dimension.name not in self._values:
                raise ValueError(
                    f"GridSearchStrategy has no grid values configured for dimension "
                    f"{dimension.name!r} -- pass values={{{dimension.name!r}: [...]}} at "
                    f"construction.",
                )

            lo, hi = dimension.bounds

            for value in self._values[dimension.name]:

                if not (lo <= value <= hi):
                    raise ValueError(
                        f"GridSearchStrategy: grid value {value!r} for dimension "
                        f"{dimension.name!r} is outside its declared bounds {dimension.bounds!r}.",
                    )

        # itertools.product's own documented, deterministic order:
        # dimensions are iterated in search_space.dimensions' own given
        # order (whatever that is -- "arbitrary ParameterDimension
        # ordering" is honored simply by never reordering it), and the
        # RIGHTMOST (last) dimension varies fastest, like an odometer.
        value_lists = [self._values[dimension.name] for dimension in search_space.dimensions]

        return list(itertools.product(*value_lists))

    def propose(
        self, *, search_space: SearchSpace, objective, history: Sequence[CalibrationSession],
    ) -> Optional[ParameterCandidate]:

        dimensions = search_space.dimensions
        grid_points = self._grid_points(search_space)

        already_evaluated = {self._signature_of(session) for session in history}
        already_evaluated.discard(None)

        for combination in grid_points:

            signature = combination[0] if len(dimensions) == 1 else combination

            if signature in already_evaluated:
                # Resume-safe / duplicate-prevention: a grid point
                # already represented in `history` (whether from this
                # same in-process run continuing, or from sessions
                # RELOADED from a previous, partially completed run --
                # see _signature_of()'s own fallback to
                # candidate_snapshot below) is skipped, never
                # re-proposed.
                continue

            if len(dimensions) == 1:
                return dimensions[0].build_candidate(combination[0])

            sub_candidates = tuple(
                dimension.build_candidate(value) for dimension, value in zip(dimensions, combination)
            )
            return _JointGridPointCandidate(sub_candidates)

        # Every grid point already appears in history -- the grid is
        # exhausted. The strategy's own generic exhaustion signal (see
        # AutoCalibrationStrategy.propose()'s own docstring).
        return None

    @staticmethod
    def _signature_of(session: CalibrationSession):

        # Reads candidate_value from whichever of the session's own two
        # representations is available -- the LIVE candidate (same-
        # process, mid-run) or candidate_snapshot (a session reloaded
        # from persistence, exactly what a genuine cross-process resume
        # would hand this strategy). Both are already-established
        # CalibrationSession properties (Calibration Studio Phase 2);
        # nothing new is read here.

        if session.candidate is not None:
            value = session.candidate.candidate_value
        elif session.candidate_snapshot is not None:
            value = session.candidate_snapshot.get("candidate_value")
        else:
            return None

        # A composite grid point's candidate_value is a tuple, which
        # JSON round-trips as a list -- normalized back to a tuple so a
        # reloaded session's signature compares equal to a freshly
        # computed one.
        if isinstance(value, list):
            value = tuple(value)

        return value

    def describe(self) -> str:

        dims = ", ".join(f"{name}={list(values)!r}" for name, values in self._values.items())

        return f"GridSearchStrategy(values={{{dims}}}) -- exhaustive Cartesian-product grid search"
