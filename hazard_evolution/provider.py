from hazard.provider import HazardProvider
from hazard.snapshot import HazardSnapshot

from hazard_evolution.engine import HazardEvolutionEngine


class EvolutionBackedHazardProvider(HazardProvider):

    # Bridges HazardEvolutionEngine back into V1's existing consumer-
    # facing seam: HazardAwareCostModel, HazardAwareCapacityModel, and
    # DecisionContext.hazard_snapshot were all built against
    # HazardProvider.snapshot_at(time) and nothing about them changes
    # to consume an evolving incident instead of ManualHazardProvider's
    # single fixed snapshot -- they only ever see HazardSnapshot
    # instances either way.
    #
    # Internally advances the engine step-by-step (in increments of at
    # most `dt`) from whatever time it last reached up to the
    # requested time, caching every snapshot it lands on exactly so a
    # later request for an already-visited time is free and returns
    # the identical instance. That caching/timeline bookkeeping is an
    # implementation detail, not part of the public API -- callers
    # only ever see snapshot_at(time).
    #
    # Forward-only, matching the engine's own evolve(): a source's
    # propose() is defined in terms of "what happens next given the
    # previous snapshot", which has no defined meaning run backwards.
    # Requesting a time before the last one reached (and not already
    # an exact, previously-visited grid point) raises ValueError rather
    # than silently fabricating a result.

    def __init__(
        self,
        engine: HazardEvolutionEngine,
        initial_snapshot: HazardSnapshot,
        dt: float,
    ):

        self.engine = engine
        self.dt = dt

        self._timeline = {initial_snapshot.timestamp: initial_snapshot}
        self._last_time = initial_snapshot.timestamp

    # =====================================================

    def snapshot_at(self, time: float) -> HazardSnapshot:

        if time in self._timeline:
            return self._timeline[time]

        if time < self._last_time:
            raise ValueError(
                f"EvolutionBackedHazardProvider cannot evolve backwards -- "
                f"requested time {time}, already advanced to {self._last_time}, "
                f"and {time} was never landed on exactly."
            )

        current_time = self._last_time
        current_snapshot = self._timeline[current_time]

        while current_time < time:

            step = min(self.dt, time - current_time)
            current_snapshot = self.engine.evolve(current_snapshot, current_time, step)
            current_time += step

            self._timeline[current_time] = current_snapshot

        self._last_time = current_time

        return current_snapshot
