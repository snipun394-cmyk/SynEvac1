from typing import Tuple

from scenario import ScenarioEvent
from scenario_runner import SimulationContext

from scenario_event_executor.handlers import execute_event


class ScenarioEventExecutor:

    # The runtime component architecture doc
    # docs/architecture/scenario_event_execution.md designs -- not a
    # second priority-queue scheduler (§4 of that document: Scenario
    # Events are static and already time-ordered, so a simple,
    # already-sorted cursor is the architecturally correct structure,
    # not a simplification of one). Owns exactly one piece of runtime
    # state -- a cursor into context.scheduled_events -- kept separate
    # from the immutable SimulationContext it was constructed from
    # (§7), the same "immutable container, mutable runtime companion"
    # split SimulationContext.simulation (itself a mutable
    # MultiAgentSimulation) already establishes.
    #
    # advance_to() mirrors hazard_evolution/provider.py::
    # EvolutionBackedHazardProvider's own calling convention exactly
    # (a target_time-driven, pull-based, forward-only advancement) --
    # the one existing precedent in this codebase for driving
    # time-based state progression independent of MultiAgentSimulation's
    # own event heap (§3/§10).
    #
    # Modifies only runtime simulation state (context.building's
    # engineering objects, context.graph's edge list) -- never
    # generates, never validates, never touches MultiAgentSimulation,
    # HazardEvolutionEngine, or HumanBehaviorLayer (§11/§12: no data
    # dependency exists in either direction), and never replans an
    # occupant's route (§2: routes are fixed at registration in the
    # already-frozen simulator/ package, unmodified and unmodifiable
    # here).

    def __init__(self, context: SimulationContext):

        self.context = context

        self._next_index = 0
        self._last_target_time = 0.0

    # =====================================================

    @property
    def is_complete(self) -> bool:

        return self._next_index >= len(self.context.scheduled_events)

    # =====================================================

    @property
    def remaining_events(self) -> Tuple[ScenarioEvent, ...]:

        return self.context.scheduled_events[self._next_index:]

    # =====================================================

    def advance_to(self, target_time: float) -> Tuple[ScenarioEvent, ...]:

        # Fires every not-yet-fired event whose time <= target_time, in
        # the order they already appear in context.scheduled_events
        # (already sorted, already conflict-checked by
        # scenario_validator upstream -- §8/§9 of the architecture
        # document: this method performs no sorting and no
        # re-validation of its own). Forward-only, matching
        # EvolutionBackedHazardProvider's own "cannot evolve backwards"
        # contract -- a smaller target_time than one already reached
        # would fire nothing new regardless (the cursor never resets),
        # so raising here turns a likely caller bug into an immediate,
        # loud error instead of a silent no-op.

        if target_time < self._last_target_time:

            raise ValueError(
                f"ScenarioEventExecutor cannot advance backwards -- requested "
                f"target_time {target_time}, already advanced to "
                f"{self._last_target_time}."
            )

        self._last_target_time = target_time

        events = self.context.scheduled_events
        fired = []

        while self._next_index < len(events) and events[self._next_index].time <= target_time:

            event = events[self._next_index]

            execute_event(self.context, event)

            fired.append(event)
            self._next_index += 1

        return tuple(fired)
