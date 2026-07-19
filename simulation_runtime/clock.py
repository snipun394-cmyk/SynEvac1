from dataclasses import dataclass
from typing import Sequence

from scenario import ScenarioEvent

from simulator import MultiAgentSimulationResult


# Simulation Runtime -- architecture doc docs/architecture/simulation_runtime.md
# SS5/SS6/SS9. One clock, one unit (seconds since scenario t=0), one fixed dt --
# this module owns only that bookkeeping, never a subsystem call (SS20's own
# package-structure note: "no subsystem calls" for this file specifically).


def resolve_default_end_time(
    movement_result: MultiAgentSimulationResult,
    scheduled_events: Sequence[ScenarioEvent],
) -> float:

    # SS9's exact documented formula: the later of "every occupant who will
    # ever arrive has arrived" and "every scheduled event has fired". Not a
    # claim this is correct for every deployment -- a caller-supplied
    # end_time always takes precedence (SimulationRuntime.__init__).

    last_arrival = movement_result.total_evacuation_time or 0.0
    last_event_time = scheduled_events[-1].time if scheduled_events else 0.0

    return max(last_arrival, last_event_time)


@dataclass
class SimulationClock:

    # Deliberately mutable -- the same "immutable container, mutable
    # runtime companion" split SimulationContext.simulation and
    # ScenarioEventExecutor's own cursor already established, one level up.
    # peek_next_time() never mutates; only advance() does, and only ever
    # called after every subsystem call for a tick has already succeeded
    # (SS8/SS15: current_time must never reflect a tick that raised).

    dt: float
    end_time: float
    current_time: float = 0.0

    # =====================================================

    def __post_init__(self):

        if self.dt <= 0.0:
            raise ValueError(f"SimulationClock requires dt > 0.0, got {self.dt}.")

        if self.end_time < 0.0:
            raise ValueError(f"SimulationClock requires end_time >= 0.0, got {self.end_time}.")

    # =====================================================

    @property
    def is_finished(self) -> bool:

        return self.current_time >= self.end_time

    # =====================================================

    def peek_next_time(self) -> float:

        return min(self.current_time + self.dt, self.end_time)

    # =====================================================

    def advance(self) -> float:

        self.current_time = self.peek_next_time()
        return self.current_time
