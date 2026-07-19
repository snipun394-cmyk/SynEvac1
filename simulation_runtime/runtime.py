from typing import Optional, Tuple

from ai_decision.engine import DecisionEngine

from perception.providers.provider import PerceptionProvider

from scenario_runner import SimulationContext

from scenario_event_executor import ScenarioEventExecutor

from hazard_evolution.provider import EvolutionBackedHazardProvider

from simulation_runtime.clock import SimulationClock, resolve_default_end_time
from simulation_runtime.occupancy_bridge import MovementTimelineOccupancyProvider
from simulation_runtime.result import TickResult


class SimulationRuntime:

    # The single orchestrator -- architecture doc
    # docs/architecture/simulation_runtime.md, this pass. Coordinates
    # ScenarioEventExecutor, EvolutionBackedHazardProvider, a movement-
    # timeline-backed OccupancyProvider (SS11), an already-constructed
    # DecisionEngine, and (optionally) an already-constructed
    # PerceptionProvider and a Dataset Logger -- it replaces none of them,
    # calling only their own already-public, already-frozen seams
    # (advance_to, snapshot_at, decide, observation_at).
    #
    # Occupant movement is NOT ticked (SS2/SS10): every occupant's route is
    # already fixed by the time a SimulationContext exists (behaviour
    # resolution is the *caller's* responsibility, completed before a
    # SimulationRuntime is ever constructed, SS7 steps 1-2). This
    # constructor performs the one activation step nothing upstream was
    # ever scoped to perform -- context.simulation.run() -- exactly once,
    # here, and the resulting MultiAgentSimulationResult is read-only for
    # the rest of this Runtime's life (SS10). This Runtime never calls
    # add_occupant()/submit_decision() itself.

    def __init__(
        self,
        context: SimulationContext,
        decision_engine: DecisionEngine,
        dt: float,
        perception_provider: Optional[PerceptionProvider] = None,
        dataset_logger=None,
        end_time: Optional[float] = None,
    ):

        self.context = context
        self.decision_engine = decision_engine
        self.perception_provider = perception_provider
        self.dataset_logger = dataset_logger

        # SS7 step 3 -- the first, and only, call this Runtime ever makes
        # to context.simulation. Drains whatever event heap
        # register_occupants() already populated, in full, once.
        self.movement_result = context.simulation.run()

        self.event_executor = ScenarioEventExecutor(context)

        self.hazard_provider = EvolutionBackedHazardProvider(
            context.hazard_engine, context.initial_hazard_snapshot, dt,
        )

        self.occupancy_provider = MovementTimelineOccupancyProvider(self.movement_result)

        resolved_end_time = (
            end_time if end_time is not None
            else resolve_default_end_time(self.movement_result, context.scheduled_events)
        )

        self.clock = SimulationClock(dt=dt, end_time=resolved_end_time)

    # =====================================================

    @property
    def current_time(self) -> float:

        return self.clock.current_time

    # =====================================================

    @property
    def end_time(self) -> float:

        return self.clock.end_time

    # =====================================================

    @property
    def is_finished(self) -> bool:

        return self.clock.is_finished

    # =====================================================

    def tick(self) -> TickResult:

        # The per-tick sequence -- SS8, in this exact order. Event
        # execution happens first, deliberately: ScenarioEventExecutor
        # mutates context.building's Camera.active/Detector.active fields
        # (among others), and the Ground Truth Perception providers this
        # tick's observation_at() call may reach read those same fields
        # live -- ordering event execution first means a same-tick
        # availability change is visible this tick, not one tick late.

        if self.clock.is_finished:
            raise ValueError(
                f"SimulationRuntime.tick() called after end_time "
                f"({self.clock.end_time}) has already been reached "
                f"(current_time={self.clock.current_time})."
            )

        next_time = self.clock.peek_next_time()

        fired_events = self.event_executor.advance_to(next_time)

        hazard_snapshot = self.hazard_provider.snapshot_at(next_time)
        occupancy_snapshot = self.occupancy_provider.snapshot_at(next_time)

        decision = self.decision_engine.decide(hazard_snapshot, occupancy_snapshot, next_time)

        observation = (
            self.perception_provider.observation_at(next_time)
            if self.perception_provider is not None
            else None
        )

        if self.dataset_logger is not None:
            self.dataset_logger.on_tick(
                next_time, fired_events, hazard_snapshot, occupancy_snapshot, decision, observation,
            )

        # Advanced only after every subsystem call above has succeeded
        # (SS15: current_time must never reflect a tick that raised).
        self.clock.advance()

        return TickResult(
            time=next_time,
            fired_events=fired_events,
            hazard_snapshot=hazard_snapshot,
            occupancy_snapshot=occupancy_snapshot,
            decision=decision,
            observation=observation,
        )

    # =====================================================

    def run(self) -> Tuple[TickResult, ...]:

        # A thin convenience loop over tick() -- no coordination logic of
        # its own beyond "keep calling tick() until end_time is reached",
        # the same primitive a future step-driven RL loop (SS17) would
        # call directly instead.

        results = []

        while not self.clock.is_finished:
            results.append(self.tick())

        return tuple(results)
