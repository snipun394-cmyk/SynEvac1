from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from scenario import ScenarioEvent

from scenario_event_executor import ScenarioEventExecutor

from scenario_runner.context import SimulationContext

from hazard_evolution.provider import EvolutionBackedHazardProvider

from simulation_interactive.action_executor import Action, ActionExecutor, ActionResult
from simulation_interactive.movement_stepper import MovementStepper
from simulation_interactive.route_manager import RouteManager
from simulation_interactive.state_snapshot import SimulationSnapshot, build_snapshot


# Engineering-mutating ScenarioEvent target_types -- events fired by
# ScenarioEventExecutor (scripted, baked into the Scenario at
# generation time) mutate the same door/exit/stair fields ActionExecutor
# mutates for externally-injected actions, so both must feed the same
# RouteManager.note_engineering_change() seam.
_ENGINEERING_EVENT_TARGET_TYPES = ("door", "exit", "stair")


@dataclass(frozen=True)
class StepResult:

    time: float
    fired_events: Tuple[ScenarioEvent, ...]
    applied_actions: Tuple[ActionResult, ...]
    replanned_occupant_ids: Tuple[str, ...]
    snapshot: SimulationSnapshot


class InteractiveSimulation:

    # Deliberately never calls context.simulation.run(). Occupants
    # only ever advance through MovementStepper.advance_to(), one
    # event at a time, driven by repeated step() calls -- the
    # prerequisite the RL-readiness investigation identified as
    # missing: routes can now be recomputed mid-episode in response to
    # door/exit/stair changes, hazard changes, and externally injected
    # recommendations, all without modifying simulator/, behavior/,
    # scenario_runner/, scenario_event_executor/, or hazard_evolution/.

    def __init__(
        self,
        context: SimulationContext,
        dt: float,
        registry=None,
        hazard_dt: Optional[float] = None,
    ):

        self._context = context
        self._dt = dt

        self._stepper = MovementStepper(context.simulation)

        self._route_manager = RouteManager(context, registry=registry)
        self._route_manager.register_all()

        self._action_executor = ActionExecutor(context, self._route_manager)
        self._event_executor = ScenarioEventExecutor(context)
        self._hazard_provider = EvolutionBackedHazardProvider(
            context.hazard_engine, context.initial_hazard_snapshot, hazard_dt or dt,
        )

        self._clock_time = 0.0
        self._pending_actions: List[Action] = []

    # =====================================================

    @property
    def current_time(self) -> float:

        return self._clock_time

    # =====================================================

    @property
    def is_finished(self) -> bool:

        return self._stepper.is_drained and not self._pending_actions

    # =====================================================

    def queue_action(self, action: Action) -> None:

        self._pending_actions.append(action)

    # =====================================================

    def step(self, dt: Optional[float] = None) -> StepResult:

        target_time = self._clock_time + (dt if dt is not None else self._dt)

        self._route_manager.start_tick()

        fired_events = self._event_executor.advance_to(target_time)

        for event in fired_events:
            if event.target_type in _ENGINEERING_EVENT_TARGET_TYPES:
                self._route_manager.note_engineering_change([event.target_id])

        pending_actions, self._pending_actions = self._pending_actions, []
        applied_actions = tuple(
            self._action_executor.apply(action, target_time) for action in pending_actions
        )

        hazard_snapshot = self._hazard_provider.snapshot_at(target_time)

        replanned = list(
            self._route_manager.replan_due_at_node_occupants(
                target_time, hazard_snapshot, self._stepper,
            )
        )

        def _on_arrive_at_node(occupant_id: str, time: float) -> None:
            if self._route_manager.maybe_replan(occupant_id, time, hazard_snapshot, self._stepper):
                replanned.append(occupant_id)

        self._stepper.advance_to(target_time, on_arrive_at_node=_on_arrive_at_node)

        self._clock_time = target_time

        snapshot = build_snapshot(self._context, self._stepper, hazard_snapshot, target_time)

        return StepResult(
            time=target_time,
            fired_events=fired_events,
            applied_actions=applied_actions,
            replanned_occupant_ids=tuple(replanned),
            snapshot=snapshot,
        )

    # =====================================================

    def run_to_completion(self, max_time: Optional[float] = None) -> Tuple[StepResult, ...]:

        # Convenience/testing helper only -- loops step() until
        # is_finished. Not used by, and does not replace,
        # SimulationRuntime.run(); exists so tests (and callers with no
        # actions left to inject) don't need to hand-loop step().

        results = []

        while not self.is_finished:

            if max_time is not None and self._clock_time >= max_time:
                break

            results.append(self.step())

        return tuple(results)
