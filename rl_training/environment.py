from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np

from models.building import Building

from scenario_definition import ScenarioDefinition

from scenario_generator import derive_scenario_seed
from scenario_pipeline import run_pipeline
from scenario_runner import run as run_scenario

from ground_truth.analyzer import SimulationArtifacts, analyze

from simulation_interactive import InteractiveSimulation
from simulation_interactive.state_snapshot import build_snapshot

from rl_training.action_space import ActionMapper
from rl_training.observation_space import (
    BROADCAST_EVACUATE,
    BROADCAST_NONE,
    BROADCAST_SHELTER,
    RECOMMENDATION_EXIT,
    RECOMMENDATION_NONE,
    RECOMMENDATION_STAIR,
    ObservationEncoder,
)
from rl_training.reward_function import ActionState, RewardConfig, RewardFunction


class SynEvacGymEnv(gymnasium.Env):

    # One env instance = one fixed Building + one fixed ScenarioDefinition
    # (topology is what fixes observation_space/action_space shape --
    # Gymnasium requires those to be stable for an env instance's whole
    # life). Each reset() generates a genuinely new Scenario (new fire,
    # new occupant placement/count, new seed) on that same building/
    # definition via scenario_pipeline.run_pipeline -- reused, not
    # reimplemented. Never modifies simulation_interactive/simulator/
    # simulation_runtime/the Scenario Engine -- only ever calls their
    # existing public entry points.

    metadata = {"render_modes": []}

    def __init__(
        self,
        building: Building,
        definition: ScenarioDefinition,
        definition_id: str,
        master_seed: int,
        dt: float = 1.0,
        max_steps: int = 500,
        reward_config: Optional[RewardConfig] = None,
        registry=None,
    ):

        super().__init__()

        self._building = building
        self._definition = definition
        self._definition_id = definition_id
        self._master_seed = master_seed
        self._dt = dt
        self._max_steps = max_steps
        self._registry = registry

        self._observation_encoder = ObservationEncoder(building)
        self._action_mapper = ActionMapper(building)
        self._reward_fn = RewardFunction(reward_config)

        self.observation_space = self._observation_encoder.space
        self.action_space = self._action_mapper.space

        self._episode_index = 0
        self._step_count = 0

        self._context = None
        self._simulation = None
        self._prev_snapshot = None
        self._action_state = ActionState()
        self._active_recommendation: str = RECOMMENDATION_NONE
        self._active_broadcast: Optional[str] = None

    # =====================================================

    def _initial_snapshot(self):

        # The one place this environment reaches past
        # InteractiveSimulation's public surface: MovementStepper is a
        # public class (simulation_interactive.MovementStepper), but
        # InteractiveSimulation exposes no public accessor for the
        # instance it already owns. simulation._stepper is read-only
        # here (never assigned into), and build_snapshot() is itself a
        # public, pure function of (context, stepper, hazard_snapshot,
        # time) -- the same read-only judgment call
        # simulation_interactive/movement_stepper.py itself made about
        # simulator/coordinator.py. Only ever used at t=0, before any
        # step() has run -- every subsequent snapshot this environment
        # uses comes straight off StepResult.snapshot (fully public).
        stepper = self._simulation._stepper

        return build_snapshot(self._context, stepper, self._context.initial_hazard_snapshot, 0.0)

    # =====================================================

    def _active_recommendation_by_zone(self) -> Dict[str, str]:

        return {zone_id: self._active_recommendation for zone_id in self._observation_encoder.schema.zone_ids}

    # =====================================================

    def _encode(self, snapshot) -> np.ndarray:

        # active_recommendation_target/steps_since_recommendation_change
        # come straight off self._action_state -- the exact same
        # ActionState the RewardFunction just updated via step_reward()
        # earlier this step (see step() below: step_reward() always
        # runs before _encode()), so the observation returned here
        # always reflects precisely the state RecommendationDiscipline
        # Component will compare the NEXT action against. Closes the
        # Markov-property gap: see observation_space.py's own docstring.
        target = None
        if self._action_state.last_signature is not None:
            target = self._action_state.last_signature[1]

        return self._observation_encoder.encode(
            snapshot,
            self._active_recommendation_by_zone(),
            self._active_broadcast,
            elapsed_time=snapshot.time,
            max_time=self._max_steps * self._dt,
            active_recommendation_target=target,
            steps_since_recommendation_change=self._action_state.steps_since_last_change,
        )

    # =====================================================

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)

        base_seed = seed if seed is not None else self._master_seed
        scenario_seed = derive_scenario_seed(base_seed, self._episode_index)
        self._episode_index += 1

        result = run_pipeline(self._definition, self._definition_id, self._building, scenario_seed)

        if not result.accepted:
            raise RuntimeError(
                f"Scenario generation failed for seed {scenario_seed!r}: "
                f"{result.failure_reason!r}. No repair/retry is performed here, matching "
                f"the Scenario Engine's own 'no repair' convention."
            )

        self._scenario = result.scenario
        self._context = run_scenario(self._scenario, self._building)
        self._simulation = InteractiveSimulation(
            self._context, dt=self._dt, registry=self._registry,
        )

        self._step_count = 0
        self._action_state = ActionState()
        self._active_recommendation = RECOMMENDATION_NONE
        self._active_broadcast = None

        self._prev_snapshot = self._initial_snapshot()

        observation = self._encode(self._prev_snapshot)

        return observation, {}

    # =====================================================

    def _update_action_bookkeeping(self, action_index: int) -> None:

        entry = self._action_mapper.schema.entries[action_index]

        if entry.action_type is None:
            return

        name = entry.action_type.name

        if name == "RECOMMEND_EXIT":
            self._active_recommendation = RECOMMENDATION_EXIT
        elif name == "RECOMMEND_STAIR":
            self._active_recommendation = RECOMMENDATION_STAIR
        elif name == "BROADCAST_EVACUATE":
            self._active_broadcast = BROADCAST_EVACUATE
        elif name == "BROADCAST_SHELTER_IN_PLACE":
            self._active_broadcast = BROADCAST_SHELTER

    # =====================================================

    def step(self, action: int):

        for queued_action in self._action_mapper.decode(action):
            self._simulation.queue_action(queued_action)

        step_result = self._simulation.step()
        self._step_count += 1

        self._update_action_bookkeeping(action)

        reward, breakdown = self._reward_fn.step_reward(
            self._prev_snapshot, step_result.snapshot, step_result.applied_actions,
            self._action_state,
        )

        terminated = bool(self._simulation.is_finished)
        truncated = self._step_count >= self._max_steps

        if terminated or truncated:
            movement_result = self._simulation._stepper.snapshot_result()
            ground_truth = analyze(
                SimulationArtifacts(
                    scenario=self._scenario, building=self._building,
                    movement_result=movement_result,
                )
            )
            terminal_bonus, terminal_breakdown = self._reward_fn.terminal_reward(ground_truth)
            reward += terminal_bonus
            breakdown.update(terminal_breakdown)

        self._prev_snapshot = step_result.snapshot

        observation = self._encode(step_result.snapshot)

        info: Dict[str, Any] = {
            "replanned_occupant_ids": step_result.replanned_occupant_ids,
            "reward_breakdown": breakdown,
        }

        return observation, reward, terminated, truncated, info

    # =====================================================

    def render(self):

        return None

    # =====================================================

    def close(self):

        pass
