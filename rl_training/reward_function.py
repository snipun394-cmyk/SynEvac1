import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from simulation_interactive import ActionResult, SimulationSnapshot


@dataclass(frozen=True)
class RewardConfig:

    # Every weight is named and defaulted here -- no inline magic
    # constants anywhere else in this module. Defaults are a
    # documented, initial engineering choice (the same honesty
    # convention HazardSeverity's score cutoffs / TSquaredFireGrowthCurve's
    # growth_time default already use elsewhere in this codebase), not a
    # validated life-safety tuning -- callers are expected to override
    # these once real training runs show what matters.

    evacuation_progress_weight: float = 1.0
    # + evacuation_progress_weight per occupant newly ARRIVED this step.

    time_penalty_weight: float = 0.01
    # - time_penalty_weight per elapsed step, while occupants remain --
    # encourages faster evacuation without waiting for the episode to
    # end to say so.

    congestion_weight: float = 0.05
    # - congestion_weight * (sum of current edge_congestion) per step.

    exit_balance_weight: float = 0.05
    # - exit_balance_weight * variance(current congestion across exits)
    # per step -- penalizes funnelling everyone through one exit while
    # others sit empty.

    unreachable_weight: float = 0.5
    # - unreachable_weight per occupant newly UNREACHABLE this step.

    redundant_action_penalty: float = 0.1
    # - flat, reissuing a recommendation/broadcast identical to the one
    # already active (wastes a decision without changing anything).

    contradictory_action_penalty: float = 0.2
    # - flat, replacing an active recommendation/broadcast with a
    # *different* one within contradiction_window_steps of the last
    # change (thrashing/back-and-forth instructions).

    contradiction_window_steps: int = 5

    completion_bonus_weight: float = 10.0
    # + flat, once, at episode end, if GroundTruth.building_cleared.

    final_evacuation_time_weight: float = 0.02
    # - final_evacuation_time_weight * GroundTruth.total_evacuation_time,
    # at episode end (0 contribution if total_evacuation_time is None).

    final_trapped_weight: float = 1.0
    # - final_trapped_weight * GroundTruth.people_trapped, at episode end.


@dataclass
class ActionState:

    # The env's own small, mutable tracker of "what have I already told
    # the simulation," passed into RewardFunction.step_reward() rather
    # than owned by it -- keeps reward components pure/testable against
    # hand-built states, with no hidden env coupling.

    last_signature: Optional[Tuple[str, Any]] = None
    steps_since_last_change: int = 10 ** 9

    def observe(
        self, action_type_name: Optional[str], target: Any, contradiction_window_steps: int,
    ) -> Tuple[bool, bool]:

        # Returns (is_redundant, is_contradictory) for this step's
        # single most-recent non-NOOP action (if any), and updates
        # this tracker's own state accordingly.

        if action_type_name is None:
            self.steps_since_last_change += 1
            return False, False

        signature = (action_type_name, target)

        is_redundant = signature == self.last_signature
        is_contradictory = (
            not is_redundant
            and self.last_signature is not None
            and self.steps_since_last_change < contradiction_window_steps
        )

        if not is_redundant:
            self.last_signature = signature
            self.steps_since_last_change = 0
        else:
            self.steps_since_last_change += 1

        return is_redundant, is_contradictory


class RewardComponent:

    name = "component"

    def compute(
        self, prev_snapshot: SimulationSnapshot, snapshot: SimulationSnapshot,
        applied_actions: Tuple[ActionResult, ...], action_state: ActionState,
        config: RewardConfig,
    ) -> float:

        raise NotImplementedError


class EvacuationProgressComponent(RewardComponent):

    name = "evacuation_progress"

    def compute(self, prev_snapshot, snapshot, applied_actions, action_state, config):

        newly_evacuated = snapshot.people_evacuated - prev_snapshot.people_evacuated

        return config.evacuation_progress_weight * max(newly_evacuated, 0)


class TimePenaltyComponent(RewardComponent):

    name = "time_penalty"

    def compute(self, prev_snapshot, snapshot, applied_actions, action_state, config):

        if snapshot.people_remaining <= 0:
            return 0.0

        return -config.time_penalty_weight


class CongestionComponent(RewardComponent):

    name = "congestion"

    def compute(self, prev_snapshot, snapshot, applied_actions, action_state, config):

        total_congestion = sum(snapshot.edge_congestion.values())

        return -config.congestion_weight * total_congestion


class ExitBalanceComponent(RewardComponent):

    name = "exit_balance"

    def compute(self, prev_snapshot, snapshot, applied_actions, action_state, config):

        exit_congestion = [
            snapshot.edge_congestion.get(exit_id, 0) for exit_id in snapshot.exit_states
        ]

        if len(exit_congestion) < 2:
            return 0.0

        return -config.exit_balance_weight * statistics.pvariance(exit_congestion)


class UnreachableOccupantsComponent(RewardComponent):

    name = "unreachable_occupants"

    def compute(self, prev_snapshot, snapshot, applied_actions, action_state, config):

        prev_unreachable = sum(1 for o in prev_snapshot.occupants if o.state == "UNREACHABLE")
        unreachable = sum(1 for o in snapshot.occupants if o.state == "UNREACHABLE")

        return -config.unreachable_weight * max(unreachable - prev_unreachable, 0)


class RecommendationDisciplineComponent(RewardComponent):

    name = "recommendation_discipline"

    def compute(self, prev_snapshot, snapshot, applied_actions, action_state, config):

        applied = [result for result in applied_actions if result.applied]

        if not applied:
            action_state.observe(None, None, config.contradiction_window_steps)
            return 0.0

        # Only the last applied action this step is scored -- an
        # environment step queues at most one logical instruction
        # (a "recommend"/"broadcast"/"deploy" choice); RECOMMEND_EXIT/
        # STAIR fan out to one Action per zone, but they all share the
        # same (action_type, target) signature, so scoring the last one
        # is equivalent to scoring the single decision that produced them.
        last = applied[-1]
        action_type_name = last.action.action_type.name
        target = last.action.parameters.get("exit_id") or last.action.parameters.get("stair_id")

        is_redundant, is_contradictory = action_state.observe(
            action_type_name, target, config.contradiction_window_steps,
        )

        penalty = 0.0
        if is_redundant:
            penalty -= config.redundant_action_penalty
        if is_contradictory:
            penalty -= config.contradictory_action_penalty

        return penalty


class RewardFunction:

    _STEP_COMPONENTS = (
        EvacuationProgressComponent(),
        TimePenaltyComponent(),
        CongestionComponent(),
        ExitBalanceComponent(),
        UnreachableOccupantsComponent(),
        RecommendationDisciplineComponent(),
    )

    def __init__(self, config: Optional[RewardConfig] = None):

        self.config = config or RewardConfig()

    # =====================================================

    def step_reward(
        self, prev_snapshot: SimulationSnapshot, snapshot: SimulationSnapshot,
        applied_actions: Tuple[ActionResult, ...], action_state: ActionState,
    ) -> Tuple[float, Dict[str, float]]:

        breakdown = {
            component.name: component.compute(
                prev_snapshot, snapshot, applied_actions, action_state, self.config,
            )
            for component in self._STEP_COMPONENTS
        }

        return sum(breakdown.values()), breakdown

    # =====================================================

    def terminal_reward(self, ground_truth) -> Tuple[float, Dict[str, float]]:

        # Uses GroundTruth -- a privileged, only-computable-after-the-
        # fact signal. This is the one place in the RL stack where that
        # is legitimate: reward shapes training, it is not what the
        # policy observes (ObservationEncoder never touches
        # ground_truth at all -- see observation_space.py's own
        # docstring and tests/test_rl_training.py::ObservationHonestyTests).

        completion_bonus = (
            self.config.completion_bonus_weight if ground_truth.building_cleared else 0.0
        )

        evacuation_time_penalty = 0.0
        if ground_truth.total_evacuation_time is not None:
            evacuation_time_penalty = (
                -self.config.final_evacuation_time_weight * ground_truth.total_evacuation_time
            )

        trapped_penalty = -self.config.final_trapped_weight * ground_truth.people_trapped

        breakdown = {
            "completion_bonus": completion_bonus,
            "final_evacuation_time": evacuation_time_penalty,
            "final_trapped": trapped_penalty,
        }

        return sum(breakdown.values()), breakdown
