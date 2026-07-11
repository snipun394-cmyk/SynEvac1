from simulator.decision import BehaviorDecision

from behavior.context import DecisionContext
from behavior.pre_movement import NoPreMovementDelay
from behavior.route_choice import ShortestRouteChoiceStrategy


class HumanBehaviorLayer:

    # The orchestrator: for one occupant, runs the Decision Stage
    # (what do they intend to do) and, only if movement is required,
    # the Navigation Stage (which goal/route, and after how long) --
    # then hands the result to Simulation as a single immutable
    # BehaviorDecision via submit_decision(). This is the *only*
    # point of contact with Simulation; the dependency direction stays
    # Behavior -> Simulation -> Pathfinding -> Navigation, since
    # BehaviorDecision itself is owned by simulator/, not behavior/.

    def __init__(self, simulation, engine=None):

        self.simulation = simulation
        self.engine = engine or simulation.engine
        self.graph = self.engine.graph

        # occupant_id -> the fully resolved BehaviorDecision made for
        # them so far in this session -- what lets a later occupant's
        # strategies (e.g. a follower's) see an earlier one's (e.g. a
        # leader's) choice via DecisionContext.decisions_so_far.
        self._decisions_so_far = {}

    # =====================================================

    def register(
        self,
        start_id,
        profile,
        decision_strategy,
        route_choice_strategy=None,
        pre_movement_strategy=None,
        base_depart_time=0.0,
    ):

        context = DecisionContext(
            graph=self.graph,
            engine=self.engine,
            profile=profile,
            start_id=start_id,
            decisions_so_far=dict(self._decisions_so_far),
        )

        intent = decision_strategy.decide(context)

        if intent.requires_movement:

            route_strategy = route_choice_strategy or ShortestRouteChoiceStrategy()
            route_choice = route_strategy.choose(context)

            delay_strategy = pre_movement_strategy or NoPreMovementDelay()
            delay = delay_strategy.delay(context)

            decision = BehaviorDecision(
                occupant_id=profile.occupant_id,
                action_type=intent.action_type,
                start_id=start_id,
                goal_id=route_choice.goal_id,
                route=route_choice.route,
                walking_speed=profile.walking_speed,
                depart_time=base_depart_time + delay,
                metadata=intent.metadata,
            )

        else:

            # WAIT/IGNORE/any non-movement intent -- Navigation Stage
            # is skipped entirely, not merely given trivial inputs.
            decision = BehaviorDecision(
                occupant_id=profile.occupant_id,
                action_type=intent.action_type,
                start_id=start_id,
                metadata=intent.metadata,
            )

        self._decisions_so_far[profile.occupant_id] = decision

        return self.simulation.submit_decision(decision)
