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
        rng=None,
    ):

        context = DecisionContext(
            graph=self.graph,
            engine=self.engine,
            profile=profile,
            start_id=start_id,
            decisions_so_far=dict(self._decisions_so_far),
            rng=rng,
        )

        intent = decision_strategy.decide(context)

        if intent.requires_movement:

            route_strategy = route_choice_strategy or ShortestRouteChoiceStrategy()
            route_choice = route_strategy.choose(context)

            delay_strategy = pre_movement_strategy or NoPreMovementDelay()
            delay = delay_strategy.delay(context)

            # Movement was required (intent.requires_movement) but the
            # Navigation Stage found no route to any exit -- a
            # structural disconnection, not a choice to stay put.
            # Flagged explicitly via route_unavailable rather than left
            # for submit_decision() to infer from goal_id/route alone:
            # both this case and a genuine WAIT/IGNORE decision produce
            # goal_id=None/route=None, which is indistinguishable
            # without this flag. See docs/validation/technical_report.md
            # §6 for how this gap was found.
            # Occupant Attributes integration -- profile.walking_speed
            # itself is never mutated (a real, protected guarantee this
            # codebase relies on); only this decision's own effective
            # speed incorporates the occupant's deterministic
            # "effective_walking_speed_multiplier" trait (combining
            # walking_speed_multiplier/stamina/fatigue_resistance/
            # mobility_factor into one already-computed factor, set by
            # the Behaviour Profile Resolver's own registration step).
            # Absent for any profile that predates this feature --
            # defaults to 1.0, i.e. exactly profile.walking_speed,
            # completely unchanged. profile.walking_speed may itself be
            # None (BehaviorProfile's own documented default, e.g. a
            # profile built without an explicit speed) -- guarded rather
            # than multiplied, the same "None means not set, never
            # coerced to a number" honesty every other Optional[float]
            # in this codebase already gets.
            #
            # Skipped for an ASSISTED occupant specifically: behavior_
            # library.assistance_strategies.AssistanceAwareRouteChoiceStrategy
            # (an existing, unmodified strategy) already overwrites
            # profile.walking_speed with their helper's own already-
            # finalized effective speed ("the assisted occupant then
            # moves at that same already-reduced pace" -- that class's
            # own docstring) -- applying this occupant's own multiplier
            # on top would double-count it. "assistance_role"/"ASSISTED"
            # are that module's own trait vocabulary, restated here as
            # plain strings (not imported) rather than reversing the
            # behavior_library-depends-on-behavior direction.
            if profile.walking_speed is None or profile.traits.get("assistance_role") == "ASSISTED":
                effective_speed = profile.walking_speed
            else:
                effective_speed = profile.walking_speed * profile.traits.get(
                    "effective_walking_speed_multiplier", 1.0,
                )

            decision = BehaviorDecision(
                occupant_id=profile.occupant_id,
                action_type=intent.action_type,
                start_id=start_id,
                goal_id=route_choice.goal_id,
                route=route_choice.route,
                walking_speed=effective_speed,
                # Edge-Type-Specific Movement Speed (Experimental Branch
                # V1) -- carried straight through from profile.stair_speed,
                # no multiplier applied, same "None stays None" contract
                # as dynamic_registrar's identical site.
                stair_speed=profile.stair_speed,
                depart_time=base_depart_time + delay,
                metadata=intent.metadata,
                route_unavailable=(route_choice.goal_id is None and route_choice.route is None),
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
