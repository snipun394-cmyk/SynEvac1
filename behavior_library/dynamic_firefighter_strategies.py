from simulator.decision import ActionType

from behavior.intent import ActionIntent, DecisionStrategy

from behavior_library.firefighter_strategies import (
    TRAIT_RESCUE_TARGET_OCCUPANT_ID,
    TRAIT_RESCUE_TARGET_ZONE_ID,
    FirefighterDecisionStrategy,
)

from human_decision_engine.firefighter_engine import (
    FirefighterDecisionEngine,
    TASK_GUIDE_CIVILIANS,
    TASK_REPORT_HAZARD,
)


# =====================================================
# Human Decision Engine Phase 3 -- Firefighter Decision Engine. Same
# drive-not-replace relationship to behavior_library.firefighter_strategies
# as DynamicAssistanceDecisionStrategy has to assistance_strategies:
# FirefighterDecisionEngine.evaluate() picks *which* task this
# firefighter pursues (search / rescue / assist wheelchair user / carry
# fallen / guide / report hazard / continue search / return outside --
# see that module's own docstring), this class writes the resulting
# rescue_target_occupant_id/rescue_target_zone_id traits onto
# BehaviorProfile using the exact same trait-key vocabulary
# FirefighterDecisionStrategy already reads, then delegates to it
# unconditionally for both the movement intent and (via
# FirefighterRouteChoiceStrategy, reused verbatim, no new route-choice
# class needed -- see behaviour_profile_resolver.dynamic_registrar) the
# combined rescue route.
#
# FirefighterDecisionStrategy's own ActionType vocabulary only
# distinguishes SEARCH vs. RESCUE (a rescue_target_zone_id trait is
# set, or it is not) -- this class overrides the returned ActionIntent's
# action_type to FIREFIGHTER_GUIDE/FIREFIGHTER_REPORT_HAZARD for the
# two patrol-task variants that need a more specific label (both
# ActionType members already existed, unused, before this phase); the
# richer task vocabulary itself (RESCUE vs. ASSIST_WHEELCHAIR_USER vs.
# CARRY_FALLEN, SEARCH vs. CONTINUE_SEARCH vs. RETURN_OUTSIDE) is
# carried in ActionIntent.metadata["task"] -- descriptive metadata,
# never branched on by Simulation, exactly as ActionType's own
# docstring already specifies.
# =====================================================

_ACTION_TYPE_OVERRIDE_BY_TASK = {
    TASK_GUIDE_CIVILIANS: ActionType.FIREFIGHTER_GUIDE,
    TASK_REPORT_HAZARD: ActionType.FIREFIGHTER_REPORT_HAZARD,
}


class DynamicFirefighterDecisionStrategy(DecisionStrategy):

    def __init__(self, engine: FirefighterDecisionEngine):

        self.engine = engine

    # =====================================================

    def decide(self, context) -> ActionIntent:

        task_decision = self.engine.evaluate(context)

        if task_decision.target_occupant_id is not None:
            context.profile.traits[TRAIT_RESCUE_TARGET_OCCUPANT_ID] = task_decision.target_occupant_id

        if task_decision.target_zone_id is not None:
            context.profile.traits[TRAIT_RESCUE_TARGET_ZONE_ID] = task_decision.target_zone_id

        intent = FirefighterDecisionStrategy().decide(context)

        intent.action_type = _ACTION_TYPE_OVERRIDE_BY_TASK.get(task_decision.task, intent.action_type)
        intent.metadata = dict(intent.metadata)
        intent.metadata["task"] = task_decision.task

        return intent
