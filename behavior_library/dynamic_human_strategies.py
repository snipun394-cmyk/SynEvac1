from behavior.intent import ActionIntent, AlwaysEvacuateDecisionStrategy, DecisionStrategy

from behavior_library.assistance_strategies import (
    ROLE_ASSISTED,
    ROLE_HELPER,
    TRAIT_ASSISTANCE_ROLE,
    TRAIT_ASSISTANCE_TARGET_ID,
    TRAIT_ASSISTANCE_TYPE,
    AssistanceAwareDecisionStrategy,
)

from human_decision_engine.engine import ACTION_BE_ASSISTED, ACTION_HELP, HumanDecisionEngine


# =====================================================
# Human Decision Engine Phase 1/2 -- Civilian Decision Engine /
# Assistance Decision. This is the *only* new civilian DecisionStrategy
# added by this phase; it does not replace AssistanceAwareDecisionStrategy
# (behavior_library.assistance_strategies) -- it *drives* it. For every
# occupant, HumanDecisionEngine.evaluate() is asked what this occupant
# should do (a dynamic, rule-based decision -- see that module's own
# docstring for exactly how); the outcome is written onto
# BehaviorProfile.traits using the *same* trait-key vocabulary
# AssistanceAwareDecisionStrategy already reads (TRAIT_ASSISTANCE_ROLE/
# TRAIT_ASSISTANCE_TARGET_ID/TRAIT_ASSISTANCE_TYPE), then delegated to
# it unconditionally -- the actual movement mechanics (speed
# multiplier, HELP/ESCORT/PUSH_WHEELCHAIR ActionType, ASSISTED
# metadata) are entirely that existing, unmodified class's own code.
# The only thing this class changes is *how* those traits get set:
# dynamically, from HumanDecisionEngine, instead of read verbatim off
# ScenarioOccupant.assisting_occupant_id.
#
# RouteChoiceStrategy needs no new class at all: by the time
# HumanBehaviorLayer.register() calls choose(), decide() has already
# run and already written the same traits AssistanceAwareRouteChoiceStrategy
# reads -- that existing class is reused completely unmodified as this
# strategy's route-choice companion (see behaviour_profile_resolver.
# dynamic_registrar).
# =====================================================


class DynamicAssistanceDecisionStrategy(DecisionStrategy):

    def __init__(self, engine: HumanDecisionEngine, fallback=None):

        self.engine = engine
        self.fallback = fallback or AlwaysEvacuateDecisionStrategy()

    # =====================================================

    def decide(self, context) -> ActionIntent:

        decision = self.engine.evaluate(context)

        if decision.action == ACTION_HELP:

            context.profile.traits[TRAIT_ASSISTANCE_ROLE] = ROLE_HELPER
            context.profile.traits[TRAIT_ASSISTANCE_TARGET_ID] = decision.target_id
            context.profile.traits[TRAIT_ASSISTANCE_TYPE] = decision.assistance_type

            return AssistanceAwareDecisionStrategy(fallback=self.fallback).decide(context)

        if decision.action == ACTION_BE_ASSISTED:

            context.profile.traits[TRAIT_ASSISTANCE_ROLE] = ROLE_ASSISTED
            context.profile.traits[TRAIT_ASSISTANCE_TARGET_ID] = decision.target_id
            context.profile.traits[TRAIT_ASSISTANCE_TYPE] = decision.assistance_type
            context.profile.traits["leader_occupant_id"] = decision.target_id

            return AssistanceAwareDecisionStrategy(fallback=self.fallback).decide(context)

        # IGNORE/DELAY/independent-EVACUATE all proceed as an ordinary,
        # unassisted occupant -- no assistance trait is written, so
        # AssistanceAwareRouteChoiceStrategy/AssistanceAwareDecisionStrategy
        # (reused by whichever *other* occupant this session, if any)
        # never sees a stale role for this one.
        return self.fallback.decide(context)
