from behavior.orchestrator import HumanBehaviorLayer

from behavior_library.assistance_strategies import ROLE_ASSISTED, TRAIT_ASSISTANCE_ROLE, TRAIT_ASSISTANCE_TARGET_ID

from scenario_runner import SimulationContext

from behaviour_profile_resolver.firefighter_registrar import register_firefighters
from behaviour_profile_resolver.registrar import _register_occupants, register_deferred_occupants


# =====================================================
# The one, necessarily thin bridge between civilian registration
# (registrar.register_occupants(), behaviour_profile_resolver.registry/
# behavior_library.assistance_strategies) and firefighter registration
# (firefighter_registrar.register_firefighters(),
# behaviour_profile_resolver.firefighter_registry/behavior_library.
# firefighter_strategies) -- Phase 4's own "keep firefighter logic
# separate from civilian behavior logic" rule is a rule about *decision-
# making* (no DecisionStrategy/RouteChoiceStrategy/registry is shared
# or reused across the civilian/firefighter boundary anywhere in this
# platform), not about whether the two populations can ever be
# deployed into one simulation together at all -- they manifestly must
# be, since a firefighter's rescue target is a civilian occupant in the
# same building. This module is the one place that knows both
# populations exist; it contains no rescue/search/assistance decision
# logic of its own, only registration *sequencing*.
#
# Sequencing, precisely: any civilian occupant a firefighter is
# assigned to rescue (ScenarioFirefighter.rescue_target_occupant_id)
# must not be registered until *after* that firefighter's own
# combined rescue route has been resolved -- the same "helper before
# assisted" ordering behavior_library.assistance_strategies.
# AssistanceAwareRouteChoiceStrategy's own docstring already requires
# for civilian-to-civilian pairs, extended here to a firefighter-to-
# civilian pair. register_occupants()'s own defer_occupant_ids/
# extra_traits_by_occupant_id parameters exist specifically so this
# module can express that without registrar.py needing to know
# firefighters exist at all.
# =====================================================


def register_population(context: SimulationContext, occupant_registry=None, firefighter_registry=None) -> HumanBehaviorLayer:

    rescue_target_ids = frozenset(
        firefighter.rescue_target_occupant_id
        for firefighter in context.firefighters
        if firefighter.rescue_target_occupant_id
    )

    behavior_layer = _register_occupants(
        context, registry=occupant_registry, defer_occupant_ids=rescue_target_ids,
    )

    register_firefighters(context, behavior_layer, registry=firefighter_registry)

    rescuer_id_by_target_id = {
        firefighter.rescue_target_occupant_id: firefighter.firefighter_id
        for firefighter in context.firefighters
        if firefighter.rescue_target_occupant_id
    }

    # A rescued civilian is registered exactly the way any other
    # ASSISTED civilian is (behavior_library.assistance_strategies
    # itself, entirely reused, not reimplemented) -- their "helper" is
    # simply a firefighter's occupant_id instead of another civilian's.
    # AssistanceAwareRouteChoiceStrategy neither knows nor cares which
    # kind of occupant_id it is reading out of decisions_so_far.
    extra_traits_by_occupant_id = {
        target_id: {
            TRAIT_ASSISTANCE_ROLE: ROLE_ASSISTED,
            TRAIT_ASSISTANCE_TARGET_ID: rescuer_id,
            "leader_occupant_id": rescuer_id,
        }
        for target_id, rescuer_id in rescuer_id_by_target_id.items()
    }

    register_deferred_occupants(
        context, behavior_layer, rescue_target_ids,
        registry=occupant_registry, extra_traits_by_occupant_id=extra_traits_by_occupant_id,
    )

    return behavior_layer
