import hashlib
import random

from behavior.orchestrator import HumanBehaviorLayer
from behavior.profile import BehaviorProfile

from behavior_library.firefighter_strategies import (
    TRAIT_RESCUE_TARGET_OCCUPANT_ID,
    TRAIT_RESCUE_TARGET_ZONE_ID,
)

from scenario_runner import SimulationContext

from behaviour_profile_resolver.firefighter_registry import DEFAULT_FIREFIGHTER_PROFILE_REGISTRY
from behaviour_profile_resolver.occupant_attributes import attribute_traits, derive_occupant_attributes
from behaviour_profile_resolver.resolver import resolve_profile


# Human Population & Assisted Evacuation Modeling Phase 4/5 --
# firefighter registration, kept in its own module, its own registry
# (behaviour_profile_resolver.firefighter_registry), and its own
# strategies (behavior_library.firefighter_strategies), per that
# phase's own "keep firefighter logic separate from civilian behavior
# logic" rule -- registrar.py (civilian occupants) never imports this
# module or firefighter_registry, and this module never imports
# registrar.py or behaviour_profile_resolver.registry. Structurally
# mirrors registrar.register_occupants() exactly (same seeding
# convention, same "purely an adapter" scope), applied to
# ScenarioFirefighter instead of ScenarioOccupant.
#
# Registers onto an *already-constructed* HumanBehaviorLayer (passed
# in, never built here) -- unlike register_occupants(), which builds
# its own -- so a caller wiring up both populations
# (behaviour_profile_resolver.combined_registrar) can register
# firefighters into the exact same HumanBehaviorLayer/
# MultiAgentSimulation civilians already share, and so firefighters'
# own resolved decisions become visible to any later civilian
# registration (e.g. a rescue target) via HumanBehaviorLayer's own
# decisions_so_far -- the same "leader/target already resolved before
# the dependent occupant is registered" ordering every other
# decisions_so_far-reading strategy in this codebase already requires.


def _derive_firefighter_seed(scenario_seed: int, firefighter_id: str) -> int:

    # Same sha256-of-a-stable-key convention
    # registrar._derive_occupant_seed() already establishes, restated
    # here (not imported) for the same "kept separate" reason every
    # other firefighter-side module in this phase restates rather than
    # imports a civilian-side helper.

    key = f"{scenario_seed}|behaviour_profile_resolver.firefighter|{firefighter_id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big")


def register_firefighters(
    context: SimulationContext, behavior_layer: HumanBehaviorLayer, registry=None,
) -> None:

    registry = registry if registry is not None else DEFAULT_FIREFIGHTER_PROFILE_REGISTRY

    for firefighter in context.firefighters:

        template = resolve_profile(firefighter.behaviour_profile_id, registry)

        traits = dict(template.traits)

        if firefighter.rescue_target_zone_id is not None:
            traits[TRAIT_RESCUE_TARGET_OCCUPANT_ID] = firefighter.rescue_target_occupant_id
            traits[TRAIT_RESCUE_TARGET_ZONE_ID] = firefighter.rescue_target_zone_id

        # Occupant Attributes Phase 1 -- firefighters get the same
        # deterministic, category-conditioned physical/behavioral
        # attributes civilians do (the brief's own "Firefighters:
        # higher movement speed, higher smoke tolerance" worked
        # example), exposed via traits only -- never grouped with
        # civilians (see occupant_grouping.py's own "civilians only"
        # rule) and no attribute-aware wrapper strategy is applied here,
        # keeping firefighter strategies (behavior_library.firefighter_
        # strategies) completely untouched, per Phase 4's own "keep
        # firefighter logic separate" rule.
        attributes = derive_occupant_attributes(
            context.metadata.seed, firefighter.firefighter_id, firefighter.behaviour_profile_id,
        )
        traits.update(attribute_traits(attributes))

        profile = BehaviorProfile(
            occupant_id=firefighter.firefighter_id,
            walking_speed=template.walking_speed,
            stair_speed=template.stair_speed,
            familiarity=dict(template.familiarity),
            compliance_level=template.compliance_level,
            role=template.role,
            traits=traits,
        )

        firefighter_seed = _derive_firefighter_seed(context.metadata.seed, firefighter.firefighter_id)

        behavior_layer.register(
            start_id=firefighter.entry_zone_id,
            profile=profile,
            decision_strategy=template.decision_strategy,
            route_choice_strategy=template.route_choice_strategy,
            pre_movement_strategy=template.pre_movement_strategy,
            base_depart_time=firefighter.arrival_time,
            rng=random.Random(firefighter_seed),
        )
