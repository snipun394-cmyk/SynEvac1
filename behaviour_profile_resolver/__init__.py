from behaviour_profile_resolver.category import OccupantCategory, occupant_category, register_category
from behaviour_profile_resolver.combined_registrar import register_population
from behaviour_profile_resolver.firefighter_registrar import register_firefighters
from behaviour_profile_resolver.firefighter_registry import DEFAULT_FIREFIGHTER_PROFILE_REGISTRY
from behaviour_profile_resolver.registrar import register_deferred_occupants, register_occupants
from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY
from behaviour_profile_resolver.resolver import UnknownBehaviourProfileError, resolve_profile
from behaviour_profile_resolver.template import BehaviorProfileTemplate

__all__ = [
    "register_occupants",
    "register_deferred_occupants",
    "resolve_profile",
    "UnknownBehaviourProfileError",
    "BehaviorProfileTemplate",
    "DEFAULT_PROFILE_REGISTRY",
    "OccupantCategory",
    "occupant_category",
    "register_category",
    "register_firefighters",
    "DEFAULT_FIREFIGHTER_PROFILE_REGISTRY",
    "register_population",
]
