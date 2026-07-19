from types import MappingProxyType

from behavior.pre_movement import NoPreMovementDelay
from behavior.profile import Role

from behavior_library.firefighter_strategies import (
    FirefighterDecisionStrategy,
    FirefighterRouteChoiceStrategy,
)

from behaviour_profile_resolver.template import BehaviorProfileTemplate


# A default, illustrative registry for firefighters -- its own,
# separate registry from behaviour_profile_resolver.registry.
# DEFAULT_PROFILE_REGISTRY (civilian occupants), per Human Population &
# Assisted Evacuation Modeling Phase 4's own "keep firefighter logic
# separate from civilian behavior logic" rule; resolved only by
# behaviour_profile_resolver.firefighter_registrar, never by
# registrar.register_occupants(). walking_speed/NoPreMovementDelay
# reflect a trained, geared, already-en-route responder (brisk pace, no
# pre-movement hesitation) -- a documented, illustrative simplification,
# not a validated fire-service performance figure, the same honesty
# convention DEFAULT_PROFILE_REGISTRY's own civilian profiles already
# carry. role=LEADER: a firefighter directs, never follows, another
# occupant's route choice (Role.FOLLOWER/FollowLeaderRouteChoiceStrategy
# are civilian-only mechanisms, never applied to a firefighter).
#
# This is a *default*, not the only possible registry --
# behaviour_profile_resolver.firefighter_registrar.register_firefighters()
# accepts any Mapping[str, BehaviorProfileTemplate] in its place, the
# same extension point DEFAULT_PROFILE_REGISTRY already offers for
# civilians.

DEFAULT_FIREFIGHTER_PROFILE_REGISTRY = MappingProxyType(
    {
        "Firefighter_Default": BehaviorProfileTemplate(
            walking_speed=1.4,
            compliance_level=1.0,
            role=Role.LEADER,
            decision_strategy=FirefighterDecisionStrategy(),
            route_choice_strategy=FirefighterRouteChoiceStrategy(),
            pre_movement_strategy=NoPreMovementDelay(),
        ),
    }
)
