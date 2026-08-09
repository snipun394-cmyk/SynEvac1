from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

from behavior.intent import AlwaysEvacuateDecisionStrategy, DecisionStrategy
from behavior.pre_movement import NoPreMovementDelay, PreMovementDelayStrategy
from behavior.profile import Role
from behavior.route_choice import RouteChoiceStrategy, ShortestRouteChoiceStrategy


@dataclass(frozen=True)
class BehaviorProfileTemplate:

    # What one behaviour_profile_id resolves to -- everything
    # HumanBehaviorLayer.register() needs except occupant_id/start_id,
    # which are per-occupant and supplied at registration time
    # (registrar.py), not stored here. Deliberately holds only already-
    # existing behavior/behavior_library types (DecisionStrategy,
    # RouteChoiceStrategy, PreMovementDelayStrategy, Role) -- this
    # package never defines a new strategy or a new behavioural
    # mechanism of its own; a template is purely a *reference* to
    # concrete instances of what already exists.
    #
    # A single template instance -- including its strategy objects --
    # is shared across every occupant resolved to the same
    # behaviour_profile_id (see registry.py's DEFAULT_PROFILE_REGISTRY),
    # matching how behavior_library's own docstrings already describe
    # their strategies as reusable across many occupants (e.g.
    # FollowLeaderRouteChoiceStrategy: "one instance is reusable across
    # every follower in every group").

    walking_speed: Optional[float] = None

    # Edge-Type-Specific Movement Speed (Experimental Branch V1) --
    # None (the default, and the only value any shipped profile in
    # registry.py sets) means "no stair-specific speed configured";
    # Stair-edge traversal then falls back to walking_speed exactly as
    # it always has (see simulator/coordinator.py's
    # _admit_onto_edge()). Never applied to Door/Exit edges. Opt-in
    # only via an explicit override registry (e.g.
    # calibration_benchmark.candidates.StairSpeedCandidate) --
    # DEFAULT_PROFILE_REGISTRY itself is not touched by this field's
    # introduction, so SynEvac Baseline V1 (docs/architecture/
    # synevac_baseline_v1.md) is unaffected.
    stair_speed: Optional[float] = None

    compliance_level: float = 1.0
    role: Role = Role.INDEPENDENT

    # Baked-in defaults a registry author may want for a given profile
    # (e.g. a Staff member's above-average familiarity) -- BehaviorProfile
    # already has both fields (behavior/profile.py); a Scenario itself
    # carries neither (ScenarioOccupant has no familiarity/traits data,
    # docs/architecture/scenario_runner.md §13's own flagged gap), so
    # these are the *only* source either can come from today.
    familiarity: Mapping[str, float] = field(default_factory=dict)
    traits: Mapping[str, Any] = field(default_factory=dict)

    decision_strategy: DecisionStrategy = field(default_factory=AlwaysEvacuateDecisionStrategy)
    route_choice_strategy: RouteChoiceStrategy = field(default_factory=ShortestRouteChoiceStrategy)
    pre_movement_strategy: PreMovementDelayStrategy = field(default_factory=NoPreMovementDelay)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "familiarity", MappingProxyType(dict(self.familiarity)))
        object.__setattr__(self, "traits", MappingProxyType(dict(self.traits)))
