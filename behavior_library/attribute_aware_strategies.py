import random

from simulator.decision import ActionType

from behavior.intent import ActionIntent, AlwaysEvacuateDecisionStrategy, DecisionStrategy
from behavior.pre_movement import NoPreMovementDelay, PreMovementDelayStrategy
from behavior.profile import Role
from behavior.route_choice import RouteChoice, RouteChoiceStrategy, ShortestRouteChoiceStrategy

from behavior_library.decision_strategies import AlwaysWaitDecisionStrategy, ComplianceDecisionStrategy
from behavior_library.pre_movement_strategies import (
    FollowLeaderPreMovementDelayStrategy,
    HesitationPreMovementDelayStrategy,
)
from behavior_library.route_choice_strategies import (
    FollowLeaderRouteChoiceStrategy,
    StaticHerdingRouteChoiceStrategy,
)


# =====================================================
# Occupant Attributes Phase 2/3 -- the two ways occupant_attributes.py/
# occupant_grouping.py's own deterministic, per-occupant values reach
# simulated behavior. Same "wraps any existing strategy, delegates to
# fallback with zero behavioral difference when the trait it looks for
# is absent" shape behavior_library.assistance_strategies.
# AssistanceAware{Decision,RouteChoice}Strategy already establish --
# these are new wrapper classes, not edits to FollowLeaderRouteChoice
# Strategy/FollowLeaderPreMovementDelayStrategy/ProbabilisticPreMovement
# Delay (all three stay completely unmodified, reused only by
# composition). Because the trait keys read here (
# "social_leader_occupant_id", "pre_movement_delay_multiplier") do not
# exist anywhere in this codebase before this feature, every existing
# scenario/test's BehaviorProfile.traits is empty for them -- these
# wrappers are a provable no-op for every caller that predates this
# feature, regardless of where they're composed into a strategy chain.
# =====================================================


class SocialGroupAwareRouteChoiceStrategy(RouteChoiceStrategy):

    # Companion to SocialGroupAwarePreMovementDelayStrategy below. An
    # occupant with no social group leader assigned this run (the
    # overwhelming majority -- see occupant_grouping.assign_occupant_
    # groups()'s own "not everyone travels in a group" docstring)
    # delegates entirely to `fallback`. A social-group follower reuses
    # FollowLeaderRouteChoiceStrategy verbatim -- the exact same
    # "leader_occupant_id" trait convention/mechanism
    # AssistanceAwareRouteChoiceStrategy's own ASSISTED branch already
    # relies on, just reached from a different, non-assistance source.
    # traits.setdefault() (never traits[...] =) means an occupant who
    # is *also* in an assistance pairing keeps their assistance-assigned
    # leader_occupant_id untouched -- assistance, being the more
    # specific, more urgent relationship, always wins.

    def __init__(self, fallback=None):

        self.fallback = fallback or ShortestRouteChoiceStrategy()
        self._follow_leader = FollowLeaderRouteChoiceStrategy(fallback=self.fallback)

    # =====================================================

    def choose(self, context) -> RouteChoice:

        social_leader_id = context.profile.traits.get("social_leader_occupant_id")

        if social_leader_id is None:
            return self.fallback.choose(context)

        context.profile.traits.setdefault("leader_occupant_id", social_leader_id)

        return self._follow_leader.choose(context)


# =====================================================


class SocialGroupAwarePreMovementDelayStrategy(PreMovementDelayStrategy):

    # Pre-movement counterpart to SocialGroupAwareRouteChoiceStrategy --
    # a social-group follower waits for their group leader to depart
    # before starting to move, the same "group observably moves
    # together" mechanic FollowLeaderPreMovementDelayStrategy already
    # provides for assistance pairs, reused unmodified here.

    def __init__(self, fallback=None, follow_gap=0.0):

        self.fallback = fallback or NoPreMovementDelay()
        self._follow_leader = FollowLeaderPreMovementDelayStrategy(
            follow_gap=follow_gap, fallback=self.fallback,
        )

    # =====================================================

    def delay(self, context) -> float:

        social_leader_id = context.profile.traits.get("social_leader_occupant_id")

        if social_leader_id is None:
            return self.fallback.delay(context)

        context.profile.traits.setdefault("leader_occupant_id", social_leader_id)

        return self._follow_leader.delay(context)


# =====================================================


class AttributeAwarePreMovementDelayStrategy(PreMovementDelayStrategy):

    # occupant_attributes.attribute_traits()'s own "pre_movement_delay_
    # multiplier" -- combines reaction_speed (faster reaction shortens
    # the delay) and panic_susceptibility (higher panic lengthens it)
    # into one scalar, applied on top of whatever `fallback` (typically
    # the occupant's own template.pre_movement_strategy, e.g. a
    # ProbabilisticPreMovementDelay) already computes. Multiplier
    # defaults to 1.0 -- an occupant/caller that never populated this
    # trait (every scenario/test that predates this feature) gets
    # fallback.delay(context) completely unchanged.

    def __init__(self, fallback=None):

        self.fallback = fallback or NoPreMovementDelay()

    # =====================================================

    def delay(self, context) -> float:

        base_delay = self.fallback.delay(context)
        multiplier = context.profile.traits.get("pre_movement_delay_multiplier", 1.0)

        return max(0.0, base_delay * multiplier)


# =====================================================


class CrowdFollowingAwareRouteChoiceStrategy(RouteChoiceStrategy):

    # crowd_following_tendency -- an occupant with no such trait
    # delegates entirely to `fallback` (byte-identical to not wrapping
    # at all). Otherwise delegates to a freshly-built
    # StaticHerdingRouteChoiceStrategy (that class's own already-tested,
    # unmodified herding logic -- reads context.decisions_so_far, the
    # same accumulated-this-session exit tally every other herding/
    # anti-herding strategy in this library already reads), using this
    # occupant's own tendency as its follow_probability instead of a
    # fixed constructor value shared by every occupant.

    def __init__(self, fallback=None):

        self.fallback = fallback or ShortestRouteChoiceStrategy()

    # =====================================================

    def choose(self, context) -> RouteChoice:

        crowd_following_tendency = context.profile.traits.get("crowd_following_tendency")

        if crowd_following_tendency is None:
            return self.fallback.choose(context)

        rng = context.rng if context.rng is not None else random.Random()

        return StaticHerdingRouteChoiceStrategy(
            follow_probability=crowd_following_tendency, rng=rng, fallback=self.fallback,
        ).choose(context)


# =====================================================


class AttributeAwareComplianceDecisionStrategy(DecisionStrategy):

    # compliance -- an occupant with no such trait delegates entirely to
    # `fallback` (byte-identical to not wrapping at all). Otherwise
    # gates on a fresh roll against this occupant's own deterministic
    # `compliance` value: on success, delegates to `fallback` (whatever
    # the template's own decision_strategy already does -- e.g. a
    # profile already using ComplianceDecisionStrategy still gets its
    # own compliance_level-driven behavior on top); on failure, reuses
    # AlwaysWaitDecisionStrategy (an existing primitive, not a new
    # mechanism) exactly as ComplianceDecisionStrategy's own default
    # noncompliant branch already does. A first, coarser "did this
    # occupant respond to the evacuation order at all" gate layered
    # ahead of whatever finer-grained decision logic the template
    # already carries -- not a replacement for it.
    #
    # Two exemptions, both deliberate, both preventing this from ever
    # *compounding* with a mechanism that already exists rather than
    # genuinely extending one that doesn't:
    #
    # 1. Role.LEADER always delegates straight to `fallback`, never
    #    rolled. DEFAULT_PROFILE_REGISTRY's own authors already made
    #    this exact distinction deliberately (Staff_Default/FireWarden_
    #    Default are the only two LEADER-role default profiles, and
    #    both are the only ones NOT wrapped in ComplianceDecisionStrategy
    #    there); restated here, not overridden, so a leader whose job is
    #    reliably directing others can never be attribute-gated into
    #    standing still forever.
    # 2. already_gated (set by the caller -- behaviour_profile_resolver.
    #    registrar._register_one() -- from `isinstance(template.
    #    decision_strategy, ComplianceDecisionStrategy)`) also delegates
    #    straight to `fallback`. A profile whose own template already
    #    rolls its own compliance_level (Adult/Child/Elderly/Visitor
    #    Default, every profile already wrapped in
    #    ComplianceDecisionStrategy) is left to that existing, unmodified
    #    mechanism alone -- stacking a second, independent roll on top
    #    would compound two probabilities into a lower effective one,
    #    silently making an already-reliable occupant meaningfully less
    #    likely to ever evacuate, which is a regression, not an
    #    extension. This trait's own generated `compliance` value only
    #    gates a profile with no existing compliance mechanism at all
    #    (Wheelchair_Default in the default registry) -- a genuinely new
    #    integration, not a duplicate of one that already exists.

    def __init__(self, fallback=None, noncompliant_strategy=None, already_gated=False):

        self.fallback = fallback or AlwaysEvacuateDecisionStrategy()
        self.noncompliant_strategy = noncompliant_strategy or AlwaysWaitDecisionStrategy()
        self.already_gated = already_gated

    # =====================================================

    def decide(self, context) -> ActionIntent:

        if self.already_gated or context.profile.role == Role.LEADER:
            return self.fallback.decide(context)

        compliance = context.profile.traits.get("compliance")

        if compliance is None:
            return self.fallback.decide(context)

        rng = context.rng if context.rng is not None else random.Random()

        if rng.random() <= compliance:
            return self.fallback.decide(context)

        return self.noncompliant_strategy.decide(context)


# =====================================================


def _sensitivity_multiplier(
    route_familiarity, smoke_tolerance, visibility_tolerance, risk_aversion, panic_susceptibility,
) -> float:

    # A bounded [0.5, 2.0] multiplier from five [0,1] attributes (each
    # defaulting to a neutral 0.5 when absent), combined by averaging
    # deviation-from-neutral rather than multiplying five factors
    # together -- keeps the effect legible and bounded (never more than
    # 2x, never less than 0.5x) regardless of how many attributes sit at
    # an extreme simultaneously, rather than compounding unboundedly.

    values = [route_familiarity, smoke_tolerance, visibility_tolerance, risk_aversion, panic_susceptibility]
    values = [0.5 if value is None else value for value in values]

    familiarity_component = 1.0 - values[0]
    tolerance_component = 1.0 - (values[1] + values[2]) / 2.0
    risk_component = values[3]
    panic_component = values[4]

    average = (familiarity_component + tolerance_component + risk_component + panic_component) / 4.0

    return 0.5 + 1.5 * average


class AttributeSensitivityAwarePreMovementDelayStrategy(PreMovementDelayStrategy):

    # route_familiarity/smoke_tolerance/visibility_tolerance/
    # risk_aversion/panic_susceptibility -- an occupant carrying none of
    # these traits gets `fallback.delay(context)` completely unchanged.
    # Otherwise adds an extra, attribute-scaled hesitation component on
    # top of `fallback`'s own delay, computed by delegating to a
    # HesitationPreMovementDelayStrategy instance (that class's own
    # already-tested, unmodified route-ambiguity detection -- reused,
    # never reimplemented) and scaling only that hesitation component by
    # _sensitivity_multiplier() above. A building with no genuinely
    # ambiguous route choice (HesitationPreMovementDelayStrategy itself
    # returns 0.0) is unaffected regardless of attribute values -- there
    # is nothing to hesitate over.

    def __init__(self, fallback=None):

        self.fallback = fallback or NoPreMovementDelay()
        self._hesitation = HesitationPreMovementDelayStrategy(fallback=NoPreMovementDelay())

    # =====================================================

    def delay(self, context) -> float:

        base_delay = self.fallback.delay(context)
        traits = context.profile.traits

        route_familiarity = traits.get("route_familiarity")
        smoke_tolerance = traits.get("smoke_tolerance")
        visibility_tolerance = traits.get("visibility_tolerance")
        risk_aversion = traits.get("risk_aversion")
        panic_susceptibility = traits.get("panic_susceptibility")

        if (
            route_familiarity is None and smoke_tolerance is None
            and visibility_tolerance is None and risk_aversion is None
            and panic_susceptibility is None
        ):
            return base_delay

        hesitation = self._hesitation.delay(context)

        if hesitation <= 0.0:
            return base_delay

        multiplier = _sensitivity_multiplier(
            route_familiarity, smoke_tolerance, visibility_tolerance, risk_aversion, panic_susceptibility,
        )

        return base_delay + hesitation * multiplier
