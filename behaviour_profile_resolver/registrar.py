import hashlib
import random

from behavior.orchestrator import HumanBehaviorLayer
from behavior.profile import BehaviorProfile

from behavior_library.assistance_strategies import (
    ASSISTANCE_HELP,
    ROLE_ASSISTED,
    ROLE_HELPER,
    TRAIT_ASSISTANCE_ROLE,
    TRAIT_ASSISTANCE_TARGET_ID,
    TRAIT_ASSISTANCE_TYPE,
    AssistanceAwareDecisionStrategy,
    AssistanceAwareRouteChoiceStrategy,
)
from behavior_library.attribute_aware_strategies import (
    AttributeAwareComplianceDecisionStrategy,
    AttributeAwarePreMovementDelayStrategy,
    AttributeSensitivityAwarePreMovementDelayStrategy,
    CrowdFollowingAwareRouteChoiceStrategy,
    SocialGroupAwarePreMovementDelayStrategy,
    SocialGroupAwareRouteChoiceStrategy,
)
from behavior_library.decision_strategies import ComplianceDecisionStrategy
from behavior_library.pre_movement_strategies import FollowLeaderPreMovementDelayStrategy

from scenario_runner import SimulationContext

from behaviour_profile_resolver.occupant_attributes import attribute_traits, derive_occupant_attributes
from behaviour_profile_resolver.occupant_grouping import assign_occupant_groups
from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY
from behaviour_profile_resolver.resolver import resolve_profile


# The corrected chain, completed -- architecture doc
# docs/architecture/scenario_runner.md §1/§7: "Scenario Runner ->
# SimulationContext -> Behaviour Profile Resolver -> Human Behavior
# Layer -> Simulation." This module is the Resolver's own entry point:
# it consumes a SimulationContext and, for every occupant, resolves
# behaviour_profile_id then registers the result onto the *same*
# MultiAgentSimulation instance the Scenario Runner already
# constructed and left empty. Importing scenario_runner here is a
# one-way, downstream dependency (Resolver depends on Runner's output
# type) -- it does not make the Runner aware of this package in any
# way; §12's "the Scenario Runner must remain completely unaware of
# this package" is a constraint on scenario_runner/'s own imports, not
# on this package's.
#
# Purely an adapter: no DecisionStrategy/RouteChoiceStrategy/
# PreMovementDelayStrategy is implemented here, no new behavioural
# mechanism is added to behavior/behavior_library -- this module only
# translates Scenario-shaped data (ScenarioOccupant) into calls against
# the existing HumanBehaviorLayer.register() API.
#
# defer_occupant_ids/extra_traits_by_occupant_id (both below) are
# generic, domain-neutral registration hooks -- this module has no
# knowledge of what a "firefighter" or a "rescue" is, or that either
# concept exists. behaviour_profile_resolver.combined_registrar is the
# one place that bridges civilian registration (here) and firefighter
# registration (behaviour_profile_resolver.firefighter_registrar) --
# see that module's own docstring for why a thin bridging module is
# unavoidable even under Phase 4's "keep firefighter logic separate"
# rule, and why it does not weaken that separation.


def _derive_occupant_seed(scenario_seed: int, occupant_id: str) -> int:

    # docs/architecture/reproducibility_review.md §7.4 -- restated
    # locally (sha256 of a stable key), not imported from
    # scenario_generator.seed_manager, the same "same algorithm,
    # independently implemented per package" precedent already used
    # for content-hashing elsewhere in this codebase. Deterministic:
    # the same (scenario_seed, occupant_id) always derives the same
    # seed, independent of registration order or which other occupants
    # exist in this Scenario.

    key = f"{scenario_seed}|behaviour_profile_resolver|{occupant_id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big")


def _ordered_for_assistance(occupants, priority_ids=frozenset()):

    # Human Population & Assisted Evacuation Modeling Phase 3: every
    # occupant assisting someone else (ScenarioOccupant.
    # assisting_occupant_id is not None) must be registered -- and
    # therefore resolved to a movement decision -- before the specific
    # occupant they assist, the same ordering dependency behavior_
    # library.assistance_strategies.AssistanceAwareRouteChoiceStrategy's
    # own docstring already documents (mirroring
    # FollowLeaderRouteChoiceStrategy's identical, pre-existing
    # requirement). A simple two-bucket stable partition -- priority
    # occupants first, in their original order, then everyone else, in
    # their original order -- rather than a full topological sort: this
    # platform's assistance/leadership model has no helper-of-a-helper
    # or leader-of-a-leader chains, so two buckets already produce a
    # valid order for every case it can generate.
    #
    # priority_ids -- Occupant Attributes Phase 3 -- generalizes the
    # original "Scenario-authored helpers first" rule to every id a
    # later occupant's strategy might look up via context.decisions_so_
    # far: Scenario-authored helpers, social-group leaders
    # (SocialGroupAware{RouteChoice,PreMovementDelay}Strategy's own
    # "leader_occupant_id" lookup needs the leader already resolved,
    # the same requirement FollowLeaderRouteChoiceStrategy's own
    # docstring already states), and helping-likelihood-driven leaders
    # (_helping_likelihood_traits_by_id()'s own ROLE_HELPER entries).
    # The caller (_register_occupants() below) is responsible for
    # supplying the full set; when it's empty (every existing caller
    # that predates this generalization), this is exactly the original
    # order -- fully backward compatible.

    priority = [occupant for occupant in occupants if occupant.occupant_id in priority_ids]
    others = [occupant for occupant in occupants if occupant.occupant_id not in priority_ids]

    return priority + others


# =====================================================


def _assistance_traits_by_occupant_id(occupants):

    # One pass, building both directions of every assistance pairing
    # from ScenarioOccupant's own single, helper-owned
    # assisting_occupant_id/assistance_type fields -- occupant_id ->
    # the BehaviorProfile.traits entries
    # AssistanceAwareDecisionStrategy/AssistanceAwareRouteChoiceStrategy
    # read. An occupant that is neither a helper nor assisted gets no
    # entry at all (not an entry with role=None) -- absence is exactly
    # what those strategies already treat as "not part of any
    # assistance pairing this run."

    traits_by_id = {}

    for occupant in occupants:

        target_id = occupant.assisting_occupant_id

        if not target_id:
            continue

        assistance_type = occupant.assistance_type

        traits_by_id[occupant.occupant_id] = {
            TRAIT_ASSISTANCE_ROLE: ROLE_HELPER,
            TRAIT_ASSISTANCE_TARGET_ID: target_id,
            TRAIT_ASSISTANCE_TYPE: assistance_type,
        }
        # "leader_occupant_id" reuses FollowLeaderPreMovementDelayStrategy's
        # own, already-established trait key exactly as-is (behavior_
        # library/pre_movement_strategies.py) -- the assisted occupant's
        # pre-movement delay is set to depart alongside their helper (see
        # register_occupants() below), the same "follower departs relative
        # to their leader" mechanism that class already implements, reused
        # for its literal intended purpose rather than reimplemented.
        traits_by_id[target_id] = {
            TRAIT_ASSISTANCE_ROLE: ROLE_ASSISTED,
            TRAIT_ASSISTANCE_TARGET_ID: occupant.occupant_id,
            TRAIT_ASSISTANCE_TYPE: assistance_type,
            "leader_occupant_id": occupant.occupant_id,
        }

    return traits_by_id


# =====================================================


def _occupant_group_traits_by_id(group_assignments):

    # Occupant Attributes Phase 3 -- Social Groups. Only followers get
    # an entry -- a group's own leader follows no one, the same
    # "absence means not part of this" rule the assistance system's own
    # traits_by_id already uses.

    traits_by_id = {}

    for occupant_id, assignment in group_assignments.items():

        if assignment.is_leader:
            continue

        traits_by_id[occupant_id] = {
            "social_leader_occupant_id": assignment.leader_occupant_id,
            "group_id": assignment.group_id,
            "group_type": assignment.group_type,
        }

    return traits_by_id


# =====================================================

_HELPING_LIKELIHOOD_THRESHOLD = 0.6


def _helping_likelihood_traits_by_id(context, group_assignments, assistance_traits_by_id):

    # Occupant Attributes Phase 3 -- helping_likelihood. Reuses
    # behavior_library.assistance_strategies in full: a group leader
    # whose own deterministic helping_likelihood clears a fixed,
    # disclosed threshold offers ASSISTANCE_HELP to their own group's
    # least-mobile member (lowest walking_speed_multiplier) -- the exact
    # same TRAIT_ASSISTANCE_ROLE/TARGET_ID/TYPE shape Scenario-authored
    # assistance (_assistance_traits_by_occupant_id) already produces,
    # so it drives the identical, already-tested speed-coupling/route-
    # copying mechanic, never a new one. Never overrides a Scenario-
    # authored assistance pairing for either party -- that pairing is an
    # explicit, authored fact and always takes precedence (checked here,
    # not left to caller merge order, so a leader already helping
    # someone else by Scenario authorship is never double-booked).

    occupants_by_id = {occupant.occupant_id: occupant for occupant in context.occupants}

    members_by_group = {}
    for occupant_id, assignment in group_assignments.items():
        members_by_group.setdefault(assignment.group_id, []).append((occupant_id, assignment))

    traits_by_id = {}

    for members in members_by_group.values():

        leader_entry = next((entry for entry in members if entry[1].is_leader), None)

        if leader_entry is None:
            continue

        leader_id = leader_entry[0]

        if leader_id in assistance_traits_by_id:
            continue

        leader_occupant = occupants_by_id[leader_id]
        leader_attributes = derive_occupant_attributes(
            context.metadata.seed, leader_id, leader_occupant.behaviour_profile_id,
        )

        if leader_attributes.helping_likelihood < _HELPING_LIKELIHOOD_THRESHOLD:
            continue

        candidates = [
            occupant_id for occupant_id, assignment in members
            if not assignment.is_leader and occupant_id not in assistance_traits_by_id
        ]

        if not candidates:
            continue

        target_id = min(
            candidates,
            key=lambda occupant_id: derive_occupant_attributes(
                context.metadata.seed, occupant_id, occupants_by_id[occupant_id].behaviour_profile_id,
            ).walking_speed_multiplier,
        )

        traits_by_id[leader_id] = {
            TRAIT_ASSISTANCE_ROLE: ROLE_HELPER,
            TRAIT_ASSISTANCE_TARGET_ID: target_id,
            TRAIT_ASSISTANCE_TYPE: ASSISTANCE_HELP,
        }
        traits_by_id[target_id] = {
            TRAIT_ASSISTANCE_ROLE: ROLE_ASSISTED,
            TRAIT_ASSISTANCE_TARGET_ID: leader_id,
            TRAIT_ASSISTANCE_TYPE: ASSISTANCE_HELP,
            "leader_occupant_id": leader_id,
        }

    return traits_by_id


# =====================================================


def _effective_walking_speed_multiplier(attributes) -> float:

    # Occupant Attributes Phase 2 -- walking_speed_multiplier, stamina,
    # fatigue_resistance, and mobility_factor all bear on how fast this
    # occupant actually moves. Combined into one factor here (registrar-
    # level integration arithmetic -- occupant_attributes.py's own
    # generation logic is never touched by this feature) rather than
    # each needing a separate mechanism: walking_speed_multiplier is the
    # dominant term (its own category-conditioned range already carries
    # most of the intended differentiation -- e.g. Wheelchair's own
    # 0.45-0.70 vs. Adult's 0.85-1.15); stamina/fatigue_resistance/
    # mobility_factor apply mild secondary damping (each in a bounded
    # [0.7, 1.0]-ish range) rather than compounding into an extreme
    # multiplier. Floored well above zero so no occupant's effective
    # speed is ever driven to (or past) a degenerate near-standstill.
    return max(
        0.15,
        attributes.walking_speed_multiplier
        * (0.85 + 0.15 * attributes.stamina)
        * (0.85 + 0.15 * attributes.fatigue_resistance)
        * (0.70 + 0.30 * attributes.mobility_factor),
    )


# =====================================================


def _register_one(behavior_layer, context, occupant, template, extra_traits):

    traits = dict(template.traits)
    traits.update(extra_traits)

    # Occupant Attributes Phase 1/2 -- every occupant automatically
    # gets a deterministic, category-conditioned spread of physical/
    # behavioral values (see occupant_attributes.py's own docstring for
    # why these are derived here, on demand, rather than ever stored on
    # Scenario/ScenarioOccupant). Merged in after assistance/group
    # traits so a same-named key can never collide with either (no
    # attribute name overlaps "assistance_role"/"leader_occupant_id"/
    # "group_id"/...), and merged before BehaviorProfile construction so
    # every strategy this profile is registered with can already see it.
    attributes = derive_occupant_attributes(
        context.metadata.seed, occupant.occupant_id, occupant.behaviour_profile_id,
    )
    traits.update(attribute_traits(attributes))
    traits["effective_walking_speed_multiplier"] = _effective_walking_speed_multiplier(attributes)

    profile = BehaviorProfile(
        occupant_id=occupant.occupant_id,
        walking_speed=template.walking_speed,
        familiarity=dict(template.familiarity),
        compliance_level=template.compliance_level,
        role=template.role,
        traits=traits,
    )

    occupant_seed = _derive_occupant_seed(context.metadata.seed, occupant.occupant_id)

    # ASSISTED occupants depart alongside their helper (follow_gap=0.0
    # -- "alongside", not "shortly after" -- see
    # FollowLeaderPreMovementDelayStrategy's own docstring), falling
    # back to the template's own pre-movement strategy if the helper
    # cannot be resolved. Every other occupant now goes through three
    # more layers, each a documented no-op unless its own trait is
    # present (see behavior_library.attribute_aware_strategies's own
    # module docstring): SocialGroupAwarePreMovementDelayStrategy (a
    # social-group follower waits for their group's leader, exactly
    # like an assisted occupant waits for their helper) wrapping
    # AttributeAwarePreMovementDelayStrategy (reaction_speed/panic_
    # susceptibility scale the sampled delay) wrapping
    # AttributeSensitivityAwarePreMovementDelayStrategy (route_
    # familiarity/smoke_tolerance/visibility_tolerance/risk_aversion/
    # panic_susceptibility scale route-ambiguity hesitation) wrapping
    # the template's own pre-movement strategy.
    if traits.get(TRAIT_ASSISTANCE_ROLE) == ROLE_ASSISTED:
        pre_movement_strategy = FollowLeaderPreMovementDelayStrategy(
            follow_gap=0.0, fallback=template.pre_movement_strategy,
        )
    else:
        pre_movement_strategy = SocialGroupAwarePreMovementDelayStrategy(
            fallback=AttributeAwarePreMovementDelayStrategy(
                fallback=AttributeSensitivityAwarePreMovementDelayStrategy(
                    fallback=template.pre_movement_strategy,
                ),
            ),
        )

    # AssistanceAware{Decision,RouteChoice}Strategy wrap the template's
    # own strategies as `fallback` unconditionally -- for the
    # (overwhelmingly common) occupant with no assistance role in
    # `traits`, both wrappers delegate straight through with zero
    # behavioral difference from using the template's strategies
    # directly (see each wrapper's own docstring).
    # AttributeAwareComplianceDecisionStrategy sits outermost on the
    # decision side (a first, coarser "did they respond to the
    # evacuation order at all" gate, ahead of assistance/template
    # logic); SocialGroupAwareRouteChoiceStrategy and
    # CrowdFollowingAwareRouteChoiceStrategy sit progressively further
    # in on the route side -- assistance (most specific) is checked
    # first, then social-group following, then generic crowd herding,
    # each falling through to the next when its own trait is absent.
    behavior_layer.register(
        start_id=occupant.zone_id,
        profile=profile,
        decision_strategy=AttributeAwareComplianceDecisionStrategy(
            fallback=AssistanceAwareDecisionStrategy(fallback=template.decision_strategy),
            already_gated=isinstance(template.decision_strategy, ComplianceDecisionStrategy),
        ),
        route_choice_strategy=AssistanceAwareRouteChoiceStrategy(
            fallback=SocialGroupAwareRouteChoiceStrategy(
                fallback=CrowdFollowingAwareRouteChoiceStrategy(fallback=template.route_choice_strategy),
            ),
        ),
        pre_movement_strategy=pre_movement_strategy,
        rng=random.Random(occupant_seed),
    )


# =====================================================


def register_occupants(context: SimulationContext, registry=None) -> HumanBehaviorLayer:

    # docs/architecture/reproducibility_review.md §7.5 (see
    # tests.test_reproducibility_fix.BackwardCompatibilityTests.
    # test_register_occupants_public_signature_is_unchanged): this
    # public signature -- exactly (context, registry) -- must never
    # grow. _register_occupants() below is where Phase 3/5's own
    # defer_occupant_ids/extra_traits_by_occupant_id hooks actually
    # live; this function is a thin, signature-preserving wrapper over
    # it, used by every caller that only ever registered occupants in
    # one pass (still every caller except
    # behaviour_profile_resolver.combined_registrar).

    return _register_occupants(context, registry=registry)


# =====================================================


def _register_occupants(
    context: SimulationContext,
    registry=None,
    defer_occupant_ids=frozenset(),
    extra_traits_by_occupant_id=None,
) -> HumanBehaviorLayer:

    # defer_occupant_ids: occupant_ids to skip entirely this call --
    # register_deferred_occupants() below registers exactly this set,
    # later, once whatever they are waiting to be resolved after
    # (behaviour_profile_resolver.combined_registrar's own use case:
    # firefighters) has been.

    registry = registry if registry is not None else DEFAULT_PROFILE_REGISTRY
    extra_traits_by_occupant_id = extra_traits_by_occupant_id or {}

    behavior_layer = HumanBehaviorLayer(context.simulation, context.engine)

    group_assignments = assign_occupant_groups(context)
    group_traits_by_id = _occupant_group_traits_by_id(group_assignments)
    assistance_traits_by_id = _assistance_traits_by_occupant_id(context.occupants)
    helping_traits_by_id = _helping_likelihood_traits_by_id(
        context, group_assignments, assistance_traits_by_id,
    )

    priority_ids = (
        {occupant.occupant_id for occupant in context.occupants if occupant.assisting_occupant_id}
        | {occupant_id for occupant_id, assignment in group_assignments.items() if assignment.is_leader}
        | {
            occupant_id for occupant_id, traits in helping_traits_by_id.items()
            if traits.get(TRAIT_ASSISTANCE_ROLE) == ROLE_HELPER
        }
    )

    for occupant in _ordered_for_assistance(context.occupants, priority_ids):

        if occupant.occupant_id in defer_occupant_ids:
            continue

        template = resolve_profile(occupant.behaviour_profile_id, registry)

        # Group assignment first, helping-likelihood-driven assistance
        # second, Scenario-authored assistance third -- dict.update()'s
        # own "later wins" semantics means an occupant covered by more
        # than one of these keeps whichever is most specific/authored,
        # never a softer one silently overwriting it (see
        # _helping_likelihood_traits_by_id()'s own docstring: it already
        # excludes anyone Scenario-authored assistance also covers, so
        # this ordering is a defensive restatement of that same rule,
        # not the only thing enforcing it). See behavior_library.
        # attribute_aware_strategies's own identical setdefault()-based
        # precedence for the same reasoning restated at the trait-read
        # site.
        extra_traits = dict(group_traits_by_id.get(occupant.occupant_id, {}))
        extra_traits.update(helping_traits_by_id.get(occupant.occupant_id, {}))
        extra_traits.update(assistance_traits_by_id.get(occupant.occupant_id, {}))
        extra_traits.update(extra_traits_by_occupant_id.get(occupant.occupant_id, {}))

        _register_one(behavior_layer, context, occupant, template, extra_traits)

    return behavior_layer


# =====================================================


def register_deferred_occupants(
    context: SimulationContext,
    behavior_layer: HumanBehaviorLayer,
    occupant_ids,
    registry=None,
    extra_traits_by_occupant_id=None,
) -> None:

    registry = registry if registry is not None else DEFAULT_PROFILE_REGISTRY
    extra_traits_by_occupant_id = extra_traits_by_occupant_id or {}

    occupants_by_id = {occupant.occupant_id: occupant for occupant in context.occupants}
    group_traits_by_id = _occupant_group_traits_by_id(assign_occupant_groups(context))

    for occupant_id in occupant_ids:

        occupant = occupants_by_id.get(occupant_id)

        if occupant is None:
            continue

        template = resolve_profile(occupant.behaviour_profile_id, registry)

        extra_traits = dict(group_traits_by_id.get(occupant_id, {}))
        extra_traits.update(extra_traits_by_occupant_id.get(occupant_id, {}))

        _register_one(behavior_layer, context, occupant, template, extra_traits)
