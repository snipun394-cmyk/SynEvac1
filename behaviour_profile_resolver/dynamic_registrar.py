import random

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from behavior.context import DecisionContext
from behavior.orchestrator import HumanBehaviorLayer
from behavior.profile import BehaviorProfile

from hazard.snapshot import HazardSnapshot

from simulator.decision import BehaviorDecision

from behavior_library.assistance_strategies import (
    ROLE_ASSISTED,
    TRAIT_ASSISTANCE_ROLE,
    TRAIT_ASSISTANCE_TARGET_ID,
    AssistanceAwareRouteChoiceStrategy,
)
from behavior_library.dynamic_firefighter_strategies import DynamicFirefighterDecisionStrategy
from behavior_library.dynamic_human_strategies import DynamicAssistanceDecisionStrategy
from behavior_library.firefighter_strategies import FirefighterRouteChoiceStrategy
from behavior_library.pre_movement_strategies import FollowLeaderPreMovementDelayStrategy

from scenario_runner import SimulationContext

from behaviour_profile_resolver.category import occupant_category
from behaviour_profile_resolver.firefighter_registrar import _derive_firefighter_seed
from behaviour_profile_resolver.firefighter_registry import DEFAULT_FIREFIGHTER_PROFILE_REGISTRY
from behaviour_profile_resolver.occupant_attributes import attribute_traits, derive_occupant_attributes
from behaviour_profile_resolver.registrar import _derive_occupant_seed, _register_one
from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY
from behaviour_profile_resolver.resolver import resolve_profile

from human_decision_engine.engine import ACTION_HELP, HumanDecisionEngine
from human_decision_engine.events import DecisionEvent, DecisionEventLog
from human_decision_engine.firefighter_engine import (
    CivilianRosterEntry,
    FirefighterDecisionEngine,
    RESCUE_TASKS,
    compute_priority_scores,
)
from human_decision_engine.groups import (
    GROUP_KIND_ASSIST,
    GROUP_KIND_FIREFIGHTER_ESCORT,
    GROUP_KIND_FOLLOW,
    Group,
    GroupRegistry,
)
from human_decision_engine.pairing import (
    ASSIST_TARGET_CATEGORIES,
    GROUP_FOLLOW_CATEGORIES,
    PAIRING_ASSIST,
    compute_dynamic_pairings,
    ordered_for_dynamic_pairing,
)


# =====================================================
# The dynamic counterpart to behaviour_profile_resolver.combined_
# registrar.register_population() -- opt-in (a caller must ask for
# this function by name; every existing caller of register_occupants()/
# register_population() is completely unaffected). Where combined_
# registrar reads assistance/rescue *assignments* straight off Scenario
# fields (ScenarioOccupant.assisting_occupant_id, ScenarioFirefighter.
# rescue_target_occupant_id), this module computes them at simulation
# time instead: human_decision_engine.pairing supplies the perceivable,
# structural "who could help whom" facts, human_decision_engine.engine/
# firefighter_engine supply the hazard-aware runtime decision, and the
# existing behavior_library.assistance_strategies/firefighter_strategies
# (entirely unmodified) still supply every actual movement mechanic --
# see behavior_library.dynamic_human_strategies/dynamic_firefighter_
# strategies for exactly how those three layers compose.
#
# No two-pass defer_occupant_ids dance is needed here (unlike combined_
# registrar's own firefighter-rescue case): BehaviorDecision's own
# contract already guarantees a later HumanBehaviorLayer.register()
# call for the same occupant_id supersedes an earlier one (see that
# type's own docstring, and MultiAgentSimulation.submit_decision()) --
# so a civilian a firefighter dynamically claims is simply *registered
# a second time*, once the firefighter's own claim is known, rather
# than deferred up front before anyone could know it would happen.
#
# hazard_snapshot: HumanBehaviorLayer.register() itself never threads
# a HazardSnapshot through to DecisionContext (see that method's own
# body) -- own-safety-aware decisions (Phase 2's "hazard, own safety"
# factors) need one, so this module builds DecisionContext directly
# and drives decide()/choose()/delay()/submit_decision() itself,
# exactly the same "second, independent orchestration of the same
# public pieces, not a change to register() itself" pattern
# simulation_interactive.replanning.replan_occupant() already
# established for its own, mid-run hazard-aware replans (see that
# function's own docstring for the identical reasoning). Optional --
# None (the default) leaves every own-safety threshold reading "clear
# to help", the same honest degrade every other hazard-aware strategy
# in this codebase already applies when no HazardSnapshot is in play.
# =====================================================


@dataclass(frozen=True)
class DynamicRegistrationResult:

    behavior_layer: HumanBehaviorLayer
    events: Tuple[DecisionEvent, ...] = field(default_factory=tuple)
    groups: Tuple[Group, ...] = field(default_factory=tuple)

    # occupant_id -> deterministic rescue priority score (Phase 4),
    # covering every civilian occupant regardless of whether any
    # firefighter ultimately claimed them -- see FirefighterDecisionEngine.
    # priority_scores()'s own docstring for why.
    priority_scores: Mapping[str, float] = field(default_factory=dict)

    # occupant_id -> the dynamic decision this occupant reached this
    # session (Command Center Phase 7's own "Current Decision"/"Current
    # Assistance Status" columns) -- civilian entries hold a
    # human_decision_engine.engine.CivilianDecision, firefighter
    # entries hold a human_decision_engine.firefighter_engine.
    # FirefighterTaskDecision. Plain Any (not a shared base class --
    # civilian and firefighter decisions are deliberately unrelated
    # types) so command_center never needs to import either engine
    # module to read them.
    decisions_by_occupant_id: Mapping[str, object] = field(default_factory=dict)

    # occupant_id -> the group_id they currently belong to, if any.
    group_id_by_occupant_id: Mapping[str, str] = field(default_factory=dict)


# =====================================================


def register_population_dynamic(
    context: SimulationContext,
    occupant_registry=None,
    firefighter_registry=None,
    known_fallen_ids: FrozenSet[str] = frozenset(),
    known_possible_injury_ids: FrozenSet[str] = frozenset(),
    hazard_snapshot: Optional[HazardSnapshot] = None,
) -> DynamicRegistrationResult:

    occupant_registry = occupant_registry if occupant_registry is not None else DEFAULT_PROFILE_REGISTRY
    firefighter_registry = (
        firefighter_registry if firefighter_registry is not None else DEFAULT_FIREFIGHTER_PROFILE_REGISTRY
    )

    event_log = DecisionEventLog()
    group_registry = GroupRegistry(event_log=event_log)

    pairings = compute_dynamic_pairings(context.occupants)
    civilian_engine = HumanDecisionEngine(pairings, event_log=event_log)

    behavior_layer = HumanBehaviorLayer(context.simulation, context.engine)

    for occupant in ordered_for_dynamic_pairing(context.occupants, pairings):

        template = resolve_profile(occupant.behaviour_profile_id, occupant_registry)
        occupant_seed = _derive_occupant_seed(context.metadata.seed, occupant.occupant_id)

        # Occupant Attributes Phase 1/2 -- same deterministic traits
        # every static-registration occupant gets (registrar.py::
        # _register_one()), computed here since this call site is the
        # one place in this function that still has context.metadata.
        # seed (the raw scenario seed, distinct from occupant_seed
        # above, which is already the occupant's own derived RNG seed).
        # No group/wrapper-strategy layering in this dynamic-
        # registration path -- deliberately deferred (see this module's
        # own docstring) to keep this already-intricate, decision-
        # engine-driven path's change surface minimal; the data is
        # still genuinely available in traits for Ground Truth/Dataset
        # Builder either way.
        extra_traits = attribute_traits(
            derive_occupant_attributes(context.metadata.seed, occupant.occupant_id, occupant.behaviour_profile_id),
        )

        _register_one_dynamic(
            behavior_layer, occupant, template, civilian_engine, occupant_seed, hazard_snapshot,
            extra_traits=extra_traits,
        )

    confirmed_civilian_target_ids = set()

    for pairing in pairings:

        decision = civilian_engine.decision_for(pairing.helper_id)

        if decision is None or decision.action != ACTION_HELP:
            continue

        confirmed_civilian_target_ids.add(pairing.target_id)

        kind = GROUP_KIND_ASSIST if pairing.kind == PAIRING_ASSIST else GROUP_KIND_FOLLOW
        group_registry.form((pairing.helper_id, pairing.target_id), kind)

    vulnerable_ids = {
        occupant.occupant_id for occupant in context.occupants
        if occupant_category(occupant.behaviour_profile_id) in (
            ASSIST_TARGET_CATEGORIES + GROUP_FOLLOW_CATEGORIES
        )
    }
    isolated_ids = vulnerable_ids - confirmed_civilian_target_ids

    full_roster = tuple(
        CivilianRosterEntry(
            occupant_id=occupant.occupant_id,
            zone_id=occupant.zone_id,
            category=occupant_category(occupant.behaviour_profile_id),
        )
        for occupant in context.occupants
    )
    firefighter_candidate_roster = tuple(
        entry for entry in full_roster if entry.occupant_id not in confirmed_civilian_target_ids
    )

    firefighter_engine = FirefighterDecisionEngine(
        firefighter_candidate_roster,
        known_fallen_ids=known_fallen_ids,
        known_possible_injury_ids=known_possible_injury_ids,
        isolated_occupant_ids=frozenset(isolated_ids),
        event_log=event_log,
    )

    occupants_by_id = {occupant.occupant_id: occupant for occupant in context.occupants}

    for firefighter in context.firefighters:

        template = resolve_profile(firefighter.behaviour_profile_id, firefighter_registry)
        firefighter_seed = _derive_firefighter_seed(context.metadata.seed, firefighter.firefighter_id)

        firefighter_traits = dict(template.traits)
        firefighter_traits.update(
            attribute_traits(
                derive_occupant_attributes(
                    context.metadata.seed, firefighter.firefighter_id, firefighter.behaviour_profile_id,
                ),
            ),
        )

        profile = BehaviorProfile(
            occupant_id=firefighter.firefighter_id,
            walking_speed=template.walking_speed,
            familiarity=dict(template.familiarity),
            compliance_level=template.compliance_level,
            role=template.role,
            traits=firefighter_traits,
        )

        _register_with_hazard(
            behavior_layer,
            occupant_id=firefighter.firefighter_id,
            start_id=firefighter.entry_zone_id,
            profile=profile,
            decision_strategy=DynamicFirefighterDecisionStrategy(firefighter_engine),
            route_choice_strategy=FirefighterRouteChoiceStrategy(),
            pre_movement_strategy=template.pre_movement_strategy,
            base_depart_time=firefighter.arrival_time,
            hazard_snapshot=hazard_snapshot,
            rng=random.Random(firefighter_seed),
        )

        task_decision = firefighter_engine.decision_for(firefighter.firefighter_id)

        if (
            task_decision is not None
            and task_decision.task in RESCUE_TASKS
            and task_decision.target_occupant_id is not None
            and task_decision.target_occupant_id in occupants_by_id
        ):

            target_occupant = occupants_by_id[task_decision.target_occupant_id]
            target_template = resolve_profile(target_occupant.behaviour_profile_id, occupant_registry)

            _register_one(
                behavior_layer, context, target_occupant, target_template,
                extra_traits={
                    TRAIT_ASSISTANCE_ROLE: ROLE_ASSISTED,
                    TRAIT_ASSISTANCE_TARGET_ID: firefighter.firefighter_id,
                    "leader_occupant_id": firefighter.firefighter_id,
                },
            )

            group_registry.form(
                (firefighter.firefighter_id, task_decision.target_occupant_id),
                GROUP_KIND_FIREFIGHTER_ESCORT,
            )

    decisions_by_occupant_id: Dict[str, object] = {}

    for occupant in context.occupants:
        decision = civilian_engine.decision_for(occupant.occupant_id)
        if decision is not None:
            decisions_by_occupant_id[occupant.occupant_id] = decision

    for firefighter in context.firefighters:
        decision = firefighter_engine.decision_for(firefighter.firefighter_id)
        if decision is not None:
            decisions_by_occupant_id[firefighter.firefighter_id] = decision

    group_id_by_occupant_id: Dict[str, str] = {
        occupant_id: group.group_id
        for group in group_registry.groups
        for occupant_id in group.member_ids
    }

    return DynamicRegistrationResult(
        behavior_layer=behavior_layer,
        events=event_log.events,
        groups=group_registry.groups,
        # The full civilian roster, not FirefighterDecisionEngine's own
        # (narrower) candidate roster -- Command Center Phase 7 needs a
        # score for every observed person, including one already
        # claimed by a civilian helper (firefighter_candidate_roster
        # deliberately excludes those from *firefighter* task
        # assignment, but that exclusion has no bearing on what their
        # own priority score honestly is).
        priority_scores=compute_priority_scores(
            full_roster, hazard_snapshot,
            known_fallen_ids=known_fallen_ids,
            known_possible_injury_ids=known_possible_injury_ids,
            isolated_occupant_ids=frozenset(isolated_ids),
        ),
        decisions_by_occupant_id=decisions_by_occupant_id,
        group_id_by_occupant_id=group_id_by_occupant_id,
    )


# =====================================================


def _register_one_dynamic(
    behavior_layer, occupant, template, engine, occupant_seed, hazard_snapshot, extra_traits=None,
) -> None:

    traits = dict(template.traits)
    traits.update(extra_traits or {})

    profile = BehaviorProfile(
        occupant_id=occupant.occupant_id,
        walking_speed=template.walking_speed,
        familiarity=dict(template.familiarity),
        compliance_level=template.compliance_level,
        role=template.role,
        traits=traits,
    )

    _register_with_hazard(
        behavior_layer,
        occupant_id=occupant.occupant_id,
        start_id=occupant.zone_id,
        profile=profile,
        decision_strategy=DynamicAssistanceDecisionStrategy(engine, fallback=template.decision_strategy),
        route_choice_strategy=AssistanceAwareRouteChoiceStrategy(fallback=template.route_choice_strategy),
        pre_movement_strategy=FollowLeaderPreMovementDelayStrategy(
            follow_gap=0.0, fallback=template.pre_movement_strategy,
        ),
        base_depart_time=0.0,
        hazard_snapshot=hazard_snapshot,
        rng=random.Random(occupant_seed),
    )


# =====================================================


def _register_with_hazard(
    behavior_layer: HumanBehaviorLayer,
    occupant_id: str,
    start_id: str,
    profile: BehaviorProfile,
    decision_strategy,
    route_choice_strategy,
    pre_movement_strategy,
    base_depart_time: float,
    hazard_snapshot: Optional[HazardSnapshot],
    rng: random.Random,
) -> None:

    # Same decide() -> choose() -> delay() -> BehaviorDecision ->
    # submit_decision() -> _decisions_so_far pipeline
    # HumanBehaviorLayer.register() runs internally, restated here (not
    # a call to register() itself) purely so a HazardSnapshot can be
    # threaded into DecisionContext -- see this module's own docstring,
    # and simulation_interactive.replanning.replan_occupant(), the
    # precedent this restatement mirrors line for line.

    decision_context = DecisionContext(
        graph=behavior_layer.graph,
        engine=behavior_layer.engine,
        profile=profile,
        start_id=start_id,
        decisions_so_far=dict(behavior_layer._decisions_so_far),
        hazard_snapshot=hazard_snapshot,
        rng=rng,
    )

    intent = decision_strategy.decide(decision_context)

    if intent.requires_movement:

        route_choice = route_choice_strategy.choose(decision_context)
        delay = pre_movement_strategy.delay(decision_context)

        # Same effective-speed read as behavior.orchestrator.
        # HumanBehaviorLayer.register()'s own identical site -- never
        # mutates profile.walking_speed, only this decision's own speed.
        # Same None-guard (profile.walking_speed may be unset) and same
        # ASSISTED exemption too (see that site's own comment): skip
        # re-multiplying a speed AssistanceAwareRouteChoiceStrategy
        # already overwrote with the helper's own finalized speed.
        if profile.walking_speed is None or profile.traits.get("assistance_role") == "ASSISTED":
            effective_speed = profile.walking_speed
        else:
            effective_speed = profile.walking_speed * profile.traits.get(
                "effective_walking_speed_multiplier", 1.0,
            )

        decision = BehaviorDecision(
            occupant_id=occupant_id,
            action_type=intent.action_type,
            start_id=start_id,
            goal_id=route_choice.goal_id,
            route=route_choice.route,
            walking_speed=effective_speed,
            depart_time=base_depart_time + delay,
            metadata=intent.metadata,
            route_unavailable=(route_choice.goal_id is None and route_choice.route is None),
        )

    else:

        decision = BehaviorDecision(
            occupant_id=occupant_id,
            action_type=intent.action_type,
            start_id=start_id,
            metadata=intent.metadata,
        )

    behavior_layer._decisions_so_far[occupant_id] = decision

    behavior_layer.simulation.submit_decision(decision)
