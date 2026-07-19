import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

from behavior.orchestrator import HumanBehaviorLayer
from behavior.profile import BehaviorProfile

from behavior_library.route_choice_strategies import HazardAwareRouteChoiceStrategy

from behaviour_profile_resolver.registrar import register_occupants
from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY
from behaviour_profile_resolver.resolver import resolve_profile

from scenario_runner.context import SimulationContext

from simulator.occupant import OccupantState

from simulation_interactive.movement_stepper import MovementStepper
from simulation_interactive.replanning import (
    BroadcastAwareDecisionStrategy,
    RecommendationAwareRouteChoiceStrategy,
    derive_replan_seed,
    replan_occupant,
)


@dataclass
class _RecommendationEntry:

    version: int
    exit_edge_id: Optional[str] = None
    stair_edge_id: Optional[str] = None


@dataclass
class _BroadcastEntry:

    version: int
    action_name: str


@dataclass
class OccupantRegistration:

    # Everything replan_occupant() needs for one occupant that
    # register_occupants() itself never hands back: the resolved
    # strategy *instances* (wrapped, so recommendations/broadcasts are
    # already reusable without per-occupant branching later), the
    # profile object replans read/mutate traits on, and per-occupant
    # bookkeeping for "have I already incorporated the current
    # recommendation/broadcast/hazard picture."

    profile: BehaviorProfile
    decision_strategy: Any
    route_choice_strategy: Any
    pre_movement_strategy: Any
    zone_id: str
    scenario_seed: int
    last_known_node_id: str = ""
    replan_count: int = 0
    last_applied_recommendation_version: int = 0
    last_applied_broadcast_version: int = 0
    last_hazard_signature: Optional[Tuple[Tuple[str, str], ...]] = None


class RouteManager:

    def __init__(self, context: SimulationContext, registry=None):

        self._context = context
        self._registry = registry if registry is not None else DEFAULT_PROFILE_REGISTRY

        self._behavior_layer: Optional[HumanBehaviorLayer] = None
        self._registrations: Dict[str, OccupantRegistration] = {}

        self._recommendation_version = 0
        self._recommendations_by_zone: Dict[str, _RecommendationEntry] = {}

        self._broadcast_version = 0
        self._broadcasts_by_zone: Dict[str, _BroadcastEntry] = {}
        self._global_broadcast: Optional[_BroadcastEntry] = None

        self._changed_edge_ids_this_tick = set()

    # =====================================================

    def register_all(self) -> HumanBehaviorLayer:

        # Resolves each occupant's template independently of
        # register_occupants() purely to remember the decision_strategy/
        # route_choice_strategy/pre_movement_strategy/profile needed for
        # later replanning -- then calls register_occupants() itself,
        # completely unmodified, to actually commit the initial routes
        # onto context.simulation. The initial route an occupant departs
        # with is therefore identical to what SimulationRuntime would
        # have produced; only a later replan uses the wrapped strategies
        # tracked here.

        scenario_seed = self._context.metadata.seed

        for occupant in self._context.occupants:

            template = resolve_profile(occupant.behaviour_profile_id, self._registry)

            profile = BehaviorProfile(
                occupant_id=occupant.occupant_id,
                walking_speed=template.walking_speed,
                familiarity=dict(template.familiarity),
                compliance_level=template.compliance_level,
                role=template.role,
                traits=dict(template.traits),
            )

            self._registrations[occupant.occupant_id] = OccupantRegistration(
                profile=profile,
                decision_strategy=BroadcastAwareDecisionStrategy(template.decision_strategy),
                route_choice_strategy=RecommendationAwareRouteChoiceStrategy(
                    template.route_choice_strategy,
                ),
                pre_movement_strategy=template.pre_movement_strategy,
                zone_id=occupant.zone_id,
                scenario_seed=scenario_seed,
                last_known_node_id=occupant.zone_id,
            )

        self._behavior_layer = register_occupants(self._context, self._registry)

        return self._behavior_layer

    # =====================================================
    # External-change notifications, applied by ActionExecutor/
    # InteractiveSimulation. Recommendations/broadcasts persist until
    # replaced; engineering changes are scoped to "since the last
    # start_tick()" so a whole tick's worth of changes (fired scenario
    # events + injected actions) are visible to both the initial
    # AT_NODE sweep and the ARRIVE_AT_NODE callback within that same
    # tick.
    # =====================================================

    def start_tick(self) -> None:

        self._changed_edge_ids_this_tick = set()

    # =====================================================

    def note_engineering_change(self, changed_edge_ids: Iterable[str]) -> None:

        self._changed_edge_ids_this_tick.update(changed_edge_ids)

    # =====================================================

    def note_recommendation(
        self,
        zone_id: str,
        recommended_exit_edge_id: Optional[str] = None,
        recommended_stair_edge_id: Optional[str] = None,
    ) -> None:

        self._recommendation_version += 1

        self._recommendations_by_zone[zone_id] = _RecommendationEntry(
            version=self._recommendation_version,
            exit_edge_id=recommended_exit_edge_id,
            stair_edge_id=recommended_stair_edge_id,
        )

    # =====================================================

    def note_broadcast(
        self, action_name: str, target_zone_ids: Optional[Iterable[str]] = None,
    ) -> None:

        self._broadcast_version += 1

        if target_zone_ids is None:
            self._global_broadcast = _BroadcastEntry(
                version=self._broadcast_version, action_name=action_name,
            )
            return

        for zone_id in target_zone_ids:
            self._broadcasts_by_zone[zone_id] = _BroadcastEntry(
                version=self._broadcast_version, action_name=action_name,
            )

    # =====================================================

    def _hazard_signature(
        self, occupant_id: str, hazard_snapshot, stepper: MovementStepper,
    ) -> Tuple[Tuple[str, str], ...]:

        if hazard_snapshot is None:
            return ()

        node_ids = stepper.occupant_remaining_node_ids(occupant_id)

        return tuple(
            (node_id, hazard_snapshot.node_state(node_id).severity.name)
            for node_id in node_ids
        )

    # =====================================================

    def maybe_replan(
        self, occupant_id: str, time: float, hazard_snapshot, stepper: MovementStepper,
    ) -> bool:

        registration = self._registrations.get(occupant_id)

        if registration is None:
            return False

        if stepper.occupant_state(occupant_id) not in (
            OccupantState.AT_NODE, OccupantState.STATIONARY,
        ):
            # Replanning is only ever triggered at natural decision
            # points -- AT_NODE (about to attempt an edge) or
            # STATIONARY (holding after a WAIT/shelter-in-place
            # decision, e.g. a prior broadcast) -- never while
            # TRAVERSING. Re-registering a TRAVERSING occupant would
            # leave a phantom entry in MultiAgentSimulation.
            # _edge_occupancy (nothing discards it -- only
            # _handle_arrive_at_node does, and that event would be
            # dropped as stale), and matches the existing "no dynamic
            # rerouting of an occupant already in flight" invariant
            # behavior_library's own strategies already rely on.
            return False

        # A STATIONARY Occupant has route=None (_register_stationary
        # never records a node), so MovementStepper can't answer
        # "where are they" from simulator state alone -- fall back to
        # the last node this manager itself last replanned them from
        # (seeded from their scenario-assigned start zone at
        # register_all() time).
        current_node_id = (
            stepper.occupant_current_node_id(occupant_id) or registration.last_known_node_id
        )
        remaining_edge_ids = set(stepper.occupant_remaining_edge_ids(occupant_id))

        trigger = False

        if remaining_edge_ids & self._changed_edge_ids_this_tick:
            trigger = True

        recommendation = self._recommendations_by_zone.get(current_node_id)

        if (
            recommendation is not None
            and recommendation.version > registration.last_applied_recommendation_version
        ):
            trigger = True
            registration.profile.traits["recommended_exit_edge_id"] = recommendation.exit_edge_id
            registration.profile.traits["recommended_stair_edge_id"] = recommendation.stair_edge_id
            registration.last_applied_recommendation_version = recommendation.version

        broadcast = self._broadcasts_by_zone.get(current_node_id) or self._global_broadcast

        if (
            broadcast is not None
            and broadcast.version > registration.last_applied_broadcast_version
        ):
            trigger = True
            registration.profile.traits["broadcast_override"] = broadcast.action_name
            registration.last_applied_broadcast_version = broadcast.version

        base_route_strategy = getattr(
            registration.route_choice_strategy, "_base", registration.route_choice_strategy,
        )
        hazard_signature = self._hazard_signature(occupant_id, hazard_snapshot, stepper)

        if (
            isinstance(base_route_strategy, HazardAwareRouteChoiceStrategy)
            and hazard_signature != registration.last_hazard_signature
        ):
            trigger = True

        registration.last_hazard_signature = hazard_signature

        if not trigger:
            return False

        registration.replan_count += 1
        seed = derive_replan_seed(
            registration.scenario_seed, occupant_id, registration.replan_count,
        )
        rng = random.Random(seed)

        replan_occupant(
            self._behavior_layer,
            registration,
            occupant_id,
            current_node_id,
            time,
            hazard_snapshot,
            stepper.snapshot_result(),
            rng,
        )

        registration.last_known_node_id = current_node_id

        return True

    # =====================================================

    def replan_due_at_node_occupants(
        self, time: float, hazard_snapshot, stepper: MovementStepper,
    ) -> Tuple[str, ...]:

        replanned = []

        for occupant_id in self._registrations:

            if stepper.occupant_state(occupant_id) not in (
                OccupantState.AT_NODE, OccupantState.STATIONARY,
            ):
                continue

            if self.maybe_replan(occupant_id, time, hazard_snapshot, stepper):
                replanned.append(occupant_id)

        return tuple(replanned)
