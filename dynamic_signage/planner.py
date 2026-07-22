import dataclasses
from typing import Dict, Iterable, Optional, Tuple

from evacuation_guidance.models import EvacuationGuidanceSnapshot, GuidanceInconsistency, NavigationStepType, RouteStatus

from models.dynamic_sign import SignIndication

from dynamic_signage.direction import relative_direction
from dynamic_signage.models import (
    DynamicSignageConfig, DynamicSignageSnapshot, SignageConflict, SignageInstruction, SignageStatus,
)
from dynamic_signage.route_mapping import next_meaningful_step_for_zone, resolve_target


# =====================================================
# Live Dynamic Evacuation Signage milestone, Phase 6/9/10/11/12 -- the
# ONE object LiveRuntime owns for this package, mirroring every prior
# live-intelligence engine's own "exactly one shared instance per live
# session" role. compute(time, guidance_snapshot, signs) is the single
# entry point.
#
# Converts an EvacuationGuidanceSnapshot ("how to reach the recommended
# exit") into a DynamicSignageSnapshot ("what each sign should
# indicate"). NEVER re-ranks or second-guesses which exit Guidance
# selected (Phase 11's own strict authority ladder: Decision Policy >
# Evacuation Recommendation > Evacuation Guidance > Dynamic Signage) --
# every instruction here is derived entirely from the CURRENT
# EvacuationGuidancePlan for a sign's zone(s), recomputed fresh every
# cycle from scratch. This is precisely what makes Phase 10's critical
# requirement true structurally rather than by convention: a sign can
# never keep showing a stale direction after the recommended exit
# changes, because nothing about a previous cycle's instruction is ever
# carried forward into this cycle's computation -- only the *revision
# counter* persists (see _next_revision), and only to detect a genuine
# change, never to reuse a stale result.
#
# ZERO execution authority: never calls DynamicSignageController/
# DynamicSignageProvider, never imports command_center/voice_evacuation/
# building_control/facp (mechanically enforced -- see tests/
# test_dynamic_signage_architecture_guards.py).
# =====================================================


@dataclasses.dataclass(frozen=True)
class _ZoneResult:

    indication: str
    status: str
    target_asset_id: Optional[str]
    target_asset_type: Optional[str]
    route_step_index: Optional[int]
    recommended_exit_id: Optional[str]
    guidance_revision: int
    confidence: Optional[float]
    reason: str


class DynamicSignagePlanner:

    def __init__(self, building, config: Optional[DynamicSignageConfig] = None):

        self.building = building
        self.config = config if config is not None else DynamicSignageConfig()

        # Phase 12 -- the ONE piece of cross-cycle state this planner
        # owns: {sign_id: (fingerprint, revision)}. Mirrors evacuation_
        # guidance.engine.EvacuationGuidanceEngine._revision_store's own
        # identical convention -- NOT a second sign registry, just a
        # revision counter.
        self._revision_store: Dict[str, Tuple[tuple, int]] = {}

    # =====================================================

    def compute(
        self, time: float, guidance_snapshot: Optional[EvacuationGuidanceSnapshot], signs: Iterable[object],
    ) -> DynamicSignageSnapshot:

        signs = tuple(signs)

        if guidance_snapshot is None:

            self._revision_store = {}
            return DynamicSignageSnapshot(timestamp=time, instructions={}, conflicts={})

        instructions = {}
        conflicts = {}

        for sign in sorted(signs, key=lambda s: s.id):

            instruction, conflict = self._plan_for_sign(sign, guidance_snapshot, time)
            instructions[sign.id] = instruction

            if conflict is not None:
                conflicts[sign.id] = conflict

        current_ids = {sign.id for sign in signs}
        self._revision_store = {sid: v for sid, v in self._revision_store.items() if sid in current_ids}

        return DynamicSignageSnapshot(timestamp=time, instructions=instructions, conflicts=conflicts)

    # =====================================================

    def _plan_for_sign(
        self, sign, guidance_snapshot: EvacuationGuidanceSnapshot, time: float,
    ) -> Tuple[SignageInstruction, Optional[SignageConflict]]:

        if not sign.active:

            return self._build(
                sign, time, zone_id=None, result=_ZoneResult(
                    indication=SignIndication.UNAVAILABLE, status=SignageStatus.UNAVAILABLE,
                    target_asset_id=None, target_asset_type=None, route_step_index=None,
                    recommended_exit_id=None, guidance_revision=0, confidence=None,
                    reason="Sign is inactive.",
                ),
            ), None

        # Phase 7 -- evaluate EVERY zone this sign is assigned to
        # independently; never assume the first one is authoritative.
        # Phase 5's own worked example ("a sign near/inside Z2 may
        # derive it from S1") means a zone need not have its OWN
        # recommendation to be a valid candidate -- a currently-valid
        # route ORIGINATING elsewhere may simply pass through it (a
        # corridor zone with no occupants of its own can still carry
        # signage for occupants flowing through). _zone_has_guidance()
        # is honestly true in either case, never fabricated for a zone
        # with neither.
        candidate_zone_ids = sorted(
            zone_id for zone_id in sign.zone_ids if self._zone_has_guidance(zone_id, guidance_snapshot)
        )

        if not candidate_zone_ids:

            return self._build(
                sign, time, zone_id=None, result=_ZoneResult(
                    indication=SignIndication.UNAVAILABLE, status=SignageStatus.UNAVAILABLE,
                    target_asset_id=None, target_asset_type=None, route_step_index=None,
                    recommended_exit_id=None, guidance_revision=0, confidence=None,
                    reason="Sign is not assigned to any zone with current guidance.",
                ),
            ), None

        results = {
            zone_id: self._resolve_for_zone(sign, zone_id, guidance_snapshot)
            for zone_id in candidate_zone_ids
        }

        # Phase 7 -- group zones by their EFFECTIVE outcome (what the
        # sign would actually have to display), never by raw guidance
        # revision. Two zones asking for the exact same indication/
        # target/exit are not a conflict even if their own guidance
        # revisions differ.
        signatures: Dict[tuple, list] = {}

        for zone_id in candidate_zone_ids:

            result = results[zone_id]
            signature = (result.indication, result.target_asset_id, result.recommended_exit_id)
            signatures.setdefault(signature, []).append(zone_id)

        if len(signatures) > 1:

            conflicting_zone_ids = tuple(candidate_zone_ids)

            conflict = SignageConflict(
                sign_id=sign.id, floor_id=sign.floor_id, conflicting_zone_ids=conflicting_zone_ids,
                reason=(
                    f"Sign {sign.id} is assigned to zones {conflicting_zone_ids} whose current guidance "
                    f"requires different signage this cycle. Never resolved arbitrarily."
                ),
                timestamp=time,
            )

            instruction = self._build(
                sign, time, zone_id=None, result=_ZoneResult(
                    indication=SignIndication.UNAVAILABLE, status=SignageStatus.CONFLICT,
                    target_asset_id=None, target_asset_type=None, route_step_index=None,
                    recommended_exit_id=None, guidance_revision=0, confidence=None,
                    reason=conflict.reason,
                ), extra_fingerprint=("CONFLICT", conflicting_zone_ids),
            )

            return instruction, conflict

        # Exactly one effective outcome -- pick the lowest zone_id as
        # the canonical zone_id reported on the instruction, purely for
        # deterministic display; the outcome itself is identical
        # regardless of which contributing zone is "the" one.
        zone_id = candidate_zone_ids[0]

        return self._build(sign, time, zone_id=zone_id, result=results[zone_id]), None

    # =====================================================

    def _zone_has_guidance(self, zone_id: str, guidance_snapshot: EvacuationGuidanceSnapshot) -> bool:

        if guidance_snapshot.zone(zone_id) is not None:
            return True

        return any(
            plan.is_valid() and zone_id in plan.ordered_zone_ids
            for plan in guidance_snapshot.zones.values()
        )

    # =====================================================

    def _resolve_for_zone(self, sign, zone_id: str, guidance_snapshot: EvacuationGuidanceSnapshot) -> _ZoneResult:

        own_plan = guidance_snapshot.zone(zone_id)

        if own_plan is not None:

            if not own_plan.is_valid():
                return self._resolve_invalid_plan(zone_id, own_plan)

            return self._resolve_meaningful_step(sign, zone_id, own_plan)

        # Phase 5 -- a zone with no recommendation of its own (a pure
        # corridor/pass-through), but genuinely on at least one OTHER
        # zone's currently-valid route. Every distinct passing-through
        # plan is resolved independently; they must all agree on the
        # EFFECTIVE outcome, or this zone's own contribution is honestly
        # reported as ambiguous (never an arbitrary pick of whichever
        # route happened to iterate first).
        passthrough_plans = [
            plan for plan in guidance_snapshot.zones.values()
            if plan.is_valid() and zone_id in plan.ordered_zone_ids
        ]

        sub_results = [self._resolve_meaningful_step(sign, zone_id, plan) for plan in passthrough_plans]

        signatures = {(r.indication, r.target_asset_id, r.recommended_exit_id) for r in sub_results}

        if len(signatures) == 1:
            return sub_results[0]

        return _ZoneResult(
            indication=SignIndication.UNAVAILABLE, status=SignageStatus.UNAVAILABLE, target_asset_id=None,
            target_asset_type=None, route_step_index=None, recommended_exit_id=None, guidance_revision=0,
            confidence=None,
            reason=f"Multiple active routes pass through zone {zone_id} requiring different signage this cycle.",
        )

    # =====================================================

    def _resolve_invalid_plan(self, zone_id: str, plan) -> _ZoneResult:

        if plan.route_status == RouteStatus.NO_SAFE_EXIT:
            indication = SignIndication.NO_SAFE_DIRECTION
            reason = f"No safe exit is currently recommended for zone {zone_id}."

        elif (
            plan.route_status == RouteStatus.ROUTE_UNAVAILABLE
            and GuidanceInconsistency.ROUTE_BECAME_UNSAFE in plan.inconsistencies
        ):
            indication = SignIndication.DO_NOT_USE
            reason = f"The previously recommended route for zone {zone_id} has become unsafe."

        else:
            indication = SignIndication.UNAVAILABLE
            reason = f"No valid guidance route for zone {zone_id} ({plan.route_status})."

        return _ZoneResult(
            indication=indication, status=SignageStatus.UNAVAILABLE, target_asset_id=None,
            target_asset_type=None, route_step_index=None, recommended_exit_id=plan.recommended_exit_id,
            guidance_revision=plan.revision, confidence=plan.confidence, reason=reason,
        )

    # =====================================================

    def _resolve_meaningful_step(self, sign, zone_id: str, plan) -> _ZoneResult:

        # `plan` is already known valid (is_valid()) by every caller.

        meaningful = next_meaningful_step_for_zone(plan, zone_id)

        if meaningful is None:

            if plan.route_status == RouteStatus.ALREADY_AT_EXIT and plan.recommended_exit_id is not None and plan.zone_id == zone_id:

                # Phase 5 -- occupants in this zone are already at the
                # recommended exit; no further decision, no directional
                # arrow, just the placard.
                return _ZoneResult(
                    indication=SignIndication.EXIT_HERE, status=SignageStatus.ACTIVE,
                    target_asset_id=plan.recommended_exit_id, target_asset_type="Exit", route_step_index=None,
                    recommended_exit_id=plan.recommended_exit_id, guidance_revision=plan.revision,
                    confidence=plan.confidence, reason="Occupants in this zone are already at the recommended exit.",
                )

            return _ZoneResult(
                indication=SignIndication.UNAVAILABLE, status=SignageStatus.UNAVAILABLE, target_asset_id=None,
                target_asset_type=None, route_step_index=None, recommended_exit_id=plan.recommended_exit_id,
                guidance_revision=plan.revision, confidence=plan.confidence,
                reason=f"No meaningful navigation decision found for zone {zone_id} on the current route.",
            )

        target = resolve_target(self.building, meaningful)

        if target is None:

            return _ZoneResult(
                indication=SignIndication.UNAVAILABLE, status=SignageStatus.UNAVAILABLE, target_asset_id=None,
                target_asset_type=None, route_step_index=meaningful.route_step_index,
                recommended_exit_id=plan.recommended_exit_id, guidance_revision=plan.revision,
                confidence=plan.confidence, reason="Target engineering asset geometry is unavailable.",
            )

        step_type = meaningful.step.step_type

        if step_type == NavigationStepType.CONTINUE_TO_EXIT:
            indication = SignIndication.EXIT_HERE

        elif step_type == NavigationStepType.USE_STAIR:
            indication = SignIndication.USE_STAIRS

        else:

            indication = relative_direction(sign.position, sign.orientation, target.position, self.config)

            if indication is None:

                return _ZoneResult(
                    indication=SignIndication.UNAVAILABLE, status=SignageStatus.UNAVAILABLE,
                    target_asset_id=target.asset_id, target_asset_type=target.asset_type,
                    route_step_index=meaningful.route_step_index, recommended_exit_id=plan.recommended_exit_id,
                    guidance_revision=plan.revision, confidence=plan.confidence,
                    reason="Sign is co-located with its target; no honest direction is derivable.",
                )

        if indication not in sign.supported_indications:

            return _ZoneResult(
                indication=SignIndication.UNAVAILABLE, status=SignageStatus.UNAVAILABLE,
                target_asset_id=target.asset_id, target_asset_type=target.asset_type,
                route_step_index=meaningful.route_step_index, recommended_exit_id=plan.recommended_exit_id,
                guidance_revision=plan.revision, confidence=plan.confidence,
                reason=f"Sign hardware does not support indication {indication!r}.",
            )

        return _ZoneResult(
            indication=indication, status=SignageStatus.ACTIVE, target_asset_id=target.asset_id,
            target_asset_type=target.asset_type, route_step_index=meaningful.route_step_index,
            recommended_exit_id=plan.recommended_exit_id, guidance_revision=plan.revision,
            confidence=plan.confidence, reason="",
        )

    # =====================================================

    def _build(
        self, sign, time: float, *, zone_id: Optional[str], result: _ZoneResult,
        extra_fingerprint: Optional[tuple] = None,
    ) -> SignageInstruction:

        # Phase 12 -- deliberately excludes raw guidance_revision from
        # the fingerprint: an unrelated guidance revision bump (e.g. a
        # STALE_GUIDANCE flag toggling) that leaves the EFFECTIVE
        # indication/target/exit unchanged must never spam a new
        # signage revision.
        fingerprint = extra_fingerprint if extra_fingerprint is not None else (
            result.indication, result.status, result.target_asset_id, result.recommended_exit_id, zone_id,
        )

        revision = self._next_revision(sign.id, fingerprint)

        return SignageInstruction(
            sign_id=sign.id, floor_id=sign.floor_id, zone_id=zone_id,
            recommended_exit_id=result.recommended_exit_id, guidance_revision=result.guidance_revision,
            indication=result.indication, target_asset_id=result.target_asset_id,
            target_asset_type=result.target_asset_type, route_step_index=result.route_step_index,
            status=result.status, confidence=result.confidence, reason=result.reason,
            signage_revision=revision, timestamp=time,
        )

    # =====================================================

    def _next_revision(self, sign_id: str, fingerprint: tuple) -> int:

        previous = self._revision_store.get(sign_id)

        if previous is not None and previous[0] == fingerprint:
            return previous[1]

        new_revision = (previous[1] + 1) if previous is not None else 1
        self._revision_store[sign_id] = (fingerprint, new_revision)

        return new_revision
