from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from voice_evacuation.adapter import civilian_announcement_to_voice_message, guidance_plan_to_voice_message
from voice_evacuation.controller import VoiceEvacuationController
from voice_evacuation.models import BroadcastInstruction, BroadcastStatus

from building_control.advisory_adapter import translate_report
from building_control.controller import BuildingControlController
from building_control.history import ControlEvent
from building_control.requests import ControlRequest
from building_control.snapshot import ControlStateEntry

from dynamic_signage.consistency import instruction_inconsistencies
from dynamic_signage.controller import DynamicSignageController, SignageControlEvent, SignageRequestStatus
from dynamic_signage.models import SignageInstruction, SignageStatus

from event_bus.bus import EventType


# =====================================================
# Live Operator Action Routing milestone -- the ONE seam Command Center
# is allowed to route explicit operator intent through to real
# execution. Lives in `command_center`, not `live_system`, for exactly
# the reason command_center/incident_data.py already does (see that
# module's own precedent): `live_system` is mechanically forbidden from
# importing `voice_evacuation`/`speaker_manager`/`building_control.
# controller`/`building_control.providers` at all (tests/
# test_live_system.py::LiveSystemPackageDependencyDirectionTests), a
# guard whose whole point is that the AI-inference/advisory-generation
# side of the platform can never reach execution authority. This
# module is deliberately the one place in `command_center` allowed to
# import those two controllers directly -- excluded from tests/
# test_live_command_center.py::CommandCenterLiveIntegrationGuardTests'
# forbidden-import scan the same way incident_data.py already is,
# because unlike every panel file that guard covers, THIS module's
# entire purpose is to be the bridge from an explicit operator action
# to a real controller.
#
# Every panel that needs to trigger an action (VoiceEvacuationPanel,
# BuildingControlsPanel -- see recommendation_center.py/
# building_controls_panel.py) is handed an already-constructed instance
# of this gateway and calls only its public methods; neither panel file
# imports voice_evacuation.controller/building_control.controller
# itself, so the pre-existing CommandCenterLiveIntegrationGuardTests
# guard continues to pass unmodified even after this milestone wires
# real execution behind an operator's Approve/Reject click.
#
# THE ONE RULE THIS MODULE HOLDS TO: every method here is called from
# exactly one place -- a UI event handler reacting to an operator's own
# explicit click (or, for tests, a direct call standing in for one).
# AdvisoryReport generation, AI inference, and Decision Policy never
# call anything in this module -- mechanically proven by
# tests/test_live_operator_action_routing.py::AIAuthorityGuardTests,
# which scans advisory_system/, live_system/live_ai_gateway.py,
# live_system/live_advisory_gateway.py, and decision_policy/ for any
# import of this module or of voice_evacuation/building_control.
# controller/building_control.providers directly.
# =====================================================


PROVIDER_CAPABILITY_NO_PROVIDER = "NO_PROVIDER"
PROVIDER_CAPABILITY_SIMULATION = "SIMULATION"
PROVIDER_CAPABILITY_LIVE_HARDWARE = "LIVE_HARDWARE"

# The one operator-identity value this milestone uses (Phase 12): no
# authentication/user-management system exists anywhere in this
# codebase, and this milestone does not build one -- COMMAND_CENTER_
# OPERATOR is a neutral, honest source label, never a fabricated
# username, passed through BuildingControlController.approve()/
# reject()'s own pre-existing `actor` parameter (building_control/
# controller.py) and this module's own voice-side OperatorActionRecord
# below.
OPERATOR_ACTOR = "COMMAND_CENTER_OPERATOR"

VOICE_STATUS_RECOMMENDED = "RECOMMENDED"
VOICE_STATUS_APPROVED = "APPROVED"
VOICE_STATUS_SENT = "SENT"
VOICE_STATUS_FAILED = "FAILED"
VOICE_STATUS_REJECTED = "REJECTED"
VOICE_STATUS_SUPERSEDED = "SUPERSEDED"

# Reuses voice_evacuation.models.BroadcastStatus's own existing
# vocabulary wherever it already means the same thing (SUPERSEDED maps
# straight across; CANCELLED is this gateway's own REJECTED, since
# nothing else in this codebase's voice-evacuation history distinguishes
# an operator-withdrawn message from a controller-cancelled one) --
# RECOMMENDED/APPROVED are the only genuinely new labels, because
# nothing before this milestone tracked "has an operator reviewed this
# recommendation yet" at all.
_VOICE_STATUS_BY_BROADCAST_STATUS = {
    BroadcastStatus.BROADCAST: VOICE_STATUS_SENT,
    BroadcastStatus.NO_SPEAKERS_AVAILABLE: VOICE_STATUS_FAILED,
    BroadcastStatus.SUPERSEDED: VOICE_STATUS_SUPERSEDED,
    BroadcastStatus.CANCELLED: VOICE_STATUS_REJECTED,
}


class OperatorActionUnavailable(Exception):

    # Raised only when an operator action was attempted with no
    # provider configured for it at all (Phase 5's "action fails
    # honestly" requirement) -- never for a provider that is configured
    # but reports failure (that is ControlResult.confirmed=False /
    # BroadcastStatus.NO_SPEAKERS_AVAILABLE, an honest execution
    # outcome, not an unavailable-action error).

    pass


class SignageApprovalBlocked(Exception):

    # Live Dynamic Sign Operator Approval & Dispatch Completion milestone,
    # Phase 7 -- raised only by approve_signage_instruction() when a
    # caller supplies the current guidance_snapshot and this exact
    # instruction disagrees with it (SIGNAGE_GUIDANCE_MISMATCH/
    # STALE_SIGNAGE_REVISION/VOICE_SIGNAGE_EXIT_MISMATCH -- see
    # dynamic_signage.consistency.instruction_inconsistencies()). A
    # provider IS configured and the instruction IS otherwise
    # PENDING_APPROVAL -- this is never OperatorActionUnavailable's own
    # "nothing to approve through" case, it is "this specific approval
    # would dispatch a sign instruction that no longer agrees with the
    # current recommended route," which must fail honestly rather than
    # silently dispatch a stale/inconsistent direction.

    pass


@dataclass(frozen=True)
class OperatorActionRecord:

    # Voice's own minimal audit-trail entry (Phase 12). building_control
    # already has ControlEvent (building_control/history.py, with its
    # own actor/note fields) for exactly this purpose on the control
    # side -- reused as-is via control_history() below, never
    # duplicated. voice_evacuation has no equivalent: BroadcastInstruction
    # (voice_evacuation/models.py) carries no actor field, and adding one
    # to that frozen, already-widely-consumed dataclass is a bigger
    # change than this milestone's own "smallest clean boundary"
    # instruction calls for. This record lives entirely on the Command-
    # Center side of the gateway instead.

    action: str
    zone_id: str
    actor: str
    time: float
    detail: str = ""


class LiveOperatorActionGateway:

    def __init__(
        self,
        *,
        voice_controller: Optional[VoiceEvacuationController] = None,
        control_controller: Optional[BuildingControlController] = None,
        signage_controller: Optional[DynamicSignageController] = None,
        event_bus: Optional[object] = None,
    ):

        self._voice_controller = voice_controller
        self._control_controller = control_controller
        self._signage_controller = signage_controller

        # Optional -- duck-typed (only .emit(event_type, payload, time)
        # is ever called), so this module never needs to import
        # live_system.event_bus.EventType itself just to accept one.
        # None is the honest default for every existing caller that
        # constructed this gateway before Live Dynamic Evacuation
        # Signage added it -- no behavior change for voice/building
        # control, which never published bus events and still don't.
        self._event_bus = event_bus

        # zone_id + announcement text -> "APPROVED"/"REJECTED". Keyed on
        # content, not just zone_id, so a genuinely NEW recommendation
        # for a zone that previously had a decided-on message (Phase 11
        # item 6, "recommendation superseded before approval") always
        # starts fresh as RECOMMENDED -- only the identical recommendation
        # reappearing across cycles keeps its prior decision.
        self._voice_decisions: Dict[str, str] = {}
        self._voice_instructions: Dict[str, Tuple[BroadcastInstruction, ...]] = {}
        self._voice_audit_log: List[OperatorActionRecord] = []

        # Live Evacuation Guidance & Zoned Message Planning milestone,
        # Phase 15/16/19 -- a SEPARATE, identically-shaped decision/
        # instruction store for guidance-derived messages, keyed on
        # zone_id + guidance_revision (never message text -- Phase 12's
        # own revision already IS the deterministic content identity).
        # A new revision always starts fresh as RECOMMENDED; the OLD
        # revision's own key/decision/instructions are never deleted
        # (Phase 16's own "sent old guidance remains in history, unsent
        # old guidance becomes stale" requirement) -- superseding a
        # message an operator never approved simply means its own key
        # is left behind at whatever RECOMMENDED/REJECTED state it was
        # in, no longer reachable from the CURRENT revision's own key.
        self._guidance_decisions: Dict[str, str] = {}
        self._guidance_instructions: Dict[str, Tuple[BroadcastInstruction, ...]] = {}

        # source_recommendation_id values an operator has explicitly
        # rejected (Phase 4/11 item 8). REJECTED is a TERMINAL status
        # (building_control/types.py) -- BuildingControlController.
        # submit()'s own dedup only skips resubmission while a prior
        # request is still LIVE, so without this, calling
        # ingest_control_recommendations() again on the very next
        # render (e.g. the show_live() re-render this panel's own
        # _on_reject_live() triggers immediately after rejecting) would
        # silently resubmit the identical recommendation as a fresh
        # PENDING_APPROVAL request, undoing the operator's own decision
        # before they ever see it take effect. A genuinely different
        # recommendation (different system/target/action ->
        # a different synthesized id) is never blocked by this.
        self._rejected_control_recommendation_ids: set = set()

    # =====================================================
    # Shared-instance identity (Production Live Runtime Composition
    # Root milestone, Phase 4) -- read-only, so a composition root/test
    # can prove the SAME VoiceEvacuationController/BuildingControlController
    # instance a live session's other components (e.g. a future
    # BuildingControlsPanel Replay-style direct read, or an identity
    # assertion in a test) reference is exactly the one this gateway
    # routes operator actions through -- never a silently-duplicated
    # second controller. Mirrors VoiceEvacuationController.provider/
    # BuildingControlController.provider's own identical "expose what is
    # already stored, add no new logic" precedent.
    # =====================================================

    @property
    def voice_controller(self) -> Optional[VoiceEvacuationController]:

        return self._voice_controller

    # =====================================================

    @property
    def control_controller(self) -> Optional[BuildingControlController]:

        return self._control_controller

    # =====================================================

    @property
    def signage_controller(self) -> Optional[DynamicSignageController]:

        return self._signage_controller

    # =====================================================
    # Provider capability (Phase 5)
    # =====================================================

    @property
    def voice_capability(self) -> str:

        if self._voice_controller is None:
            return PROVIDER_CAPABILITY_NO_PROVIDER

        return (
            PROVIDER_CAPABILITY_SIMULATION if self._voice_controller.provider.is_simulation_only
            else PROVIDER_CAPABILITY_LIVE_HARDWARE
        )

    # =====================================================

    @property
    def control_capability(self) -> str:

        if self._control_controller is None:
            return PROVIDER_CAPABILITY_NO_PROVIDER

        return (
            PROVIDER_CAPABILITY_SIMULATION if self._control_controller.provider.is_simulation_only
            else PROVIDER_CAPABILITY_LIVE_HARDWARE
        )

    # =====================================================

    @property
    def signage_capability(self) -> str:

        if self._signage_controller is None:
            return PROVIDER_CAPABILITY_NO_PROVIDER

        return (
            PROVIDER_CAPABILITY_SIMULATION if self._signage_controller.provider.is_simulation_only
            else PROVIDER_CAPABILITY_LIVE_HARDWARE
        )

    # =====================================================
    # Voice Evacuation operator workflow (Phase 3)
    # =====================================================

    def _voice_key(self, announcement) -> str:

        return f"{announcement.zone_id}::{announcement.announcement}"

    # =====================================================

    def voice_recommendation_status(self, announcement) -> str:

        # Pure derivation, safe to call every render cycle -- never
        # mutates state, never touches the controller/provider.

        key = self._voice_key(announcement)
        decision = self._voice_decisions.get(key)

        if decision is None:
            return VOICE_STATUS_RECOMMENDED

        if decision == VOICE_STATUS_REJECTED:
            return VOICE_STATUS_REJECTED

        instructions = self._voice_instructions.get(key, ())

        if not instructions:
            return VOICE_STATUS_APPROVED

        relevant = tuple(i for i in instructions if i.target_zone_id == announcement.zone_id)
        latest = relevant[-1] if relevant else instructions[-1]

        return _VOICE_STATUS_BY_BROADCAST_STATUS.get(latest.status, VOICE_STATUS_APPROVED)

    # =====================================================

    def approve_voice_message(self, announcement, time: float) -> Tuple[BroadcastInstruction, ...]:

        if self._voice_controller is None:

            raise OperatorActionUnavailable(
                "No voice output provider is configured -- cannot approve/send this message."
            )

        key = self._voice_key(announcement)

        if self._voice_decisions.get(key) == VOICE_STATUS_APPROVED:

            # Duplicate click / a caller re-approving the identical,
            # already-approved recommendation (Phase 11 item 7) -- never
            # re-broadcast, just hand back what already happened.
            return self._voice_instructions.get(key, ())

        message = civilian_announcement_to_voice_message(announcement, timestamp=time)
        instructions = self._voice_controller.broadcast(message, time)

        self._voice_decisions[key] = VOICE_STATUS_APPROVED
        self._voice_instructions[key] = instructions

        self._voice_audit_log.append(
            OperatorActionRecord(
                action="APPROVE_VOICE_MESSAGE", zone_id=announcement.zone_id,
                actor=OPERATOR_ACTOR, time=time, detail=announcement.announcement,
            )
        )

        return instructions

    # =====================================================

    def reject_voice_message(self, announcement, time: float) -> None:

        # Rejecting never requires a provider -- it is purely "the
        # operator chooses not to approve this," never touches
        # VoiceEvacuationController/the output provider at all.

        key = self._voice_key(announcement)
        self._voice_decisions[key] = VOICE_STATUS_REJECTED

        self._voice_audit_log.append(
            OperatorActionRecord(
                action="REJECT_VOICE_MESSAGE", zone_id=announcement.zone_id,
                actor=OPERATOR_ACTOR, time=time, detail=announcement.announcement,
            )
        )

    # =====================================================

    def voice_audit_log(self) -> Tuple[OperatorActionRecord, ...]:

        return tuple(self._voice_audit_log)

    # =====================================================

    def voice_broadcast_history(self) -> Tuple[BroadcastInstruction, ...]:

        if self._voice_controller is None:
            return ()

        return self._voice_controller.broadcast_log.all_instructions()

    # =====================================================
    # Live Evacuation Guidance & Zoned Message Planning milestone,
    # Phase 14/19 -- the guidance-message counterpart to the civilian-
    # announcement workflow immediately above, same shape, same
    # requirement ("every method here is called from exactly one
    # place -- an operator's own explicit click"). Reuses the SAME
    # self._voice_controller (never a second controller) -- "one final
    # operator-visible message per zone" is enforced by that
    # controller's own pre-existing per-zone priority supersession
    # (see voice_evacuation/adapter.py's own guidance_plan_to_voice_
    # message() docstring), not by any new logic here.
    # =====================================================

    def _guidance_key(self, plan) -> str:

        return f"{plan.zone_id}::{plan.guidance_revision}"

    # =====================================================

    def guidance_recommendation_status(self, plan) -> str:

        # Pure derivation, safe to call every render cycle -- never
        # mutates state, never touches the controller/provider. Mirrors
        # voice_recommendation_status() exactly.

        key = self._guidance_key(plan)
        decision = self._guidance_decisions.get(key)

        if decision is None:
            return VOICE_STATUS_RECOMMENDED

        if decision == VOICE_STATUS_REJECTED:
            return VOICE_STATUS_REJECTED

        instructions = self._guidance_instructions.get(key, ())

        if not instructions:
            return VOICE_STATUS_APPROVED

        relevant = tuple(i for i in instructions if i.target_zone_id == plan.zone_id)
        latest = relevant[-1] if relevant else instructions[-1]

        return _VOICE_STATUS_BY_BROADCAST_STATUS.get(latest.status, VOICE_STATUS_APPROVED)

    # =====================================================

    def approve_guidance_message(self, plan, time: float) -> Tuple[BroadcastInstruction, ...]:

        if self._voice_controller is None:

            raise OperatorActionUnavailable(
                "No voice output provider is configured -- cannot approve/send this guidance message."
            )

        key = self._guidance_key(plan)

        if self._guidance_decisions.get(key) == VOICE_STATUS_APPROVED:

            # Duplicate click / re-approving the identical, already-
            # approved revision -- never re-broadcast.
            return self._guidance_instructions.get(key, ())

        message = guidance_plan_to_voice_message(plan, timestamp=time)
        instructions = self._voice_controller.broadcast(message, time)

        self._guidance_decisions[key] = VOICE_STATUS_APPROVED
        self._guidance_instructions[key] = instructions

        self._voice_audit_log.append(
            OperatorActionRecord(
                action="APPROVE_GUIDANCE_MESSAGE", zone_id=plan.zone_id,
                actor=OPERATOR_ACTOR, time=time, detail=plan.message_text,
            )
        )

        return instructions

    # =====================================================

    def reject_guidance_message(self, plan, time: float) -> None:

        # Rejecting never requires a provider -- purely "the operator
        # chooses not to approve this," never touches the controller/
        # output provider at all.

        key = self._guidance_key(plan)
        self._guidance_decisions[key] = VOICE_STATUS_REJECTED

        self._voice_audit_log.append(
            OperatorActionRecord(
                action="REJECT_GUIDANCE_MESSAGE", zone_id=plan.zone_id,
                actor=OPERATOR_ACTOR, time=time, detail=plan.message_text,
            )
        )

    # =====================================================
    # Building Control operator workflow (Phase 4)
    # =====================================================

    def ingest_control_recommendations(self, report) -> Tuple[ControlRequest, ...]:

        # Mirrors command_center/incident_data.py's own
        # ingest_control_recommendations() precedent exactly: translate,
        # then submit() each -- BuildingControlController.submit() itself
        # already performs deduplication (Phase 11 item 5) and target
        # validation, so calling this once per render cycle is always
        # safe (never creates a duplicate live PENDING_APPROVAL request
        # for the same system/target/action).

        if self._control_controller is None or report is None:
            return ()

        requests = translate_report(report)

        submitted = []

        for request in requests:

            if request.source_recommendation_id in self._rejected_control_recommendation_ids:
                # An operator already said no to exactly this
                # recommendation -- never silently resubmit it just
                # because the same AdvisoryReport was re-ingested on a
                # later render (see this gateway's own __init__ docstring
                # for _rejected_control_recommendation_ids).
                continue

            submitted.append(self._control_controller.submit(request))

        return tuple(submitted)

    # =====================================================

    def approve_control_request(self, request_id: str) -> ControlRequest:

        if self._control_controller is None:

            raise OperatorActionUnavailable(
                "No building control provider is configured -- cannot approve this request."
            )

        return self._control_controller.approve(request_id, actor=OPERATOR_ACTOR)

    # =====================================================

    def reject_control_request(self, request_id: str) -> ControlRequest:

        if self._control_controller is None:

            raise OperatorActionUnavailable(
                "No building control provider is configured -- cannot reject this request."
            )

        request = self._control_controller.reject(request_id, actor=OPERATOR_ACTOR)

        if request.source_recommendation_id is not None:
            self._rejected_control_recommendation_ids.add(request.source_recommendation_id)

        return request

    # =====================================================

    def pending_control_requests(self) -> Tuple[ControlRequest, ...]:

        if self._control_controller is None:
            return ()

        return self._control_controller.pending_requests()

    # =====================================================

    def all_control_requests(self) -> Tuple[ControlRequest, ...]:

        if self._control_controller is None:
            return ()

        return self._control_controller.all_requests()

    # =====================================================

    def confirmed_control_entries(self) -> Tuple[ControlStateEntry, ...]:

        if self._control_controller is None:
            return ()

        return self._control_controller.snapshot().entries

    # =====================================================

    def control_history(self) -> Tuple[ControlEvent, ...]:

        if self._control_controller is None:
            return ()

        return self._control_controller.history()

    # =====================================================
    # Dynamic Evacuation Signage operator workflow -- Live Dynamic
    # Evacuation Signage milestone, Phase 17/18. Mirrors the Building
    # Control workflow immediately above in shape (submit-then-decide,
    # via DynamicSignageController's own PENDING_APPROVAL/APPROVED/
    # REJECTED lifecycle -- unlike voice, which tracks the decision on
    # THIS gateway, DynamicSignageController tracks it itself, per
    # Phase 13's own explicit design). Only ACTIVE instructions (a
    # genuine, actionable indication) are ever submitted for approval --
    # an UNAVAILABLE/CONFLICT instruction is display-only, never
    # something an operator can "approve" into existence.
    # =====================================================

    def ingest_signage_instructions(self, snapshot) -> Tuple[SignageInstruction, ...]:

        if self._signage_controller is None or snapshot is None:
            return ()

        submitted = []

        for instruction in snapshot.instructions.values():

            if instruction.status != SignageStatus.ACTIVE:
                continue

            submitted.append(self._signage_controller.submit(instruction, instruction.timestamp))

        return tuple(submitted)

    # =====================================================

    def signage_instruction_status(self, instruction: SignageInstruction) -> str:

        # Pure derivation, safe to call every render cycle -- never
        # mutates state, never touches the controller/provider itself.

        if self._signage_controller is None:
            return SignageRequestStatus.PENDING_APPROVAL

        try:
            return self._signage_controller.status_of(instruction.sign_id, instruction.signage_revision)
        except KeyError:
            return SignageRequestStatus.PENDING_APPROVAL

    # =====================================================

    def approve_signage_instruction(
        self, instruction: SignageInstruction, time: float, guidance_snapshot=None,
    ) -> SignageInstruction:

        if self._signage_controller is None:

            raise OperatorActionUnavailable(
                "No dynamic signage provider is configured -- cannot approve this sign instruction."
            )

        # Phase 7 -- an OPTIONAL, additive guard: a caller (the real
        # Command Center panel, always) that has the current
        # EvacuationGuidanceSnapshot in hand passes it here, and this
        # exact instruction is re-checked against it one last time,
        # right at the one seam that can actually reach a provider --
        # never trusted merely because it was PENDING_APPROVAL. Callers
        # that don't have one (every pre-existing test in
        # tests/test_dynamic_signage_command_center.py and tests/
        # test_dynamic_signage_e2e.py) keep their prior behavior exactly.
        #
        # Only checked while the instruction is still genuinely
        # PENDING_APPROVAL -- an already-resolved (CONFIRMED/REJECTED/
        # SUPERSEDED/FAILED) request fails for the more fundamentally
        # correct reason (DynamicSignageController.approve()'s own
        # "not PENDING_APPROVAL" error below), never relabeled as a
        # guidance mismatch merely because a since-superseded
        # instruction's own recommended_exit_id no longer agrees with
        # the CURRENT cycle's Guidance.
        if guidance_snapshot is not None and self._signage_controller.status_of(
            instruction.sign_id, instruction.signage_revision,
        ) == SignageRequestStatus.PENDING_APPROVAL:

            codes = instruction_inconsistencies(guidance_snapshot, instruction)

            if codes:

                raise SignageApprovalBlocked(
                    f"cannot approve signage instruction for sign {instruction.sign_id!r}: "
                    f"disagrees with current Evacuation Guidance ({', '.join(codes)})."
                )

        result = self._signage_controller.approve(
            instruction.sign_id, instruction.signage_revision, time, actor=OPERATOR_ACTOR,
        )

        if self._event_bus is not None:
            self._event_bus.emit(EventType.SIGNAGE_INSTRUCTION_APPROVED, result, time)

        return result

    # =====================================================

    def reject_signage_instruction(self, instruction: SignageInstruction, time: float) -> SignageInstruction:

        if self._signage_controller is None:

            raise OperatorActionUnavailable(
                "No dynamic signage provider is configured -- cannot reject this sign instruction."
            )

        result = self._signage_controller.reject(
            instruction.sign_id, instruction.signage_revision, time, actor=OPERATOR_ACTOR,
        )

        if self._event_bus is not None:
            self._event_bus.emit(EventType.SIGNAGE_INSTRUCTION_REJECTED, result, time)

        return result

    # =====================================================

    def pending_signage_instructions(self) -> Tuple[SignageInstruction, ...]:

        if self._signage_controller is None:
            return ()

        return self._signage_controller.pending_instructions()

    # =====================================================

    def all_signage_instructions(self) -> Tuple[SignageInstruction, ...]:

        if self._signage_controller is None:
            return ()

        return self._signage_controller.all_instructions()

    # =====================================================

    def signage_history(self) -> Tuple[SignageControlEvent, ...]:

        if self._signage_controller is None:
            return ()

        return self._signage_controller.history()
