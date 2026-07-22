from typing import Mapping, Optional, Tuple

from advisory_system.recommendation_models import CivilianAnnouncement

from decision_policy.zone_policy import EVACUATE_IMMEDIATELY, SHELTER_IN_PLACE, WAIT

from evacuation_guidance.models import VoiceGuidanceMessagePlan

from voice_evacuation.models import VoiceMessage, VoiceMessageType, priority_for_message_type


# decision_policy.zone_policy's own already-computed zone action ->
# VoiceMessageType. Restated as a lookup table, not a new decision --
# advisory_system.CivilianAnnouncement itself carries no action/type
# field (only the fully-formatted announcement text; the action string
# lives one layer up, on decision_policy.zone_decisions, and is never
# copied onto CivilianAnnouncement -- confirmed by inspection of
# advisory_system/recommendation_models.py). WAIT maps to ROUTE_GUIDANCE
# ("hold your position, route is congested" is guidance about routing,
# not a full evacuate/shelter directive) -- the one interpretive choice
# this module makes, documented here rather than left implicit.
_MESSAGE_TYPE_BY_ZONE_ACTION = {
    SHELTER_IN_PLACE: VoiceMessageType.SHELTER_IN_PLACE,
    EVACUATE_IMMEDIATELY: VoiceMessageType.EVACUATE,
    WAIT: VoiceMessageType.ROUTE_GUIDANCE,
}


def civilian_announcement_to_voice_message(
    announcement: CivilianAnnouncement, timestamp: float, zone_action: Optional[str] = None,
) -> VoiceMessage:

    # The one adapter from advisory_system.CivilianAnnouncement to the
    # canonical VoiceMessage -- Phase 5's own required shape. Preserves
    # target zone, confidence, and a reference back to the source
    # recommendation exactly; carries the announcement's own already-
    # generated text through VERBATIM (advisory_engine.
    # build_civilian_announcements()/_format_announcement() already
    # produce a complete, zone-addressed sentence -- "Attention occupants
    # in <zone>. ..." -- this function never rewrites, paraphrases, or
    # regenerates that text, and never invents recommendation logic of
    # its own). zone_action is OPTIONAL and caller-supplied (typically
    # `decision["action"] for decision in decision_policy.zone_decisions
    # if decision["zone_id"] == announcement.zone_id`) purely so
    # message_type/priority can be classified honestly when the caller
    # happens to have it; when omitted, message_type stays None (never
    # guessed from the announcement text) and priority falls back to
    # priority_for_message_type(None)'s own documented default.
    #
    # source_recommendation_id is a composite, stable reference
    # (zone_id + timestamp) -- CivilianAnnouncement itself carries no
    # standalone recommendation id field to copy, so this is the
    # closest honest identifier back to "this exact recommendation,
    # for this zone, at this simulation time" without fabricating a
    # new unrelated id space.
    #
    # Explicit architecture guard: nothing here ever reads or copies an
    # occupant_id -- CivilianAnnouncement itself carries none (Phase 2 of
    # the Advisory System's own zone-only design), and this function
    # only ever touches zone_id/zone_name/announcement/reason/confidence.

    message_type = _MESSAGE_TYPE_BY_ZONE_ACTION.get(zone_action) if zone_action else None

    return VoiceMessage(
        timestamp=timestamp,
        target_zone_ids=(announcement.zone_id,),
        message_text=announcement.announcement,
        message_type=message_type,
        priority=priority_for_message_type(message_type),
        source_recommendation_id=f"{announcement.zone_id}@{timestamp}",
        confidence=announcement.confidence,
    )


def civilian_announcements_to_voice_messages(
    announcements: Tuple[CivilianAnnouncement, ...],
    timestamp: float,
    zone_actions: Optional[Mapping[str, str]] = None,
) -> Tuple[VoiceMessage, ...]:

    # Batch form -- Phase 6's "simultaneous messages to different
    # zones" starts here: every CivilianAnnouncement in one AdvisoryReport
    # (one per zone, computed independently by advisory_engine.
    # build_civilian_announcements()) becomes its own independent
    # VoiceMessage, never merged or reconciled against each other.

    zone_actions = zone_actions or {}

    return tuple(
        civilian_announcement_to_voice_message(
            announcement, timestamp, zone_action=zone_actions.get(announcement.zone_id),
        )
        for announcement in announcements
    )


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone, Phase 15
# -- the SECOND, separately-labelled voice-message source this
# milestone's own required "explicit architecture decision" resolves
# on: guidance messages are NEVER merged into CivilianAnnouncement.
# announcement text, and never broadcast independently either. They
# become their own VoiceMessage, always VoiceMessageType.ROUTE_GUIDANCE
# (priority 50 -- see voice_evacuation.models._PRIORITY_BY_TYPE),
# reaching VoiceEvacuationController through the EXACT SAME operator-
# approval seam a civilian announcement already does (command_center.
# live_operator_action_gateway.LiveOperatorActionGateway.
# approve_guidance_message()). "One final operator-visible message per
# zone" is then enforced by VoiceEvacuationController's OWN pre-existing
# per-zone priority supersession (voice_evacuation/controller.py's
# _route_to_zone()) -- an EVACUATE_IMMEDIATELY civilian announcement
# (priority 90) for the same zone always outranks and supersedes a
# ROUTE_GUIDANCE message (priority 50), never the reverse, without this
# module needing any reconciliation logic of its own.
# =====================================================


def guidance_plan_to_voice_message(plan: VoiceGuidanceMessagePlan, timestamp: float) -> VoiceMessage:

    # message_text is carried through VERBATIM from evacuation_guidance.
    # message_planner.build_voice_message_text() -- this function never
    # generates, rewrites, or paraphrases wording of its own, mirroring
    # civilian_announcement_to_voice_message()'s own identical discipline.

    return VoiceMessage(
        timestamp=timestamp,
        target_zone_ids=(plan.zone_id,),
        message_text=plan.message_text,
        message_type=VoiceMessageType.ROUTE_GUIDANCE,
        priority=priority_for_message_type(VoiceMessageType.ROUTE_GUIDANCE),
        source_recommendation_id=f"{plan.zone_id}@guidance-rev-{plan.guidance_revision}",
        confidence=None,
    )
