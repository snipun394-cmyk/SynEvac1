from typing import Tuple

from voice_evacuation.models import BroadcastStatus

from execution_layer.models import ExecutionCategory, ExecutionRequest, ExecutionStatus, RecommendationIdProvenance


# =====================================================
# Voice Evacuation adapter -- READ-ONLY. Reads VoiceEvacuationController.
# broadcast_log (already-recorded state) and normalizes it into the
# unified ExecutionRequest vocabulary. Never calls broadcast()/cancel()
# -- VoiceEvacuationController remains the sole execution authority for
# this category.
#
# Disclosed limitation: unlike Building Control/Signage, Voice
# Evacuation's own controller has no separate PENDING_APPROVAL/APPROVED
# phase of its own -- broadcast() dispatches synchronously the moment
# it is called (the pre-dispatch "has an operator reviewed this yet"
# bookkeeping lives one layer up, inside command_center.
# live_operator_action_gateway.LiveOperatorActionGateway's own state,
# which this read-only adapter deliberately does not reach into to
# stay minimal-footprint). created_at/approved_at are therefore always
# None for this category in V1 -- an honest gap, never fabricated.
# =====================================================


_STATUS_BY_BROADCAST_STATUS = {
    BroadcastStatus.BROADCAST: ExecutionStatus.CONFIRMED,
    BroadcastStatus.NO_SPEAKERS_AVAILABLE: ExecutionStatus.FAILED,
    BroadcastStatus.SUPERSEDED: ExecutionStatus.SUPERSEDED,
    BroadcastStatus.CANCELLED: ExecutionStatus.CANCELLED,
}


def build_execution_requests(controller) -> Tuple[ExecutionRequest, ...]:

    if controller is None:
        return ()

    provider_source = type(controller.provider).__name__

    requests = []

    for instruction in controller.broadcast_log.all_instructions():

        source_recommendation_id = instruction.message.source_recommendation_id if instruction.message is not None else None

        requests.append(ExecutionRequest(
            execution_request_id=instruction.instruction_id,
            category=ExecutionCategory.VOICE_EVACUATION,
            status=_STATUS_BY_BROADCAST_STATUS.get(instruction.status, ExecutionStatus.FAILED),
            provider_source=provider_source,
            originating_recommendation_id=source_recommendation_id,
            recommendation_id_provenance=(
                RecommendationIdProvenance.ADVISORY_SYSTEM if source_recommendation_id is not None
                else RecommendationIdProvenance.UNAVAILABLE
            ),
            target_description=f"Zone {instruction.target_zone_id}",
            created_at=None,
            approved_at=None,
            dispatched_at=instruction.timestamp,
            completed_at=instruction.timestamp,
            result_message="",
            result_confirmed=instruction.status == BroadcastStatus.BROADCAST,
        ))

    return tuple(requests)
