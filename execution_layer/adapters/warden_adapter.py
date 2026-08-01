from typing import Optional, Tuple

from warden_notification.requests import WardenNotificationRequest
from warden_notification.types import WardenNotificationStatus

from recommendation_layer.models import RecommendationType

from execution_layer.models import ExecutionCategory, ExecutionRequest, RecommendationIdProvenance


# =====================================================
# Warden Notification adapter -- the one category with two directions:
#
#   (a) translate()/translate_recommendation_set(): the SUBMIT side.
#       Converts a recommendation_layer.Recommendation (type
#       WARDEN_DISPATCH) into a WardenNotificationRequest, tagged with
#       the REAL recommendation_id -- built correctly from day one,
#       unlike the other three categories' advisory-sourced/absent
#       traceability. This function only CONSTRUCTS the request; it
#       never calls controller.submit() itself -- that happens in
#       command_center.live_operator_action_gateway.
#       LiveOperatorActionGateway.ingest_warden_recommendations(),
#       mirroring ingest_control_recommendations()'s own precedent.
#
#   (b) build_execution_requests(): READ-ONLY, mirrors the other three
#       adapters exactly -- reads WardenNotificationController.
#       all_requests()/history(), never calls approve()/notify() itself.
# =====================================================


_TRANSITION_TIMESTAMP_FIELDS = {
    WardenNotificationStatus.PENDING_APPROVAL: "created_at",
    WardenNotificationStatus.APPROVED: "approved_at",
    WardenNotificationStatus.DISPATCHED: "dispatched_at",
}

_TERMINAL_STATUSES = (WardenNotificationStatus.CONFIRMED, WardenNotificationStatus.FAILED)


def translate(recommendation) -> Optional[WardenNotificationRequest]:

    if recommendation.type != RecommendationType.WARDEN_DISPATCH:
        return None

    return WardenNotificationRequest(
        zone_id=recommendation.affected_zones[0] if recommendation.affected_zones else None,
        reason=recommendation.recommended_action or recommendation.explanation,
        source_recommendation_id=recommendation.recommendation_id,
        confidence=recommendation.confidence,
    )


def translate_recommendation_set(recommendation_set) -> Tuple[WardenNotificationRequest, ...]:

    if recommendation_set is None:
        return ()

    candidates = (translate(rec) for rec in recommendation_set.by_type(RecommendationType.WARDEN_DISPATCH))

    return tuple(request for request in candidates if request is not None)


def build_execution_requests(controller) -> Tuple[ExecutionRequest, ...]:

    if controller is None:
        return ()

    provider_source = type(controller.provider).__name__

    requests = []

    for warden_request in controller.all_requests():

        events = [e for e in controller.history() if e.request_id == warden_request.request_id]

        timestamps = {"created_at": None, "approved_at": None, "dispatched_at": None, "completed_at": None}
        result_message = ""
        result_confirmed = None

        for event in events:

            field_name = _TRANSITION_TIMESTAMP_FIELDS.get(event.to_status)

            if field_name is not None:
                timestamps[field_name] = event.timestamp

            if event.to_status in _TERMINAL_STATUSES:

                timestamps["completed_at"] = event.timestamp
                result_message = event.note
                result_confirmed = event.to_status == WardenNotificationStatus.CONFIRMED

        requests.append(ExecutionRequest(
            execution_request_id=warden_request.request_id,
            category=ExecutionCategory.WARDEN_NOTIFICATION,
            status=controller.status_of(warden_request.request_id).name,
            provider_source=provider_source,
            originating_recommendation_id=warden_request.source_recommendation_id,
            recommendation_id_provenance=(
                RecommendationIdProvenance.RECOMMENDATION_LAYER if warden_request.source_recommendation_id is not None
                else RecommendationIdProvenance.UNAVAILABLE
            ),
            target_description=f"Warden for zone {warden_request.zone_id}",
            result_message=result_message,
            result_confirmed=result_confirmed,
            **timestamps,
        ))

    return tuple(requests)
