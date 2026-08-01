from typing import Tuple

from building_control.types import RequestStatus

from execution_layer.models import ExecutionCategory, ExecutionRequest, RecommendationIdProvenance


# =====================================================
# Building Control adapter -- READ-ONLY. Reads BuildingControlController.
# all_requests()/history() (already-recorded state) and normalizes it
# into the unified ExecutionRequest vocabulary. Never calls submit()/
# approve()/reject()/execute() -- BuildingControlController remains the
# sole execution authority for this category.
# =====================================================


_TRANSITION_TIMESTAMP_FIELDS = {
    RequestStatus.PENDING_APPROVAL: "created_at",
    RequestStatus.APPROVED: "approved_at",
    RequestStatus.DISPATCHED: "dispatched_at",
}

_TERMINAL_STATUSES = (RequestStatus.CONFIRMED, RequestStatus.FAILED)


def _recommendation_id_provenance(source_recommendation_id):

    if source_recommendation_id is None:
        return RecommendationIdProvenance.UNAVAILABLE

    if source_recommendation_id.startswith("rec-"):
        # advisory_adapter._synthesize_recommendation_id()'s own
        # content-derived hash prefix -- never a real recommendation_
        # layer.Recommendation.recommendation_id.
        return RecommendationIdProvenance.ADVISORY_SYSTEM

    return RecommendationIdProvenance.UNAVAILABLE


def build_execution_requests(controller) -> Tuple[ExecutionRequest, ...]:

    if controller is None:
        return ()

    provider_source = type(controller.provider).__name__

    requests = []

    for control_request in controller.all_requests():

        events = [e for e in controller.history() if e.request_id == control_request.request_id]

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
                result_confirmed = event.to_status == RequestStatus.CONFIRMED

        requests.append(ExecutionRequest(
            execution_request_id=control_request.request_id,
            category=ExecutionCategory.BUILDING_CONTROL,
            status=controller.status_of(control_request.request_id).name,
            provider_source=provider_source,
            originating_recommendation_id=control_request.source_recommendation_id,
            recommendation_id_provenance=_recommendation_id_provenance(control_request.source_recommendation_id),
            target_description=f"{control_request.system_type.name} {control_request.target_id}",
            result_message=result_message,
            result_confirmed=result_confirmed,
            **timestamps,
        ))

    return tuple(requests)
