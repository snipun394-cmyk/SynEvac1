from typing import Tuple

from dynamic_signage.controller import SignageRequestStatus

from execution_layer.models import ExecutionCategory, ExecutionRequest, RecommendationIdProvenance


# =====================================================
# Dynamic Signage adapter -- READ-ONLY. Reads DynamicSignageController.
# all_instructions()/history() (already-recorded state) and normalizes
# it into the unified ExecutionRequest vocabulary. Never calls submit()/
# approve()/reject()/apply() -- DynamicSignageController remains the
# sole execution authority for this category.
# =====================================================


_TRANSITION_TIMESTAMP_FIELDS = {
    SignageRequestStatus.PENDING_APPROVAL: "created_at",
    SignageRequestStatus.APPROVED: "approved_at",
    SignageRequestStatus.DISPATCHED: "dispatched_at",
}

_TERMINAL_STATUSES = (SignageRequestStatus.CONFIRMED, SignageRequestStatus.FAILED)


def build_execution_requests(controller) -> Tuple[ExecutionRequest, ...]:

    if controller is None:
        return ()

    provider_source = type(controller.provider).__name__

    requests = []

    for instruction in controller.all_instructions():

        key = controller.request_key(instruction)
        events = [e for e in controller.history() if e.request_key == key]

        timestamps = {"created_at": None, "approved_at": None, "dispatched_at": None, "completed_at": None}
        result_message = ""
        result_confirmed = None

        for event in events:

            field_name = _TRANSITION_TIMESTAMP_FIELDS.get(event.to_status)

            if field_name is not None:
                timestamps[field_name] = event.time

            if event.to_status in _TERMINAL_STATUSES:

                timestamps["completed_at"] = event.time
                result_message = event.note
                result_confirmed = event.to_status == SignageRequestStatus.CONFIRMED

        requests.append(ExecutionRequest(
            execution_request_id=key,
            category=ExecutionCategory.DYNAMIC_SIGNAGE,
            status=controller.status_of(instruction.sign_id, instruction.signage_revision),
            provider_source=provider_source,
            # SignageInstruction carries no recommendation traceability
            # field at all today (confirmed: no source_recommendation_id
            # on dynamic_signage.models.SignageInstruction) -- always
            # honestly UNAVAILABLE, never fabricated.
            originating_recommendation_id=None,
            recommendation_id_provenance=RecommendationIdProvenance.UNAVAILABLE,
            target_description=f"Sign {instruction.sign_id} (zone {instruction.zone_id})",
            result_message=result_message,
            result_confirmed=result_confirmed,
            **timestamps,
        ))

    return tuple(requests)
