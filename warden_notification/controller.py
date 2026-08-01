from typing import Dict, List, Optional, Tuple

from warden_notification.history import WardenNotificationEvent
from warden_notification.provider import WardenNotificationProvider
from warden_notification.requests import WardenNotificationInstruction, WardenNotificationRequest, WardenNotificationResult
from warden_notification.types import TERMINAL_STATUSES, WardenNotificationStatus


# Non-terminal statuses a pending/in-flight request may already be in --
# mirrors building_control.controller._LIVE_STATUSES exactly.
_LIVE_STATUSES = (
    WardenNotificationStatus.PENDING_APPROVAL,
    WardenNotificationStatus.APPROVED,
    WardenNotificationStatus.DISPATCHED,
    WardenNotificationStatus.CONFIRMED,
)


class WardenNotificationController(object):

    # Accepts WardenNotificationRequests, tracks lifecycle, approves/
    # rejects/cancels, converts approved requests into
    # WardenNotificationInstructions, dispatches through a
    # WardenNotificationProvider, and records provider results into an
    # append-only history. Mirrors building_control.controller.
    # BuildingControlController exactly, minus target-category
    # validation (a warden notification targets a zone_id -- no door/
    # exit/stair category check is meaningful here).
    #
    # Deliberately does NOT: make evacuation decisions, run AI, run RL,
    # create recommendations, or talk to a real notification transport
    # -- nothing in this class imports ai_decision, rl_training, or any
    # protocol/transport library.

    def __init__(self, provider: WardenNotificationProvider):

        self._provider = provider

        self._requests: Dict[str, WardenNotificationRequest] = {}
        self._status: Dict[str, WardenNotificationStatus] = {}
        self._history: List[WardenNotificationEvent] = []

    # =====================================================

    def submit(self, request: WardenNotificationRequest) -> WardenNotificationRequest:

        duplicate = self._find_live_duplicate(request)
        if duplicate is not None:
            return duplicate

        self._requests[request.request_id] = request
        self._set_status(request.request_id, WardenNotificationStatus.PENDING_APPROVAL, actor="system", note="submitted")

        return request

    # =====================================================

    def approve(self, request_id: str, actor: str = "operator", note: str = "") -> WardenNotificationRequest:

        request = self._require_request(request_id)

        if self._status[request_id] != WardenNotificationStatus.PENDING_APPROVAL:

            raise ValueError(
                f"cannot approve request {request_id!r}: status is "
                f"{self._status[request_id].name}, not PENDING_APPROVAL"
            )

        self._set_status(request_id, WardenNotificationStatus.APPROVED, actor=actor, note=note)
        self._dispatch(request)

        return request

    # =====================================================

    def reject(self, request_id: str, actor: str = "operator", note: str = "") -> WardenNotificationRequest:

        request = self._require_request(request_id)

        if self._status[request_id] != WardenNotificationStatus.PENDING_APPROVAL:

            raise ValueError(
                f"cannot reject request {request_id!r}: status is "
                f"{self._status[request_id].name}, not PENDING_APPROVAL"
            )

        self._set_status(request_id, WardenNotificationStatus.REJECTED, actor=actor, note=note)

        return request

    # =====================================================

    def cancel(self, request_id: str, actor: str = "operator", note: str = "") -> WardenNotificationRequest:

        request = self._require_request(request_id)
        current = self._status[request_id]

        if current in TERMINAL_STATUSES or current == WardenNotificationStatus.CONFIRMED:

            raise ValueError(
                f"cannot cancel request {request_id!r}: status is {current.name}, already final"
            )

        self._set_status(request_id, WardenNotificationStatus.CANCELLED, actor=actor, note=note)

        return request

    # =====================================================

    def _dispatch(self, request: WardenNotificationRequest) -> None:

        self._set_status(request.request_id, WardenNotificationStatus.DISPATCHED, actor="system", note="dispatching to provider")

        instruction = WardenNotificationInstruction(
            request_id=request.request_id, zone_id=request.zone_id, reason=request.reason,
        )

        result: WardenNotificationResult = self._provider.notify(instruction)

        if result.confirmed:
            self._set_status(request.request_id, WardenNotificationStatus.CONFIRMED, actor="provider", note=result.message)
        else:
            self._set_status(request.request_id, WardenNotificationStatus.FAILED, actor="provider", note=result.message)

    # =====================================================

    def _find_live_duplicate(self, request: WardenNotificationRequest) -> Optional[WardenNotificationRequest]:

        for existing in self._requests.values():

            if (
                existing.zone_id == request.zone_id
                and existing.source_recommendation_id == request.source_recommendation_id
                and self._status[existing.request_id] in _LIVE_STATUSES
            ):
                return existing

        return None

    # =====================================================

    def _set_status(self, request_id: str, status: WardenNotificationStatus, *, actor: str, note: str = "") -> None:

        previous = self._status.get(request_id)
        self._status[request_id] = status

        self._history.append(
            WardenNotificationEvent(
                request_id=request_id, from_status=previous, to_status=status, actor=actor, note=note,
            )
        )

    # =====================================================

    def _require_request(self, request_id: str) -> WardenNotificationRequest:

        request = self._requests.get(request_id)

        if request is None:
            raise KeyError(f"no such warden notification request: {request_id!r}")

        return request

    # =====================================================
    # Read-only views
    # =====================================================

    @property
    def provider(self) -> WardenNotificationProvider:

        # Read-only visibility onto the injected provider -- mirrors
        # BuildingControlController.provider's own reasoning: a caller
        # (command_center.live_operator_action_gateway.
        # LiveOperatorActionGateway) can honestly report provider
        # capability without this controller exposing that judgement
        # itself.
        return self._provider

    # =====================================================

    def status_of(self, request_id: str) -> WardenNotificationStatus:

        return self._status[self._require_request(request_id).request_id]

    def get_request(self, request_id: str) -> WardenNotificationRequest:

        return self._require_request(request_id)

    def pending_requests(self) -> Tuple[WardenNotificationRequest, ...]:

        return tuple(
            sorted(
                (r for r in self._requests.values() if self._status[r.request_id] == WardenNotificationStatus.PENDING_APPROVAL),
                key=lambda r: (r.timestamp, r.request_id),
            )
        )

    def all_requests(self) -> Tuple[WardenNotificationRequest, ...]:

        return tuple(sorted(self._requests.values(), key=lambda r: (r.timestamp, r.request_id)))

    def history(self) -> Tuple[WardenNotificationEvent, ...]:

        return tuple(self._history)
