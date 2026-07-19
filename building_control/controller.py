from typing import Dict, List, Optional, Tuple

from building_control.history import ControlEvent
from building_control.providers import BuildingControlProvider
from building_control.requests import ControlInstruction, ControlRequest, ControlResult
from building_control.snapshot import ControlStateEntry, ControlStateSnapshot
from building_control.types import (
    TERMINAL_STATUSES,
    ApprovalMode,
    ControlAction,
    ControlSystemType,
    RequestStatus,
    VALID_ACTIONS,
)


# Non-terminal statuses a pending/in-flight request may already be in
# -- used by submit() to decide whether an identical incoming request
# is a duplicate of live work (skip it) or of something that already
# concluded negatively (allow a fresh ask, preserving the old one's
# history untouched). See types.TERMINAL_STATUSES for the converse set.
_LIVE_STATUSES = (
    RequestStatus.PENDING_APPROVAL,
    RequestStatus.APPROVED,
    RequestStatus.DISPATCHED,
    RequestStatus.CONFIRMED,
)

_EXPECTED_CATEGORY = {
    ControlSystemType.DOOR: "door",
    ControlSystemType.EXIT: "exit",
    ControlSystemType.STAIR_PRESSURIZATION: "stair",
    ControlSystemType.SMOKE_EXHAUST: "zone",
    ControlSystemType.DELUGE: "zone",
}


def _categorize_target(building, target_id: Optional[str]) -> Optional[str]:

    # No generic "get object by id" exists on Building/Floor today
    # (scenario_runner.building_initializer.find_door/find_exit only
    # cover those two categories) -- this is the small, additive
    # lookup needed for stair/zone target validation, kept local to
    # this controller rather than added to models/building.py, since
    # nothing else in the platform needs a cross-category id lookup.

    if target_id is None:
        return None

    for floor in building.floors:

        for door in floor.doors:
            if door.id == target_id:
                return "door"

        for exit_obj in floor.exits:
            if exit_obj.id == target_id:
                return "exit"

        for stair in floor.stairs:
            if stair.id == target_id:
                return "stair"

        for zone in floor.zones:
            if zone.id == target_id:
                return "zone"

    return None


class BuildingControlController(object):

    # Accepts ControlRequests, validates targets, tracks lifecycle,
    # approves/rejects/cancels, converts approved requests into
    # ControlInstructions, dispatches through a BuildingControlProvider,
    # and records provider results into an append-only history.
    #
    # Deliberately does NOT: make evacuation decisions, run AI, run RL,
    # create Advisory recommendations, or talk to real hardware --
    # nothing in this class imports ai_decision, rl_training, or any
    # protocol library (see tests/test_building_control.py's
    # architecture guards).

    def __init__(
        self,
        building,
        provider: BuildingControlProvider,
        *,
        approval_mode: ApprovalMode = ApprovalMode.REQUIRES_APPROVAL,
    ):

        if approval_mode == ApprovalMode.AUTO_APPROVE_SIMULATION and not provider.is_simulation_only:

            # The one guard preventing simulation auto-approval from
            # ever reaching a Live provider by accident (Phase 5/14).
            raise ValueError(
                "ApprovalMode.AUTO_APPROVE_SIMULATION requires a provider with "
                "is_simulation_only = True; refusing to pair it with a non-simulation provider."
            )

        self._building = building
        self._provider = provider
        self._approval_mode = approval_mode

        self._requests: Dict[str, ControlRequest] = {}
        self._status: Dict[str, RequestStatus] = {}
        self._history: List[ControlEvent] = []

    # =====================================================

    def submit(self, request: ControlRequest) -> ControlRequest:

        duplicate = self._find_live_duplicate(request)
        if duplicate is not None:
            return duplicate

        self._requests[request.request_id] = request

        validation_failure = self._validate_target(request)

        if validation_failure is not None:

            self._set_status(request.request_id, RequestStatus.REJECTED, actor="system", note=validation_failure)
            return request

        self._set_status(request.request_id, RequestStatus.PENDING_APPROVAL, actor="system", note="submitted")

        if self._approval_mode == ApprovalMode.AUTO_APPROVE_SIMULATION:
            self.approve(request.request_id, actor="auto_approve_simulation")

        return request

    # =====================================================

    def approve(self, request_id: str, actor: str = "operator", note: str = "") -> ControlRequest:

        request = self._require_request(request_id)

        if self._status[request_id] != RequestStatus.PENDING_APPROVAL:

            raise ValueError(
                f"cannot approve request {request_id!r}: status is "
                f"{self._status[request_id].name}, not PENDING_APPROVAL"
            )

        self._set_status(request_id, RequestStatus.APPROVED, actor=actor, note=note)
        self._dispatch(request)

        return request

    # =====================================================

    def reject(self, request_id: str, actor: str = "operator", note: str = "") -> ControlRequest:

        request = self._require_request(request_id)

        if self._status[request_id] != RequestStatus.PENDING_APPROVAL:

            raise ValueError(
                f"cannot reject request {request_id!r}: status is "
                f"{self._status[request_id].name}, not PENDING_APPROVAL"
            )

        self._set_status(request_id, RequestStatus.REJECTED, actor=actor, note=note)

        return request

    # =====================================================

    def cancel(self, request_id: str, actor: str = "operator", note: str = "") -> ControlRequest:

        request = self._require_request(request_id)
        current = self._status[request_id]

        if current in TERMINAL_STATUSES or current == RequestStatus.CONFIRMED:

            raise ValueError(
                f"cannot cancel request {request_id!r}: status is {current.name}, already final"
            )

        self._set_status(request_id, RequestStatus.CANCELLED, actor=actor, note=note)

        return request

    # =====================================================

    def _dispatch(self, request: ControlRequest) -> None:

        self._set_status(request.request_id, RequestStatus.DISPATCHED, actor="system", note="dispatching to provider")

        instruction = ControlInstruction(
            request_id=request.request_id,
            system_type=request.system_type,
            target_id=request.target_id,
            action=request.requested_action,
        )

        result: ControlResult = self._provider.execute(instruction)

        if result.confirmed:
            self._set_status(request.request_id, RequestStatus.CONFIRMED, actor="provider", note=result.message)
        else:
            self._set_status(request.request_id, RequestStatus.FAILED, actor="provider", note=result.message)

    # =====================================================

    def _validate_target(self, request: ControlRequest) -> Optional[str]:

        if request.requested_action not in VALID_ACTIONS[request.system_type]:

            return (
                f"{request.requested_action.name} is not a valid action for "
                f"{request.system_type.name}"
            )

        actual_category = _categorize_target(self._building, request.target_id)
        expected_category = _EXPECTED_CATEGORY[request.system_type]

        if actual_category is None:
            return f"target {request.target_id!r} does not exist"

        if actual_category != expected_category:

            return (
                f"target {request.target_id!r} is a {actual_category}, "
                f"not a {expected_category} (required for {request.system_type.name})"
            )

        return None

    # =====================================================

    def _find_live_duplicate(self, request: ControlRequest) -> Optional[ControlRequest]:

        for existing in self._requests.values():

            if (
                existing.system_type == request.system_type
                and existing.target_id == request.target_id
                and existing.requested_action == request.requested_action
                and self._status[existing.request_id] in _LIVE_STATUSES
            ):
                return existing

        return None

    # =====================================================

    def _set_status(self, request_id: str, status: RequestStatus, *, actor: str, note: str = "") -> None:

        previous = self._status.get(request_id)
        self._status[request_id] = status

        self._history.append(
            ControlEvent(
                request_id=request_id, from_status=previous, to_status=status, actor=actor, note=note,
            )
        )

    # =====================================================

    def _require_request(self, request_id: str) -> ControlRequest:

        request = self._requests.get(request_id)

        if request is None:
            raise KeyError(f"no such control request: {request_id!r}")

        return request

    # =====================================================
    # Read-only views
    # =====================================================

    def status_of(self, request_id: str) -> RequestStatus:

        return self._status[self._require_request(request_id).request_id]

    def get_request(self, request_id: str) -> ControlRequest:

        return self._require_request(request_id)

    def pending_requests(self) -> Tuple[ControlRequest, ...]:

        return tuple(
            sorted(
                (r for r in self._requests.values() if self._status[r.request_id] == RequestStatus.PENDING_APPROVAL),
                key=lambda r: (r.timestamp, r.request_id),
            )
        )

    def all_requests(self) -> Tuple[ControlRequest, ...]:

        return tuple(sorted(self._requests.values(), key=lambda r: (r.timestamp, r.request_id)))

    def history(self) -> Tuple[ControlEvent, ...]:

        return tuple(self._history)

    # =====================================================

    def snapshot(self) -> ControlStateSnapshot:

        # Only CONFIRMED requests ever contribute an entry -- pending/
        # approved/dispatched/rejected/cancelled/failed requests are
        # never reported as current control state (Phase 9/11: "Do not
        # report CONFIRMED before provider response").

        latest_confirmed: Dict[Tuple[ControlSystemType, Optional[str]], ControlRequest] = {}

        for request in self.all_requests():

            if self._status[request.request_id] != RequestStatus.CONFIRMED:
                continue

            key = (request.system_type, request.target_id)
            existing = latest_confirmed.get(key)

            if existing is None or request.timestamp >= existing.timestamp:
                latest_confirmed[key] = request

        entries = tuple(
            ControlStateEntry(
                system_type=request.system_type, target_id=request.target_id,
                action=request.requested_action, confirmed_at=request.timestamp,
            )
            for request in latest_confirmed.values()
        )

        pending_count = len(self.pending_requests())

        return ControlStateSnapshot(entries=entries, pending_count=pending_count)
