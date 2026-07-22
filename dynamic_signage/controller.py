from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from dynamic_signage.models import SignageInstruction
from dynamic_signage.provider import DynamicSignageProvider


class SignageRequestStatus:

    # The smallest honest state machine covering this sign instruction's
    # own approval lifecycle -- mirrors building_control.types.
    # RequestStatus's own vocabulary exactly, restated independently
    # here (this package must not import building_control -- see
    # tests/test_dynamic_signage_architecture_guards.py).

    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    DISPATCHED = "DISPATCHED"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"


_TERMINAL_STATUSES = (
    SignageRequestStatus.REJECTED, SignageRequestStatus.SUPERSEDED,
    SignageRequestStatus.CONFIRMED, SignageRequestStatus.FAILED,
)


@dataclass(frozen=True)
class SignageControlEvent:

    # One append-only audit-trail entry -- mirrors building_control.
    # history.ControlEvent's own shape.

    request_key: str
    from_status: Optional[str]
    to_status: str
    actor: str
    time: float
    note: str = ""


class DynamicSignageController:

    # Phase 13 -- analogous in philosophy to VoiceEvacuationController/
    # BuildingControlController: accepts planned SignageInstructions,
    # tracks a pending-approval lifecycle per sign, approves/rejects,
    # supersedes a stale pending instruction the moment a NEWER one is
    # submitted for the same sign, dispatches an APPROVED instruction to
    # a DynamicSignageProvider, and records everything in an append-only
    # history. This class NEVER computes a direction, NEVER decides
    # which exit to recommend, and NEVER dispatches without an explicit
    # approve() call (Phase 17/27 -- mechanically enforced, see
    # tests/test_dynamic_signage_architecture_guards.py's AI/Advisory/
    # FACP/Building-Control "cannot dispatch" guards).
    #
    # Requests are keyed by "{sign_id}::{signage_revision}" -- Phase 12's
    # own signage_revision already IS the deterministic content identity
    # for a sign's instruction, the same "keyed on revision, never on
    # message text" discipline command_center.live_operator_action_
    # gateway.LiveOperatorActionGateway._guidance_key() already
    # establishes for guidance messages.

    def __init__(self, provider: DynamicSignageProvider):

        self._provider = provider

        self._instructions: Dict[str, SignageInstruction] = {}
        self._status: Dict[str, str] = {}
        self._current_key_by_sign: Dict[str, str] = {}
        self._history: List[SignageControlEvent] = []

    # =====================================================

    @property
    def provider(self) -> DynamicSignageProvider:

        return self._provider

    # =====================================================

    @staticmethod
    def request_key(instruction: SignageInstruction) -> str:

        return f"{instruction.sign_id}::{instruction.signage_revision}"

    # =====================================================

    def submit(self, instruction: SignageInstruction, time: float) -> SignageInstruction:

        key = self.request_key(instruction)

        if key in self._status:
            # Duplicate submission of the identical sign+revision (e.g.
            # the same planning cycle re-ingested) -- never resubmitted,
            # never re-superseded.
            return instruction

        previous_key = self._current_key_by_sign.get(instruction.sign_id)

        if (
            previous_key is not None
            and previous_key != key
            and self._status.get(previous_key) == SignageRequestStatus.PENDING_APPROVAL
        ):
            # Phase 16 -- a NEWER instruction for this sign supersedes an
            # old one an operator never got to (or chose not to) approve
            # yet. An already-applied (CONFIRMED/DISPATCHED) instruction
            # is left exactly as it is in history -- never rewritten,
            # never retroactively marked superseded.
            self._set_status(previous_key, SignageRequestStatus.SUPERSEDED, actor="system", note="superseded by a newer signage revision", time=time)

        self._instructions[key] = instruction
        self._current_key_by_sign[instruction.sign_id] = key
        self._set_status(key, SignageRequestStatus.PENDING_APPROVAL, actor="system", note="submitted", time=time)

        return instruction

    # =====================================================

    def approve(self, sign_id: str, signage_revision: int, time: float, actor: str = "operator") -> SignageInstruction:

        key = f"{sign_id}::{signage_revision}"
        instruction = self._require(key)

        if self._status[key] != SignageRequestStatus.PENDING_APPROVAL:

            raise ValueError(
                f"cannot approve signage instruction {key!r}: status is {self._status[key]}, not PENDING_APPROVAL"
            )

        self._set_status(key, SignageRequestStatus.APPROVED, actor=actor, note="", time=time)
        self._dispatch(key, instruction, time)

        return instruction

    # =====================================================

    def reject(self, sign_id: str, signage_revision: int, time: float, actor: str = "operator") -> SignageInstruction:

        key = f"{sign_id}::{signage_revision}"
        instruction = self._require(key)

        if self._status[key] != SignageRequestStatus.PENDING_APPROVAL:

            raise ValueError(
                f"cannot reject signage instruction {key!r}: status is {self._status[key]}, not PENDING_APPROVAL"
            )

        self._set_status(key, SignageRequestStatus.REJECTED, actor=actor, note="", time=time)

        return instruction

    # =====================================================

    def _dispatch(self, key: str, instruction: SignageInstruction, time: float) -> None:

        self._set_status(key, SignageRequestStatus.DISPATCHED, actor="system", note="dispatching to provider", time=time)

        result = self._provider.apply(instruction)

        if result.confirmed:
            self._set_status(key, SignageRequestStatus.CONFIRMED, actor="provider", note=result.message, time=time)
        else:
            self._set_status(key, SignageRequestStatus.FAILED, actor="provider", note=result.message, time=time)

    # =====================================================
    # Read-only views
    # =====================================================

    def status_of(self, sign_id: str, signage_revision: int) -> str:

        key = f"{sign_id}::{signage_revision}"
        return self._status[self._require_key(key)]

    # =====================================================

    def current_instruction_for_sign(self, sign_id: str) -> Optional[SignageInstruction]:

        key = self._current_key_by_sign.get(sign_id)

        if key is None:
            return None

        return self._instructions.get(key)

    # =====================================================

    def current_status_for_sign(self, sign_id: str) -> Optional[str]:

        key = self._current_key_by_sign.get(sign_id)

        if key is None:
            return None

        return self._status.get(key)

    # =====================================================

    def pending_instructions(self) -> Tuple[SignageInstruction, ...]:

        return tuple(
            sorted(
                (
                    instruction for key, instruction in self._instructions.items()
                    if self._status[key] == SignageRequestStatus.PENDING_APPROVAL
                ),
                key=lambda i: (i.sign_id, i.signage_revision),
            )
        )

    # =====================================================

    def all_instructions(self) -> Tuple[SignageInstruction, ...]:

        return tuple(
            sorted(self._instructions.values(), key=lambda i: (i.timestamp, i.sign_id, i.signage_revision))
        )

    # =====================================================

    def history(self) -> Tuple[SignageControlEvent, ...]:

        return tuple(self._history)

    # =====================================================
    # Internals
    # =====================================================

    def _set_status(self, key: str, status: str, *, actor: str, note: str, time: float) -> None:

        previous = self._status.get(key)
        self._status[key] = status

        self._history.append(
            SignageControlEvent(request_key=key, from_status=previous, to_status=status, actor=actor, time=time, note=note)
        )

    # =====================================================

    def _require(self, key: str) -> SignageInstruction:

        instruction = self._instructions.get(key)

        if instruction is None:
            raise KeyError(f"no such signage instruction: {key!r}")

        return instruction

    # =====================================================

    def _require_key(self, key: str) -> str:

        if key not in self._status:
            raise KeyError(f"no such signage instruction: {key!r}")

        return key
