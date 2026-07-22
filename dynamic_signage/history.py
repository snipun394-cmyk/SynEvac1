from typing import Tuple

from dynamic_signage.models import SignageInstruction


class SignageHistory:

    # An append-only store for every SignageInstruction the Controller
    # ever submitted -- mirrors voice_evacuation.broadcast_log.
    # BroadcastLog's exact shape and role (Phase 16's own "previously
    # applied signage retained in history, never rewritten"
    # requirement). Independent of DynamicSignageController itself, so
    # a future Live/Replay adapter reusing the same query surface never
    # needs to depend on the Controller's own approval/supersession
    # logic.

    def __init__(self):

        self._instructions: list = []

    # =====================================================

    def append(self, instruction: SignageInstruction) -> None:

        self._instructions.append(instruction)

    # =====================================================

    def all_instructions(self) -> Tuple[SignageInstruction, ...]:

        return tuple(self._instructions)

    # =====================================================

    def by_sign(self, sign_id: str) -> Tuple[SignageInstruction, ...]:

        return tuple(i for i in self._instructions if i.sign_id == sign_id)

    # =====================================================

    def by_status(self, status: str) -> Tuple[SignageInstruction, ...]:

        return tuple(i for i in self._instructions if i.status == status)

    # =====================================================

    def between(self, start_time: float, end_time: float) -> Tuple[SignageInstruction, ...]:

        return tuple(i for i in self._instructions if start_time <= i.timestamp <= end_time)

    # =====================================================

    def recent(self, count: int) -> Tuple[SignageInstruction, ...]:

        return tuple(self._instructions[-count:]) if count > 0 else ()

    # =====================================================

    def __len__(self) -> int:

        return len(self._instructions)
