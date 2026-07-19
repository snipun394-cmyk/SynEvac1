from typing import Tuple

from voice_evacuation.models import BroadcastInstruction, BroadcastStatus


class BroadcastLog:

    # An append-only store for BroadcastInstruction history (Phase 8's
    # own "this history can later be used for validation/research/
    # Command Center playback" requirement) -- mirrors facp.event_log.
    # PanelEventLog's exact shape and role. Its own class, independent
    # of VoiceEvacuationController, so a future Live/Replay adapter that
    # produces canonical BroadcastInstructions some other way can reuse
    # the same query surface without depending on
    # VoiceEvacuationController's own routing/supersession logic at all.

    def __init__(self):

        self._instructions: list = []

    # =====================================================

    def append(self, instruction: BroadcastInstruction) -> None:

        self._instructions.append(instruction)

    # =====================================================

    def all_instructions(self) -> Tuple[BroadcastInstruction, ...]:

        return tuple(self._instructions)

    # =====================================================

    def by_zone(self, zone_id: str) -> Tuple[BroadcastInstruction, ...]:

        return tuple(i for i in self._instructions if i.target_zone_id == zone_id)

    # =====================================================

    def by_status(self, status: BroadcastStatus) -> Tuple[BroadcastInstruction, ...]:

        return tuple(i for i in self._instructions if i.status == status)

    # =====================================================

    def by_message_id(self, message_id: str) -> Tuple[BroadcastInstruction, ...]:

        return tuple(
            i for i in self._instructions if i.message is not None and i.message.message_id == message_id
        )

    # =====================================================

    def between(self, start_time: float, end_time: float) -> Tuple[BroadcastInstruction, ...]:

        return tuple(
            i for i in self._instructions if start_time <= i.timestamp <= end_time
        )

    # =====================================================

    def recent(self, count: int) -> Tuple[BroadcastInstruction, ...]:

        return tuple(self._instructions[-count:]) if count > 0 else ()

    # =====================================================

    def __len__(self) -> int:

        return len(self._instructions)
