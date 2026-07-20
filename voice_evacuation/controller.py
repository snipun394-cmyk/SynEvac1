from dataclasses import replace
from typing import Dict, Optional, Tuple

from voice_evacuation.broadcast_log import BroadcastLog
from voice_evacuation.models import BroadcastInstruction, BroadcastStatus, VoiceMessage
from voice_evacuation.provider import VoiceOutputProvider


class VoiceEvacuationController:

    # Routes canonical VoiceMessages to Speaker assets and records what
    # happened -- Phase 6's own required responsibility, and nothing
    # more. This class NEVER generates a recommendation, an evacuation
    # decision, or message text of its own; every VoiceMessage it routes
    # was already fully formed by voice_evacuation.adapter (or a caller
    # constructing one directly) before it ever reaches broadcast().
    # Consumes speaker identity/status exclusively through a
    # speaker_manager.manager.SpeakerManager instance (constructor-
    # injected, never constructed here) -- this class does not discover,
    # register, enable/disable, or otherwise own Speaker assets itself.
    #
    # ===================================================================
    # Priority / supersession rules (Phase 7) -- deterministic, and the
    # only place this logic lives:
    # ===================================================================
    #
    # Each zone tracks its own single "active message" independently
    # (self._active_message_by_zone) -- messages for different zones
    # never interact, so "Zone A: Use Exit 3" and "Zone B: Avoid Exit 3,
    # Use Exit 5" coexist exactly as Phase 6 requires, with no risk of
    # being merged into one contradictory building-wide message.
    #
    # For a given zone:
    #   - No existing active message -> the new message always wins.
    #   - New message.priority >= existing.priority -> the new message
    #     wins; the existing one is retroactively recorded SUPERSEDED
    #     (via its own synthetic BroadcastInstruction -- VoiceMessage is
    #     immutable and is never mutated in place, the same "operational
    #     state tracked at the coordinator level, not on the frozen
    #     event/message object" discipline facp.engine.SimulatedFACP
    #     already established for PanelEvent/PanelState).
    #   - New message.priority < existing.priority -> the new message
    #     itself is recorded SUPERSEDED (it never becomes active, the
    #     existing higher-priority message keeps the zone).
    #
    # Priority itself is never invented here -- see voice_evacuation.
    # models.priority_for_message_type()'s own fixed, documented ladder,
    # generally derived from the same decision_policy.zone_policy action
    # every CivilianAnnouncement was already built from.

    def __init__(self, speaker_manager, output_provider: VoiceOutputProvider):

        self._speaker_manager = speaker_manager
        self._output_provider = output_provider

        self._active_message_by_zone: Dict[str, VoiceMessage] = {}
        self._broadcast_log = BroadcastLog()

    # =====================================================

    @property
    def broadcast_log(self) -> BroadcastLog:

        return self._broadcast_log

    # =====================================================

    @property
    def provider(self) -> VoiceOutputProvider:

        # Read-only visibility onto the injected output provider --
        # added so a caller (command_center.live_operator_action_
        # gateway.LiveOperatorActionGateway) can honestly report
        # provider capability (is_simulation_only) without this
        # controller needing to expose that judgement itself.
        return self._output_provider

    # =====================================================

    def active_message_for_zone(self, zone_id: str) -> Optional[VoiceMessage]:

        return self._active_message_by_zone.get(zone_id)

    # =====================================================

    def active_messages(self) -> Dict[str, VoiceMessage]:

        # A shallow snapshot of every zone's current active message --
        # e.g. for a Command Center panel to display "what is currently
        # being said, where" without reaching into this controller's
        # own private state. A plain dict copy, not a live view: later
        # broadcast()/cancel() calls never retroactively change a
        # snapshot already handed out.

        return dict(self._active_message_by_zone)

    # =====================================================

    def broadcast(self, message: VoiceMessage, time: float) -> Tuple[BroadcastInstruction, ...]:

        # Deterministic ordering -- target_zone_ids is iterated in
        # sorted order regardless of the tuple's own construction order,
        # so the same VoiceMessage always produces the same instruction
        # sequence (Phase 13's own "deterministic broadcast ordering"
        # requirement).

        return tuple(
            self._route_to_zone(message, zone_id, time)
            for zone_id in sorted(message.target_zone_ids)
        )

    # =====================================================

    def broadcast_building_wide(self, message: VoiceMessage, time: float) -> Tuple[BroadcastInstruction, ...]:

        # Expands message onto every zone at least one currently-known
        # Speaker asset is assigned to -- "every zone with delivery
        # capability," not "every Zone in the Building Model" (this
        # controller has no reference to Building/Zone at all, only to
        # SpeakerManager -- claiming coverage of a zone with no speaker
        # hardware in it would misrepresent what a building-wide
        # broadcast can actually reach).

        all_zone_ids = sorted({
            zone_id
            for speaker in self._speaker_manager.all_speakers()
            for zone_id in speaker.zone_ids
        })

        return self.broadcast(replace(message, target_zone_ids=tuple(all_zone_ids)), time)

    # =====================================================

    def cancel(self, zone_id: str, time: float) -> Optional[BroadcastInstruction]:

        message = self._active_message_by_zone.pop(zone_id, None)

        if message is None:
            return None

        instruction = BroadcastInstruction(
            timestamp=time, message=message, target_zone_id=zone_id,
            speaker_ids=(), status=BroadcastStatus.CANCELLED,
        )

        return self._dispatch(instruction)

    # =====================================================
    # Internals
    # =====================================================

    def _route_to_zone(self, message: VoiceMessage, zone_id: str, time: float) -> BroadcastInstruction:

        existing = self._active_message_by_zone.get(zone_id)

        if existing is not None and existing.message_id == message.message_id:
            # Re-broadcasting the exact same message (e.g. a caller
            # re-running the same tick's report) is a no-op re-affirmation,
            # not a supersession of itself.
            existing = None

        if existing is not None and existing.priority > message.priority:

            instruction = BroadcastInstruction(
                timestamp=time, message=message, target_zone_id=zone_id,
                speaker_ids=(), status=BroadcastStatus.SUPERSEDED,
            )
            return self._dispatch(instruction)

        if existing is not None:

            superseded_instruction = BroadcastInstruction(
                timestamp=time, message=existing, target_zone_id=zone_id,
                speaker_ids=(), status=BroadcastStatus.SUPERSEDED,
            )
            self._dispatch(superseded_instruction)

        self._active_message_by_zone[zone_id] = message

        speakers = self._speaker_manager.active_speakers_in_zone(zone_id)
        speaker_ids = tuple(sorted(speaker.id for speaker in speakers))

        status = BroadcastStatus.BROADCAST if speaker_ids else BroadcastStatus.NO_SPEAKERS_AVAILABLE

        instruction = BroadcastInstruction(
            timestamp=time, message=message, target_zone_id=zone_id,
            speaker_ids=speaker_ids, status=status,
        )
        return self._dispatch(instruction)

    # =====================================================

    def _dispatch(self, instruction: BroadcastInstruction) -> BroadcastInstruction:

        result = self._output_provider.send(instruction)
        self._broadcast_log.append(result)

        return result
