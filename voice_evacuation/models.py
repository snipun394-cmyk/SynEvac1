from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
from uuid import uuid4


class VoiceMessageType(Enum):

    # Only the types the current Advisory System can honestly produce a
    # signal for -- see voice_evacuation/adapter.py's own docstring for
    # exactly which decision_policy.zone_policy action each one is
    # derived from. ALL_CLEAR/AVOID_ROUTE have no CivilianAnnouncement-
    # sourced signal today (no existing Advisory output distinguishes
    # "avoid this specific route" from a plain evacuate instruction, and
    # nothing currently declares an incident over) -- both remain
    # constructible for a caller that DOES know one occurred (e.g. a
    # manual test, or a future richer Advisory output), never fabricated
    # by the adapter itself.

    EVACUATE = auto()
    ROUTE_GUIDANCE = auto()
    AVOID_ROUTE = auto()
    SHELTER_IN_PLACE = auto()
    ALL_CLEAR = auto()


class VoiceMessageStatus(Enum):

    # The message's own lifecycle -- distinct from BroadcastStatus below
    # (which describes what happened when THIS controller tried to route
    # THIS message to speakers). A message can be ACTIVE (currently the
    # governing recommendation for its zone(s)), SUPERSEDED (a newer,
    # equal-or-higher-priority message replaced it for at least one of
    # its zones), or CANCELLED (explicitly withdrawn, see
    # VoiceEvacuationController.cancel()).

    ACTIVE = auto()
    SUPERSEDED = auto()
    CANCELLED = auto()


# Fixed, disclosed priority ladder -- Phase 7's own explicit requirement
# ("derive priority from existing Advisory recommendation types/status
# where possible", "document every rule"). Higher number wins. Mirrors
# the real-world FACP/PA convention this milestone's own Phase 7 example
# states directly: Shelter-in-place and full evacuation guidance outrank
# routine route guidance, which in turn outranks an all-clear notice.
# An unclassified message (message_type=None -- the adapter could not
# determine a type, see DetectorConditionReport-style "never guessed"
# convention elsewhere in this codebase) is deliberately given the SAME
# priority as ROUTE_GUIDANCE, not EVACUATE -- an unknown type must never
# be assumed to be the more urgent one.
_PRIORITY_BY_TYPE = {
    VoiceMessageType.SHELTER_IN_PLACE: 100,
    VoiceMessageType.EVACUATE: 90,
    VoiceMessageType.AVOID_ROUTE: 70,
    VoiceMessageType.ROUTE_GUIDANCE: 50,
    VoiceMessageType.ALL_CLEAR: 10,
}

_DEFAULT_PRIORITY = _PRIORITY_BY_TYPE[VoiceMessageType.ROUTE_GUIDANCE]


def priority_for_message_type(message_type: Optional[VoiceMessageType]) -> int:

    if message_type is None:
        return _DEFAULT_PRIORITY

    return _PRIORITY_BY_TYPE[message_type]


@dataclass(frozen=True)
class VoiceMessage:

    # The canonical, hardware-agnostic voice-evacuation message --
    # Phase 4's own required shape. Explicit architecture guard: NO
    # field on this type may ever identify an individual occupant. Every
    # message is addressed to one or more ZONES, never a person -- see
    # tests.test_voice_evacuation.ArchitectureGuardTests.
    # test_voice_message_cannot_carry_an_occupant_id, which fails the
    # whole suite the moment an occupant_id-shaped field is added here.
    #
    # message_text is carried through verbatim from whatever produced it
    # (advisory_system.CivilianAnnouncement.announcement today) -- this
    # type never generates, rewrites, or paraphrases wording of its own;
    # see voice_evacuation/adapter.py's own docstring.

    message_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0

    target_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    message_text: str = ""

    message_type: Optional[VoiceMessageType] = None
    priority: int = _DEFAULT_PRIORITY

    source_recommendation_id: Optional[str] = None
    confidence: Optional[float] = None

    status: VoiceMessageStatus = VoiceMessageStatus.ACTIVE


class BroadcastStatus(Enum):

    # What actually happened when VoiceEvacuationController routed one
    # VoiceMessage to one zone's speakers -- see BroadcastInstruction
    # below. BROADCAST means at least one active speaker in this zone
    # received the instruction (Simulation mode: was recorded by
    # SimulationVoiceOutputProvider -- never "was physically heard",
    # see that class's own docstring). NO_SPEAKERS_AVAILABLE is the
    # honest outcome for a zone with no active speaker at all (Phase 13's
    # "zone with no available speaker reports delivery unavailable
    # honestly" requirement) -- never silently dropped, never
    # misreported as BROADCAST. SUPERSEDED means this exact instruction
    # was not sent because a higher-or-equal-priority message already
    # owned this zone at the time. CANCELLED means an operator/caller
    # explicitly withdrew it via VoiceEvacuationController.cancel().

    BROADCAST = auto()
    NO_SPEAKERS_AVAILABLE = auto()
    SUPERSEDED = auto()
    CANCELLED = auto()


@dataclass(frozen=True)
class BroadcastInstruction:

    # One zone's worth of one VoiceMessage's routing outcome -- a
    # multi-zone VoiceMessage produces one BroadcastInstruction PER
    # zone (see VoiceEvacuationController.broadcast()), since speaker
    # routing and per-zone supersession are both inherently zone-scoped;
    # this is deliberate, not an oversight -- Phase 6's own "these must
    # be allowed to exist simultaneously because recommendations are
    # zone-specific" requirement is exactly what per-zone instructions
    # (rather than one instruction covering every target zone) makes
    # possible: Zone A's instruction is completely independent of Zone
    # B's, even when both originated from messages issued in the same
    # broadcast() call.

    instruction_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0

    message: Optional[VoiceMessage] = None
    target_zone_id: str = ""

    speaker_ids: Tuple[str, ...] = field(default_factory=tuple)

    status: BroadcastStatus = BroadcastStatus.NO_SPEAKERS_AVAILABLE
