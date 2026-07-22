from typing import Optional

from navigation.edge import Edge

from evacuation_guidance.instruction_builder import asset_label
from evacuation_guidance.models import DeliveryStatus, RouteStatus, SpeakerCoverage, VoiceGuidanceMessagePlan


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone, Phase
# 13/14 -- speaker coverage lookup (delivery-planning) and the single
# combined voice sentence built deterministically from a valid route's
# own structured Door/Stair/Exit edges (message-planning). Deliberately
# TWO separate concerns, per Phase 13's own "keep route planning and
# delivery planning separate" requirement -- neither function below
# needs the other's result to run.
#
# No LLM anywhere in this module -- every sentence is assembled from
# fixed templates and real asset labels/ids only.
# =====================================================


def speaker_coverage_for_zone(zone_id: str, speaker_manager) -> SpeakerCoverage:

    # speaker_manager is accepted duck-typed (only .active_speakers_in_zone()
    # is ever called) -- this module never imports speaker_manager/,
    # mirroring emergency_response.engine's own "receive a collaborator,
    # never import its package" convention, and keeping this package's
    # own dependency surface narrow (Phase 28).

    if speaker_manager is None:
        return SpeakerCoverage(zone_id=zone_id, speaker_ids=(), delivery_status=DeliveryStatus.NO_SPEAKER_COVERAGE)

    speakers = speaker_manager.active_speakers_in_zone(zone_id)
    speaker_ids = tuple(sorted(speaker.id for speaker in speakers))

    status = DeliveryStatus.PLANNED if speaker_ids else DeliveryStatus.NO_SPEAKER_COVERAGE

    return SpeakerCoverage(zone_id=zone_id, speaker_ids=speaker_ids, delivery_status=status)


# =====================================================


def build_voice_message_text(zone_id: str, route, route_status: str) -> str:

    if route_status not in (RouteStatus.ROUTE_AVAILABLE, RouteStatus.ALREADY_AT_EXIT) or route is None:
        return f"Zone {zone_id}: No safe evacuation route is currently available."

    # Walks route.edges in their ACTUAL sequence order (never grouped by
    # type) -- a route that descends a stair before passing through a
    # door must say so in that order, never always "door, then stair"
    # regardless of which one genuinely comes first (Phase 8's own "no
    # hallucinated" discipline applies to sequence, not just wording).
    non_exit_edges = [edge for edge in route.edges if edge.edge_type != Edge.EXIT]
    exit_edges = [edge for edge in route.edges if edge.edge_type == Edge.EXIT]
    exit_label = asset_label(exit_edges[-1]) if exit_edges else "the nearest safe exit"

    _PREPOSITION = {Edge.DOOR: "through", Edge.STAIR: "toward"}

    # Clauses are built lowercase (joined into one sentence, capitalized
    # only once at the end) -- the LAST clause is always the exit itself,
    # preceded by "and"; every clause before it is simply space-joined.
    # This is what produces the milestone's own exact worked example
    # ("Proceed through Door D4 toward Stair S2 and continue to Exit E2.")
    # while still reading naturally for any other edge order/mix.
    clauses = []

    for index, edge in enumerate(non_exit_edges):

        preposition = _PREPOSITION[edge.edge_type]
        label = asset_label(edge)

        clauses.append(f"proceed {preposition} {label}" if index == 0 else f"{preposition} {label}")

    clauses.append(f"continue to {exit_label}")

    if len(clauses) == 1:
        sentence = clauses[0]
    else:
        sentence = " ".join(clauses[:-1]) + " and " + clauses[-1]

    sentence = sentence[0].upper() + sentence[1:]

    return f"Zone {zone_id}: {sentence}."


# =====================================================


def build_voice_plan(
    zone_id: str, route, route_status: str, recommended_exit_id: Optional[str], revision: int,
    speaker_coverage: SpeakerCoverage, timestamp: float,
) -> Optional[VoiceGuidanceMessagePlan]:

    # Phase 15's own "voice plan not generated from invalid guidance"
    # requirement -- a route that is not genuinely valid never produces
    # a VoiceGuidanceMessagePlan at all (None, not a fabricated one).
    if route_status not in (RouteStatus.ROUTE_AVAILABLE, RouteStatus.ALREADY_AT_EXIT):
        return None

    message_text = build_voice_message_text(zone_id, route, route_status)

    return VoiceGuidanceMessagePlan(
        zone_id=zone_id, message_text=message_text, recommended_exit_id=recommended_exit_id,
        guidance_revision=revision, speaker_ids=speaker_coverage.speaker_ids,
        delivery_status=speaker_coverage.delivery_status, route_status=route_status, timestamp=timestamp,
    )
