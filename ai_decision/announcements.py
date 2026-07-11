from typing import Optional

from hazard.severity import HazardSeverity

from ai_decision.recommendation import ZoneRecommendation


class VoiceAnnouncementTemplate:

    # The one interface AIDecisionEngine delegates to for "what should
    # the Voice Evacuation System say about this zone" -- mirrors
    # EvacuationPriorityRule's role: a future richer template (e.g.
    # localized phrasing, a text-to-speech-tuned script) plugs in
    # without touching AIDecisionEngine. Returning None means "say
    # nothing about this zone" -- a safe zone simply gets no
    # announcement rather than an empty/placeholder message.

    def announce(
        self, zone_recommendation: ZoneRecommendation, zone_name: str, exit_name: Optional[str],
    ) -> Optional[str]:

        raise NotImplementedError


class DefaultVoiceAnnouncementTemplate(VoiceAnnouncementTemplate):

    # V1's whole announcement policy: simple, deterministic f-string
    # templates, never generated text -- a documented placeholder, not
    # a validated public-address script. Only ever called for zones
    # AIDecisionEngine has already flagged unsafe; announce() itself
    # still guards on is_unsafe so it never fabricates an announcement
    # if called directly against a safe zone.

    def announce(self, zone_recommendation, zone_name, exit_name):

        if not zone_recommendation.is_unsafe:
            return None

        if not zone_recommendation.is_reachable:
            return (
                f"{zone_name}: no safe route to an exit -- "
                f"shelter in place and await rescue."
            )

        urgency = (
            "URGENT: Evacuate"
            if zone_recommendation.severity == HazardSeverity.CRITICAL
            else "Evacuate"
        )

        exit_clause = f" via {exit_name}" if exit_name else ""

        return f"{urgency} {zone_name} now{exit_clause}."
