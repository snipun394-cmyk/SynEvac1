from typing import Dict, Tuple

from evacuation_guidance.models import EvacuationGuidanceSnapshot

from dynamic_signage.models import DynamicSignageSnapshot, SignageInconsistency, SignageStatus


# =====================================================
# Live Dynamic Evacuation Signage milestone, Phase 21 -- an INDEPENDENT
# cross-check between Signage's own output and the SAME
# EvacuationGuidanceSnapshot it (and voice_evacuation, via
# EvacuationGuidanceSnapshot.voice_plan()) were built from. Detects,
# never "fixes" -- a mismatch is reported, Guidance is never mutated to
# make it go away (Phase 21's own explicit requirement).
#
# Deliberately reads ONLY evacuation_guidance.models (already an
# allowed dependency -- Signage consumes Guidance directly) and this
# package's own models -- never voice_evacuation/speaker_manager
# (VoiceGuidanceMessagePlan.recommended_exit_id is already exposed on
# EvacuationGuidanceSnapshot itself, so no import of voice_evacuation is
# ever needed to compare against it).
# =====================================================


def detect_inconsistencies(
    guidance_snapshot: EvacuationGuidanceSnapshot, signage_snapshot: DynamicSignageSnapshot,
) -> Dict[str, Tuple[str, ...]]:

    result: Dict[str, Tuple[str, ...]] = {}

    for sign_id, instruction in signage_snapshot.instructions.items():

        if instruction.status != SignageStatus.ACTIVE or instruction.zone_id is None:
            continue

        codes = []

        plan = guidance_snapshot.zone(instruction.zone_id)

        if plan is not None:

            if instruction.recommended_exit_id != plan.recommended_exit_id:
                codes.append(SignageInconsistency.SIGNAGE_GUIDANCE_MISMATCH)

            if instruction.guidance_revision != plan.revision:
                codes.append(SignageInconsistency.STALE_SIGNAGE_REVISION)

        voice_plan = guidance_snapshot.voice_plan(instruction.zone_id)

        if voice_plan is not None and instruction.recommended_exit_id != voice_plan.recommended_exit_id:
            codes.append(SignageInconsistency.VOICE_SIGNAGE_EXIT_MISMATCH)

        if codes:
            result[sign_id] = tuple(codes)

    return result
