from typing import Dict, List, Optional, Tuple

from behaviour_profile_resolver.category import OccupantCategory, occupant_category

from decision_policy.exit_policy import CLOSE, KEEP_OPEN, OPEN
from decision_policy.stair_policy import AVOID, USE
from decision_policy.zone_policy import (
    CRITICAL_RISK_THRESHOLD, ELEVATED_RISK_THRESHOLD, EVACUATE_IMMEDIATELY, SHELTER_IN_PLACE, WAIT,
)
from decision_policy.rescue_policy import CRITICAL_THRESHOLD, MODERATE_THRESHOLD

from advisory_system.confidence_engine import (
    combine_confidence, occupancy_confidence, recommendation_confidence,
)
from advisory_system.explanation_engine import (
    compute_redirect_target, estimate_rset_improvement, explain_zone_recommendation,
)
from advisory_system.recommendation_models import (
    AdvisoryInputs, BuildingRecommendation, CivilianAnnouncement,
    FirefighterIntelligenceReport, IncidentCommanderDashboard,
)


# =====================================================
# Advisory Engine -- the four per-audience builders. Every function
# below reads only AdvisoryInputs' own already-computed fields
# (GroundTruth, DecisionPolicy, HumanObservation, AI Prediction/
# ConfidenceReport, an already-decided RL action) -- none of them fits
# a model, runs a simulation, or steps an environment. See each
# function's own docstring for the specific "never" this module holds
# to for its audience.
# =====================================================


def _ordered_zones(building):

    return [zone for floor in building.ordered_floors() for zone in floor.zones]


def _zone_name_lookup(building) -> Dict[str, str]:

    return {zone.id: (zone.name or zone.id) for floor in building.ordered_floors() for zone in floor.zones}


def _exit_name_lookup(building) -> Dict[str, str]:

    return {
        exit_obj.id: (exit_obj.name or exit_obj.id)
        for floor in building.ordered_floors() for exit_obj in floor.exits
    }


def _stair_name_lookup(building) -> Dict[str, str]:

    return {
        stair.id: (stair.name or stair.id)
        for floor in building.ordered_floors() for stair in floor.stairs
    }


def _floor_id_for_zone(building, zone_id: str) -> Optional[str]:

    for floor in building.ordered_floors():
        for zone in floor.zones:
            if zone.id == zone_id:
                return floor.id

    return None


def _assembly_points_for_floor(building, floor_id: Optional[str]) -> List:

    # Assembly points are floor-scoped (models.floor.Floor.assembly_points),
    # not zone-scoped -- there is no per-zone "your assigned assembly
    # point" field anywhere in the Building model, so this reports every
    # assembly point already designated for the zone's own floor, never
    # a guessed single one. An empty list (a floor with none configured)
    # simply means no assembly-point line is added to that zone's
    # announcement -- see build_civilian_announcements() below.

    if floor_id is None:
        return []

    for floor in building.ordered_floors():
        if floor.id == floor_id:
            return list(floor.assembly_points)

    return []


def _occupant_counts_by_zone(scenario, zones) -> Dict[str, int]:

    counts = {zone.id: 0 for zone in zones}

    for occupant in scenario.occupants:
        if occupant.zone_id in counts:
            counts[occupant.zone_id] += 1

    return counts


def _exit_status_by_id(decision_policy) -> Dict[str, str]:

    return {entry["exit_id"]: entry["status"] for entry in decision_policy.exit_decisions}


def _stair_status_by_id(decision_policy) -> Dict[str, str]:

    return {entry["stair_id"]: entry["status"] for entry in decision_policy.stair_decisions}


def _risk_by_zone(ground_truth) -> Dict[str, Optional[float]]:

    return {entry["zone_id"]: entry["risk_score"] for entry in ground_truth.zone_risk_scores}


def _ai_signal_for(inputs: AdvisoryInputs, location_id: Optional[str]) -> Tuple[Optional[float], Optional[bool]]:

    # Cross-references an already-computed AI Inference prediction
    # against this exact location, the same "flagged/agreement" shape
    # ai_inference.recommendation.decision_policy_flagged() already
    # establishes -- never re-derives a location prediction of its own.
    # AdvisoryInputs.ai_predictions is typed Any (a Prediction-like
    # object exposing .value/.probability, or a plain dict), so both
    # shapes are read defensively via getattr/dict.get.

    if location_id is None or not inputs.ai_predictions:
        return None, None

    for prediction_type, prediction in inputs.ai_predictions.items():

        value = getattr(prediction, "value", None)
        if value is None and isinstance(prediction, dict):
            value = prediction.get("value")

        if value != location_id:
            continue

        confidence = inputs.ai_confidence.get(prediction_type)
        if confidence is None:
            confidence = getattr(prediction, "probability", None)

        return confidence, True

    return None, None


def _confidence_source(
    ai_confidence: Optional[float], rl_confidence: Optional[float], crowd_confidence: Optional[float] = None,
    progress_confidence: Optional[float] = None, response_confidence: Optional[float] = None,
) -> Tuple[str, ...]:

    # What confidence_engine.recommendation_confidence() actually blended
    # in for THIS recommendation, beyond the deterministic rule-based
    # floor every recommendation always carries -- "ai"/"rl"/"crowd"
    # appear only when a genuine, non-None signal was supplied, never
    # guessed. This is the one place command_center.recommendation_center
    # is allowed to trust for whether it may say "AI confidence"/"RL
    # influence"/"Crowd Intelligence" -- see CivilianAnnouncement.
    # confidence_source's own docstring. "crowd" is deterministic
    # analytics, never AI -- see docs/architecture/
    # crowd_aware_advisory_integration.md's own Safety Precedence section
    # for why the two are never conflated under the same label.

    sources = []

    if ai_confidence is not None:
        sources.append("ai")

    if rl_confidence is not None:
        sources.append("rl")

    if crowd_confidence is not None:
        sources.append("crowd")

    if progress_confidence is not None:
        sources.append("progress")

    if response_confidence is not None:
        sources.append("response")

    return tuple(sources)


def _ai_bottleneck_confidence_for_wait_zone(action: str, evidence) -> Optional[float]:

    # AI-Augmented Decision Policy & Advisory Integration milestone --
    # the ONE place AI Decision Evidence is allowed to touch a Civilian
    # Announcement's confidence, and only for a zone Decision Policy's
    # own zone_policy already, independently, marked WAIT (i.e. its
    # recommended_exit is already in ground_truth.exits_exceeding_
    # capacity -- a deterministic, GroundTruth-derived congestion
    # signal, computed with zero knowledge of this AI model). AI never
    # creates a WAIT status here, never changes which exit/stair is
    # recommended, and never touches SHELTER_IN_PLACE/EVACUATE_
    # IMMEDIATELY zones (the two life-safety-critical actions) at all --
    # it may only strengthen the confidence behind a congestion warning
    # Decision Policy already, independently decided to issue (Phase 6's
    # own explicit example: "AI may strengthen an existing deterministic
    # congestion warning").

    if action != WAIT or evidence is None or not evidence.available:
        return None

    return evidence.bottleneck_occurrence_probability


# =====================================================
# Live Crowd Intelligence -> Operational Advisory Integration milestone
# -- SAFETY PRECEDENCE (Phase 4, the most important requirement in this
# module): crowd congestion is SECONDARY, SUPPORTING evidence. Every
# function below reads decision_policy's own exit_decisions/
# stair_decisions/zone_decisions status FIRST and unconditionally
# defers to it -- none of them can create, remove, or change a CLOSE/
# AVOID/EVACUATE_IMMEDIATELY/SHELTER_IN_PLACE decision. Crowd evidence
# may only ever: (a) strengthen/note confidence on a WAIT zone
# announcement decision_policy ITSELF already produced (the identical
# restriction _ai_bottleneck_confidence_for_wait_zone already applies to
# AI, one line above), or (b) add a wholly NEW, separate
# BuildingRecommendation (never replacing or suppressing an existing
# one -- see build_building_recommendations() below).

# A documented, deterministic, configurable-in-spirit mapping from a
# crowd_intelligence.models.IntensityLevel name (as a plain string --
# advisory_system never imports crowd_intelligence itself, see
# advisory_system.crowd_evidence's own docstring) to a confidence
# contribution -- the same "disclosed policy threshold, not a fabricated
# data value" convention DETERMINISTIC_RULE_BASE_CONFIDENCE/
# risk_based_confidence already establish. LOW/MODERATE are not
# operationally noteworthy enough to contribute (None); only HIGH and
# above ever raise a recommendation's confidence.
_CROWD_LEVEL_CONFIDENCE = {
    "HIGH": 0.60,
    "VERY_HIGH": 0.75,
    "CRITICAL": 0.90,
}


def _crowd_confidence_for_level(level: Optional[str]) -> Optional[float]:

    if level is None:
        return None

    return _CROWD_LEVEL_CONFIDENCE.get(level)


def _crowd_congestion_confidence_for_wait_zone(
    action: str, recommended_exit: Optional[str], recommended_stair: Optional[str], evidence,
) -> Optional[float]:

    # The ONE place Crowd Decision Evidence is allowed to touch a
    # Civilian Announcement's confidence -- mirrors
    # _ai_bottleneck_confidence_for_wait_zone exactly, restricted to a
    # zone Decision Policy's own zone_policy already, independently,
    # marked WAIT. Crowd evidence never creates a WAIT status here,
    # never changes which exit/stair is recommended, and never touches
    # SHELTER_IN_PLACE/EVACUATE_IMMEDIATELY zones at all -- it may only
    # strengthen the confidence behind a congestion warning Decision
    # Policy already, independently decided to issue, and only when THIS
    # exact zone's own recommended exit/stair is one crowd intelligence
    # has independently, live-observed to be congested (never a
    # building-wide signal applied indiscriminately to every WAIT zone,
    # unlike the AI signal above, which crowd intelligence's own
    # per-asset localization makes possible to avoid).

    if action != WAIT or evidence is None or not evidence.available:
        return None

    for asset_type, asset_id in (("Exit", recommended_exit), ("Stair", recommended_stair)):

        if asset_id is None or not evidence.is_congested(asset_type, asset_id):
            continue

        detail = evidence.asset_details.get(asset_id)
        level = detail.congestion_level if detail is not None else None

        # A congestion level should always be present once is_congested()
        # is True (see the adapter's own contract -- congested asset ids
        # always get a matching asset_details entry); "HIGH" is a
        # defensive, disclosed floor only, never silently assumed CRITICAL.
        return _crowd_confidence_for_level(level) if level is not None else _CROWD_LEVEL_CONFIDENCE["HIGH"]

    return None


def _crowd_wait_zone_reason_note(
    action: str, recommended_exit: Optional[str], recommended_stair: Optional[str], evidence,
) -> Optional[str]:

    # Phase 5/12's own explainability requirement -- a plain, structured-
    # evidence-derived sentence, appended to CivilianAnnouncement.reason
    # ONLY (never to the broadcast announcement text itself -- see
    # _format_announcement(), untouched by this milestone). Crowd
    # evidence explains WHY a WAIT recommendation is additionally
    # supported; it never rewrites WHAT occupants are told to do.

    if action != WAIT or evidence is None or not evidence.available:
        return None

    for asset_type, asset_id, noun in (("Exit", recommended_exit, "Exit"), ("Stair", recommended_stair, "Stair")):

        if asset_id is None or not evidence.is_congested(asset_type, asset_id):
            continue

        detail = evidence.asset_details.get(asset_id)
        level = detail.congestion_level if detail is not None else None
        trend = detail.trend if detail is not None else None

        level_text = level or "elevated"
        trend_text = f" and a {trend} trend" if trend else ""

        return f"Live crowd intelligence observes {level_text} congestion{trend_text} at {noun} {asset_id}."

    return None


# =====================================================
# Live Evacuation Progress, Flow & Clearance Intelligence milestone --
# SAFETY PRECEDENCE (Phase 14), mirroring the crowd-congestion functions
# immediately above EXACTLY: evacuation progress is SECONDARY,
# SUPPORTING evidence. It may only ever (a) strengthen/note confidence
# on a WAIT-zone announcement decision_policy itself already produced,
# or (b) add wholly NEW, separate BuildingRecommendations. It never
# creates, removes, or changes a CLOSE/AVOID/EVACUATE_IMMEDIATELY/
# SHELTER_IN_PLACE decision.

# A documented, deterministic, single flat confidence contribution for
# any progress-sourced finding -- deliberately simpler than crowd's own
# three-tier _CROWD_LEVEL_CONFIDENCE, since evacuation-progress findings
# (stalled zone, high-queue-low-flow exit) are themselves already
# binary (either flagged or not -- there is no HIGH/CRITICAL gradient
# analogous to crowd density/congestion levels). Set above
# DETERMINISTIC_RULE_BASE_CONFIDENCE (0.70) deliberately -- a genuine,
# independently-observed operational finding is corroborating evidence
# that should STRENGTHEN a recommendation's confidence once blended in
# (matching _CROWD_LEVEL_CONFIDENCE's own HIGH tier), never weaken it by
# averaging in a below-floor number.
_PROGRESS_FINDING_CONFIDENCE = 0.80


def _progress_confidence_for_wait_zone(
    action: str, recommended_exit: Optional[str], recommended_stair: Optional[str], evidence,
) -> Optional[float]:

    # Mirrors _crowd_congestion_confidence_for_wait_zone exactly,
    # restricted to a zone Decision Policy's own zone_policy already,
    # independently, marked WAIT, and only when THIS exact zone's own
    # recommended exit is one evacuation progress has independently
    # flagged as high-queue-low-flow (a genuine bottleneck signal, not a
    # building-wide generalization).

    if action != WAIT or evidence is None or not evidence.available:
        return None

    if recommended_exit is not None and recommended_exit in evidence.high_queue_low_flow_exit_ids:
        return _PROGRESS_FINDING_CONFIDENCE

    return None


def _progress_wait_zone_reason_note(
    action: str, recommended_exit: Optional[str], recommended_stair: Optional[str], evidence,
) -> Optional[str]:

    # Mirrors _crowd_wait_zone_reason_note exactly -- appended to
    # CivilianAnnouncement.reason ONLY, never to the broadcast
    # announcement text itself.

    if action != WAIT or evidence is None or not evidence.available:
        return None

    if recommended_exit is not None and recommended_exit in evidence.high_queue_low_flow_exit_ids:

        detail = evidence.exit_details.get(recommended_exit)
        queue = detail.queue_candidate_count if detail is not None else None
        queue_text = f" ({queue} occupant(s) queueing, low observed throughput)" if queue is not None else ""

        return f"Observed evacuation progress shows Exit {recommended_exit} has a high queue with low throughput{queue_text}."

    return None


# =====================================================
# Live Emergency Response & Rescue Priority Intelligence milestone --
# SAFETY PRECEDENCE (Phase 19), mirroring the crowd/progress functions
# immediately above EXACTLY: emergency response priority is SECONDARY,
# SUPPORTING evidence. It may only ever (a) strengthen/note confidence
# on a WAIT-zone announcement decision_policy itself already produced,
# or (b) add wholly NEW, separate BuildingRecommendations. It never
# creates, removes, or changes a CLOSE/AVOID/EVACUATE_IMMEDIATELY/
# SHELTER_IN_PLACE decision, and it is deterministic analytics, never AI
# (Phase 20 -- confidence_source below tags it "response", never "ai").

_RESPONSE_FINDING_CONFIDENCE = 0.80  # same "above the 0.70 floor, strengthens rather than weakens" reasoning as _PROGRESS_FINDING_CONFIDENCE


def _response_confidence_for_wait_zone(action: str, zone_id: str, evidence) -> Optional[float]:

    # Mirrors _progress_confidence_for_wait_zone exactly, restricted to
    # a zone Decision Policy's own zone_policy already, independently,
    # marked WAIT, and only when THIS exact zone is one emergency
    # response intelligence has independently flagged CRITICAL/HIGH
    # priority (never a building-wide generalization).

    if action != WAIT or evidence is None or not evidence.available:
        return None

    if zone_id in evidence.critical_zone_ids or zone_id in evidence.high_priority_zone_ids:
        return _RESPONSE_FINDING_CONFIDENCE

    return None


def _response_wait_zone_reason_note(action: str, zone_id: str, evidence) -> Optional[str]:

    # Mirrors _progress_wait_zone_reason_note exactly -- appended to
    # CivilianAnnouncement.reason ONLY, never to the broadcast
    # announcement text itself.

    if action != WAIT or evidence is None or not evidence.available:
        return None

    detail = evidence.zone_details.get(zone_id)

    if detail is None or detail.priority_level not in ("CRITICAL", "HIGH"):
        return None

    reason_text = ", ".join(code.replace("_", " ").title() for code in detail.reason_codes) or "elevated response priority"

    return f"Live emergency response intelligence flags Zone {zone_id} as {detail.priority_level} priority ({reason_text})."


# =====================================================
# Phase 2 -- Civilian Advisory. Zone-based announcements ONLY: every
# CivilianAnnouncement below is addressed to "occupants in <zone>," and
# no field anywhere on it references an individual occupant_id. The
# underlying action/recommended_exit/recommended_stair are read
# straight off DecisionPolicy.zone_decisions (already deterministic,
# already computed) -- this function only formats them into an
# announcement and resolves any exit/stair Decision Policy itself
# marked unsafe/congested after zone_decisions were computed (the
# "automatically avoid contradictory announcements" requirement).
# =====================================================


def build_civilian_announcements(inputs: AdvisoryInputs) -> Tuple[CivilianAnnouncement, ...]:

    ground_truth = inputs.ground_truth
    decision_policy = inputs.decision_policy
    building = inputs.building

    zone_names = _zone_name_lookup(building)
    exit_names = _exit_name_lookup(building)
    stair_names = _stair_name_lookup(building)

    exit_status = _exit_status_by_id(decision_policy)
    stair_status = _stair_status_by_id(decision_policy)
    risk_by_zone = _risk_by_zone(ground_truth)

    announcements = []

    for zone_decision in decision_policy.zone_decisions:

        zone_id = zone_decision["zone_id"]
        action, recommended_exit, recommended_stair, contradiction_notes = _resolve_zone_action(
            zone_id=zone_id,
            action=zone_decision["action"],
            recommended_exit=zone_decision.get("recommended_exit"),
            recommended_stair=zone_decision.get("recommended_stair"),
            ground_truth=ground_truth,
            exit_status=exit_status,
            stair_status=stair_status,
        )

        announcement_text = _format_announcement(
            zone_name=zone_names.get(zone_id, zone_id),
            action=action,
            exit_name=exit_names.get(recommended_exit, recommended_exit) if recommended_exit else None,
            stair_name=stair_names.get(recommended_stair, recommended_stair) if recommended_stair else None,
            avoided_stair_name=(
                stair_names.get(zone_decision.get("recommended_stair"), zone_decision.get("recommended_stair"))
                if "stair_avoided" in contradiction_notes else None
            ),
            assembly_points=_assembly_points_for_floor(building, _floor_id_for_zone(building, zone_id)),
        )

        rset_improvement = (
            estimate_rset_improvement(ground_truth, zone_id, recommended_exit)
            if "redirected" in contradiction_notes else None
        )

        ai_confidence, ai_agrees = _ai_signal_for(inputs, recommended_exit or zone_id)
        wait_zone_ai_confidence = _ai_bottleneck_confidence_for_wait_zone(action, inputs.ai_decision_evidence)
        combined_ai_confidence = combine_confidence(ai_confidence, wait_zone_ai_confidence)

        # Live Crowd Intelligence -> Operational Advisory Integration
        # milestone -- kept as its OWN, separate confidence contribution
        # (never folded into combined_ai_confidence above), so
        # _confidence_source()/CivilianAnnouncement.confidence_source can
        # honestly distinguish "ai" from "crowd" (Phase 11's own "keep
        # provenance separate" requirement).
        crowd_wait_confidence = _crowd_congestion_confidence_for_wait_zone(
            action, recommended_exit, recommended_stair, inputs.crowd_decision_evidence,
        )

        # Live Evacuation Progress, Flow & Clearance Intelligence
        # milestone -- a THIRD, independently-tracked confidence
        # contribution, alongside ai/crowd, never merged with either.
        progress_wait_confidence = _progress_confidence_for_wait_zone(
            action, recommended_exit, recommended_stair, inputs.evacuation_progress_evidence,
        )

        # Live Emergency Response & Rescue Priority Intelligence
        # milestone -- a FOURTH, independently-tracked confidence
        # contribution, keyed on the ZONE itself (never an asset).
        response_wait_confidence = _response_confidence_for_wait_zone(
            action, zone_id, inputs.emergency_response_evidence,
        )

        threshold = CRITICAL_RISK_THRESHOLD if action == SHELTER_IN_PLACE else ELEVATED_RISK_THRESHOLD
        confidence = recommendation_confidence(
            risk_score=risk_by_zone.get(zone_id), risk_threshold=threshold,
            ai_confidence=combined_ai_confidence, rl_confidence=inputs.rl_confidence,
            crowd_confidence=crowd_wait_confidence, progress_confidence=progress_wait_confidence,
            response_confidence=response_wait_confidence,
            agreement_signals=[ai_agrees] if ai_agrees is not None else [],
        )
        confidence_source = _confidence_source(
            combined_ai_confidence, inputs.rl_confidence, crowd_wait_confidence, progress_wait_confidence,
            response_wait_confidence,
        )

        explanation = explain_zone_recommendation(
            zone_id=zone_id, action=action, recommended_exit=recommended_exit,
            ground_truth=ground_truth, confidence=confidence, rset_improvement=rset_improvement,
        )

        reason_parts = list(explanation.reasons)

        crowd_note = _crowd_wait_zone_reason_note(action, recommended_exit, recommended_stair, inputs.crowd_decision_evidence)
        if crowd_note is not None:
            reason_parts.append(crowd_note)

        progress_note = _progress_wait_zone_reason_note(
            action, recommended_exit, recommended_stair, inputs.evacuation_progress_evidence,
        )
        if progress_note is not None:
            reason_parts.append(progress_note)

        response_note = _response_wait_zone_reason_note(action, zone_id, inputs.emergency_response_evidence)
        if response_note is not None:
            reason_parts.append(response_note)

        announcements.append(
            CivilianAnnouncement(
                zone_id=zone_id,
                zone_name=zone_names.get(zone_id, zone_id),
                announcement=announcement_text,
                reason=" ".join(reason_parts),
                confidence=confidence,
                predicted_rset_improvement_seconds=rset_improvement,
                confidence_source=confidence_source,
            )
        )

    return tuple(announcements)


def _resolve_zone_action(
    *, zone_id: str, action: str, recommended_exit: Optional[str], recommended_stair: Optional[str],
    ground_truth, exit_status: Dict[str, str], stair_status: Dict[str, str],
) -> Tuple[str, Optional[str], Optional[str], "frozenset"]:

    # Contradiction avoidance: Decision Policy's own zone_decisions and
    # exit_decisions/stair_decisions are computed independently (see
    # decision_policy/policy.py's own call order) and can disagree --
    # e.g. a zone recommended toward an exit that exit_decisions itself
    # separately marks CLOSE. Rather than ever announcing "proceed to"
    # an exit/stair Decision Policy itself considers unsafe, this
    # degrades the announcement honestly: an EVACUATE_IMMEDIATELY zone
    # whose only recommended exit is CLOSE becomes SHELTER_IN_PLACE (no
    # fabricated alternative route is invented); a recommended stair
    # marked AVOID is simply dropped from the announcement (the zone
    # still evacuates via its exit, just without a stair instruction
    # Decision Policy itself flagged unsafe). A recommended exit that
    # is merely overloaded (not CLOSE) and has a genuine underutilized
    # alternative is redirected -- ground_truth.recommendations' own
    # "Redirect occupants" rule, independently reapplied here (see
    # explanation_engine.compute_redirect_target()) so the announcement
    # itself, not just its stated reason, reflects the better route.

    notes = set()

    if recommended_stair is not None and stair_status.get(recommended_stair) == AVOID:
        recommended_stair = None
        notes.add("stair_avoided")

    if action == EVACUATE_IMMEDIATELY and recommended_exit is not None:

        if exit_status.get(recommended_exit) == CLOSE:
            return SHELTER_IN_PLACE, None, None, frozenset({"exit_closed"})

        alternative_exit = compute_redirect_target(ground_truth, zone_id)

        if alternative_exit is not None:
            notes.add("redirected")
            recommended_exit = alternative_exit

    return action, recommended_exit, recommended_stair, frozenset(notes)


def _format_announcement(
    *, zone_name: str, action: str, exit_name: Optional[str], stair_name: Optional[str],
    avoided_stair_name: Optional[str], assembly_points: List,
) -> str:

    lines = [f"Attention occupants in {zone_name}."]

    if action == SHELTER_IN_PLACE:

        lines.append("Remain in place. Await further instructions.")

    elif action == WAIT:

        lines.append("Hold your position briefly. Your evacuation route is currently congested.")

    else:  # EVACUATE_IMMEDIATELY

        if avoided_stair_name:
            lines.append(f"Avoid {avoided_stair_name}.")

        if exit_name:
            lines.append(f"Proceed to {exit_name}.")
            if stair_name:
                lines.append(f"Use {stair_name}.")
        elif stair_name:
            lines.append(f"Proceed through {stair_name}.")
        else:
            lines.append("Proceed to your nearest available exit.")

        if assembly_points:
            names = " or ".join(point.name or point.id for point in assembly_points)
            lines.append(f"Move to {names}.")

    return " ".join(lines)


# =====================================================
# Phase 3 -- Firefighter Intelligence. Information only -- there is no
# field anywhere on FirefighterIntelligenceReport that names a
# firefighter_id or assigns a task; rescue_priority_areas/
# suggested_access_routes are ranked *locations*, read straight off
# DecisionPolicy.rescue_priorities/rescue_order (already computed,
# never re-derived), not directives to any specific person.
# =====================================================


def build_firefighter_intelligence(inputs: AdvisoryInputs) -> FirefighterIntelligenceReport:

    ground_truth = inputs.ground_truth
    decision_policy = inputs.decision_policy
    scenario = inputs.scenario
    building = inputs.building

    zones = _ordered_zones(building)
    occupants_by_zone = _occupant_counts_by_zone(scenario, zones)

    children_count = sum(
        1 for occupant in scenario.occupants
        if occupant_category(occupant.behaviour_profile_id) == OccupantCategory.CHILD
    )
    wheelchair_users_count = sum(
        1 for occupant in scenario.occupants
        if occupant_category(occupant.behaviour_profile_id) == OccupantCategory.WHEELCHAIR_USER
    )

    risk_by_zone = _risk_by_zone(ground_truth)
    smoke_by_zone = _latest_zone_smoke_levels(inputs)

    exit_status = _exit_status_by_id(decision_policy)
    stair_status = _stair_status_by_id(decision_policy)

    blocked_routes = tuple(
        [f"Exit {exit_id}" for exit_id, status in exit_status.items() if status == CLOSE]
        + [f"Stair {stair_id}" for stair_id, status in stair_status.items() if status == AVOID]
    )
    available_routes = tuple(
        [f"Exit {exit_id}" for exit_id, status in exit_status.items() if status in (KEEP_OPEN, OPEN)]
        + [f"Stair {stair_id}" for stair_id, status in stair_status.items() if status == USE]
    )

    max_risk = max((score for score in risk_by_zone.values() if score is not None), default=None)
    ai_confidence, ai_agrees = _ai_signal_for(inputs, ground_truth.maximum_hazard_zone)

    evidence = inputs.ai_decision_evidence
    evidence_confidence = evidence.bottleneck_occurrence_probability if evidence is not None and evidence.available else None
    combined_ai_confidence = combine_confidence(ai_confidence, evidence_confidence)

    # Live Crowd Intelligence -> Operational Advisory Integration
    # milestone -- Phase 8's "firefighter awareness" of high zone
    # density/congestion, contributed as its own, separately-tracked
    # confidence signal (never merged into combined_ai_confidence).
    crowd_confidence = _crowd_confidence_for_level(
        inputs.crowd_decision_evidence.most_congested_level
        if inputs.crowd_decision_evidence is not None and inputs.crowd_decision_evidence.available else None
    )

    # Live Emergency Response & Rescue Priority Intelligence milestone --
    # a further, separately-tracked confidence signal (Phase 17's own
    # "particularly useful to firefighter recommendations").
    response_evidence = inputs.emergency_response_evidence
    response_confidence = (
        _RESPONSE_FINDING_CONFIDENCE
        if response_evidence is not None and response_evidence.available
        and (response_evidence.critical_zone_ids or response_evidence.high_priority_zone_ids)
        else None
    )

    confidence = recommendation_confidence(
        risk_score=max_risk, risk_threshold=MODERATE_THRESHOLD,
        ai_confidence=combined_ai_confidence, rl_confidence=inputs.rl_confidence,
        crowd_confidence=crowd_confidence, response_confidence=response_confidence,
        agreement_signals=[ai_agrees] if ai_agrees is not None else [],
    )
    confidence_source = _confidence_source(
        combined_ai_confidence, inputs.rl_confidence, crowd_confidence, response_confidence=response_confidence,
    )

    building_status = (
        "Building cleared" if ground_truth.building_cleared
        else f"{ground_truth.people_trapped} occupant(s) not yet evacuated"
    )

    route_stats_by_zone = {entry["zone_id"]: entry for entry in ground_truth.zone_route_stats}

    suggested_access_routes = tuple(
        {
            "zone_id": zone_id,
            "via_exit": route_stats_by_zone.get(zone_id, {}).get("preferred_exit"),
            "via_stair": route_stats_by_zone.get(zone_id, {}).get("preferred_stair"),
            "estimated_travel_time_seconds": route_stats_by_zone.get(zone_id, {}).get("average_travel_time"),
        }
        for zone_id in decision_policy.rescue_order
    )

    return FirefighterIntelligenceReport(
        building_status=building_status,
        occupants_remaining=ground_truth.people_trapped,
        occupants_by_zone=occupants_by_zone,
        children_count=children_count,
        wheelchair_users_count=wheelchair_users_count,
        possible_injury_count=ground_truth.possible_injury_count,
        fallen_count=ground_truth.fallen_count,
        helping_groups_count=ground_truth.helping_group_count,
        hazard_severity_by_zone=risk_by_zone,
        smoke_conditions_by_zone=smoke_by_zone,
        available_routes=available_routes,
        blocked_routes=blocked_routes,
        building_system_status=inputs.building_system_state,
        predicted_rset_seconds=ground_truth.total_evacuation_time,
        confidence=confidence,
        rescue_priority_areas=decision_policy.rescue_priorities,
        suggested_access_routes=suggested_access_routes,
        ai_bottleneck_probability=evidence_confidence,
        ai_bottleneck_model_id=evidence.model_id if evidence is not None and evidence.available else None,
        confidence_source=confidence_source,
        live_priority_zone_ids=(
            tuple(sorted(set(response_evidence.critical_zone_ids) | set(response_evidence.high_priority_zone_ids)))
            if response_evidence is not None and response_evidence.available else ()
        ),
        live_possible_assistance_zone_ids=(
            response_evidence.possible_assistance_zone_ids
            if response_evidence is not None and response_evidence.available else ()
        ),
    )


def _latest_zone_smoke_levels(inputs: AdvisoryInputs) -> Dict[str, Optional[float]]:

    # Smoke level has no per-zone field on GroundTruth (only per-tick,
    # on a Timeline Dataset row) -- read from the most recent supplied
    # timeline row, the same "latest row" convention decision_policy.
    # policy._current_fire_zone_ids() already establishes for its own
    # timeline-sourced field. Empty when no timeline_rows were supplied
    # -- never guessed from risk_score.

    if not inputs.timeline_rows:
        return {}

    from dataset_builder.schema import ordered_zones as _dataset_ordered_zones
    from dataset_builder.timeline_schema import zone_smoke_columns

    zones = _dataset_ordered_zones(inputs.building)
    columns = zone_smoke_columns(inputs.building)
    latest_row = inputs.timeline_rows[-1]

    return {zone.id: latest_row.get(column) for column, zone in zip(columns, zones)}


# =====================================================
# Phase 4 -- Building Advisory. Recommendation only -- no function in
# this module (or anywhere else in this package) executes a control;
# every entry below is a BuildingRecommendation, an inert record.
# =====================================================

_BENEFIT_BY_ACTION_PREFIX = (
    ("Open Exit", "Provides an additional evacuation route for occupants still inside."),
    ("Close Door", "Prevents further hazard spread through a congested bottleneck."),
    ("Deploy staff", "Reduces stair queueing through active crowd management."),
    ("Increase exit width", "Reduces future capacity-driven congestion at this exit."),
    ("Additional detector", "Improves early hazard detection coverage for this zone."),
    ("Additional camera", "Improves visual monitoring coverage for this zone."),
    ("Redirect occupants", "Balances load away from an overloaded exit toward an underused one."),
)


def build_building_recommendations(inputs: AdvisoryInputs) -> Tuple[BuildingRecommendation, ...]:

    ground_truth = inputs.ground_truth
    decision_policy = inputs.decision_policy

    risk_by_zone = _risk_by_zone(ground_truth)
    risk_by_stair = {entry["stair_id"]: entry["risk_score"] for entry in ground_truth.stair_risk_scores}

    recommendations: List[BuildingRecommendation] = []

    # ---- Already-computed engineering findings (ground_truth.recommendations) ----

    for entry in ground_truth.recommendations:

        location_risk = None
        if entry["target_type"] == "zone":
            location_risk = risk_by_zone.get(entry["target_id"])
        elif entry["target_type"] == "stair":
            location_risk = risk_by_stair.get(entry["target_id"])

        confidence = recommendation_confidence(risk_score=location_risk, risk_threshold=MODERATE_THRESHOLD)

        recommendations.append(
            BuildingRecommendation(
                action=entry["action"], target_type=entry["target_type"], target_id=entry["target_id"],
                reason=entry["reason"], confidence=confidence,
                expected_engineering_benefit=_benefit_for(entry["action"]),
            )
        )

    # ---- Stair Pressurization -- stairs Decision Policy marked AVOID
    # because they connect to a hazardous zone (the same connects_to_
    # hazard signal decision_policy.stair_policy already computed). ----

    hazardous_zone_ids = set(ground_truth.hazard_spread_order)
    if ground_truth.maximum_hazard_zone:
        hazardous_zone_ids.add(ground_truth.maximum_hazard_zone)

    for entry in decision_policy.stair_decisions:

        if entry["status"] != AVOID:
            continue

        recommendations.append(
            BuildingRecommendation(
                action=f"Activate Stair Pressurization for Stair {entry['stair_id']}",
                target_type="stair", target_id=entry["stair_id"],
                reason=f"Stair {entry['stair_id']} was marked AVOID (unsafe or severely bottlenecked).",
                confidence=recommendation_confidence(
                    risk_score=risk_by_stair.get(entry["stair_id"]), risk_threshold=MODERATE_THRESHOLD,
                ),
                expected_engineering_benefit=(
                    "Prevents smoke migration into the stairwell, preserving it as a viable egress route."
                ),
            )
        )

    # ---- Smoke Exhaust -- zones on the hazard spread path. ----

    for zone_id in sorted(hazardous_zone_ids):

        recommendations.append(
            BuildingRecommendation(
                action=f"Activate Smoke Exhaust in Zone {zone_id}",
                target_type="zone", target_id=zone_id,
                reason=f"Zone {zone_id} is on the fire's own observed hazard spread path.",
                confidence=recommendation_confidence(
                    risk_score=risk_by_zone.get(zone_id), risk_threshold=MODERATE_THRESHOLD,
                ),
                expected_engineering_benefit=(
                    "Reduces smoke accumulation, improving visibility along evacuation routes."
                ),
            )
        )

    # ---- Deluge -- the single most hazardous zone only. ----

    if ground_truth.maximum_hazard_zone:

        recommendations.append(
            BuildingRecommendation(
                action=f"Activate Deluge in Zone {ground_truth.maximum_hazard_zone}",
                target_type="zone", target_id=ground_truth.maximum_hazard_zone,
                reason=f"Zone {ground_truth.maximum_hazard_zone} is the most hazardous zone in this incident.",
                confidence=recommendation_confidence(
                    risk_score=risk_by_zone.get(ground_truth.maximum_hazard_zone),
                    risk_threshold=CRITICAL_THRESHOLD,
                ),
                expected_engineering_benefit=(
                    "Suppresses fire growth in the most hazardous zone, reducing further hazard spread."
                ),
            )
        )

    # ---- Unlock Exit -- exits at least one zone currently recommends
    # but Decision Policy's own exit_decisions still reports as
    # something other than open (a live-state signal ground_truth.
    # recommendations' own "Open Exit" rule -- built from historical
    # under-use -- does not evaluate). ----

    recommended_exit_ids = {
        decision["recommended_exit"] for decision in decision_policy.zone_decisions
        if decision.get("recommended_exit")
    }
    exit_status = _exit_status_by_id(decision_policy)

    for exit_id in sorted(recommended_exit_ids):

        if exit_status.get(exit_id) == CLOSE:

            recommendations.append(
                BuildingRecommendation(
                    action=f"Unlock Exit {exit_id}",
                    target_type="exit", target_id=exit_id,
                    reason=f"Exit {exit_id} is recommended by at least one zone but is currently CLOSE.",
                    confidence=recommendation_confidence(risk_score=None),
                    expected_engineering_benefit="Provides occupants a viable route through this exit.",
                )
            )

    # ---- Update Dynamic Exit Signs -- whenever any zone has a
    # currently recommended exit at all. ----

    if recommended_exit_ids:

        recommendations.append(
            BuildingRecommendation(
                action="Update Dynamic Exit Signs",
                target_type="building", target_id=None,
                reason=(
                    f"{len(recommended_exit_ids)} exit(s) are currently recommended across "
                    f"{len(decision_policy.zone_decisions)} zone(s)."
                ),
                confidence=recommendation_confidence(risk_score=None),
                expected_engineering_benefit="Directs occupants toward the currently recommended exit in real time.",
            )
        )

    # ---- Broadcast Voice Message -- whenever Decision Policy itself
    # produced any announcement. ----

    if decision_policy.announcements:

        recommendations.append(
            BuildingRecommendation(
                action="Broadcast Voice Message",
                target_type="building", target_id=None,
                reason=f"{len(decision_policy.announcements)} announcement(s) are currently active.",
                confidence=recommendation_confidence(risk_score=None),
                expected_engineering_benefit="Ensures civilian announcements reach the intended zones.",
            )
        )

    # ---- Monitor for Building-Wide Congestion -- AI-Augmented Decision
    # Policy & Advisory Integration milestone. The ONE new recommendation
    # AI Decision Evidence alone can add -- deliberately a "monitor"
    # action, never a control action (no door/deluge/smoke-exhaust/
    # stair-pressurization/voice-broadcast/exit-closure recommendation is
    # ever generated from this signal, per Phase 8's own explicit list),
    # deliberately building-wide/target_type="building" (never a
    # fabricated zone/stair/exit target -- see ai_evidence.py's own
    # docstring for why no localization exists to target), and
    # deliberately additive -- appended only, never replacing or
    # suppressing any deterministic recommendation above. ----

    if inputs.ai_decision_evidence is not None and inputs.ai_decision_evidence.bottleneck_predicted:

        evidence = inputs.ai_decision_evidence

        recommendations.append(
            BuildingRecommendation(
                action="Monitor for Building-Wide Congestion",
                target_type="building", target_id=None,
                reason=(
                    f"Bottleneck Occurrence Model predicts elevated building-wide bottleneck "
                    f"probability ({evidence.bottleneck_occurrence_probability:.2f})."
                ),
                confidence=recommendation_confidence(
                    risk_score=None, ai_confidence=evidence.bottleneck_occurrence_probability,
                ),
                expected_engineering_benefit=(
                    "Prompts proactive monitoring/staffing attention before congestion is directly observed."
                ),
                confidence_source=("ai",),
            )
        )

    # =====================================================
    # Live Crowd Intelligence -> Operational Advisory Integration
    # milestone. SAFETY PRECEDENCE (Phase 4): every recommendation below
    # is either (a) purely informational (a "Monitor" note -- valid
    # regardless of an asset's own safety status; Phase 8's own "do NOT
    # modify emergency evacuation priority purely because a zone is
    # crowded"), or (b) an alternate-route SUPPORT recommendation
    # restricted to assets decision_policy ITSELF already marks usable
    # (status != CLOSE for exits, != AVOID for stairs) -- crowd
    # intelligence never determines route safety on its own (Phase 5/6),
    # only ever chooses among options Decision Policy already considers
    # safe. All of these are additive-only -- appended, never replacing
    # or suppressing any recommendation above (including the "ai"-sourced
    # one immediately above, which coexists independently -- Phase 11).
    # =====================================================

    crowd_evidence = inputs.crowd_decision_evidence

    if crowd_evidence is not None and crowd_evidence.available:

        stair_status = _stair_status_by_id(decision_policy)
        coverage_caveat = _crowd_coverage_caveat(crowd_evidence)

        # ---- Monitor congestion -- one per congested Door/Exit/Stair,
        # informational only, valid regardless of decision_policy status
        # (a firefighter/commander benefits from knowing a CLOSED exit is
        # also crowding up against it). ----

        for asset_type_label, congested_ids, target_type in (
            ("Door", crowd_evidence.congested_door_ids, "door"),
            ("Exit", crowd_evidence.congested_exit_ids, "exit"),
            ("Stair", crowd_evidence.congested_stair_ids, "stair"),
        ):

            for asset_id in sorted(congested_ids):

                recommendations.append(
                    _crowd_monitor_asset_recommendation(asset_type_label, asset_id, target_type, crowd_evidence, coverage_caveat)
                )

        # ---- Monitor high crowd density -- one per zone above the
        # configured density threshold, informational only. ----

        for zone_id in sorted(crowd_evidence.zones_above_density_threshold):

            recommendations.append(_crowd_monitor_zone_recommendation(zone_id, crowd_evidence, coverage_caveat))

        # ---- Prefer an alternate SAFE exit/stair -- ONLY ever among
        # assets decision_policy itself already marks usable. Never
        # invented when no safe, uncongested alternative genuinely
        # exists. ----

        exit_status = _exit_status_by_id(decision_policy)

        for status_by_id, congested_ids, unsafe_value, asset_type in (
            (exit_status, crowd_evidence.congested_exit_ids, CLOSE, "exit"),
            (stair_status, crowd_evidence.congested_stair_ids, AVOID, "stair"),
        ):

            recommendations.extend(
                _crowd_prefer_alternative_recommendations(status_by_id, congested_ids, unsafe_value, asset_type, crowd_evidence, coverage_caveat)
            )

    # =====================================================
    # Live Evacuation Progress, Flow & Clearance Intelligence milestone.
    # SAFETY PRECEDENCE (Phase 14): every recommendation below is purely
    # informational -- a "stalled"/"uncertain"/"slow flow" note, never a
    # route/safety decision. None of these ever touch decision_policy's
    # own CLOSE/AVOID/action fields; they are appended only.
    # =====================================================

    progress_evidence = inputs.evacuation_progress_evidence

    if progress_evidence is not None and progress_evidence.available:

        for zone_id in sorted(progress_evidence.stalled_zone_ids):

            detail = progress_evidence.zone_details.get(zone_id)
            fraction_text = (
                f" ({detail.clearance_fraction:.0%} of observed occupants cleared so far)"
                if detail is not None and detail.clearance_fraction is not None else ""
            )

            recommendations.append(
                BuildingRecommendation(
                    action=f"Review Stalled Evacuation Progress in Zone {zone_id}",
                    target_type="zone", target_id=zone_id,
                    reason=f"Observed evacuation progress in Zone {zone_id} appears stalled{fraction_text}.",
                    confidence=recommendation_confidence(risk_score=None, progress_confidence=_PROGRESS_FINDING_CONFIDENCE),
                    expected_engineering_benefit="Prompts operator attention to a zone that has stopped clearing despite known remaining occupants.",
                    confidence_source=("progress",),
                )
            )

        for exit_id in sorted(progress_evidence.high_queue_low_flow_exit_ids):

            detail = progress_evidence.exit_details.get(exit_id)
            queue_text = f" ({detail.queue_candidate_count} occupant(s) queueing)" if detail is not None else ""

            recommendations.append(
                BuildingRecommendation(
                    action=f"Review Exit {exit_id} Throughput",
                    target_type="exit", target_id=exit_id,
                    reason=f"Exit {exit_id} has high queue demand but low observed throughput{queue_text}.",
                    confidence=recommendation_confidence(risk_score=None, progress_confidence=_PROGRESS_FINDING_CONFIDENCE),
                    expected_engineering_benefit="Distinguishes a genuinely bottlenecked exit from one that is heavily used but functioning.",
                    confidence_source=("progress",),
                )
            )

        for zone_id in sorted(progress_evidence.zones_clearance_unknown):

            recommendations.append(
                BuildingRecommendation(
                    action=f"Confirm Clearance Status for Zone {zone_id}",
                    target_type="zone", target_id=zone_id,
                    reason=f"Clearance status for Zone {zone_id} is uncertain due to insufficient camera coverage.",
                    confidence=recommendation_confidence(risk_score=None, progress_confidence=_PROGRESS_FINDING_CONFIDENCE),
                    expected_engineering_benefit="Flags a zone whose true occupancy cannot currently be confirmed, never assumed clear.",
                    confidence_source=("progress",),
                )
            )

        if progress_evidence.overall_progress_trend in ("SLOWING", "STALLED"):

            recommendations.append(
                BuildingRecommendation(
                    action="Monitor Slowing Building-Wide Evacuation Progress",
                    target_type="building", target_id=None,
                    reason=f"Observed evacuation progress has a {progress_evidence.overall_progress_trend.lower()} trend.",
                    confidence=recommendation_confidence(risk_score=None, progress_confidence=_PROGRESS_FINDING_CONFIDENCE),
                    expected_engineering_benefit="Prompts proactive monitoring before evacuation progress deteriorates further.",
                    confidence_source=("progress",),
                )
            )

    # =====================================================
    # Live Emergency Response & Rescue Priority Intelligence milestone.
    # SAFETY PRECEDENCE (Phase 19): every recommendation below is purely
    # informational -- a priority-search/assistance/uncertainty/clear
    # note, never a route/safety decision. None of these ever touch
    # decision_policy's own CLOSE/AVOID/action fields.
    # =====================================================

    if inputs.emergency_response_evidence is not None and inputs.emergency_response_evidence.available:

        response_evidence = inputs.emergency_response_evidence

        for zone_id in sorted(set(response_evidence.critical_zone_ids) | set(response_evidence.high_priority_zone_ids)):

            detail = response_evidence.zone_details.get(zone_id)
            level = detail.priority_level if detail is not None else None
            reason_codes = detail.reason_codes if detail is not None else ()
            reason_text = ", ".join(code.replace("_", " ").title() for code in reason_codes) or "elevated response priority"

            recommendations.append(
                BuildingRecommendation(
                    action=f"Priority Search: Zone {zone_id}",
                    target_type="zone", target_id=zone_id,
                    reason=f"Zone {zone_id} is {level or 'elevated'} response priority. Evidence: {reason_text}.",
                    confidence=recommendation_confidence(risk_score=None, response_confidence=_RESPONSE_FINDING_CONFIDENCE),
                    expected_engineering_benefit="Directs incident command attention to the zone with the strongest combined evidence of remaining risk.",
                    confidence_source=("response",),
                )
            )

        for zone_id in sorted(response_evidence.possible_assistance_zone_ids):

            detail = response_evidence.zone_details.get(zone_id)
            possible = detail.possible_assistance_count if detail is not None else 0
            confirmed = detail.confirmed_assistance_count if detail is not None else 0
            certainty = "Confirmed" if confirmed > 0 else "Possible"

            recommendations.append(
                BuildingRecommendation(
                    action=f"Possible Assistance Required in Zone {zone_id}",
                    target_type="zone", target_id=zone_id,
                    reason=f"{certainty} assistance required in Zone {zone_id} ({possible} possible, {confirmed} confirmed signal(s)).",
                    confidence=recommendation_confidence(risk_score=None, response_confidence=_RESPONSE_FINDING_CONFIDENCE),
                    expected_engineering_benefit="Flags occupants who may be unable to self-evacuate without ever asserting a medical diagnosis.",
                    confidence_source=("response",),
                )
            )

        for zone_id in sorted(response_evidence.uncertain_search_zone_ids):

            recommendations.append(
                BuildingRecommendation(
                    action=f"Verify Occupancy in Zone {zone_id}",
                    target_type="zone", target_id=zone_id,
                    reason=f"Clearance of Zone {zone_id} is uncertain due to unavailable camera coverage.",
                    confidence=recommendation_confidence(risk_score=None, response_confidence=_RESPONSE_FINDING_CONFIDENCE),
                    expected_engineering_benefit="Flags a zone that may still require a physical search, never assumed safely clear.",
                    confidence_source=("response",),
                )
            )

    return tuple(recommendations)


def _crowd_coverage_caveat(evidence) -> str:

    # Phase 10's own explicit honesty requirement: never let an absence
    # of observed congestion read as a confirmed "clear" when position/
    # calibration coverage is incomplete. None (full or unknown coverage)
    # adds no caveat; anything below 100% gets an explicit, disclosed
    # note -- this is the ONE sentence every crowd-sourced recommendation
    # below shares, so a caller can never see one without the other.

    if evidence.position_coverage_fraction is None:
        return ""

    if evidence.position_coverage_fraction >= 1.0:
        return " Position coverage: 100%."

    return (
        f" Position coverage: {evidence.position_coverage_fraction:.0%} -- absence of observed "
        f"congestion elsewhere does not confirm those routes are clear."
    )


def _crowd_monitor_asset_recommendation(asset_type_label, asset_id, target_type, evidence, coverage_caveat) -> BuildingRecommendation:

    detail = evidence.asset_details.get(asset_id)
    level = detail.congestion_level if detail is not None else None
    trend = detail.trend if detail is not None else None

    reason = f"Live crowd intelligence reports {level or 'elevated'} congestion at {asset_type_label} {asset_id}"
    if trend:
        reason += f" with a {trend} trend"
    reason += "."
    if detail is not None:
        reason += (
            f" {detail.approaching_count} occupant(s) approaching, "
            f"{detail.queue_candidate_count} in an estimated queue."
        )
    reason += coverage_caveat

    return BuildingRecommendation(
        action=f"Monitor Congestion at {asset_type_label} {asset_id}",
        target_type=target_type, target_id=asset_id,
        reason=reason,
        confidence=recommendation_confidence(risk_score=None, crowd_confidence=_crowd_confidence_for_level(level)),
        expected_engineering_benefit="Prompts proactive crowd management attention at a live-observed congestion point.",
        confidence_source=("crowd",),
    )


def _crowd_monitor_zone_recommendation(zone_id, evidence, coverage_caveat) -> BuildingRecommendation:

    detail = evidence.zone_details.get(zone_id)
    level = detail.density_classification if detail is not None else None
    trend = detail.trend if detail is not None else None

    reason = f"Live crowd intelligence reports {level or 'elevated'} occupant density in Zone {zone_id}"
    if trend:
        reason += f" with a {trend} trend"
    reason += "."
    reason += coverage_caveat

    return BuildingRecommendation(
        action=f"Monitor High Crowd Density in Zone {zone_id}",
        target_type="zone", target_id=zone_id,
        reason=reason,
        confidence=recommendation_confidence(risk_score=None, crowd_confidence=_crowd_confidence_for_level(level)),
        expected_engineering_benefit="Prompts proactive crowd management attention in a densely occupied zone.",
        confidence_source=("crowd",),
    )


def _crowd_prefer_alternative_recommendations(
    status_by_id, congested_ids, unsafe_value, asset_type, evidence, coverage_caveat,
) -> List[BuildingRecommendation]:

    # SAFETY GATE: `usable_ids` excludes anything decision_policy itself
    # marked unsafe (CLOSE for exits, AVOID for stairs) BEFORE congestion
    # is even considered -- an unsafe asset can never appear as either
    # the "congested" side or the "alternative" side below, satisfying
    # Phase 16 test #3 ("unsafe exit + alternative safe congested exit ->
    # unsafe exit still never selected") structurally, not by a
    # special-cased check.

    usable_ids = sorted(asset_id for asset_id, status in status_by_id.items() if status != unsafe_value)
    congested_usable = [asset_id for asset_id in usable_ids if asset_id in congested_ids]

    # CONFIRMED clear only -- an asset absent from congested_ids is
    # honestly "clear" ONLY when it also has real position coverage
    # (Phase 10/13: "no position coverage" and "confirmed uncongested"
    # are never the same thing). An asset with zero coverage is neither
    # recommended as an alternative nor implied to be safe from a
    # congestion standpoint -- it is simply not evidence either way.
    clear_usable = [
        asset_id for asset_id in usable_ids
        if asset_id not in congested_ids and asset_id not in evidence.position_unavailable_asset_ids
    ]

    if not congested_usable or not clear_usable:
        return []

    alternative = clear_usable[0]
    recommendations = []

    for congested_id in congested_usable:

        detail = evidence.asset_details.get(congested_id)
        level = detail.congestion_level if detail is not None else None

        reason = (
            f"{asset_type.title()} {congested_id} is currently congested; {asset_type.title()} {alternative} "
            f"is an equally Decision-Policy-usable route with no congestion currently observed."
        )
        reason += coverage_caveat

        recommendations.append(
            BuildingRecommendation(
                action=f"Prefer {asset_type.title()} {alternative} over {asset_type.title()} {congested_id}",
                target_type=asset_type, target_id=alternative,
                reason=reason,
                confidence=recommendation_confidence(risk_score=None, crowd_confidence=_crowd_confidence_for_level(level)),
                expected_engineering_benefit=(
                    f"Balances occupant load away from a congested {asset_type} toward an uncongested, "
                    f"equally safe alternative."
                ),
                confidence_source=("crowd",),
            )
        )

    return recommendations


def _benefit_for(action: str) -> str:

    for prefix, benefit in _BENEFIT_BY_ACTION_PREFIX:
        if action.startswith(prefix):
            return benefit

    return "Addresses an engineering finding identified during simulation analysis."


# =====================================================
# Phase 5 -- Incident Commander Dashboard.
# =====================================================


def build_commander_dashboard(
    inputs: AdvisoryInputs,
    civilian_announcements: Tuple[CivilianAnnouncement, ...],
    building_recommendations: Tuple[BuildingRecommendation, ...],
) -> IncidentCommanderDashboard:

    ground_truth = inputs.ground_truth
    decision_policy = inputs.decision_policy
    scenario = inputs.scenario
    building = inputs.building

    zones = _ordered_zones(building)
    occupants_by_zone = _occupant_counts_by_zone(scenario, zones)
    risk_by_zone = _risk_by_zone(ground_truth)

    safe_zones, warning_zones, critical_zones = [], [], []

    for zone in zones:

        risk_score = risk_by_zone.get(zone.id)

        if risk_score is None:
            continue

        if risk_score >= CRITICAL_THRESHOLD:
            critical_zones.append(zone.id)
        elif risk_score >= MODERATE_THRESHOLD:
            warning_zones.append(zone.id)
        else:
            safe_zones.append(zone.id)

    exit_status = _exit_status_by_id(decision_policy)
    stair_status = _stair_status_by_id(decision_policy)

    blocked_routes = tuple(
        [f"Exit {exit_id}" for exit_id, status in exit_status.items() if status == CLOSE]
        + [f"Stair {stair_id}" for stair_id, status in stair_status.items() if status == AVOID]
    )
    available_exits = tuple(
        exit_id for exit_id, status in exit_status.items() if status in (KEEP_OPEN, OPEN)
    )

    predicted_bottlenecks = tuple(
        [f"Door {door_id}" for door_id in ground_truth.doors_that_became_bottlenecks]
        + ([f"Exit {ground_truth.worst_exit}"] if ground_truth.worst_exit else [])
        + ([f"Stair {ground_truth.worst_stair}"] if ground_truth.worst_stair else [])
    )

    evidence = inputs.ai_decision_evidence
    evidence_confidence = evidence.bottleneck_occurrence_probability if evidence is not None and evidence.available else None

    crowd_evidence = inputs.crowd_decision_evidence
    crowd_available = crowd_evidence is not None and crowd_evidence.available
    crowd_confidence = _crowd_confidence_for_level(crowd_evidence.most_congested_level) if crowd_available else None

    progress_evidence = inputs.evacuation_progress_evidence
    progress_available = progress_evidence is not None and progress_evidence.available
    progress_confidence = (
        _PROGRESS_FINDING_CONFIDENCE
        if progress_available and (progress_evidence.stalled_zone_ids or progress_evidence.high_queue_low_flow_exit_ids)
        else None
    )

    response_evidence = inputs.emergency_response_evidence
    response_available = response_evidence is not None and response_evidence.available
    response_confidence = (
        _RESPONSE_FINDING_CONFIDENCE
        if response_available and (response_evidence.critical_zone_ids or response_evidence.high_priority_zone_ids)
        else None
    )

    rec_confidence = combine_confidence(
        *[entry.confidence for entry in civilian_announcements],
        *[entry.confidence for entry in building_recommendations],
        evidence_confidence,
        crowd_confidence,
        progress_confidence,
        response_confidence,
    )

    overall_severity = _overall_incident_severity(
        building_cleared=ground_truth.building_cleared,
        people_trapped=ground_truth.people_trapped,
        critical_zone_count=len(critical_zones),
        warning_zone_count=len(warning_zones),
    )

    return IncidentCommanderDashboard(
        safe_zones=tuple(safe_zones),
        critical_zones=tuple(critical_zones),
        warning_zones=tuple(warning_zones),
        occupants_remaining=ground_truth.people_trapped,
        occupants_by_zone=occupants_by_zone,
        predicted_bottlenecks=predicted_bottlenecks,
        blocked_routes=blocked_routes,
        available_exits=available_exits,
        building_system_status=inputs.building_system_state,
        predicted_rset_seconds=ground_truth.total_evacuation_time,
        highest_priority_rescue_areas=decision_policy.rescue_order,
        occupancy_confidence=occupancy_confidence(inputs.human_observations),
        recommendation_confidence=rec_confidence,
        overall_incident_severity=overall_severity,
        ai_bottleneck_probability=evidence_confidence,
        ai_bottleneck_model_id=evidence.model_id if evidence is not None and evidence.available else None,
        crowd_highest_density_zone_id=crowd_evidence.highest_density_zone_id if crowd_available else None,
        crowd_most_congested_asset_id=crowd_evidence.most_congested_asset_id if crowd_available else None,
        crowd_most_congested_level=crowd_evidence.most_congested_level if crowd_available else None,
        evacuation_progress_fraction=progress_evidence.overall_progress_fraction if progress_available else None,
        evacuation_stalled_zone_ids=progress_evidence.stalled_zone_ids if progress_available else (),
        evacuation_clearance_unknown_zone_ids=progress_evidence.zones_clearance_unknown if progress_available else (),
        response_highest_priority_zone_id=response_evidence.highest_priority_zone_id if response_available else None,
        response_critical_zone_ids=response_evidence.critical_zone_ids if response_available else (),
        response_possible_assistance_zone_ids=response_evidence.possible_assistance_zone_ids if response_available else (),
        response_being_assisted_zone_ids=response_evidence.being_assisted_zone_ids if response_available else (),
    )


GT_5_TRAPPED_IS_CRITICAL = 5


def _overall_incident_severity(
    *, building_cleared: bool, people_trapped: int, critical_zone_count: int, warning_zone_count: int,
) -> str:

    # A disclosed, deterministic severity ladder over already-computed
    # values -- never a fabricated judgment call. Any critical zone, or
    # a large-enough trapped headcount (GT_5_TRAPPED_IS_CRITICAL, the
    # same "fixed, disclosed parameter" convention every other threshold
    # in this codebase already uses), is CRITICAL outright; any warning
    # zone or nonzero trapped count is at least ELEVATED; a fully
    # resolved incident with nothing outstanding is STABLE.

    if building_cleared:
        return "RESOLVED"

    if critical_zone_count > 0 or people_trapped >= GT_5_TRAPPED_IS_CRITICAL:
        return "CRITICAL"

    if warning_zone_count > 0 or people_trapped > 0:
        return "ELEVATED"

    return "STABLE"
