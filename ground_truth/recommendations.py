from typing import Any, Dict, List


# A disclosed policy threshold this ruleset uses to decide "high
# enough to act on" -- not a fabricated data value, the same kind of
# fixed parameter any deterministic engineering rule needs (e.g. a
# fire code's own numeric thresholds).
HIGH_RISK_THRESHOLD = 0.5


# =====================================================
# Deterministic engineering recommendations -- every rule below reads
# only metrics analyzer.py already computed (congestion/engineering/
# risk dicts, zone route stats) and Building's own static fields
# (Door.zone_a_id/zone_b_id, Zone/Detector/Camera.floor_id). No AI, no
# learned weights, no randomness: the same inputs always produce the
# same recommendations.
# =====================================================


def generate_recommendations(
    *,
    zones,
    doors,
    detectors,
    cameras,
    zone_route_stats,
    engineering,
    congestion,
    zone_risk_scores,
    stair_risk_scores,
    maximum_hazard_zone,
    people_trapped,
    exit_open_lookup,
) -> List[Dict[str, Any]]:

    recommendations: List[Dict[str, Any]] = []

    recommendations.extend(
        _open_exit_recommendations(engineering, exit_open_lookup, people_trapped),
    )
    recommendations.extend(_close_door_recommendations(doors, engineering, maximum_hazard_zone))
    recommendations.extend(
        _deploy_staff_recommendations(stair_risk_scores, engineering["stairs_exceeding_capacity"]),
    )
    recommendations.extend(_redirect_recommendations(zone_route_stats, engineering, congestion))
    recommendations.extend(_increase_exit_width_recommendations(engineering))
    recommendations.extend(_additional_detector_recommendations(zones, zone_risk_scores, detectors))
    recommendations.extend(_additional_camera_recommendations(zones, zone_risk_scores, cameras))

    return recommendations


# =====================================================


def _open_exit_recommendations(engineering, exit_open_lookup, people_trapped) -> List[Dict[str, Any]]:

    if people_trapped <= 0:
        return []

    recommendations = []

    for exit_id in engineering["exits_underutilized"]:

        if not exit_open_lookup.get(exit_id, True):

            recommendations.append(
                {
                    "action": f"Open Exit {exit_id}",
                    "target_type": "exit",
                    "target_id": exit_id,
                    "reason": (
                        f"Exit {exit_id} was closed/blocked and never used while "
                        f"{people_trapped} occupant(s) did not evacuate."
                    ),
                }
            )

    return recommendations


# =====================================================


def _close_door_recommendations(doors, engineering, maximum_hazard_zone) -> List[Dict[str, Any]]:

    if maximum_hazard_zone is None:
        return []

    door_by_id = {door.id: door for door in doors}
    recommendations = []

    for door_id in engineering["doors_that_became_bottlenecks"]:

        door = door_by_id.get(door_id)

        if door is None:
            continue

        if maximum_hazard_zone in (door.zone_a_id, door.zone_b_id):

            recommendations.append(
                {
                    "action": f"Close Door {door_id}",
                    "target_type": "door",
                    "target_id": door_id,
                    "reason": (
                        f"Door {door_id} was a congestion bottleneck and connects "
                        f"directly to the most hazardous zone ({maximum_hazard_zone})."
                    ),
                }
            )

    return recommendations


# =====================================================


def _deploy_staff_recommendations(stair_risk_scores, stairs_exceeding_capacity) -> List[Dict[str, Any]]:

    # stair_risk_scores' risk_score is now the worse of two signals
    # (queueing fraction, derived-capacity exceedance -- see
    # ground_truth/risk_analysis.py::compute_stair_risk_scores), so the
    # reason text is worded generically around "risk score" rather
    # than claiming queueing specifically, with an added, honest note
    # when capacity exceedance is what actually drove it.

    exceeding = set(stairs_exceeding_capacity)
    recommendations = []

    for entry in stair_risk_scores:

        risk_score = entry["risk_score"]

        if risk_score is not None and risk_score >= HIGH_RISK_THRESHOLD:

            stair_id = entry["stair_id"]

            reason = (
                f"Stair {stair_id} has a risk score of {risk_score:.0%} "
                f"(queueing and/or occupancy relative to its derived capacity)."
            )

            if stair_id in exceeding:
                reason += " Peak occupancy exceeded the stair's derived capacity."

            recommendations.append(
                {
                    "action": f"Deploy staff to Stair {stair_id}",
                    "target_type": "stair",
                    "target_id": stair_id,
                    "reason": reason,
                }
            )

    return recommendations


# =====================================================


def _redirect_recommendations(zone_route_stats, engineering, congestion) -> List[Dict[str, Any]]:

    underutilized = engineering["exits_underutilized"]

    if not underutilized:
        return []

    overloaded = set(engineering["exits_exceeding_capacity"])
    if congestion.get("worst_exit"):
        overloaded.add(congestion["worst_exit"])

    if not overloaded:
        return []

    alternative_exit = underutilized[0]
    recommendations = []

    for entry in zone_route_stats:

        preferred_exit = entry["preferred_exit"]

        if preferred_exit and preferred_exit in overloaded and preferred_exit != alternative_exit:

            recommendations.append(
                {
                    "action": (
                        f"Redirect occupants from Zone {entry['zone_id']} to "
                        f"Exit {alternative_exit}"
                    ),
                    "target_type": "zone",
                    "target_id": entry["zone_id"],
                    "reason": (
                        f"Zone {entry['zone_id']}'s preferred Exit {preferred_exit} was "
                        f"overloaded while Exit {alternative_exit} was never used."
                    ),
                }
            )

    return recommendations


# =====================================================


def _increase_exit_width_recommendations(engineering) -> List[Dict[str, Any]]:

    return [
        {
            "action": f"Increase exit width recommendation for Exit {exit_id}",
            "target_type": "exit",
            "target_id": exit_id,
            "reason": f"Exit {exit_id}'s peak occupancy exceeded its modeled capacity.",
        }
        for exit_id in engineering["exits_exceeding_capacity"]
    ]


# =====================================================


def _additional_detector_recommendations(zones, zone_risk_scores, detectors) -> List[Dict[str, Any]]:

    risk_by_zone_id = {entry["zone_id"]: entry["risk_score"] for entry in zone_risk_scores}
    floors_with_detectors = {detector.floor_id for detector in detectors}

    recommendations = []

    for zone in zones:

        risk_score = risk_by_zone_id.get(zone.id)

        if (
            risk_score is not None
            and risk_score >= HIGH_RISK_THRESHOLD
            and zone.floor_id not in floors_with_detectors
        ):

            recommendations.append(
                {
                    "action": f"Additional detector recommendation for Zone {zone.id}",
                    "target_type": "zone",
                    "target_id": zone.id,
                    "reason": (
                        f"Zone {zone.id} scored {risk_score:.2f} risk with no detector "
                        f"present anywhere on its floor."
                    ),
                }
            )

    return recommendations


# =====================================================


def _additional_camera_recommendations(zones, zone_risk_scores, cameras) -> List[Dict[str, Any]]:

    risk_by_zone_id = {entry["zone_id"]: entry["risk_score"] for entry in zone_risk_scores}
    floors_with_cameras = {camera.floor_id for camera in cameras}

    recommendations = []

    for zone in zones:

        risk_score = risk_by_zone_id.get(zone.id)

        if (
            risk_score is not None
            and risk_score >= HIGH_RISK_THRESHOLD
            and zone.floor_id not in floors_with_cameras
        ):

            recommendations.append(
                {
                    "action": f"Additional camera recommendation for Zone {zone.id}",
                    "target_type": "zone",
                    "target_id": zone.id,
                    "reason": (
                        f"Zone {zone.id} scored {risk_score:.2f} risk with no camera "
                        f"present anywhere on its floor."
                    ),
                }
            )

    return recommendations
