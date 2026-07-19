from typing import Any, Dict, List

from decision_policy.zone_policy import EVACUATE_IMMEDIATELY, SHELTER_IN_PLACE, WAIT


# =====================================================
# Groups already-computed per-zone actions (zone_policy.py) into
# public-address announcements -- one announcement per distinct
# action that at least one zone actually has, each carrying a fixed
# priority and the list of zones it applies to. No new decision logic
# lives here; this only reshapes zone_policy's output for broadcast.
# =====================================================

CRITICAL = "CRITICAL"
HIGH = "HIGH"
NORMAL = "NORMAL"

_MESSAGE_BY_ACTION = {
    SHELTER_IN_PLACE: (
        "Shelter in place. Do not attempt to evacuate through hazardous conditions."
    ),
    EVACUATE_IMMEDIATELY: "Evacuate immediately using your nearest recommended exit.",
    WAIT: "Hold your position briefly. Your evacuation route is currently congested.",
}

_PRIORITY_BY_ACTION = {
    SHELTER_IN_PLACE: CRITICAL,
    EVACUATE_IMMEDIATELY: HIGH,
    WAIT: NORMAL,
}

# Fixed broadcast order -- most urgent message first, regardless of
# zone iteration order.
_ACTION_ORDER = (SHELTER_IN_PLACE, EVACUATE_IMMEDIATELY, WAIT)


def generate_announcements(zone_decisions) -> List[Dict[str, Any]]:

    target_zones_by_action = {action: [] for action in _ACTION_ORDER}

    for decision in zone_decisions:

        action = decision["action"]

        if action in target_zones_by_action:
            target_zones_by_action[action].append(decision["zone_id"])

    announcements = []

    for action in _ACTION_ORDER:

        target_zones = target_zones_by_action[action]

        if not target_zones:
            continue

        announcements.append(
            {
                "priority": _PRIORITY_BY_ACTION[action],
                "message": _MESSAGE_BY_ACTION[action],
                "target_zones": target_zones,
            }
        )

    return announcements
