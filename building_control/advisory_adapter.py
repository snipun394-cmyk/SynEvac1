import hashlib
from typing import Optional, Tuple

from building_control.requests import ControlRequest
from building_control.types import ControlAction, ControlSystemType, RequestSource


# Translates advisory_system.recommendation_models.BuildingRecommendation
# (a free-text action string + target_type/target_id, never a structured
# action_type -- see that dataclass's own definition) into ControlRequests.
# This is a pure, additive read of an already-produced recommendation --
# it never generates a new recommendation, and it never executes anything
# itself (Phase 4). Every pattern below was confirmed to exist verbatim in
# advisory_system/advisory_engine.py::build_building_recommendations() and
# ground_truth/recommendations.py during this milestone's own investigation
# phase; matching is (target_type, action-prefix), not a full-string match,
# so the exit/door/stair/zone id embedded in the action text never needs to
# be re-parsed -- target_id is already carried on the recommendation itself.
#
# Deliberately NOT translated here:
#   - "Broadcast Voice Message" -- already fully wired end-to-end via
#     AdvisoryReport.civilian_announcements -> voice_evacuation (see
#     command_center/incident_data.py's own voice broadcast pipeline).
#     Translating the BuildingRecommendation copy of this action here
#     would create a second, duplicate control path for the same intent.
#   - "Update Dynamic Exit Signs" -- no digital exit-sign asset/control
#     abstraction exists anywhere in this codebase (confirmed during
#     investigation); this milestone does not fabricate one.
#   - "Deploy staff to Stair ...", "Increase exit width ...",
#     "Additional detector ...", "Additional camera ...",
#     "Redirect occupants ..." -- none of these correspond to a
#     ControlSystemType this platform can honestly represent as a
#     dispatchable building-control action.
_PATTERNS: Tuple[Tuple[str, str, ControlSystemType, ControlAction], ...] = (
    ("exit", "Open Exit", ControlSystemType.EXIT, ControlAction.OPEN),
    ("exit", "Unlock Exit", ControlSystemType.EXIT, ControlAction.OPEN),
    ("exit", "Close Exit", ControlSystemType.EXIT, ControlAction.CLOSE),
    ("door", "Open Door", ControlSystemType.DOOR, ControlAction.OPEN),
    ("door", "Close Door", ControlSystemType.DOOR, ControlAction.CLOSE),
    ("stair", "Activate Stair Pressurization", ControlSystemType.STAIR_PRESSURIZATION, ControlAction.ACTIVATE),
    ("zone", "Activate Smoke Exhaust", ControlSystemType.SMOKE_EXHAUST, ControlAction.ACTIVATE),
    ("zone", "Activate Deluge", ControlSystemType.DELUGE, ControlAction.ACTIVATE),
)


def _synthesize_recommendation_id(recommendation) -> str:

    # BuildingRecommendation carries no id field of its own (confirmed
    # during investigation) -- this is a deterministic, content-derived
    # stand-in purely for traceability (so the same recommendation seen
    # again across frames maps to the same source_recommendation_id),
    # never presented as a native identifier.

    raw = f"{recommendation.target_type}:{recommendation.target_id}:{recommendation.action}"
    return "rec-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def translate_recommendation(recommendation) -> Optional[ControlRequest]:

    for target_type, prefix, system_type, action in _PATTERNS:

        if recommendation.target_type != target_type:
            continue

        if not recommendation.action.startswith(prefix):
            continue

        return ControlRequest(
            system_type=system_type,
            target_id=recommendation.target_id,
            requested_action=action,
            reason=recommendation.reason,
            source=RequestSource.ADVISORY_ADAPTER,
            source_recommendation_id=_synthesize_recommendation_id(recommendation),
            confidence=recommendation.confidence,
        )

    return None


def translate_report(report) -> Tuple[ControlRequest, ...]:

    # One ControlRequest per translatable BuildingRecommendation on
    # this AdvisoryReport, in the same order they appear on
    # report.building_recommendations. Untranslatable recommendations
    # are silently skipped -- see the module docstring above for
    # exactly which ones and why.

    translated = (translate_recommendation(entry) for entry in report.building_recommendations)

    return tuple(request for request in translated if request is not None)
