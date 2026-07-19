from typing import Any, Dict, Sequence


# =====================================================
# Human Decision Engine Phase 6 -- summarizes a plain list of decision-
# event dicts (human_decision_engine.events.DecisionEvent.to_dict()'s
# own shape) into per-scenario counts. Consumes only plain
# Dict[str, Any] -- ground_truth/ never imports human_decision_engine,
# the same "typed Any, not imported for its own sake" convention
# command_center.incident_data.IncidentData.ground_truth already
# follows for its own cross-package boundaries. Event type strings are
# restated as literals here (not imported) -- the same "same algorithm/
# vocabulary, independently restated per package" precedent this
# codebase already applies throughout (registrar._derive_occupant_seed,
# firefighter_registrar._derive_firefighter_seed, ...).
#
# events defaults to empty (no dynamic registration was ever used this
# run, the overwhelmingly common case for every scenario predating this
# phase) -- every count below is honestly 0, never fabricated.
# =====================================================

_COLUMN_BY_EVENT_TYPE = {
    "Help_Decision": "help_decision_count",
    "Help_Rejected": "help_rejected_count",
    "Firefighter_Task": "firefighter_task_count",
    "Group_Formed": "group_formed_count",
    "Group_Dissolved": "group_dissolved_count",
    "Rescue_Initiated": "rescue_initiated_count",
    "Rescue_Completed": "rescue_completed_count",
}


def summarize_decision_events(events: Sequence[Dict[str, Any]]) -> Dict[str, int]:

    counts = {column: 0 for column in _COLUMN_BY_EVENT_TYPE.values()}

    for event in events:

        column = _COLUMN_BY_EVENT_TYPE.get(event.get("event_type"))

        if column is not None:
            counts[column] += 1

    return counts
