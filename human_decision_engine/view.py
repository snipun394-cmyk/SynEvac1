from typing import Any, Dict


# =====================================================
# Phase 7 -- Command Center. The one place allowed to translate this
# package's own decision types (human_decision_engine.engine.
# CivilianDecision / human_decision_engine.firefighter_engine.
# FirefighterTaskDecision) into the plain, id-keyed dict shape
# command_center.incident_data.IncidentFrame.human_decisions expects
# (see that field's own docstring) -- so IncidentFrame/HumanPanel
# themselves never need to import human_decision_engine at all, the
# same "typed Any, not imported for its own sake" boundary discipline
# IncidentData.ground_truth/decision_policy already establish for
# their own cross-package fields.
#
# A caller wiring a live or replayed dynamic-registration run into the
# Command Center (mirroring live_system.integration.
# DashboardCommandCenterGateway's own "build an IncidentFrame directly"
# pattern for perception data) passes this function's own output
# straight into IncidentFrame(human_decisions=...).
# =====================================================


def build_human_decisions_view(result) -> Dict[str, Dict[str, Any]]:

    view: Dict[str, Dict[str, Any]] = {}

    for occupant_id, decision in result.decisions_by_occupant_id.items():

        entry: Dict[str, Any] = {}

        if hasattr(decision, "action"):

            # human_decision_engine.engine.CivilianDecision
            entry["decision"] = decision.action

            if decision.target_id:
                entry["goal"] = decision.target_id

            if decision.action in ("HELP", "BE_ASSISTED"):
                entry["assistance_status"] = decision.action

        elif hasattr(decision, "task"):

            # human_decision_engine.firefighter_engine.FirefighterTaskDecision
            entry["decision"] = decision.task
            entry["firefighter_task"] = decision.task

            if decision.target_occupant_id:
                entry["goal"] = decision.target_occupant_id

        group_id = result.group_id_by_occupant_id.get(occupant_id)

        if group_id:
            entry["group"] = group_id

        priority = result.priority_scores.get(occupant_id)

        if priority is not None:
            entry["rescue_priority"] = priority

        if entry:
            view[occupant_id] = entry

    return view
