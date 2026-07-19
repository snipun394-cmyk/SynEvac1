from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from human_decision_engine.events import DecisionEventLog, GROUP_DISSOLVED, GROUP_FORMED, RESCUE_COMPLETED


# =====================================================
# Phase 5 -- Group Behavior. A temporary Group is nothing more than a
# labeled set of occupant_ids that formed together this session --
# GroupRegistry itself makes no movement/routing decision (that
# already happens via the assistance/rescue pairing + AssistanceAware*
# mechanics this whole package reuses); it only records the fact that
# a group exists, for Command Center display and Dataset Builder
# export.
#
# Formation is driven by whatever already decided two occupants belong
# together -- an ASSIST/GROUP_FOLLOW civilian pairing (human_decision_
# engine.pairing) or a firefighter's rescue claim (human_decision_
# engine.firefighter_engine) -- GroupRegistry.form() is called by
# behaviour_profile_resolver.dynamic_registrar once each such pairing
# is *confirmed* (the helper actually decided HELP, not merely
# proposed), never speculatively.
#
# Dissolution is necessarily post-hoc: "this group is done" can only
# be known once every member's own OccupantTimeline reaches a terminal
# state (ARRIVED or UNREACHABLE) -- the same authoritative signal
# ground_truth.evacuation_metrics.compute_evacuation_metrics() already
# uses to decide who evacuated. compute_group_dissolution() below is
# analysis over an already-completed MultiAgentSimulationResult, the
# same "post-hoc derivation over a completed run" shape ground_truth.
# human_behavior.compute_occupant_behavior() already establishes --
# never a live, intra-run event this run-to-completion engine cannot
# honestly produce (see human_decision_engine.firefighter_engine's own
# docstring for the identical reasoning about "task priority evolving
# intra-run").
#
# Known, documented simplification: every group formed here has
# exactly two members (one helper, one target) -- Phase 5's own "Two
# civilians carrying an injured person" example (a three-member group:
# two helpers, one target) is not produced by this phase's 1:1 pairing
# model (human_decision_engine.pairing) and is not fabricated here.
# Group is not restricted to two members structurally (member_ids is
# an arbitrary tuple), so a future pairing model that allows a second
# helper to join an existing group needs no change to this module.
# =====================================================

GROUP_KIND_ASSIST = "ASSIST"
GROUP_KIND_FOLLOW = "FOLLOW"
GROUP_KIND_FIREFIGHTER_ESCORT = "FIREFIGHTER_ESCORT"


@dataclass(frozen=True)
class Group:

    group_id: str
    member_ids: Tuple[str, ...]
    kind: str


class GroupRegistry:

    def __init__(self, event_log: Optional[DecisionEventLog] = None):

        self._groups: Dict[str, Group] = {}
        self._counter = 0

        self.event_log = event_log if event_log is not None else DecisionEventLog()

    # =====================================================

    def form(self, member_ids: Sequence[str], kind: str) -> str:

        self._counter += 1
        group_id = f"group-{self._counter}"

        member_ids = tuple(member_ids)
        self._groups[group_id] = Group(group_id=group_id, member_ids=member_ids, kind=kind)

        self.event_log.record(
            GROUP_FORMED,
            member_ids[0],
            related_occupant_id=member_ids[1] if len(member_ids) > 1 else None,
            reason=kind,
            metadata={"group_id": group_id, "member_ids": list(member_ids)},
        )

        return group_id

    # =====================================================

    @property
    def groups(self) -> Tuple[Group, ...]:

        return tuple(self._groups.values())

    # =====================================================

    def group_for(self, occupant_id: str) -> Optional[Group]:

        for group in self._groups.values():
            if occupant_id in group.member_ids:
                return group

        return None


# =====================================================


def compute_group_dissolution(
    groups: Sequence[Group], movement_result, event_log: Optional[DecisionEventLog] = None,
) -> Tuple[Dict[str, Any], ...]:

    from simulator.occupant import OccupantState

    records = []

    for group in groups:

        timelines = [movement_result.occupants.get(member_id) for member_id in group.member_ids]

        if any(timeline is None for timeline in timelines):
            continue

        terminal = all(
            timeline.state in (OccupantState.ARRIVED, OccupantState.UNREACHABLE)
            for timeline in timelines
        )

        dissolution_time = None

        if terminal:

            arrived_timelines = [t for t in timelines if t.state == OccupantState.ARRIVED]
            arrival_times = [t.arrival_time for t in arrived_timelines if t.arrival_time is not None]

            if arrived_timelines and len(arrival_times) == len(arrived_timelines):
                dissolution_time = max(arrival_times)

        records.append(
            {
                "group_id": group.group_id,
                "member_ids": list(group.member_ids),
                "kind": group.kind,
                "dissolved": terminal,
                "dissolution_time": dissolution_time,
            }
        )

        if terminal and event_log is not None:

            event_log.record(
                GROUP_DISSOLVED,
                group.member_ids[0],
                related_occupant_id=group.member_ids[1] if len(group.member_ids) > 1 else None,
                reason=group.kind,
                metadata={"group_id": group.group_id, "dissolution_time": dissolution_time},
            )

    return tuple(records)


# =====================================================
# RESCUE_COMPLETED -- the post-hoc counterpart to RESCUE_INITIATED
# (logged live, at assignment time, by FirefighterDecisionEngine).
# "Completed" uses the same authoritative ARRIVED signal as everywhere
# else in this codebase; a target who never reaches ARRIVED (including
# one whose firefighter themselves became unreachable) is never
# reported complete.
# =====================================================


def compute_rescue_completion_events(
    rescue_target_by_firefighter_id: Dict[str, str], movement_result, event_log: DecisionEventLog,
) -> None:

    from simulator.occupant import OccupantState

    for firefighter_id, target_id in rescue_target_by_firefighter_id.items():

        target_timeline = movement_result.occupants.get(target_id)

        if target_timeline is not None and target_timeline.state == OccupantState.ARRIVED:

            event_log.record(
                RESCUE_COMPLETED, firefighter_id, related_occupant_id=target_id,
                reason="target_arrived", metadata={"arrival_time": target_timeline.arrival_time},
            )
