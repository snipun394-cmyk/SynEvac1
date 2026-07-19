from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Sequence, Set

from hazard.severity import HazardSeverity

from behaviour_profile_resolver.category import OccupantCategory

from human_decision_engine.events import (
    DecisionEventLog,
    FIREFIGHTER_TASK,
    RESCUE_INITIATED,
)
from human_decision_engine.priority import RescuePriorityFactors, compute_rescue_priority


# =====================================================
# Phase 3 -- Firefighter Decision Engine. Deliberately its own module,
# sharing no class with human_decision_engine.engine (the civilian
# engine) -- same "keep firefighter logic separate from civilian
# behavior logic" rule Human Population & Assisted Evacuation
# Modeling's own Phase 4 already established for behavior_library, now
# restated one layer up. The only things shared with the civilian
# engine are neutral primitives (human_decision_engine.events/
# priority), never a firefighter-specific mechanism reused by, or
# reusing, a civilian-specific one.
#
# "Task priority should change as the incident evolves": within one
# registration session, each firefighter's own decision is made once
# (this platform's run-to-completion Simulation Runtime resolves every
# decision before movement begins -- see DecisionContext's own
# docstring on the multi-pass/prior_result extension point for how a
# *second* registration pass, against an updated HazardSnapshot,
# genuinely re-evaluates every task from scratch). Across a session,
# priority still evolves *between* firefighters in registration order:
# each rescue candidate can be claimed by at most one firefighter, so
# the second firefighter registered necessarily re-ranks the remaining,
# unclaimed population -- an honest, incremental form of the incident
# "evolving" as more of it becomes assigned, without claiming a live,
# intra-run task-switching capability this run-to-completion engine
# was never built to provide (that would be a simulator redesign).
# =====================================================

TASK_RESCUE = "RESCUE"
TASK_ASSIST_WHEELCHAIR_USER = "ASSIST_WHEELCHAIR_USER"
TASK_CARRY_FALLEN = "CARRY_FALLEN"
TASK_GUIDE_CIVILIANS = "GUIDE_CIVILIANS"
TASK_REPORT_HAZARD = "REPORT_HAZARD"
TASK_SEARCH_UNCLEARED_ZONE = "SEARCH_UNCLEARED_ZONE"
TASK_CONTINUE_SEARCH = "CONTINUE_SEARCH"
TASK_RETURN_OUTSIDE = "RETURN_OUTSIDE"

RESCUE_TASKS = (TASK_RESCUE, TASK_ASSIST_WHEELCHAIR_USER, TASK_CARRY_FALLEN)

_SEVERITY_ORDINAL = {
    HazardSeverity.NONE: 0,
    HazardSeverity.LOW: 1,
    HazardSeverity.MODERATE: 2,
    HazardSeverity.HIGH: 3,
    HazardSeverity.CRITICAL: 4,
}

# A candidate must clear this minimum urgency score to be worth
# diverting a firefighter toward -- an ordinary, unremarkable Adult in
# a clear zone (score 0.0) is never a rescue target merely for
# existing.
_MINIMUM_URGENCY_SCORE = 0.0


@dataclass(frozen=True)
class CivilianRosterEntry:

    occupant_id: str
    zone_id: str
    category: OccupantCategory


@dataclass(frozen=True)
class FirefighterTaskDecision:

    task: str
    target_occupant_id: Optional[str] = None
    target_zone_id: Optional[str] = None
    reason: str = ""


def _own_hazard_severity(context) -> Optional[HazardSeverity]:

    if context.hazard_snapshot is None:
        return None

    return context.hazard_snapshot.node_state(context.start_id).severity


def _factors_for(
    hazard_snapshot, entry: CivilianRosterEntry,
    known_possible_injury_ids: FrozenSet[str], known_fallen_ids: FrozenSet[str],
    isolated_occupant_ids: FrozenSet[str],
) -> RescuePriorityFactors:

    hazard_score = 0.0

    if hazard_snapshot is not None:
        hazard_score = hazard_snapshot.node_state(entry.zone_id).hazard_score

    return RescuePriorityFactors(
        category=entry.category,
        is_wheelchair_user=entry.category == OccupantCategory.WHEELCHAIR_USER,
        possible_injury=entry.occupant_id in known_possible_injury_ids,
        fallen=entry.occupant_id in known_fallen_ids,
        smoke_exposure=hazard_score,
        distance_from_hazard=1.0 - hazard_score,
        accessibility_score=0.5 if entry.category == OccupantCategory.WHEELCHAIR_USER else 1.0,
        isolation_score=1.0 if entry.occupant_id in isolated_occupant_ids else 0.0,
    )


def compute_priority_scores(
    roster: Sequence[CivilianRosterEntry], hazard_snapshot,
    known_fallen_ids: FrozenSet[str] = frozenset(),
    known_possible_injury_ids: FrozenSet[str] = frozenset(),
    isolated_occupant_ids: FrozenSet[str] = frozenset(),
) -> Dict[str, float]:

    return {
        entry.occupant_id: compute_rescue_priority(
            _factors_for(
                hazard_snapshot, entry,
                known_possible_injury_ids=known_possible_injury_ids,
                known_fallen_ids=known_fallen_ids,
                isolated_occupant_ids=isolated_occupant_ids,
            )
        )
        for entry in roster
    }


def _task_for_factors(factors: RescuePriorityFactors) -> str:

    if factors.fallen or factors.possible_injury:
        return TASK_CARRY_FALLEN

    if factors.is_wheelchair_user:
        return TASK_ASSIST_WHEELCHAIR_USER

    return TASK_RESCUE


class FirefighterDecisionEngine:

    def __init__(
        self,
        civilian_roster: Sequence[CivilianRosterEntry],
        known_fallen_ids: FrozenSet[str] = frozenset(),
        known_possible_injury_ids: FrozenSet[str] = frozenset(),
        isolated_occupant_ids: FrozenSet[str] = frozenset(),
        event_log: Optional[DecisionEventLog] = None,
    ):

        self._roster = list(civilian_roster)
        self._known_fallen_ids = frozenset(known_fallen_ids)
        self._known_possible_injury_ids = frozenset(known_possible_injury_ids)
        self._isolated_occupant_ids = frozenset(isolated_occupant_ids)

        self._claimed_target_ids: Set[str] = set()
        self._searched_zones_by_firefighter: Dict[str, Set[str]] = {}
        self._decisions: Dict[str, FirefighterTaskDecision] = {}

        self.event_log = event_log if event_log is not None else DecisionEventLog()

    # =====================================================

    def evaluate(self, context) -> FirefighterTaskDecision:

        firefighter_id = context.profile.occupant_id

        cached = self._decisions.get(firefighter_id)

        if cached is not None:
            return cached

        searched = self._searched_zones_by_firefighter.setdefault(firefighter_id, set())
        is_first_decision = len(searched) == 0
        searched.add(context.start_id)

        candidate = self._best_candidate(context.hazard_snapshot)

        if candidate is not None:
            decision = self._commit_rescue(firefighter_id, candidate)
        else:
            decision = self._commit_patrol_task(firefighter_id, context, is_first_decision)

        self._decisions[firefighter_id] = decision

        return decision

    # =====================================================

    def decision_for(self, occupant_id: str) -> Optional[FirefighterTaskDecision]:

        return self._decisions.get(occupant_id)

    # =====================================================

    def priority_scores(self, hazard_snapshot=None) -> Dict[str, float]:

        # Every roster entry, scored, regardless of claim state -- the
        # Command Center's own "Current Rescue Priority" column (Phase
        # 7) needs a score for every observed person, not only whoever
        # a firefighter happens to have claimed.

        return compute_priority_scores(
            self._roster, hazard_snapshot,
            known_fallen_ids=self._known_fallen_ids,
            known_possible_injury_ids=self._known_possible_injury_ids,
            isolated_occupant_ids=self._isolated_occupant_ids,
        )

    # =====================================================

    def _best_candidate(self, hazard_snapshot):

        best = None
        best_factors = None

        for entry in self._roster:

            if entry.occupant_id in self._claimed_target_ids:
                continue

            factors = _factors_for(
                hazard_snapshot, entry,
                known_possible_injury_ids=self._known_possible_injury_ids,
                known_fallen_ids=self._known_fallen_ids,
                isolated_occupant_ids=self._isolated_occupant_ids,
            )
            score = compute_rescue_priority(factors)

            if best is None or score > best[1] or (
                score == best[1] and entry.occupant_id < best[0].occupant_id
            ):
                best = (entry, score)
                best_factors = factors

        if best is None or best[1] <= _MINIMUM_URGENCY_SCORE:
            return None

        entry, score = best

        return entry, score, _task_for_factors(best_factors)

    # =====================================================

    def _commit_rescue(self, firefighter_id, candidate) -> FirefighterTaskDecision:

        entry, score, task = candidate

        self._claimed_target_ids.add(entry.occupant_id)

        decision = FirefighterTaskDecision(
            task=task,
            target_occupant_id=entry.occupant_id,
            target_zone_id=entry.zone_id,
            reason=f"highest_priority_score:{score}",
        )

        self.event_log.record(
            FIREFIGHTER_TASK, firefighter_id, related_occupant_id=entry.occupant_id,
            reason=decision.reason, metadata={"task": task, "priority_score": score},
        )

        if task in RESCUE_TASKS:
            self.event_log.record(
                RESCUE_INITIATED, firefighter_id, related_occupant_id=entry.occupant_id, reason=task,
            )

        return decision

    # =====================================================

    def _commit_patrol_task(self, firefighter_id, context, is_first_decision) -> FirefighterTaskDecision:

        severity = _own_hazard_severity(context)
        ordinal = _SEVERITY_ORDINAL.get(severity, 0)
        remaining = any(
            entry.occupant_id not in self._claimed_target_ids for entry in self._roster
        )

        if ordinal >= _SEVERITY_ORDINAL[HazardSeverity.HIGH]:
            task = TASK_REPORT_HAZARD
        elif remaining and is_first_decision:
            task = TASK_GUIDE_CIVILIANS
        elif not remaining and not is_first_decision:
            task = TASK_RETURN_OUTSIDE
        elif is_first_decision:
            task = TASK_SEARCH_UNCLEARED_ZONE
        else:
            task = TASK_CONTINUE_SEARCH

        decision = FirefighterTaskDecision(task=task, reason="no_unclaimed_rescue_candidate")

        self.event_log.record(
            FIREFIGHTER_TASK, firefighter_id, reason=decision.reason, metadata={"task": task},
        )

        return decision
