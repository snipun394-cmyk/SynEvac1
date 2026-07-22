import dataclasses
from typing import Optional

from trajectory_intelligence.anomaly import (
    currently_in_hazardous_zone, movement_stalled, moving_away_from_safe_exit, repeated_route_reversal,
)
from trajectory_intelligence.flow_alignment import compute_flow_alignment
from trajectory_intelligence.history import TrajectoryHistoryStore
from trajectory_intelligence.models import (
    AnomalyFlag, AnomalySeverityThresholds, AnomalyWeights, GroupAnomalyRecord, RouteProgressStatus,
    TrajectoryConfig, TrajectoryIntelligenceSnapshot, TrajectoryResult,
)
from trajectory_intelligence.route_progress import SafeRouteCalculator, compute_route_progress
from trajectory_intelligence.trajectory import compute_movement_facts, is_stale


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone, Phase 2 -- the ONE object LiveRuntime owns
# for this package, mirroring crowd_intelligence.engine.
# CrowdIntelligenceEngine/evacuation_progress.engine.
# EvacuationProgressEngine/emergency_response.engine.
# EmergencyResponseIntelligenceEngine's own "exactly one shared instance
# per live session" role. compute(time, building_state, crowd_snapshot,
# evacuation_progress_snapshot) is the single entry point.
#
# LiveOccupantManager remains the canonical persistent occupant owner
# (Phase 2's own explicit "do not create another occupant registry") --
# this engine only ever reads live_occupant_manager.active_occupants()
# each cycle and owns exactly one small piece of its own cross-cycle
# state (TrajectoryHistoryStore, see history.py), pruned to the same
# active occupant_id set every cycle.
#
# Never makes an evacuation decision, never broadcasts, never executes
# a control, never dispatches personnel, and never overrides a
# deterministic safety decision -- purely a deterministic, geometry-
# only reporting layer (mirrors emergency_response.engine.
# EmergencyResponseIntelligenceEngine's own docstring almost exactly;
# see tests/test_trajectory_intelligence_architecture_guards.py for the
# mechanical enforcement).
# =====================================================


class TrajectoryIntelligenceEngine:

    def __init__(
        self, building, navigation_graph, live_occupant_manager,
        config: Optional[TrajectoryConfig] = None,
        weights: Optional[AnomalyWeights] = None,
        thresholds: Optional[AnomalySeverityThresholds] = None,
    ):

        self.building = building
        self.graph = navigation_graph
        self.live_occupant_manager = live_occupant_manager

        self.config = config if config is not None else TrajectoryConfig()
        self.weights = weights if weights is not None else AnomalyWeights()
        self.thresholds = thresholds if thresholds is not None else AnomalySeverityThresholds()

        self._route_calculator = SafeRouteCalculator(navigation_graph, self.config)
        self._history_store = TrajectoryHistoryStore(max_length=self.config.max_route_distance_samples)

    # =====================================================

    def compute(
        self, time: float, building_state=None, crowd_snapshot=None, evacuation_progress_snapshot=None,
    ) -> TrajectoryIntelligenceSnapshot:

        # crowd_snapshot/evacuation_progress_snapshot are accepted
        # (Phase 19/20's own "may consume where useful, keep ownership
        # separate" requirement) but deliberately UNUSED in this
        # implementation -- every raw TrajectoryResult field is derived
        # solely from LiveOccupant/NavigationGraph/BuildingState, never
        # reinterpreted through crowd/evacuation-progress context, so
        # that context stays available for a future consumer (e.g.
        # Advisory combining "stalled near a congested exit") without
        # this package ever duplicating density/clearance logic itself.

        occupants = self.live_occupant_manager.active_occupants()
        distance_cache = self._route_calculator.compute(building_state)

        results = {}
        movement_directions = {}
        speeds = {}

        for occupant in occupants:

            facts = compute_movement_facts(occupant, self.config)
            movement_directions[occupant.occupant_id] = facts["movement_direction"]
            speeds[occupant.occupant_id] = facts["current_speed"]

            history_state = self._history_store.get(occupant.occupant_id)

            status, distance, trend, exit_id, history_state = compute_route_progress(
                occupant, time, distance_cache, self.graph, history_state, self.config,
            )

            flags = []

            if status == RouteProgressStatus.NO_SAFE_ROUTE:
                flags.append(AnomalyFlag.NO_SAFE_ROUTE)

            if moving_away_from_safe_exit(history_state, time, self.config):
                flags.append(AnomalyFlag.MOVING_AWAY_FROM_SAFE_EXIT)

            if movement_stalled(history_state, self.config):
                flags.append(AnomalyFlag.MOVEMENT_STALLED)

            if repeated_route_reversal(occupant, self.config):
                flags.append(AnomalyFlag.REPEATED_ROUTE_REVERSAL)

            hazardous_now = currently_in_hazardous_zone(occupant, building_state, self.config)

            if hazardous_now:
                flags.append(
                    AnomalyFlag.REMAINS_IN_HAZARDOUS_ZONE if history_state.was_in_hazardous_zone
                    else AnomalyFlag.ENTERED_HAZARDOUS_ZONE
                )

            history_state = history_state.with_hazard_zone_flag(hazardous_now)
            self._history_store.set(occupant.occupant_id, history_state)

            results[occupant.occupant_id] = TrajectoryResult(
                occupant_id=occupant.occupant_id,
                floor_id=occupant.current_floor_id,
                zone_id=occupant.current_zone_id,
                **facts,
                route_progress_status=status,
                route_distance_m=distance,
                route_distance_trend=trend,
                nearest_safe_exit_id=exit_id,
                anomaly_flags=tuple(flags),
                stale=is_stale(occupant, time, self.config),
                last_updated_at=time,
            )

        dominant_by_floor, coverage_by_floor, against_flow_ids = compute_flow_alignment(
            occupants, movement_directions, speeds, self.config,
        )

        for occupant_id in against_flow_ids:

            result = results[occupant_id]
            results[occupant_id] = dataclasses.replace(
                result, anomaly_flags=result.anomaly_flags + (AnomalyFlag.AGAINST_DOMINANT_FLOW,),
            )

        group_anomalies = self._compute_group_anomalies(results)
        results = self._apply_group_flags(results, group_anomalies)

        results = {
            occupant_id: dataclasses.replace(result, anomaly_severity=self._score_severity(result))
            for occupant_id, result in results.items()
        }

        self._history_store.prune(occupant.occupant_id for occupant in occupants)

        return TrajectoryIntelligenceSnapshot(
            timestamp=time,
            occupants=results,
            group_anomalies=group_anomalies,
            dominant_flow_direction_by_floor=dominant_by_floor,
            dominant_flow_coverage_by_floor=coverage_by_floor,
        )

    # =====================================================

    def _compute_group_anomalies(self, results):

        # Phase 13 -- aggregate-only evidence, produced ONLY when at
        # least config.group_min_size occupants sharing the SAME zone
        # genuinely share the same individual anomaly this cycle
        # (never a single occupant, never a fabricated shared cause).
        # Grouped/sorted deterministically so result order never depends
        # on dict/occupant iteration order (Phase 27 test 45).

        by_zone_moving_away = {}
        by_zone_reversal = {}

        for occupant_id, result in results.items():

            if result.zone_id is None:
                continue

            if AnomalyFlag.MOVING_AWAY_FROM_SAFE_EXIT in result.anomaly_flags:
                by_zone_moving_away.setdefault(result.zone_id, []).append(occupant_id)

            if AnomalyFlag.REPEATED_ROUTE_REVERSAL in result.anomaly_flags:
                by_zone_reversal.setdefault(result.zone_id, []).append(occupant_id)

        records = []

        for zone_id in sorted(by_zone_moving_away):

            occupant_ids = tuple(sorted(by_zone_moving_away[zone_id]))

            if len(occupant_ids) >= self.config.group_min_size:
                records.append(GroupAnomalyRecord(
                    anomaly_type="SHARED_ROUTE_DEVIATION", occupant_ids=occupant_ids, zone_id=zone_id,
                    severity=self.thresholds.classify(self.weights.shared_deviation_weight),
                ))

        for zone_id in sorted(by_zone_reversal):

            occupant_ids = tuple(sorted(by_zone_reversal[zone_id]))

            if len(occupant_ids) >= self.config.group_min_size:
                records.append(GroupAnomalyRecord(
                    anomaly_type="MULTI_OCCUPANT_REVERSAL", occupant_ids=occupant_ids, zone_id=zone_id,
                    severity=self.thresholds.classify(self.weights.shared_deviation_weight),
                ))

        return tuple(records)

    # =====================================================

    def _apply_group_flags(self, results, group_anomalies):

        flag_for_type = {
            "SHARED_ROUTE_DEVIATION": AnomalyFlag.SHARED_ROUTE_DEVIATION,
            "MULTI_OCCUPANT_REVERSAL": AnomalyFlag.MULTI_OCCUPANT_REVERSAL,
        }

        updated = dict(results)

        for record in group_anomalies:

            flag = flag_for_type[record.anomaly_type]

            for occupant_id in record.occupant_ids:

                result = updated[occupant_id]

                if flag not in result.anomaly_flags:
                    updated[occupant_id] = dataclasses.replace(result, anomaly_flags=result.anomaly_flags + (flag,))

        return updated

    # =====================================================

    def _score_severity(self, result: TrajectoryResult) -> str:

        # Phase 14 -- deterministic, documented weighted sum. No ML.

        w = self.weights
        score = 0.0
        flags = result.anomaly_flags

        if AnomalyFlag.MOVING_AWAY_FROM_SAFE_EXIT in flags:
            score += w.moving_away_weight

        if AnomalyFlag.REPEATED_ROUTE_REVERSAL in flags:
            score += w.reversal_weight

        if AnomalyFlag.MOVEMENT_STALLED in flags:
            score += w.movement_stalled_weight

        if AnomalyFlag.ENTERED_HAZARDOUS_ZONE in flags or AnomalyFlag.REMAINS_IN_HAZARDOUS_ZONE in flags:
            score += w.hazardous_zone_weight

        if AnomalyFlag.AGAINST_DOMINANT_FLOW in flags:
            score += w.against_flow_weight

        if AnomalyFlag.NO_SAFE_ROUTE in flags:
            score += w.no_safe_route_weight

        if AnomalyFlag.SHARED_ROUTE_DEVIATION in flags or AnomalyFlag.MULTI_OCCUPANT_REVERSAL in flags:
            score += w.shared_deviation_weight

        return self.thresholds.classify(score)
