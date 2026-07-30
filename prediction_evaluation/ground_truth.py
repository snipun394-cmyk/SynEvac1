from typing import Optional

from prediction_evaluation.models import GroundTruthSample


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 3 -- the
# ONE place a GroundTruthSample is derived from already-computed
# snapshots. Every argument is duck-typed and OPTIONAL -- this function
# never re-simulates, never re-runs Crowd Intelligence, never queries a
# LiveOccupantManager itself; a caller (Simulation OR Live Runtime, via
# the exact same call shape -- Phase 0's own "identical evaluation
# logic" requirement) supplies whatever it already has this cycle.
# Absence of an argument honestly leaves the corresponding
# GroundTruthSample field None, never fabricated.
#
# `crowd_snapshot` is the one argument doing most of the work --
# crowd_intelligence.models.CrowdIntelligenceSnapshot is produced
# IDENTICALLY by CrowdIntelligenceEngine.compute() in both a Simulation
# context (simulation_runtime's own per-tick composition) and a Live
# Runtime context (live_runtime.factory.build_live_runtime()'s own
# EngineCrowdIntelligenceGateway) -- there is no second, live-only or
# sim-only crowd intelligence computation anywhere in this codebase for
# this function to accidentally diverge from.
# =====================================================


def extract_ground_truth_sample(
    timestamp: float,
    source: str = "unknown",
    *,
    crowd_snapshot=None,
    occupancy_facts=None,
    hazard_summary=None,
    evacuation_progress_snapshot=None,
    total_evacuation_time: Optional[float] = None,
) -> GroundTruthSample:

    total_occupant_count = None
    if occupancy_facts is not None:
        total_occupant_count = getattr(occupancy_facts, "total_observed_count", None)
    elif crowd_snapshot is not None and crowd_snapshot.building_summary is not None:
        total_occupant_count = crowd_snapshot.building_summary.total_observed_occupants

    congestion_detected = None
    congested_asset_ids = ()

    if crowd_snapshot is not None and crowd_snapshot.building_summary is not None:

        summary = crowd_snapshot.building_summary
        congested_asset_ids = tuple(summary.congested_doors) + tuple(summary.congested_exits) + tuple(summary.congested_stairs)
        congestion_detected = len(congested_asset_ids) > 0

    queue_lengths = {}
    flow_rates = {}

    if crowd_snapshot is not None:

        for metrics_map in (crowd_snapshot.door_metrics, crowd_snapshot.exit_metrics, crowd_snapshot.stair_metrics):
            for asset_id, metrics in metrics_map.items():
                if metrics.position_available and metrics.queue_candidate_count is not None:
                    queue_lengths[asset_id] = metrics.queue_candidate_count

        for stair_id, flow_metrics in crowd_snapshot.stair_flow_metrics.items():
            if flow_metrics.exit_rate_per_minute is not None:
                flow_rates[stair_id] = flow_metrics.exit_rate_per_minute

    if evacuation_progress_snapshot is not None:

        for exit_id, exit_flow in getattr(evacuation_progress_snapshot, "exits", {}).items():
            if exit_flow.recent_flow_per_minute is not None:
                flow_rates[exit_id] = exit_flow.recent_flow_per_minute

    hazard_overall_severity = None
    if hazard_summary is not None and getattr(hazard_summary, "overall_severity", None) is not None:
        hazard_overall_severity = hazard_summary.overall_severity.name

    evacuation_complete = None
    evacuation_time_seconds = None

    if total_evacuation_time is not None:

        evacuation_complete = True
        evacuation_time_seconds = total_evacuation_time

    elif evacuation_progress_snapshot is not None:

        total_observed = getattr(evacuation_progress_snapshot, "known_total_observed_occupants", 0)

        if total_observed > 0:
            # Honest only when someone was genuinely observed at all --
            # zero observed occupants is "nothing to evacuate," never
            # silently read as "evacuation complete."
            evacuation_complete = getattr(evacuation_progress_snapshot, "known_active_occupants", None) == 0

    return GroundTruthSample(
        timestamp=timestamp,
        source=source,
        total_occupant_count=total_occupant_count,
        congestion_detected=congestion_detected,
        congested_asset_ids=congested_asset_ids,
        queue_lengths=queue_lengths,
        flow_rates=flow_rates,
        hazard_overall_severity=hazard_overall_severity,
        evacuation_complete=evacuation_complete,
        evacuation_time_seconds=evacuation_time_seconds,
    )
