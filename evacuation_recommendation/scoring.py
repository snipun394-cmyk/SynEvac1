from typing import Optional, Tuple

from evacuation_recommendation.models import RecommendationConfig, RecommendationReason, RecommendationWeights


# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone, Phase 4/5/7
# -- deterministic, disclosed, no-ML scoring. Every contribution below
# is a documented weight (evacuation_recommendation.models.
# RecommendationWeights) with its own named reason code, mirroring
# emergency_response.engine._score_zone()'s own transparency
# discipline exactly.
# =====================================================


_INTENSITY_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "VERY_HIGH": 3, "CRITICAL": 4}


def score_candidate(
    *, route_distance_m: float, min_distance: float, max_distance: float,
    congestion_level: Optional[str], queue_count: Optional[int], throughput: Optional[float],
    trajectory_support: Optional[str], emergency_response_elevated: bool,
    ai_bottleneck_probability: Optional[float],
    weights: RecommendationWeights, config: RecommendationConfig,
) -> Tuple[float, Tuple[str, ...]]:

    reason_codes = []
    score = 0.0

    # Route distance -- normalized RELATIVE to this zone's own other
    # safe candidates only (Phase 4's own "shortest safe exit
    # preferred" requirement); a zone with only one reachable safe exit
    # gets full distance credit (no basis to rank it against anything).
    distance_range = max_distance - min_distance
    distance_goodness = 1.0 if distance_range <= 0 else 1.0 - (route_distance_m - min_distance) / distance_range
    score += weights.route_distance_weight * distance_goodness

    if route_distance_m == min_distance:
        reason_codes.append(RecommendationReason.SHORTEST_SAFE_ROUTE)

    # Congestion -- None (no crowd evidence) is scored neutrally (0.5),
    # never assumed clear.
    if congestion_level is not None:

        congestion_rank = _INTENSITY_RANK.get(congestion_level, 0)
        max_rank = _INTENSITY_RANK.get(config.congestion_normalization_level, 4)
        congestion_goodness = 1.0 - min(congestion_rank / max_rank, 1.0) if max_rank else 1.0

        score += weights.congestion_weight * congestion_goodness

        if congestion_rank <= _INTENSITY_RANK["MODERATE"]:
            reason_codes.append(RecommendationReason.LOW_CONGESTION)
        elif congestion_rank >= max_rank:
            reason_codes.append(RecommendationReason.HIGH_CONGESTION)

    else:
        score += weights.congestion_weight * 0.5

    # Queue length.
    if queue_count is not None:

        queue_goodness = 1.0 - min(queue_count / config.queue_normalization_count, 1.0) if config.queue_normalization_count else 1.0
        score += weights.queue_weight * queue_goodness

        if queue_count > 0:
            reason_codes.append(RecommendationReason.QUEUE_PRESENT)

    else:
        score += weights.queue_weight * 0.5

    # Throughput -- higher recent_flow_per_minute is better.
    if throughput is not None:

        throughput_goodness = min(throughput / config.throughput_normalization_per_minute, 1.0) if config.throughput_normalization_per_minute else 0.5
        score += weights.throughput_weight * throughput_goodness

        if throughput_goodness >= 0.66:
            reason_codes.append(RecommendationReason.HIGH_THROUGHPUT)
        elif throughput_goodness <= 0.2:
            reason_codes.append(RecommendationReason.LOW_THROUGHPUT)

    else:
        score += weights.throughput_weight * 0.5

    # Trajectory -- per-exit occupant movement evidence (Phase 4).
    if trajectory_support == "PROGRESSING":

        score += weights.trajectory_weight * 1.0
        reason_codes.append(RecommendationReason.TRAJECTORY_PROGRESSING_NORMALLY)

    elif trajectory_support == "MOVING_AWAY":

        score += weights.trajectory_weight * 0.0
        reason_codes.append(RecommendationReason.TRAJECTORY_MOVING_AWAY)

    else:
        score += weights.trajectory_weight * 0.5

    # Emergency response -- a binary, conservative penalty when this
    # exit's own zone carries an elevated response priority.
    if emergency_response_elevated:
        reason_codes.append(RecommendationReason.EMERGENCY_RESPONSE_ZONE_ELEVATED)
    else:
        score += weights.emergency_response_weight

    # AI evidence -- Phase 7/13's own "AI only supports" requirement.
    # bottleneck_probability is a SINGLE, building-wide value (never
    # per-exit -- see advisory_system.ai_evidence's own documented
    # boundary), so this contribution is IDENTICAL across every
    # candidate for this zone this cycle -- it can never, by
    # construction, change which candidate ranks higher than another,
    # only the absolute score/explanation/confidence context.
    if ai_bottleneck_probability is not None:

        ai_goodness = 1.0 - ai_bottleneck_probability
        score += weights.ai_support_weight * ai_goodness

        reason_codes.append(
            RecommendationReason.AI_BOTTLENECK_RISK_LOW if ai_bottleneck_probability < 0.5
            else RecommendationReason.AI_BOTTLENECK_RISK_ELEVATED
        )

    else:
        score += weights.ai_support_weight * 0.5

    return score, tuple(reason_codes)


# =====================================================


def compute_confidence(
    *, position_coverage_fraction: Optional[float], route_distance_derived: bool, hazard_evidence_available: bool,
) -> Optional[float]:

    # Phase 7 -- evidence QUALITY, never AI certainty. Each available
    # component contributes its own honest [0, 1] reading (a boolean
    # component contributes 1.0/0.5, never 0.0 -- "evidence absent" is
    # scored neutrally, not as proof of danger); None only when there
    # is genuinely zero basis to estimate confidence at all.

    components = []

    if position_coverage_fraction is not None:
        components.append(position_coverage_fraction)

    components.append(1.0 if route_distance_derived else 0.5)
    components.append(1.0 if hazard_evidence_available else 0.5)

    if not components:
        return None

    return sum(components) / len(components)
