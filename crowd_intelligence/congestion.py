from dataclasses import dataclass
from typing import Optional

from crowd_intelligence.models import IntensityLevel
from crowd_intelligence.queue import QueueMetrics


@dataclass(frozen=True)
class CongestionThresholds:

    # Phase 6's own live congestion classification -- a SEPARATE,
    # differently-UNITED scale from crowd_intelligence.models.
    # DensityThresholds (people/m^2): here the input is a dimensionless
    # demand-to-capacity RATIO (queue_candidate_count + approaching_count)
    # / simulation_style_capacity, deliberately allowed to exceed 1.0 --
    # demand is not capped just because capacity is finite, mirroring
    # simulator.congestion.DefaultCongestionModel's own
    # occupancy_ratio = other_occupants / capacity (also uncapped before
    # its own min(..., 1.0) clamp). Same explicit "project assumption,
    # not a validated standard" disclosure as DensityThresholds.

    moderate_at: float = 0.5
    high_at: float = 1.0
    very_high_at: float = 1.5
    critical_at: float = 2.0

    def classify(self, ratio: Optional[float]) -> Optional[IntensityLevel]:

        if ratio is None:
            return None

        if ratio >= self.critical_at:
            return IntensityLevel.CRITICAL

        if ratio >= self.very_high_at:
            return IntensityLevel.VERY_HIGH

        if ratio >= self.high_at:
            return IntensityLevel.HIGH

        if ratio >= self.moderate_at:
            return IntensityLevel.MODERATE

        return IntensityLevel.LOW


def compute_congestion_level(
    queue_metrics: QueueMetrics, simulation_style_capacity: Optional[int], thresholds: CongestionThresholds,
) -> Optional[IntensityLevel]:

    # Live-observable demand (queueing + actively approaching occupants)
    # against the reused SIMULATION-STYLE capacity estimate
    # (crowd_intelligence.capacity) -- explicitly two DIFFERENT signals
    # combined into one live-only classification (Phase 6's own "clearly
    # separate SIMULATION CAPACITY from LIVE ESTIMATED CONGESTION" --
    # this ratio is the live estimate; simulation_style_capacity is only
    # its denominator, never claimed to be a measured live quantity
    # itself). None whenever capacity itself is not honestly known, or
    # there is no demand evidence at all to rate.

    if simulation_style_capacity is None or simulation_style_capacity <= 0:
        return None

    demand = queue_metrics.queue_candidate_count + queue_metrics.approaching_count

    if demand == 0:
        return IntensityLevel.LOW

    ratio = demand / simulation_style_capacity

    return thresholds.classify(ratio)
