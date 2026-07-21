from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from crowd_intelligence.models import TrendDirection


@dataclass(frozen=True)
class TrendConfig:

    # Phase 8's own requirements: "maintain BOUNDED temporal history"
    # (max_history_length caps memory regardless of how long a
    # deployment runs) and "use configurable time windows"
    # (trend_window_seconds -- how far back "recent" looks, independent
    # of how many samples happen to have arrived in that time). Both a
    # relative and an absolute tolerance are used for the RISING/STABLE/
    # FALLING cutoff so a metric near zero (e.g. density in a nearly-
    # empty zone) does not flicker between RISING/FALLING from
    # meaningless noise -- whichever tolerance is larger applies.
    # Documented project assumptions, same disclosure as
    # DensityThresholds/CongestionThresholds.

    max_history_length: int = 20
    trend_window_seconds: float = 30.0
    stable_relative_tolerance: float = 0.10
    stable_absolute_tolerance: float = 0.05


class TrendTracker:

    # Phase 8 -- one instance is meant to be owned by crowd_intelligence.
    # engine.CrowdIntelligenceEngine (one per live session, exactly the
    # same "single shared, long-lived owner" convention live_occupants.
    # manager.LiveOccupantManager already established), keyed by
    # whatever string identifies the metric being tracked (e.g.
    # f"zone_density:{zone_id}", f"asset_congestion:{asset_id}") so one
    # tracker instance can serve every zone/door/exit/stair at once
    # without the caller managing a dict of trackers itself.

    def __init__(self, config: Optional[TrendConfig] = None):

        self.config = config if config is not None else TrendConfig()
        self._history: Dict[str, List[Tuple[float, float]]] = {}

    # =====================================================

    def observe(self, key: str, timestamp: float, value: Optional[float]) -> TrendDirection:

        if value is None:
            # No honest numeric reading this cycle -- recorded nowhere,
            # never treated as a 0.0 that would corrupt the trend of
            # whichever real readings came before/after it.
            return TrendDirection.UNKNOWN

        samples = self._history.setdefault(key, [])
        samples.append((timestamp, value))

        if len(samples) > self.config.max_history_length:
            del samples[: len(samples) - self.config.max_history_length]

        latest_time, latest_value = samples[-1]
        window_start = latest_time - self.config.trend_window_seconds

        baseline = next((sample for sample in samples if sample[0] >= window_start), None)

        if baseline is None or baseline[0] == latest_time:
            return TrendDirection.UNKNOWN

        baseline_time, baseline_value = baseline
        delta = latest_value - baseline_value

        tolerance = max(self.config.stable_absolute_tolerance, abs(baseline_value) * self.config.stable_relative_tolerance)

        if abs(delta) <= tolerance:
            return TrendDirection.STABLE

        return TrendDirection.RISING if delta > 0 else TrendDirection.FALLING
