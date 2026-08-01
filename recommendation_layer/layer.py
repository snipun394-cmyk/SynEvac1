from typing import Optional

from recommendation_layer.adapters import (
    congestion_mitigation_adapter, exit_utilization_adapter, hazard_avoidance_adapter, occupant_routing_adapter,
    system_warning_adapter, warden_dispatch_adapter,
)
from recommendation_layer.manager import DEFAULT_GRACE_PERIOD_SECONDS, RecommendationManager
from recommendation_layer.models import RecommendationSet


# =====================================================
# RecommendationLayer -- the ONE public facade this package exposes.
# Every future consumer (Studio's Recommendation panel, a future
# Guidance v2, Command Center, a Dashboard) should import THIS class,
# never reach into evacuation_recommendation/advisory_system directly.
#
# compute() never raises and always returns a RecommendationSet
# (possibly empty) -- a bug in one category's adapter must never blank
# the other five; the None-means-"skip"/"failed" convention belongs to
# the live_system gateway one layer up (see live_system.
# recommendation_layer_gateway), which has actual I/O to fail on.
# =====================================================


class RecommendationLayer:

    def __init__(self, grace_period_seconds: float = DEFAULT_GRACE_PERIOD_SECONDS):

        self._manager = RecommendationManager(grace_period_seconds=grace_period_seconds)
        self._latest: Optional[RecommendationSet] = None

    # =====================================================

    @property
    def latest(self) -> Optional[RecommendationSet]:

        return self._latest

    # =====================================================

    def compute(
        self, time: float, *,
        evacuation_recommendation_snapshot=None,
        evacuation_guidance_snapshot=None,
        emergency_response_snapshot=None,
        crowd_intelligence_snapshot=None,
        ai_prediction_snapshot=None,
        advisory_report=None,
    ) -> RecommendationSet:

        candidates = []

        adapter_calls = (
            lambda: occupant_routing_adapter.adapt(evacuation_recommendation_snapshot),
            lambda: hazard_avoidance_adapter.adapt(
                evacuation_recommendation_snapshot, emergency_response_snapshot, advisory_report,
            ),
            lambda: congestion_mitigation_adapter.adapt(
                evacuation_recommendation_snapshot, crowd_intelligence_snapshot, advisory_report,
            ),
            lambda: exit_utilization_adapter.adapt(evacuation_recommendation_snapshot),
            lambda: warden_dispatch_adapter.adapt(emergency_response_snapshot, advisory_report),
            lambda: system_warning_adapter.adapt(
                evacuation_recommendation_snapshot, evacuation_guidance_snapshot, ai_prediction_snapshot,
            ),
        )

        for call in adapter_calls:

            try:
                candidates.extend(call())
            except Exception:  # noqa: BLE001 -- one category's bug must never blank the other five
                continue

        self._latest = self._manager.ingest(candidates, time)

        return self._latest
