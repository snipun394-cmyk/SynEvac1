from typing import Optional, Protocol

from recommendation_layer.layer import RecommendationLayer
from recommendation_layer.models import RecommendationSet


# =====================================================
# The Recommendation Layer milestone -- the seam LiveOrchestrator uses
# to reach recommendation_layer/, mirroring live_system.evacuation_
# recommendation_gateway's own "thin Protocol + real adapter" shape
# exactly.
# =====================================================


class RecommendationLayerGateway(Protocol):

    def compute(
        self, time: float, evacuation_recommendation_snapshot=None, evacuation_guidance_snapshot=None,
        emergency_response_snapshot=None, crowd_intelligence_snapshot=None, ai_prediction_snapshot=None,
        advisory_report=None,
    ) -> Optional[RecommendationSet]: ...


# =====================================================


class EngineRecommendationLayerGateway:

    # The real adapter -- never allowed to raise out of compute() and
    # crash the live cycle, exactly the same discipline every sibling
    # Engine*Gateway in live_system/ already established.

    def __init__(self, layer: RecommendationLayer):

        self._layer = layer

    # =====================================================

    def compute(
        self, time: float, evacuation_recommendation_snapshot=None, evacuation_guidance_snapshot=None,
        emergency_response_snapshot=None, crowd_intelligence_snapshot=None, ai_prediction_snapshot=None,
        advisory_report=None,
    ) -> Optional[RecommendationSet]:

        try:

            return self._layer.compute(
                time,
                evacuation_recommendation_snapshot=evacuation_recommendation_snapshot,
                evacuation_guidance_snapshot=evacuation_guidance_snapshot,
                emergency_response_snapshot=emergency_response_snapshot,
                crowd_intelligence_snapshot=crowd_intelligence_snapshot,
                ai_prediction_snapshot=ai_prediction_snapshot,
                advisory_report=advisory_report,
            )

        except Exception:  # noqa: BLE001 -- an unexpected recommendation-layer failure must never crash the live cycle

            return None
