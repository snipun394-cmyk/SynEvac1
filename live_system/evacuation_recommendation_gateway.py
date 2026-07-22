from typing import Optional, Protocol

from evacuation_recommendation.engine import EvacuationRecommendationEngine
from evacuation_recommendation.models import EvacuationRecommendationSnapshot


# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone -- the seam
# LiveOrchestrator uses to reach evacuation_recommendation/, mirroring
# live_system.trajectory_intelligence_gateway/emergency_response_
# gateway's own "thin Protocol + real adapter" shape exactly.
# =====================================================


class EvacuationRecommendationGateway(Protocol):

    def compute(
        self, time: float, building_state, crowd_snapshot=None, evacuation_progress_snapshot=None,
        trajectory_snapshot=None, emergency_response_snapshot=None, ai_prediction_snapshot=None,
    ) -> Optional[EvacuationRecommendationSnapshot]: ...


# =====================================================


class EngineEvacuationRecommendationGateway:

    # The real adapter -- never allowed to raise out of compute() and
    # crash the live cycle, exactly the same discipline every sibling
    # Engine*Gateway in this module already established.

    def __init__(self, engine: EvacuationRecommendationEngine):

        self._engine = engine

    # =====================================================

    def compute(
        self, time: float, building_state, crowd_snapshot=None, evacuation_progress_snapshot=None,
        trajectory_snapshot=None, emergency_response_snapshot=None, ai_prediction_snapshot=None,
    ) -> Optional[EvacuationRecommendationSnapshot]:

        try:

            return self._engine.compute(
                time, building_state, crowd_snapshot, evacuation_progress_snapshot,
                trajectory_snapshot, emergency_response_snapshot, ai_prediction_snapshot,
            )

        except Exception:  # noqa: BLE001 -- an unexpected recommendation failure must never crash the live cycle

            return None
