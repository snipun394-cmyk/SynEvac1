from typing import Optional, Protocol

from evacuation_guidance.engine import EvacuationGuidanceEngine
from evacuation_guidance.models import EvacuationGuidanceSnapshot


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone -- the
# seam LiveOrchestrator uses to reach evacuation_guidance/, mirroring
# live_system.evacuation_recommendation_gateway/trajectory_intelligence_
# gateway's own "thin Protocol + real adapter" shape exactly.
# =====================================================


class EvacuationGuidanceGateway(Protocol):

    def compute(
        self, time: float, recommendation_snapshot=None, building_state=None, speaker_manager=None,
    ) -> Optional[EvacuationGuidanceSnapshot]: ...


# =====================================================


class EngineEvacuationGuidanceGateway:

    # The real adapter -- never allowed to raise out of compute() and
    # crash the live cycle, exactly the same discipline every sibling
    # Engine*Gateway in this module already established. speaker_manager
    # is forwarded through unchanged, duck-typed (see evacuation_
    # guidance.message_planner.speaker_coverage_for_zone) -- this
    # gateway never imports speaker_manager/ itself either.

    def __init__(self, engine: EvacuationGuidanceEngine, speaker_manager=None):

        self._engine = engine
        self._speaker_manager = speaker_manager

    # =====================================================

    def compute(
        self, time: float, recommendation_snapshot=None, building_state=None, speaker_manager=None,
    ) -> Optional[EvacuationGuidanceSnapshot]:

        try:

            return self._engine.compute(
                time, recommendation_snapshot, building_state,
                speaker_manager if speaker_manager is not None else self._speaker_manager,
            )

        except Exception:  # noqa: BLE001 -- an unexpected guidance failure must never crash the live cycle

            return None
