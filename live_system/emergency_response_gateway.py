from typing import Optional, Protocol

from emergency_response.engine import EmergencyResponseIntelligenceEngine
from emergency_response.models import EmergencyResponseSnapshot


# =====================================================
# Live Emergency Response & Rescue Priority Intelligence milestone,
# Phase 14 -- the seam LiveOrchestrator uses to reach emergency_
# response/, mirroring live_system.crowd_intelligence_gateway/
# evacuation_progress_gateway's own "thin Protocol + real adapter"
# shape exactly.
# =====================================================


class EmergencyResponseGateway(Protocol):

    def compute(
        self, time: float, building_state, crowd_snapshot, evacuation_progress_snapshot,
    ) -> Optional[EmergencyResponseSnapshot]: ...


# =====================================================


class EngineEmergencyResponseGateway:

    # The real adapter -- never allowed to raise out of compute() and
    # crash the live cycle, exactly the same discipline
    # EngineCrowdIntelligenceGateway/EngineEvacuationProgressGateway
    # already established.

    def __init__(self, engine: EmergencyResponseIntelligenceEngine):

        self._engine = engine

    # =====================================================

    def compute(
        self, time: float, building_state, crowd_snapshot, evacuation_progress_snapshot,
    ) -> Optional[EmergencyResponseSnapshot]:

        try:

            return self._engine.compute(time, building_state, crowd_snapshot, evacuation_progress_snapshot)

        except Exception:  # noqa: BLE001 -- an unexpected emergency-response failure must never crash the live cycle

            return None
