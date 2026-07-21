from typing import Optional, Protocol

from evacuation_progress.engine import EvacuationProgressEngine
from evacuation_progress.models import EvacuationProgressSnapshot


# =====================================================
# Live Evacuation Progress, Flow & Clearance Intelligence milestone,
# Phase 11 -- the seam LiveOrchestrator uses to reach evacuation_
# progress/, mirroring live_system.crowd_intelligence_gateway's own
# "thin Protocol + real adapter, orchestrator never constructs the
# underlying package itself" shape exactly. LiveOrchestrator holds only
# an EvacuationProgressGateway -- never an EvacuationProgressEngine,
# Building, or LiveOccupantManager directly; composing those is a
# caller's job (live_runtime.factory.build_live_runtime()).
# =====================================================


class EvacuationProgressGateway(Protocol):

    def compute(self, time: float, building_state, crowd_snapshot) -> Optional[EvacuationProgressSnapshot]: ...


# =====================================================


class EngineEvacuationProgressGateway:

    # The real adapter -- never allowed to raise out of compute() and
    # crash the live cycle, exactly the same discipline
    # EngineCrowdIntelligenceGateway already established.

    def __init__(self, engine: EvacuationProgressEngine):

        self._engine = engine

    # =====================================================

    def compute(self, time: float, building_state, crowd_snapshot) -> Optional[EvacuationProgressSnapshot]:

        try:

            return self._engine.compute(time, building_state, crowd_snapshot)

        except Exception:  # noqa: BLE001 -- an unexpected evacuation-progress failure must never crash the live cycle

            return None
