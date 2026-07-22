from typing import Optional, Protocol

from trajectory_intelligence.engine import TrajectoryIntelligenceEngine
from trajectory_intelligence.models import TrajectoryIntelligenceSnapshot


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone, Phase 25 -- the seam LiveOrchestrator uses to
# reach trajectory_intelligence/, mirroring live_system.
# crowd_intelligence_gateway/evacuation_progress_gateway/
# emergency_response_gateway's own "thin Protocol + real adapter" shape
# exactly.
# =====================================================


class TrajectoryIntelligenceGateway(Protocol):

    def compute(
        self, time: float, building_state, crowd_snapshot=None, evacuation_progress_snapshot=None,
    ) -> Optional[TrajectoryIntelligenceSnapshot]: ...


# =====================================================


class EngineTrajectoryIntelligenceGateway:

    # The real adapter -- never allowed to raise out of compute() and
    # crash the live cycle, exactly the same discipline every sibling
    # Engine*Gateway in this module already established.

    def __init__(self, engine: TrajectoryIntelligenceEngine):

        self._engine = engine

    # =====================================================

    def compute(
        self, time: float, building_state, crowd_snapshot=None, evacuation_progress_snapshot=None,
    ) -> Optional[TrajectoryIntelligenceSnapshot]:

        try:

            return self._engine.compute(time, building_state, crowd_snapshot, evacuation_progress_snapshot)

        except Exception:  # noqa: BLE001 -- an unexpected trajectory-intelligence failure must never crash the live cycle

            return None
