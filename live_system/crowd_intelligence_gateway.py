from typing import Optional, Protocol

from crowd_intelligence.engine import CrowdIntelligenceEngine
from crowd_intelligence.models import CrowdIntelligenceSnapshot


# =====================================================
# Live Occupancy, Crowd Density & Congestion Intelligence milestone,
# Phase 10/11 -- the seam LiveOrchestrator uses to reach crowd_
# intelligence/, mirroring live_system.live_ai_gateway's own "thin
# Protocol + real adapter, orchestrator never constructs the underlying
# package itself" shape exactly. LiveOrchestrator holds only a
# CrowdIntelligenceGateway -- never a CrowdIntelligenceEngine, Building,
# or LiveOccupantManager directly; composing those is a caller's job
# (live_runtime.factory.build_live_runtime(), the same way it already
# composes SensorFusionEngine/LivePerceptionFusionCoordinator).
#
# Investigated and rejected: stuffing this onto BuildingState itself
# (Phase 11's first suggested direction). BuildingState's own docstring
# is explicit that it only ever answers "what is true right now" via
# BuildingStateEstimator.estimate()'s already-fixed parameter list
# (facp_status/control_status are PASSTHROUGH estimate() parameters,
# not independently computed); crowd_intelligence's own computation
# happens in a wholly separate engine, never inside estimate() itself,
# and this milestone must not add a new estimate() parameter (equivalent
# to redesigning BuildingState). The established alternative already
# used by ai_prediction_snapshot/advisory_report on live_system.
# state_manager.LiveBuildingSnapshot -- a sibling, additive snapshot
# field with its own update_*()/latest_*() pair -- is architecturally
# identical to what crowd_intelligence needs, so it is reused unchanged
# here rather than inventing a third pattern.
# =====================================================


class CrowdIntelligenceGateway(Protocol):

    def compute(self, time: float) -> Optional[CrowdIntelligenceSnapshot]: ...


# =====================================================


class EngineCrowdIntelligenceGateway:

    # The real adapter -- composes an already-constructed
    # CrowdIntelligenceEngine (itself already holding the ONE shared
    # Building/LiveOccupantManager this live session uses). Never
    # allowed to raise out of compute() and crash the live cycle,
    # exactly the same "an analytics/derived-signal failure must never
    # affect the rest of the live system" discipline live_ai_gateway.
    # RegistryLiveAIInferenceGateway already established for AI
    # inference failures.

    def __init__(self, engine: CrowdIntelligenceEngine):

        self._engine = engine

    # =====================================================

    def compute(self, time: float) -> Optional[CrowdIntelligenceSnapshot]:

        try:

            return self._engine.compute(time)

        except Exception:  # noqa: BLE001 -- an unexpected crowd-intelligence failure must never crash the live cycle

            return None
