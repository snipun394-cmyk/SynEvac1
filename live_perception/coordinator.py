from typing import Optional, Sequence

from sensor_fusion.engine import SensorFusionEngine
from sensor_fusion.provider import ObservationProvider

from live_perception.building_state_adapter import BuildingStateInputAdapter
from live_perception.snapshot import FusedPerceptionSnapshot


class LivePerceptionFusionCoordinator:

    # Live Perception -> BuildingState Integration Bridge milestone,
    # Phase 2 -- the production composition of the three architectural
    # layers this milestone keeps deliberately separate:
    #
    #   sensor_fusion.engine.SensorFusionEngine  = generic fusion logic
    #   live_perception.providers.*               = SynEvac-specific
    #                                                Observation sources
    #   live_perception.building_state_adapter.
    #     BuildingStateInputAdapter                = FusedObservation ->
    #                                                canonical BuildingState
    #                                                inputs
    #
    # collect(time) is the ONE real fusion entry point -- it runs
    # provider collection + fusion + adaptation EXACTLY ONCE per
    # distinct `time` value, memoized, because live_system.
    # building_state_gateway.EstimatorBuildingStateGateway.collect()
    # calls hazard_snapshot_provider(time) and occupancy_snapshot_
    # provider(time) as two SEPARATE callables within the same
    # BuildingStateEstimator.estimate() call -- without memoization,
    # every cycle would silently run fusion twice for identical input.
    # hazard_snapshot_provider()/occupancy_snapshot_provider() below are
    # the exact two callables live_runtime.factory.build_live_runtime()
    # wires directly into EstimatorBuildingStateGateway's own
    # constructor.

    def __init__(
        self,
        providers: Sequence[ObservationProvider],
        engine: Optional[SensorFusionEngine] = None,
        adapter: Optional[BuildingStateInputAdapter] = None,
    ):

        self.providers = list(providers)
        self.engine = engine if engine is not None else SensorFusionEngine()
        self.adapter = adapter if adapter is not None else BuildingStateInputAdapter()

        self._cached_time: Optional[float] = None
        self._cached_snapshot: Optional[FusedPerceptionSnapshot] = None

    # =====================================================

    def collect(self, time: float) -> FusedPerceptionSnapshot:

        if self._cached_time == time and self._cached_snapshot is not None:
            return self._cached_snapshot

        observations = self.engine.collect(self.providers, time)
        fused_observations = self.engine.fuse(observations, time)

        hazard_snapshot = self.adapter.to_hazard_snapshot(fused_observations, time)
        occupancy_snapshot = self.adapter.to_occupancy_snapshot(fused_observations, time)

        snapshot = FusedPerceptionSnapshot(
            timestamp=time, fused_observations=fused_observations,
            hazard_snapshot=hazard_snapshot, occupancy_snapshot=occupancy_snapshot,
        )

        self._cached_time = time
        self._cached_snapshot = snapshot

        return snapshot

    # =====================================================
    # The two callables EstimatorBuildingStateGateway actually needs.
    # =====================================================

    def hazard_snapshot_provider(self, time: float):

        return self.collect(time).hazard_snapshot

    # =====================================================

    def occupancy_snapshot_provider(self, time: float):

        return self.collect(time).occupancy_snapshot
