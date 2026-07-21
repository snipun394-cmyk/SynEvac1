from dataclasses import dataclass, field
from typing import Tuple

from hazard.snapshot import HazardSnapshot

from occupancy.snapshot import OccupancySnapshot

from sensor_fusion.observation import FusedObservation


@dataclass(frozen=True)
class FusedPerceptionSnapshot:

    # Live Perception -> BuildingState Integration Bridge milestone --
    # one live_perception.coordinator.LivePerceptionFusionCoordinator.
    # collect() call's complete result: the raw FusedObservations (kept
    # for diagnostics/future consumers -- e.g. a future dashboard
    # wanting to show ALARM-kind observations BuildingStateEstimator
    # itself never receives, see building_state_adapter.py's own
    # docstring on why) alongside the two derived, canonical snapshots
    # BuildingStateEstimator.estimate() actually consumes. Immutable,
    # matching every other snapshot type in this pipeline.

    timestamp: float
    fused_observations: Tuple[FusedObservation, ...] = field(default_factory=tuple)
    hazard_snapshot: HazardSnapshot = field(default_factory=HazardSnapshot)
    occupancy_snapshot: OccupancySnapshot = field(default_factory=OccupancySnapshot)
