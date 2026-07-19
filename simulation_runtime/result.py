from dataclasses import dataclass
from typing import Optional, Tuple

from hazard.snapshot import HazardSnapshot

from occupancy.snapshot import OccupancySnapshot

from perception.models.building_observation import BuildingObservation

from ai_decision.recommendation import DecisionRecommendation

from scenario import ScenarioEvent


@dataclass(frozen=True)
class TickResult:

    # One tick's complete set of artifacts -- architecture doc
    # docs/architecture/simulation_runtime.md SS8/SS18, exactly what a
    # future Dataset Logger's on_tick() receives (SimulationRuntime.tick()
    # returns the identical bundle it hands the logger, so a caller that
    # isn't using a logger at all can still inspect every tick's outcome).
    # observation is None whenever no perception_provider was configured
    # (SS13 -- Perception composition remains an optional, injected
    # dependency this Runtime does not build).

    time: float

    fired_events: Tuple[ScenarioEvent, ...]
    hazard_snapshot: HazardSnapshot
    occupancy_snapshot: OccupancySnapshot
    decision: DecisionRecommendation
    observation: Optional[BuildingObservation] = None
