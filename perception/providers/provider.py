from perception.models.building_observation import BuildingObservation


class PerceptionProvider:

    # The one seam every consumer of Perception (the Rule-Based
    # Decision Engine, Dataset Generation, the Firefighter Dashboard,
    # Logging/Replay, and -- only through a future ObservationEncoder
    # -- an RL Agent) is meant to depend on -- mirrors
    # hazard.provider.HazardProvider/occupancy.provider.OccupancyProvider's
    # snapshot_at(time) role exactly, one layer over on the Perception
    # side. A concrete implementation is deliberately out of scope for
    # this phase: no Ground Truth adapter, no Sensor Fusion, and no
    # Occupancy Estimation exist yet to assemble a BuildingObservation
    # out of -- this is only the contract a future implementation will
    # satisfy.

    def observation_at(self, time: float) -> BuildingObservation:

        raise NotImplementedError
