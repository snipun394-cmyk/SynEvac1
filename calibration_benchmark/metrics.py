from dataclasses import dataclass
from typing import Any, Dict, Optional

from ground_truth.analyzer import ordered_doors, ordered_exits, ordered_stairs
from navigation.edge import Edge


# =====================================================
# Calibration Benchmark Framework V1 -- the five paired-comparison
# metrics this milestone's own worked example names (Evacuation Time,
# Maximum Density, Congestion Duration, Queue Length, Exit Utilization).
# Every field here is read directly off objects a normal simulation run
# already produces (GroundTruth, MultiAgentSimulationResult) -- nothing
# is resimulated or independently recomputed except "Maximum Density",
# which GroundTruth has no field for at all (it tracks a peak
# *occupancy count*, never an areal people/m^2 density -- see
# ground_truth/bottleneck.py's own compute_congestion()). That one
# metric is therefore an explicitly-labeled PROXY: the worst
# occupancy-ratio (peak occupants on an edge / that edge's own modeled
# capacity) across every Door/Exit/Stair in the building, using
# whichever CapacityModel the run itself used -- the same
# occupancy_ratio concept simulator.congestion.DefaultCongestionModel
# already computes internally, just read back out and maximized across
# every edge instead of one.
# =====================================================


@dataclass(frozen=True)
class MetricSample:

    scenario_id: str
    evacuation_time: Optional[float]
    congestion_duration: Optional[float]
    queue_length: Optional[int]
    peak_occupancy_ratio: Optional[float]
    exit_utilization_balance: Optional[float]
    people_trapped: int
    people_evacuated: int

    def to_dict(self) -> Dict[str, Any]:

        return {
            "scenario_id": self.scenario_id,
            "evacuation_time": self.evacuation_time,
            "congestion_duration": self.congestion_duration,
            "queue_length": self.queue_length,
            "peak_occupancy_ratio": self.peak_occupancy_ratio,
            "exit_utilization_balance": self.exit_utilization_balance,
            "people_trapped": self.people_trapped,
            "people_evacuated": self.people_evacuated,
        }


METRIC_FIELDS = (
    "evacuation_time", "congestion_duration", "queue_length",
    "peak_occupancy_ratio", "exit_utilization_balance",
)

METRIC_LABELS = {
    "evacuation_time": "Evacuation Time (s)",
    "congestion_duration": "Congestion Duration (s)",
    "queue_length": "Queue Length (peak simultaneous occupants, worst edge)",
    "peak_occupancy_ratio": "Maximum Density (peak occupancy ratio, proxy)",
    "exit_utilization_balance": "Exit Utilization Balance (1.0 = every exit used near capacity)",
}


# =====================================================


def _edge_for(obj, edge_type, walking_distance=None) -> Edge:

    return Edge(
        id=obj.id, edge_type=edge_type, from_node="", to_node="", reference=obj,
        walking_distance=walking_distance,
    )


def compute_peak_occupancy_ratio(building, movement_result, capacity_model) -> Optional[float]:

    ratios = []

    for door in ordered_doors(building):

        capacity = capacity_model.capacity(_edge_for(door, Edge.DOOR))
        if capacity:
            ratios.append(movement_result.peak_edge_occupancy.get(door.id, 0) / capacity)

    for exit_obj in ordered_exits(building):

        capacity = capacity_model.capacity(_edge_for(exit_obj, Edge.EXIT))
        if capacity:
            ratios.append(movement_result.peak_edge_occupancy.get(exit_obj.id, 0) / capacity)

    for stair in ordered_stairs(building):

        walking_distance = stair.travel_distance(building)
        capacity = capacity_model.capacity(_edge_for(stair, Edge.STAIR, walking_distance=walking_distance))
        if capacity:
            ratios.append(movement_result.peak_edge_occupancy.get(stair.id, 0) / capacity)

    return max(ratios) if ratios else None


def compute_exit_utilization_balance(ground_truth, building) -> Optional[float]:

    total_exits = len(ordered_exits(building))

    if total_exits == 0:
        return None

    imbalanced = set(ground_truth.exits_underutilized) | set(ground_truth.exits_exceeding_capacity)

    return 1.0 - (len(imbalanced) / total_exits)


def extract_metrics(scenario_id: str, ground_truth, movement_result, building, capacity_model) -> MetricSample:

    return MetricSample(
        scenario_id=scenario_id,
        evacuation_time=ground_truth.total_evacuation_time,
        congestion_duration=ground_truth.congestion_duration,
        queue_length=ground_truth.peak_congestion_value,
        peak_occupancy_ratio=compute_peak_occupancy_ratio(building, movement_result, capacity_model),
        exit_utilization_balance=compute_exit_utilization_balance(ground_truth, building),
        people_trapped=ground_truth.people_trapped,
        people_evacuated=ground_truth.people_evacuated,
    )
