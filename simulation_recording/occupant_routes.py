from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from serialization.json_reader import JsonReader
from serialization.json_writer import JsonWriter


# =====================================================
# Simulation Replay Studio V1 -- the one new seam this milestone adds
# near the frozen Simulation package. This module is a pure, read-only
# consumer of simulator.multi_agent_result.MultiAgentSimulationResult's
# already-public fields (OccupantTimeline/OccupantTimelineStep) -- it
# never imports simulator.coordinator, never constructs a
# MultiAgentSimulation, and never re-derives movement. It exists solely
# because that result is otherwise read once for aggregates (Dataset
# Builder/Ground Truth) and discarded -- nothing before this milestone
# ever persisted an occupant's own hop-by-hop route/timing to disk, so
# there was nothing for a replay viewer to draw an occupant's movement
# from.
#
# OccupantTimelineStep.from_node/to_node/edge are live Node/Edge objects
# (each carrying a `reference` back to the live engineering object, and
# Edge additionally carrying live `blocking_obstacles` references) --
# none of that is JSON-safe. Only the already-existing plain-id/scalar
# surface (`.id`, `.edge_type`, `.distance`, `.start_time`, `.end_time`,
# `.queue_wait_time`) is ever read here, the same "plain-id extraction,
# never the live graph object" discipline Route.node_ids/Route.edge_ids
# already establish.
#
# to_dict()/from_dict() + JsonWriter/JsonReader is the exact convention
# scenario_storage/storage.py already uses for persisting a Scenario
# (architecture doc: "built on serialization/json_writer.py/
# json_reader.py, not Serializer") -- restated here for this artifact,
# not imported, matching this codebase's own established habit of
# restating a small persistence convention independently per package.
# =====================================================


@dataclass(frozen=True)
class OccupantRouteHop:

    # One realized hop, exactly mirroring simulator.multi_agent_result.
    # OccupantTimelineStep's own shape -- but by plain id/scalar only,
    # never by live Node/Edge reference.

    from_node_id: str
    to_node_id: str
    edge_id: str
    edge_type: str

    start_time: float
    end_time: float

    # None only when the source Edge's own walking_distance was itself
    # None (not derivable) -- never fabricated, mirrors
    # OccupantTimelineStep.distance's own honesty.
    distance: Optional[float]

    queue_wait_time: float = 0.0

    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "distance": self.distance,
            "queue_wait_time": self.queue_wait_time,
        }

    # =====================================================

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OccupantRouteHop":

        return OccupantRouteHop(
            from_node_id=data["from_node_id"],
            to_node_id=data["to_node_id"],
            edge_id=data["edge_id"],
            edge_type=data["edge_type"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            distance=data.get("distance"),
            queue_wait_time=data.get("queue_wait_time", 0.0),
        )


@dataclass(frozen=True)
class OccupantRouteRecord:

    # One occupant's full, replayable route -- state is the source
    # OccupantTimeline.state's own .name (ARRIVED/UNREACHABLE/
    # STATIONARY/...), never re-interpreted. An occupant whose Behavior
    # Decision never required movement (STATIONARY) or was never
    # resolvable at all (UNREACHABLE) has an empty hops tuple -- exactly
    # as honest as OccupantTimeline.route being None for both of those
    # cases; a replay viewer resolves their displayed position from the
    # Scenario's own authored starting zone instead (see
    # simulation_recording.occupant_position), never fabricated here.

    occupant_id: str
    state: str
    depart_time: float
    arrival_time: Optional[float]
    hops: Tuple[OccupantRouteHop, ...] = field(default_factory=tuple)

    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "occupant_id": self.occupant_id,
            "state": self.state,
            "depart_time": self.depart_time,
            "arrival_time": self.arrival_time,
            "hops": [hop.to_dict() for hop in self.hops],
        }

    # =====================================================

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OccupantRouteRecord":

        return OccupantRouteRecord(
            occupant_id=data["occupant_id"],
            state=data["state"],
            depart_time=data["depart_time"],
            arrival_time=data.get("arrival_time"),
            hops=tuple(OccupantRouteHop.from_dict(entry) for entry in data.get("hops", [])),
        )


# =====================================================


def build_occupant_route_records(movement_result) -> Tuple[OccupantRouteRecord, ...]:

    # `movement_result` is a simulator.multi_agent_result.
    # MultiAgentSimulationResult -- this function only ever reads its
    # already-public `.occupants` mapping and the already-public fields
    # of each OccupantTimeline/OccupantTimelineStep within it. No
    # simulation is run or re-run here.

    records = []

    for occupant_id, timeline in movement_result.occupants.items():

        hops = tuple(
            OccupantRouteHop(
                from_node_id=step.from_node.id,
                to_node_id=step.to_node.id,
                edge_id=step.edge.id,
                edge_type=step.edge.edge_type,
                start_time=step.start_time,
                end_time=step.end_time,
                distance=step.distance,
                queue_wait_time=step.queue_wait_time,
            )
            for step in timeline.steps
        )

        records.append(
            OccupantRouteRecord(
                occupant_id=occupant_id,
                state=timeline.state.name,
                depart_time=timeline.depart_time,
                arrival_time=timeline.arrival_time,
                hops=hops,
            )
        )

    return tuple(records)


# =====================================================


def save_occupant_routes(records: Tuple[OccupantRouteRecord, ...], path: str) -> None:

    JsonWriter.write(path, [record.to_dict() for record in records])


def load_occupant_routes(path: str) -> Tuple[OccupantRouteRecord, ...]:

    data = JsonReader.read(path)

    return tuple(OccupantRouteRecord.from_dict(entry) for entry in data)
