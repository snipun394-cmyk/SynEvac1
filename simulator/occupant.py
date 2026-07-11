from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from pathfinding.route import Route


class OccupantState(Enum):

    PENDING = auto()        # registered, not yet departed
    AT_NODE = auto()        # holding at a node, about to attempt the next edge
    QUEUED = auto()         # waiting for edge capacity to free up
    TRAVERSING = auto()     # currently crossing an edge
    ARRIVED = auto()        # reached the final node of their route
    UNREACHABLE = auto()    # OccupantSimulator found no route at registration time
    STATIONARY = auto()     # decided not to move (e.g. WAIT/IGNORE) -- not a failure


@dataclass
class Occupant:

    # A single agent's runtime record inside a MultiAgentSimulation --
    # not an engineering model, not touching models/. Never mutated in
    # place: the route is fixed for the lifetime of *this* Occupant
    # instance, and MultiAgentSimulation only ever advances
    # current_edge_index forward through it. A later BehaviorDecision
    # for the same occupant_id (see simulator/coordinator.py's
    # generation-based supersession) replaces this instance wholesale
    # rather than editing it -- there is still no dynamic rerouting of
    # an Occupant already in flight.

    occupant_id: str
    walking_speed: float

    route: Optional[Route]
    depart_time: float

    current_edge_index: int = 0
    state: OccupantState = OccupantState.PENDING
