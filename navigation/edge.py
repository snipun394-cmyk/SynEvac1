from dataclasses import dataclass
from typing import Any


@dataclass
class Edge:

    # An Edge is a thin, read-only view over the engineering object
    # that creates it (a Door, an Exit, or a Staircase) -- the
    # Navigation Graph never owns or duplicates Designer data. `id`
    # is always that object's own id, for the same reason Node.id
    # always reuses the Zone's id.
    #
    # Every V1 edge is inherently bidirectional (a Door, Exit, or
    # Stair can be walked both ways), so there is no separate
    # direction flag -- from_node/to_node just name the two ends.

    id: str
    edge_type: str
    from_node: str
    to_node: str
    reference: Any = None

    DOOR = "Door"
    EXIT = "Exit"
    STAIR = "Stair"

    EDGE_TYPES = (
        DOOR,
        EXIT,
        STAIR,
    )
