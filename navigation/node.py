from dataclasses import dataclass
from typing import Any


@dataclass
class Node:

    # A Node is a thin, read-only view over an engineering object (or,
    # for "Outside", over nothing at all) -- the Navigation Graph never
    # owns or duplicates Designer data. `id` is always the underlying
    # model's own id (e.g. the Zone's id), never a freshly generated
    # one, so a Node and the object it represents are always trivially
    # the same thing looked at two ways.

    id: str
    name: str
    floor_id: str
    node_type: str
    reference: Any = None

    # Node Types. ASSEMBLY_POINT is the same string as
    # connectable_space.ASSEMBLY_POINT (models/connectable_space.py) --
    # that module is the registry of what a Door can connect; this is
    # the graph's own type tag for the Node it produces. More will be
    # added (e.g. "Outdoor Area") only if a future version needs
    # another distinct navigable space.
    ZONE = "Zone"
    OUTSIDE = "Outside"
    ASSEMBLY_POINT = "AssemblyPoint"

    NODE_TYPES = (
        ZONE,
        OUTSIDE,
        ASSEMBLY_POINT,
    )

    # The whole graph shares exactly one "Outside" node -- every Exit
    # leads to the same, single exterior world, so there is no per-
    # floor or per-Exit Outside node. Fixed rather than generated so
    # it stays stable/predictable across rebuilds and in tests.
    OUTSIDE_NODE_ID = "outside"
