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

    # Node Types -- V1 scope only. More will be added (e.g. "Stair
    # Landing") only if a future version needs a distinct navigable
    # space that isn't a Zone.
    ZONE = "Zone"
    OUTSIDE = "Outside"

    NODE_TYPES = (
        ZONE,
        OUTSIDE,
    )

    # The whole graph shares exactly one "Outside" node -- every Exit
    # leads to the same, single exterior world, so there is no per-
    # floor or per-Exit Outside node. Fixed rather than generated so
    # it stays stable/predictable across rebuilds and in tests.
    OUTSIDE_NODE_ID = "outside"
