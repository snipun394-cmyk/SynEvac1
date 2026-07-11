from dataclasses import dataclass, field
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

    # Reserved for future subsystems (Simulation, RL, Live AI Decision
    # Support) to attach transient, per-node runtime state -- e.g.
    # occupancy, smoke density, temperature, visibility. Never
    # populated or read by the Navigation Graph itself; a plain dict
    # rather than dedicated fields so those future properties can
    # arrive without another Node schema change.
    dynamic_state: dict = field(default_factory=dict)

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

    # =====================================================
    # Engineering properties -- all derived from `reference`, never
    # stored, so they can never drift out of sync with the Building
    # Model. None means "not applicable/not derivable for this node",
    # not "zero".
    # =====================================================

    @property
    def space_type(self):

        # The finer engineering classification of a Zone (Room,
        # Corridor, Lobby, ...) -- distinct from `node_type`, which is
        # only the graph's own topological category (Zone/Outside/
        # AssemblyPoint). Only Zone carries this subtype today.
        if self.node_type == self.ZONE:
            return getattr(self.reference, "zone_type", None)

        return None

    # =====================================================

    @property
    def area(self):

        # Zone and AssemblyPoint both already expose a derived `area`
        # property on their own model -- reused as-is, never
        # recomputed here.
        if self.reference is not None:
            return getattr(self.reference, "area", None)

        return None

    # =====================================================

    @property
    def capacity(self):

        # Zone and AssemblyPoint spell "how many people fit here"
        # differently (max_occupancy vs. capacity) -- this just reads
        # whichever one applies, it never stores its own value.
        if self.node_type == self.ZONE:
            return getattr(self.reference, "max_occupancy", None)

        if self.node_type == self.ASSEMBLY_POINT:
            return getattr(self.reference, "capacity", None)

        return None
