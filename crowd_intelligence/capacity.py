from typing import Optional

from navigation.edge import Edge

from simulator.capacity import DefaultCapacityModel, StairCapacityModel


# =====================================================
# Phase 6 -- SIMULATION-STYLE capacity, reused directly from simulator.
# capacity rather than re-derived (Phase 6's own "investigate reusing
# DefaultCapacityModel/StairCapacityModel"). navigation.edge.Edge is a
# thin, stateless reference wrapper (its width/capacity properties are
# plain getattr passthroughs onto the Door/Exit/Staircase itself) -- no
# NavigationGraph, no MultiAgentSimulation, no simulation state of any
# kind is constructed here, only this one read-only view object.
#
# Named and returned as `simulation_style_capacity` everywhere it is
# consumed (crowd_intelligence.models.AssetApproachMetrics) specifically
# so it is never confused with a measured live flow rate -- it is the
# same documented, non-validated ENGINEERING ESTIMATE simulator.capacity
# itself already discloses ("not a validated life-safety flow-rate
# model, just a reasonable default"), carried over unchanged. LIVE
# ESTIMATED CONGESTION (crowd_intelligence.congestion) is a distinct,
# separately-computed signal that uses this capacity as only ONE input
# among several -- the two are related but never treated as identical
# (Phase 6's own explicit requirement).
# =====================================================


_capacity_model = DefaultCapacityModel()
_stair_capacity_model = StairCapacityModel()


def door_capacity(door) -> Optional[int]:

    edge = Edge(id=door.id, edge_type=Edge.DOOR, from_node="", to_node="", reference=door)

    return _capacity_model.capacity(edge)


def exit_capacity(exit_obj) -> Optional[int]:

    edge = Edge(id=exit_obj.id, edge_type=Edge.EXIT, from_node="", to_node="", reference=exit_obj)

    return _capacity_model.capacity(edge)


def stair_capacity(stair, building) -> Optional[int]:

    walking_distance = stair.travel_distance(building)

    edge = Edge(
        id=stair.id, edge_type=Edge.STAIR, from_node="", to_node="", reference=stair, walking_distance=walking_distance,
    )

    return _stair_capacity_model.capacity(edge)
