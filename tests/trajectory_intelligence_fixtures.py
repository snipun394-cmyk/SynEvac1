from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from live_occupants.manager import LiveOccupantManager

from live_system.event_bus import EventBus

from building_state.models import BuildingState, HazardSummary

from trajectory_intelligence.engine import TrajectoryIntelligenceEngine
from trajectory_intelligence.models import TrajectoryConfig


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone -- one shared, hand-built fixture, mirroring
# tests.live_runtime_fixtures/tests.test_emergency_response's own
# established "one shared fixture module per concern" convention.
#
# Topology:
#
#   Floor f1: z1 (Lobby, has EXIT-1) -- DOOR-1 -- z2 (Hall) -- DOOR-2 -- z4 (Annex, has EXIT-2)
#   Floor f2: z3 (Room2F) -- STAIR-1 -- z1 (f1)
#
# Two independent routes to Outside exist from z2/z4 (via EXIT-1
# through z1, or via EXIT-2 directly) -- deliberately, so an unsafe-zone
# exclusion test can prove an alternate route is still found rather than
# collapsing straight to NO_SAFE_ROUTE.
# =====================================================


def make_trajectory_building() -> Building:

    f1 = Floor(
        id="f1", name="Floor 1",
        zones=[
            Zone(id="z1", name="Lobby", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            Zone(id="z2", name="Hall", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            Zone(id="z4", name="Annex", x=40.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
        ],
        doors=[
            Door(id="DOOR-1", name="Lobby-Hall Door", floor_id="f1", zone_a_id="z1", zone_b_id="z2"),
            Door(id="DOOR-2", name="Hall-Annex Door", floor_id="f1", zone_a_id="z2", zone_b_id="z4"),
        ],
        exits=[
            Exit(id="EXIT-1", name="Main Exit", floor_id="f1", zone_id="z1"),
            Exit(id="EXIT-2", name="Annex Exit", floor_id="f1", zone_id="z4"),
        ],
    )

    f2 = Floor(
        id="f2", name="Floor 2",
        zones=[
            Zone(id="z3", name="Room2F", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f2"),
        ],
        stairs=[
            Staircase(id="STAIR-1", name="Stair 1", from_zone_id="z3", to_zone_id="z1", to_floor_id="f1"),
        ],
    )

    return Building(id="trajectory-test-building", name="Trajectory Test Building", floors=[f1, f2])


def make_trajectory_graph(building=None):

    building = building if building is not None else make_trajectory_building()

    return NavigationGraphGenerator().build(building)


def make_engine(building=None, graph=None, config=None, manager=None, event_bus=None):

    building = building if building is not None else make_trajectory_building()
    graph = graph if graph is not None else make_trajectory_graph(building)
    event_bus = event_bus if event_bus is not None else EventBus()
    manager = manager if manager is not None else LiveOccupantManager(event_bus=event_bus, exits=[], expire_after_seconds=100000.0)
    config = config if config is not None else TrajectoryConfig()

    engine = TrajectoryIntelligenceEngine(building, graph, manager, config=config)

    return engine, manager


def make_building_state(hazard_severity_by_zone=None):

    hazard_summary = HazardSummary(zone_severities=dict(hazard_severity_by_zone or {}))

    return BuildingState(hazard_summary=hazard_summary)
