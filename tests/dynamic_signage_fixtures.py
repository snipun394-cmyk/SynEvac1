from models.building import Building
from models.door import Door
from models.dynamic_sign import DynamicEvacuationSign
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from evacuation_guidance.engine import EvacuationGuidanceEngine
from evacuation_guidance.models import GuidanceConfig

from evacuation_recommendation.models import EvacuationRecommendationSnapshot, RecommendationStatus, ZoneEvacuationRecommendation


# =====================================================
# Live Dynamic Evacuation Signage milestone -- a small, hand-built
# fixture with REAL door/exit/stair geometry (unlike tests.
# trajectory_intelligence_fixtures.make_trajectory_building, whose
# doors/exits are pure topology with every position defaulted to
# (0.0, 0.0) -- fine for graph-shape tests, useless for direction
# geometry, which is exactly what this milestone needs to verify).
#
# Topology:
#
#   Floor f1: z1 (Lobby, has EXIT-1 at (0, 5)) -- DOOR-1 at (15, 5) --
#             z2 (Hall, spans x 20-30)
#   Floor f2: z3 (Room2F) -- STAIR-1 -- z1 (landing at (8, 8) on f1)
#
# All positions in meters, (x, y) with y increasing downward -- the
# same convention models.camera.Camera.coverage_polygon()/models.
# dynamic_sign.DynamicEvacuationSign.orientation already use.
# =====================================================


def make_signage_building() -> Building:

    f1 = Floor(
        id="f1", name="Floor 1",
        zones=[
            Zone(id="z1", name="Lobby", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            Zone(id="z2", name="Hall", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
        ],
        doors=[
            Door(id="DOOR-1", name="Lobby-Hall Door", floor_id="f1", zone_a_id="z1", zone_b_id="z2",
                 start_point=(15.0, 3.0), end_point=(15.0, 7.0)),
        ],
        exits=[
            Exit(id="EXIT-1", name="Main Exit", floor_id="f1", zone_id="z1",
                 start_point=(0.0, 3.0), end_point=(0.0, 7.0)),
        ],
    )

    f2 = Floor(
        id="f2", name="Floor 2",
        zones=[
            Zone(id="z3", name="Room2F", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f2"),
        ],
        stairs=[
            Staircase(id="STAIR-1", name="Stair 1", from_zone_id="z3", to_zone_id="z1",
                      from_floor_id="f2", to_floor_id="f1",
                      from_position=(5.0, 5.0), to_position=(8.0, 8.0)),
        ],
    )

    return Building(id="signage-test-building", name="Signage Test Building", floors=[f1, f2])


def make_signage_graph(building=None):

    building = building if building is not None else make_signage_building()

    return NavigationGraphGenerator().build(building)


def make_signage_engine(building=None, graph=None, config=None):

    building = building if building is not None else make_signage_building()
    graph = graph if graph is not None else make_signage_graph(building)
    config = config if config is not None else GuidanceConfig()

    return EvacuationGuidanceEngine(building, graph, config=config)


def make_recommendation_snapshot(zone_id, floor_id, exit_id, timestamp=0.0, status=RecommendationStatus.RECOMMENDED, confidence=0.8):

    recommendation = ZoneEvacuationRecommendation(
        zone_id=zone_id, floor_id=floor_id, status=status, recommended_exit_id=exit_id,
        ranked_exit_ids=(exit_id,) if exit_id else (), confidence=confidence, occupant_count=1, timestamp=timestamp,
    )

    return EvacuationRecommendationSnapshot(timestamp=timestamp, zones={zone_id: recommendation}, safe_exit_ids=(exit_id,) if exit_id else ())


def make_guidance_snapshot(building=None, graph=None, zone_id="z2", floor_id="f1", exit_id="EXIT-1", time=0.0, building_state=None, engine=None):

    engine = engine if engine is not None else make_signage_engine(building, graph)
    recommendation = make_recommendation_snapshot(zone_id, floor_id, exit_id, timestamp=time)

    return engine, engine.compute(time, recommendation, building_state)


def make_sign(sign_id, floor_id="f1", zone_ids=("z2",), position=(22.0, 5.0), orientation=180.0, active=True, supported_indications=None):

    kwargs = dict(
        id=sign_id, name=sign_id, floor_id=floor_id, zone_ids=tuple(zone_ids),
        position=position, orientation=orientation, active=active,
    )

    if supported_indications is not None:
        kwargs["supported_indications"] = tuple(supported_indications)

    return DynamicEvacuationSign(**kwargs)
