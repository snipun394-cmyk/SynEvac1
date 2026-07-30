import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from live_occupants.manager import LiveOccupantManager

from evacuation_recommendation.engine import EvacuationRecommendationEngine
from evacuation_recommendation.models import RecommendationStatus

from evacuation_guidance.engine import EvacuationGuidanceEngine
from evacuation_guidance.models import RouteStatus

from building_state.models import BuildingState


# =====================================================
# Stair Simulation Reliability & Multi-Floor Reachability Audit
# milestone -- Phases 14/15. Both engines already route through the
# SAME shared pathfinding.engine.PathfindingEngine (see evacuation_
# recommendation.ranking.SafeExitDistanceCalculator, confirmed by direct
# code trace) -- these tests are the concrete, executable proof for an
# upper-floor occupant specifically, not just an architectural
# inference. Nothing in either package needed to change.
# =====================================================


def make_zone(zone_id, floor_id, x=0.0, y=0.0, width=10.0, height=10.0):
    return Zone(id=zone_id, name=zone_id, floor_id=floor_id, x=x, y=y, width=width, height=height)


def make_upper_floor_building():

    building = Building(name="Upper Floor Recommendation Building")
    ground = building.create_floor(name="Ground", height=3.0)
    floor1 = building.create_floor(name="Floor 1", height=3.0)

    upstairs = make_zone("UPSTAIRS", floor1.id)
    lobby = make_zone("LOBBY", ground.id, x=0.0, y=0.0)
    corridor_a = make_zone("CORRIDOR-A", ground.id, x=20.0, y=0.0)
    corridor_b = make_zone("CORRIDOR-B", ground.id, x=0.0, y=500.0)

    floor1.add_zone(upstairs)
    ground.add_zone(lobby)
    ground.add_zone(corridor_a)
    ground.add_zone(corridor_b)

    ground.add_door(Door(
        id="D-A", name="D-A", floor_id=ground.id, start_point=(10.0, 5.0), end_point=(20.0, 5.0),
        zone_a_id=lobby.id, zone_b_id=corridor_a.id,
    ))
    ground.add_door(Door(
        id="D-B", name="D-B", floor_id=ground.id, start_point=(5.0, 10.0), end_point=(5.0, 500.0),
        zone_a_id=lobby.id, zone_b_id=corridor_b.id,
    ))

    # Exit A is the SHORT physical path from the stair landing; Exit B
    # is deliberately much farther on the ground floor's own geometry
    # -- but a naive floor-local-only Euclidean check could not even
    # SEE Exit B is farther, since "distance from the upper floor" has
    # no meaning without first crossing the Stair. Both exits are only
    # ever reachable from UPSTAIRS via the Stair -> Lobby -> Door path.
    ground.add_exit(Exit(id="EXIT-A", name="Exit A", zone_id=corridor_a.id, floor_id=ground.id, start_point=(20.0, 0.0), end_point=(20.0, 5.0)))
    ground.add_exit(Exit(id="EXIT-B", name="Exit B", zone_id=corridor_b.id, floor_id=ground.id, start_point=(0.0, 500.0), end_point=(0.0, 505.0)))

    stair = Staircase(
        id="STAIR-1", name="Stair", from_floor_id=ground.id, to_floor_id=floor1.id,
        from_zone_id=lobby.id, to_zone_id=upstairs.id, width=1.5,
    )
    ground.add_stair(stair)

    return building, ground, floor1, upstairs, lobby, corridor_a, corridor_b


class RecommendationUnderstandsStairRoutesTests(unittest.TestCase):

    def test_recommendation_ranks_an_exit_reachable_only_via_stair(self):

        building, ground, floor1, upstairs, lobby, corridor_a, corridor_b = make_upper_floor_building()
        graph = NavigationGraphGenerator().build(building)

        manager = LiveOccupantManager()
        manager.update("OCC-1", None, None, upstairs.id, floor1.id, None, None, None, 0.9, 0.0)

        engine = EvacuationRecommendationEngine(building, graph, manager)
        snapshot = engine.compute(0.0, building_state=BuildingState())

        recommendation = snapshot.zones[upstairs.id]

        self.assertEqual(recommendation.status, RecommendationStatus.RECOMMENDED)
        # Exit A is the genuinely shorter total route (short stair +
        # short ground-floor leg) -- proves the ranking is graph-cost-
        # based (Stair-aware), never a floor-local Euclidean shortcut
        # that would have no way to even express "via the Stair."
        self.assertEqual(recommendation.recommended_exit_id, "EXIT-A")

    def test_recommendation_migrates_when_the_only_stair_route_is_blocked(self):

        from models.obstacle import Obstacle

        building, ground, floor1, upstairs, lobby, corridor_a, corridor_b = make_upper_floor_building()

        # Block the lobby -> corridor_a door -- the exit reachable via
        # the short path from the Stair landing is no longer usable;
        # ONLY the far exit via corridor_b remains reachable at all.
        ground.obstacles.append(
            Obstacle(id="OBS-1", name="Blockage", floor_id=ground.id, x=14.0, y=4.0, length=2.0, width=2.0, active=True, traversability="Blocked"),
        )

        graph = NavigationGraphGenerator().build(building)
        manager = LiveOccupantManager()
        manager.update("OCC-1", None, None, upstairs.id, floor1.id, None, None, None, 0.9, 0.0)

        engine = EvacuationRecommendationEngine(building, graph, manager)
        snapshot = engine.compute(0.0, building_state=BuildingState())
        recommendation = snapshot.zones[upstairs.id]

        self.assertEqual(recommendation.status, RecommendationStatus.RECOMMENDED)
        self.assertEqual(recommendation.recommended_exit_id, "EXIT-B")


class GuidancePreservesStairTraversalTests(unittest.TestCase):

    def test_guidance_route_includes_the_required_stair_hop(self):

        building, ground, floor1, upstairs, lobby, corridor_a, corridor_b = make_upper_floor_building()
        graph = NavigationGraphGenerator().build(building)

        manager = LiveOccupantManager()
        manager.update("OCC-1", None, None, upstairs.id, floor1.id, None, None, None, 0.9, 0.0)

        rec_engine = EvacuationRecommendationEngine(building, graph, manager)
        rec_snapshot = rec_engine.compute(0.0, building_state=BuildingState())

        guidance_engine = EvacuationGuidanceEngine(building, graph)
        guidance_snapshot = guidance_engine.compute(0.0, rec_snapshot, BuildingState())

        plan = guidance_snapshot.zones[upstairs.id]

        self.assertEqual(plan.route_status, RouteStatus.ROUTE_AVAILABLE)
        self.assertIn("STAIR-1", plan.ordered_stair_ids)

        # The route must never jump straight from the upper floor's own
        # zone to a ground-floor zone without the Stair between them,
        # never omit the Stair, and never reference an inaccessible one.
        self.assertEqual(plan.ordered_zone_ids[0], upstairs.id)
        self.assertIn(lobby.id, plan.ordered_zone_ids)
        self.assertEqual(plan.recommended_exit_id, "EXIT-A")

    def test_guidance_never_references_an_inaccessible_stair(self):

        building, ground, floor1, upstairs, lobby, corridor_a, corridor_b = make_upper_floor_building()

        # A second, unconnected Stair (its own approach zone has no
        # Door to anything) must never appear in a guidance route.
        isolated_landing_ground = make_zone("ISOLATED-GROUND", ground.id, x=500.0, y=500.0)
        isolated_landing_up = make_zone("ISOLATED-UP", floor1.id, x=500.0, y=500.0)
        ground.add_zone(isolated_landing_ground)
        floor1.add_zone(isolated_landing_up)

        ghost_stair = Staircase(
            id="GHOST-STAIR", name="Ghost", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=isolated_landing_ground.id, to_zone_id=isolated_landing_up.id, width=1.5,
        )
        ground.add_stair(ghost_stair)

        graph = NavigationGraphGenerator().build(building)
        manager = LiveOccupantManager()
        manager.update("OCC-1", None, None, upstairs.id, floor1.id, None, None, None, 0.9, 0.0)

        rec_engine = EvacuationRecommendationEngine(building, graph, manager)
        rec_snapshot = rec_engine.compute(0.0, building_state=BuildingState())
        guidance_engine = EvacuationGuidanceEngine(building, graph)
        guidance_snapshot = guidance_engine.compute(0.0, rec_snapshot, BuildingState())

        plan = guidance_snapshot.zones[upstairs.id]

        self.assertEqual(plan.ordered_stair_ids, ("STAIR-1",))
        self.assertNotIn("GHOST-STAIR", plan.ordered_stair_ids)


if __name__ == "__main__":
    unittest.main()
