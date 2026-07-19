import unittest

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from models.building import Building
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge
from navigation.node import Node

from occupancy.snapshot import OccupancySnapshot

from pathfinding.route import Route

from scenario.fire import ScenarioFire
from scenario.metadata import ScenarioMetadata
from scenario.occupant import ScenarioOccupant
from scenario.scenario import Scenario

from simulator.multi_agent_result import (
    MultiAgentSimulationResult,
    OccupantTimeline,
    OccupantTimelineStep,
)
from simulator.occupant import OccupantState

from ground_truth import GroundTruth, SimulationArtifacts, analyze
from ground_truth import bottleneck, evacuation_metrics, risk_analysis
from ground_truth.recommendations import generate_recommendations


# =====================================================
# Fixtures. Everything is hand-built (no real pathfinder/simulator
# run) since ground_truth only ever reads an already-completed result.
# =====================================================


def make_building():

    floor_1 = Floor(
        name="Ground", id="floor-1", display_order=0,
        zones=[
            Zone(id="zone-0", name="Zone 0", x=0.0, y=0.0, width=4.0, height=5.0, floor_id="floor-1"),
            Zone(id="zone-0b", name="Zone 0b", x=5.0, y=0.0, width=4.0, height=5.0, floor_id="floor-1"),
        ],
        doors=[
            Door(id="door-0", floor_id="floor-1", zone_a_id="zone-0", zone_b_id="zone-0b"),
        ],
        exits=[
            Exit(id="exit-0", floor_id="floor-1", zone_id="zone-0b", capacity=1, is_blocked=False),
            Exit(id="exit-1", floor_id="floor-1", zone_id="zone-0b", capacity=5, is_blocked=True),
        ],
        stairs=[
            Staircase(
                id="stair-0", from_floor_id="floor-1", to_floor_id="floor-2",
                from_zone_id="zone-0b", to_zone_id="zone-1",
            ),
        ],
    )

    floor_2 = Floor(
        name="First Floor", id="floor-2", display_order=1,
        zones=[
            Zone(id="zone-1", name="Zone 1", x=0.0, y=0.0, width=4.0, height=5.0, floor_id="floor-2"),
        ],
        exits=[
            Exit(id="exit-2", floor_id="floor-2", zone_id="zone-1", capacity=5, is_blocked=False),
        ],
        detectors=[Detector(id="detector-0", floor_id="floor-2", active=True)],
        cameras=[Camera(id="camera-0", floor_id="floor-2", active=True)],
    )

    return Building(name="Test Building", id="building-1", floors=[floor_1, floor_2])


def make_scenario(building):

    metadata = ScenarioMetadata(
        scenario_id="scn-1", definition_id="def-1", definition_content_hash="hash-1",
        generation_version="v1", seed=7, created_at="2026-01-01T00:00:00",
    )

    fire = ScenarioFire(
        ignition_zone_id="zone-0", ignition_floor_id="floor-1",
        fire_profile="Electrical", growth_parameters={"growth_time": 200.0},
    )

    occupants = (
        ScenarioOccupant(
            occupant_id="occ-1", zone_id="zone-0", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-2", zone_id="zone-0", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-3", zone_id="zone-1", floor_id="floor-2",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-4", zone_id="zone-0b", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-5", zone_id="zone-0b", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
    )

    return Scenario(metadata=metadata, occupants=occupants, fire=fire)


def _node(node_id, floor_id, node_type=Node.ZONE):

    return Node(id=node_id, name=node_id, floor_id=floor_id, node_type=node_type)


def _outside():

    return Node(id=Node.OUTSIDE_NODE_ID, name="Outside", floor_id="", node_type=Node.OUTSIDE)


def make_movement_result(building):

    exit_0 = building.floors[0].exits[0]
    exit_1 = building.floors[0].exits[1]
    exit_2 = building.floors[1].exits[0]
    door_0 = building.floors[0].doors[0]
    stair_0 = building.floors[0].stairs[0]

    zone_0 = _node("zone-0", "floor-1")
    zone_0b = _node("zone-0b", "floor-1")
    zone_1 = _node("zone-1", "floor-2")
    outside = _outside()

    door_edge = Edge(id="door-0", edge_type=Edge.DOOR, from_node="zone-0", to_node="zone-0b",
                      walking_distance=2.0, reference=door_0)
    exit_0_edge = Edge(id="exit-0", edge_type=Edge.EXIT, from_node="zone-0b", to_node="outside",
                        walking_distance=6.0, reference=exit_0)
    stair_edge = Edge(id="stair-0", edge_type=Edge.STAIR, from_node="zone-0b", to_node="zone-1",
                       walking_distance=5.0, reference=stair_0)
    exit_2_edge = Edge(id="exit-2", edge_type=Edge.EXIT, from_node="zone-1", to_node="outside",
                        walking_distance=3.0, reference=exit_2)

    def door_exit_route():
        return Route(
            nodes=[zone_0, zone_0b, outside], edges=[door_edge, exit_0_edge],
            total_cost=8.0, total_distance=8.0,
        )

    def stair_exit_route():
        return Route(
            nodes=[zone_0b, zone_1, outside], edges=[stair_edge, exit_2_edge],
            total_cost=8.0, total_distance=8.0,
        )

    occ_1 = OccupantTimeline(
        occupant_id="occ-1", route=door_exit_route(),
        steps=[
            OccupantTimelineStep(index=0, from_node=zone_0, to_node=zone_0b, edge=door_edge,
                                  queue_wait_time=1.0, start_time=1.0, end_time=2.0),
            OccupantTimelineStep(index=1, from_node=zone_0b, to_node=outside, edge=exit_0_edge,
                                  queue_wait_time=0.0, start_time=2.0, end_time=10.0),
        ],
        state=OccupantState.ARRIVED, depart_time=0.0, arrival_time=10.0,
    )

    occ_2 = OccupantTimeline(
        occupant_id="occ-2", route=door_exit_route(),
        steps=[
            OccupantTimelineStep(index=0, from_node=zone_0, to_node=zone_0b, edge=door_edge,
                                  queue_wait_time=0.0, start_time=3.0, end_time=4.0),
            OccupantTimelineStep(index=1, from_node=zone_0b, to_node=outside, edge=exit_0_edge,
                                  queue_wait_time=2.0, start_time=4.0, end_time=12.0),
        ],
        state=OccupantState.ARRIVED, depart_time=3.0, arrival_time=12.0,
    )

    occ_3 = OccupantTimeline(
        occupant_id="occ-3", route=None, steps=[],
        state=OccupantState.UNREACHABLE, depart_time=0.0, arrival_time=None,
    )

    occ_4 = OccupantTimeline(
        occupant_id="occ-4", route=stair_exit_route(),
        steps=[
            OccupantTimelineStep(index=0, from_node=zone_0b, to_node=zone_1, edge=stair_edge,
                                  queue_wait_time=3.0, start_time=0.0, end_time=5.0),
            OccupantTimelineStep(index=1, from_node=zone_1, to_node=outside, edge=exit_2_edge,
                                  queue_wait_time=0.0, start_time=5.0, end_time=8.0),
        ],
        state=OccupantState.ARRIVED, depart_time=0.0, arrival_time=8.0,
    )

    occ_5 = OccupantTimeline(
        occupant_id="occ-5", route=stair_exit_route(),
        steps=[
            OccupantTimelineStep(index=0, from_node=zone_0b, to_node=zone_1, edge=stair_edge,
                                  queue_wait_time=0.0, start_time=1.0, end_time=6.0),
            OccupantTimelineStep(index=1, from_node=zone_1, to_node=outside, edge=exit_2_edge,
                                  queue_wait_time=0.0, start_time=6.0, end_time=9.0),
        ],
        state=OccupantState.ARRIVED, depart_time=1.0, arrival_time=9.0,
    )

    return MultiAgentSimulationResult(
        occupants={
            "occ-1": occ_1, "occ-2": occ_2, "occ-3": occ_3, "occ-4": occ_4, "occ-5": occ_5,
        },
        total_evacuation_time=12.0,
        unreachable_occupant_ids=["occ-3"],
        peak_edge_occupancy={"door-0": 1, "exit-0": 2, "stair-0": 2, "exit-2": 2},
        peak_node_occupancy={},
    )


def make_tick_results():

    from ai_decision.recommendation import DecisionRecommendation
    from simulation_runtime.result import TickResult

    tick_1 = TickResult(
        time=5.0, fired_events=(),
        hazard_snapshot=HazardSnapshot(node_states={"zone-0": HazardNodeState(hazard_score=0.3)}),
        occupancy_snapshot=OccupancySnapshot(),
        decision=DecisionRecommendation(),
    )
    tick_2 = TickResult(
        time=10.0, fired_events=(),
        hazard_snapshot=HazardSnapshot(node_states={"zone-0": HazardNodeState(hazard_score=1.0)}),
        occupancy_snapshot=OccupancySnapshot(),
        decision=DecisionRecommendation(),
    )

    return (tick_1, tick_2)


def make_artifacts(with_tick_results=True):

    building = make_building()
    scenario = make_scenario(building)
    movement_result = make_movement_result(building)
    tick_results = make_tick_results() if with_tick_results else ()

    return SimulationArtifacts(
        scenario=scenario, building=building, movement_result=movement_result,
        tick_results=tick_results,
    )


# =====================================================


class EvacuationMetricsTests(unittest.TestCase):

    def test_evacuation_counts(self):

        ground_truth = analyze(make_artifacts())

        self.assertEqual(ground_truth.people_evacuated, 4)
        self.assertEqual(ground_truth.people_trapped, 1)
        self.assertEqual(ground_truth.reachable_occupants, 4)
        self.assertEqual(ground_truth.unreachable_occupants, 1)
        self.assertFalse(ground_truth.building_cleared)
        self.assertEqual(ground_truth.total_evacuation_time, 12.0)


class RouteStatisticsTests(unittest.TestCase):

    def setUp(self):

        self.ground_truth = analyze(make_artifacts())
        self.stats_by_zone = {
            entry["zone_id"]: entry for entry in self.ground_truth.zone_route_stats
        }

    def test_preferred_exit_and_stair(self):

        self.assertEqual(self.stats_by_zone["zone-0"]["preferred_exit"], "exit-0")
        self.assertIsNone(self.stats_by_zone["zone-0"]["preferred_stair"])

        self.assertEqual(self.stats_by_zone["zone-0b"]["preferred_exit"], "exit-2")
        self.assertEqual(self.stats_by_zone["zone-0b"]["preferred_stair"], "stair-0")

        self.assertIsNone(self.stats_by_zone["zone-1"]["preferred_exit"])

    def test_average_travel_distance_and_time(self):

        self.assertAlmostEqual(self.stats_by_zone["zone-0"]["average_travel_distance"], 8.0)
        self.assertAlmostEqual(self.stats_by_zone["zone-0"]["average_travel_time"], 9.5)

        self.assertAlmostEqual(self.stats_by_zone["zone-0b"]["average_travel_distance"], 8.0)
        self.assertAlmostEqual(self.stats_by_zone["zone-0b"]["average_travel_time"], 8.0)

        self.assertIsNone(self.stats_by_zone["zone-1"]["average_travel_distance"])
        self.assertIsNone(self.stats_by_zone["zone-1"]["average_travel_time"])


class BottleneckDetectionTests(unittest.TestCase):

    def setUp(self):

        self.ground_truth = analyze(make_artifacts())

    def test_worst_objects(self):

        self.assertEqual(self.ground_truth.worst_exit, "exit-0")
        self.assertEqual(self.ground_truth.worst_stair, "stair-0")
        self.assertEqual(self.ground_truth.worst_door, "door-0")

    def test_peak_congestion_location_and_duration(self):

        self.assertEqual(self.ground_truth.peak_congestion_location_id, "exit-0")
        self.assertEqual(self.ground_truth.peak_congestion_location_type, "exit")
        self.assertEqual(self.ground_truth.peak_congestion_value, 2)
        self.assertAlmostEqual(self.ground_truth.congestion_duration, 6.0)

    def test_engineering_findings(self):

        self.assertEqual(list(self.ground_truth.doors_that_became_bottlenecks), ["door-0"])
        self.assertEqual(list(self.ground_truth.exits_underutilized), ["exit-1"])
        self.assertEqual(list(self.ground_truth.exits_exceeding_capacity), ["exit-0"])
        # stair-0: width 1.5 -> base 1.5*1.2=1.8, walking_distance 5.0
        # -> one 4.0m penalty step, derived capacity floors to 1;
        # peak_edge_occupancy 2 > 1, so it is flagged.
        self.assertEqual(list(self.ground_truth.stairs_exceeding_capacity), ["stair-0"])

    def test_congestion_duration_zero_for_an_edge_never_shared(self):

        building = make_building()
        movement_result = make_movement_result(building)

        duration = bottleneck._congestion_duration_for_edge(movement_result, "door-0")

        self.assertEqual(duration, 0.0)


class RiskScoreTests(unittest.TestCase):

    def setUp(self):

        self.ground_truth = analyze(make_artifacts())
        self.zone_risk = {e["zone_id"]: e["risk_score"] for e in self.ground_truth.zone_risk_scores}
        self.exit_risk = {e["exit_id"]: e["risk_score"] for e in self.ground_truth.exit_risk_scores}
        self.stair_risk = {e["stair_id"]: e["risk_score"] for e in self.ground_truth.stair_risk_scores}

    def test_zone_risk_scores(self):

        self.assertAlmostEqual(self.zone_risk["zone-0"], 0.5)
        self.assertAlmostEqual(self.zone_risk["zone-0b"], 0.0)
        self.assertAlmostEqual(self.zone_risk["zone-1"], 0.5)

    def test_exit_risk_scores_are_capacity_ratios_clamped_to_one(self):

        self.assertAlmostEqual(self.exit_risk["exit-0"], 1.0)  # peak 2 / capacity 1, clamped
        self.assertAlmostEqual(self.exit_risk["exit-1"], 0.0)
        self.assertAlmostEqual(self.exit_risk["exit-2"], 0.4)  # peak 2 / capacity 5

    def test_stair_risk_score_is_worse_of_queue_fraction_and_capacity_fraction(self):

        # queue_fraction = 0.5 (1 of 2 crossings queued), capacity_fraction
        # = min(2 / 1, 1.0) = 1.0 (derived stair capacity is 1, see
        # test_engineering_findings) -- the max of the two wins.
        self.assertAlmostEqual(self.stair_risk["stair-0"], 1.0)

    def test_zone_risk_is_none_without_occupants_or_hazard_data(self):

        building = make_building()
        building.floors[0].zones.append(
            Zone(id="zone-empty", name="Empty", floor_id="floor-1", width=1.0, height=1.0),
        )
        scenario = make_scenario(building)
        movement_result = make_movement_result(building)

        rows = risk_analysis.compute_zone_risk_scores(scenario, movement_result, (), [
            zone for floor in building.ordered_floors() for zone in floor.zones
        ])
        empty_zone_row = next(row for row in rows if row["zone_id"] == "zone-empty")

        self.assertIsNone(empty_zone_row["risk_score"])


class FireAnalysisTests(unittest.TestCase):

    def test_hazard_spread_and_maximum_zone(self):

        ground_truth = analyze(make_artifacts())

        self.assertEqual(ground_truth.first_hazardous_zone, "zone-0")
        self.assertEqual(list(ground_truth.hazard_spread_order), ["zone-0"])
        self.assertEqual(ground_truth.maximum_hazard_zone, "zone-0")

    def test_highest_risk_floor(self):

        ground_truth = analyze(make_artifacts())

        self.assertEqual(ground_truth.highest_risk_floor, "floor-2")

    def test_fire_fields_are_absent_without_tick_results(self):

        ground_truth = analyze(make_artifacts(with_tick_results=False))

        self.assertIsNone(ground_truth.first_hazardous_zone)
        self.assertEqual(list(ground_truth.hazard_spread_order), [])
        self.assertIsNone(ground_truth.maximum_hazard_zone)

        # Zone risk scores still compute (from trapped_fraction alone),
        # they just have no hazard component blended in.
        zone_risk = {e["zone_id"]: e["risk_score"] for e in ground_truth.zone_risk_scores}
        self.assertAlmostEqual(zone_risk["zone-0"], 0.0)


class RecommendationGenerationTests(unittest.TestCase):

    def setUp(self):

        self.ground_truth = analyze(make_artifacts())
        self.by_action = {r["action"]: r for r in self.ground_truth.recommendations}

    def test_open_exit_recommendation(self):

        self.assertIn("Open Exit exit-1", self.by_action)
        self.assertEqual(self.by_action["Open Exit exit-1"]["target_type"], "exit")
        self.assertEqual(self.by_action["Open Exit exit-1"]["target_id"], "exit-1")

    def test_close_door_recommendation(self):

        self.assertIn("Close Door door-0", self.by_action)

    def test_deploy_staff_recommendation(self):

        self.assertIn("Deploy staff to Stair stair-0", self.by_action)

    def test_redirect_recommendation(self):

        self.assertIn("Redirect occupants from Zone zone-0 to Exit exit-1", self.by_action)

    def test_increase_exit_width_recommendation(self):

        self.assertIn("Increase exit width recommendation for Exit exit-0", self.by_action)

    def test_additional_detector_and_camera_recommendations_only_for_uncovered_high_risk_zone(self):

        self.assertIn("Additional detector recommendation for Zone zone-0", self.by_action)
        self.assertIn("Additional camera recommendation for Zone zone-0", self.by_action)

        # zone-1 is also high risk, but floor-2 already has a detector
        # and camera -- no recommendation should fire for it.
        self.assertNotIn("Additional detector recommendation for Zone zone-1", self.by_action)
        self.assertNotIn("Additional camera recommendation for Zone zone-1", self.by_action)

    def test_no_open_exit_recommendation_when_nobody_is_trapped(self):

        recs = generate_recommendations(
            zones=[], doors=[], detectors=[], cameras=[],
            zone_route_stats=[], engineering={
                "exits_underutilized": ["exit-1"], "doors_that_became_bottlenecks": [],
                "exits_exceeding_capacity": [], "stairs_exceeding_capacity": [],
            },
            congestion={"worst_exit": None},
            zone_risk_scores=[], stair_risk_scores=[],
            maximum_hazard_zone=None, people_trapped=0,
            exit_open_lookup={"exit-1": False},
        )

        self.assertEqual(recs, [])


class DeterminismTests(unittest.TestCase):

    def test_repeated_analysis_is_identical(self):

        artifacts = make_artifacts()

        first = analyze(artifacts)
        second = analyze(artifacts)

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_independently_built_artifacts_produce_identical_ground_truth(self):

        first = analyze(make_artifacts())
        second = analyze(make_artifacts())

        self.assertEqual(first.to_dict(), second.to_dict())


class SerializationTests(unittest.TestCase):

    def test_round_trip_preserves_all_fields(self):

        ground_truth = analyze(make_artifacts())

        restored = GroundTruth.from_dict(ground_truth.to_dict())

        self.assertEqual(ground_truth.to_dict(), restored.to_dict())
        self.assertEqual(ground_truth, restored)

    def test_to_dict_uses_plain_json_safe_types(self):

        ground_truth = analyze(make_artifacts())
        data = ground_truth.to_dict()

        self.assertIsInstance(data["zone_route_stats"], list)
        self.assertIsInstance(data["zone_route_stats"][0], dict)
        self.assertIsInstance(data["recommendations"], list)
        self.assertIsInstance(data["doors_that_became_bottlenecks"], list)

    def test_from_dict_defaults_missing_optional_fields(self):

        minimal = GroundTruth.from_dict({"scenario_id": "scn-x", "definition_id": "def-x"})

        self.assertIsNone(minimal.total_evacuation_time)
        self.assertEqual(minimal.zone_route_stats, ())
        self.assertEqual(minimal.recommendations, ())


if __name__ == "__main__":
    unittest.main()
