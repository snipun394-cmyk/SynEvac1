import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.coordinator import MultiAgentSimulation


# Edge-Type-Specific Movement Speed (Experimental Branch V1) --
# end-to-end tests through MultiAgentSimulation itself, exercising the
# real seam (simulator/coordinator.py's _admit_onto_edge()), not a
# mocked stand-in. `stair_speed` is passed straight to add_occupant()
# here (the same simulator-level API walking_speed already uses in
# tests/test_multi_agent_simulation.py) -- the behavior/behaviour_
# profile_resolver propagation layers above this are exercised by the
# full existing test suite, since every registration path that
# constructs BehaviorProfile/BehaviorDecision/Occupant now threads
# stair_speed through unconditionally.


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


class _StairAndDoorScenario(unittest.TestCase):

    # One building, two floors: a wide Stair (Lobby <-> Upstairs) and a
    # wide Door (Lobby <-> Corridor) on the same ground floor, so
    # Stair- and Door-edge behavior can be compared within one graph.
    # Both wide enough (width=3.0 -> capacity 4) that a handful of
    # concurrent occupants never queue -- any timing difference is
    # purely the congestion speed effect, matching
    # tests/test_multi_agent_simulation.py::CongestionEffectTests.

    def setUp(self):

        self.building = Building(name="B")
        self.ground = self.building.create_floor(name="Ground Floor")
        self.floor1 = self.building.create_floor(name="Floor 1", height=3.0)

        self.lobby = make_zone("Lobby", x=0.0, y=0.0)
        self.corridor = make_zone("Corridor", x=20.0, y=0.0)
        self.upstairs = make_zone("Upstairs", x=0.0, y=0.0, floor_id=self.floor1.id)

        self.ground.add_zone(self.lobby)
        self.ground.add_zone(self.corridor)
        self.floor1.add_zone(self.upstairs)

        self.door = Door(
            name="D1", zone_a_id=self.lobby.id, zone_b_id=self.corridor.id,
            floor_id=self.ground.id, width=3.0,
        )
        self.ground.add_door(self.door)

        self.stair = Staircase(
            name="S1", from_floor_id=self.ground.id, to_floor_id=self.floor1.id,
            from_zone_id=self.lobby.id, to_zone_id=self.upstairs.id, width=3.0,
        )
        self.ground.add_stair(self.stair)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

        self.stair_edge = next(e for e in self.graph.edges if e.edge_type == Edge.STAIR)
        self.door_edge = next(e for e in self.graph.edges if e.edge_type == Edge.DOOR)


class BackwardCompatibilityTests(_StairAndDoorScenario):

    def test_stair_speed_none_reproduces_the_pre_existing_walking_speed_duration(self):

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(self.lobby.id, self.upstairs.id, occupant_id="solo", walking_speed=1.4)

        step = sim.run().occupants["solo"].steps[0]

        expected_duration = self.stair_edge.traversal_cost / 1.4
        self.assertAlmostEqual(step.end_time - step.start_time, expected_duration)

    def test_explicitly_passing_stair_speed_none_matches_omitting_it_entirely(self):

        sim_omitted = MultiAgentSimulation(self.engine)
        sim_omitted.add_occupant(self.lobby.id, self.upstairs.id, occupant_id="occ", walking_speed=1.4)

        sim_explicit = MultiAgentSimulation(self.engine)
        sim_explicit.add_occupant(
            self.lobby.id, self.upstairs.id, occupant_id="occ", walking_speed=1.4, stair_speed=None,
        )

        self.assertAlmostEqual(
            sim_omitted.run().occupants["occ"].arrival_time,
            sim_explicit.run().occupants["occ"].arrival_time,
        )


class StairOverrideTests(_StairAndDoorScenario):

    def test_stair_edge_uses_stair_speed_not_walking_speed_when_configured(self):

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(
            self.lobby.id, self.upstairs.id, occupant_id="solo",
            walking_speed=1.4, stair_speed=0.5,
        )

        step = sim.run().occupants["solo"].steps[0]
        actual_duration = step.end_time - step.start_time

        self.assertAlmostEqual(actual_duration, self.stair_edge.traversal_cost / 0.5)
        self.assertNotAlmostEqual(actual_duration, self.stair_edge.traversal_cost / 1.4)


class HorizontalPreservationTests(_StairAndDoorScenario):

    def test_door_edge_still_uses_walking_speed_when_stair_speed_is_configured(self):

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(
            self.lobby.id, self.corridor.id, occupant_id="solo",
            walking_speed=1.4, stair_speed=0.5,
        )

        step = sim.run().occupants["solo"].steps[0]

        self.assertEqual(step.edge.id, self.door_edge.id)
        self.assertAlmostEqual(step.end_time - step.start_time, self.door_edge.traversal_cost / 1.4)

    def test_exit_edge_still_uses_walking_speed_when_stair_speed_is_configured(self):

        self.ground.add_exit(Exit(name="Ex", zone_id=self.corridor.id, floor_id=self.ground.id))

        graph = NavigationGraphGenerator().build(self.building)
        engine = PathfindingEngine(graph)
        exit_edge = next(e for e in graph.edges if e.edge_type == Edge.EXIT)

        sim = MultiAgentSimulation(engine)
        sim.add_occupant(self.corridor.id, occupant_id="solo", walking_speed=1.4, stair_speed=0.5)

        step = sim.run().occupants["solo"].steps[0]

        self.assertEqual(step.edge.id, exit_edge.id)
        self.assertAlmostEqual(step.end_time - step.start_time, exit_edge.traversal_cost / 1.4)


class CongestionInteractionTests(_StairAndDoorScenario):

    def test_congestion_still_degrades_stair_speed_not_just_walking_speed(self):

        solo_sim = MultiAgentSimulation(self.engine)
        solo_sim.add_occupant(
            self.lobby.id, self.upstairs.id, occupant_id="solo",
            walking_speed=1.4, stair_speed=0.5,
        )
        solo_duration = solo_sim.run().occupants["solo"].steps[0].end_time

        crowded_sim = MultiAgentSimulation(self.engine)
        for i in range(4):
            crowded_sim.add_occupant(
                self.lobby.id, self.upstairs.id, occupant_id=f"p{i}",
                walking_speed=1.4, stair_speed=0.5,
            )
        crowded_result = crowded_sim.run()

        # Capacity 4 (width 3.0), 4 occupants -- nobody queues, so any
        # extra time is purely congestion_model.speed_factor(), not
        # admission-control queueing.
        for timeline in crowded_result.occupants.values():
            self.assertEqual(timeline.steps[0].queue_wait_time, 0.0)

        # First admitted, nobody else yet present -- zero congestion
        # penalty, same as crossing alone.
        self.assertAlmostEqual(crowded_result.occupants["p0"].steps[0].end_time, solo_duration)

        # Every later occupant shares the edge with more people already
        # on it -- speed_factor < 1.0 must still apply on top of the
        # stair_speed base, exactly as it already does for walking_speed.
        durations = [crowded_result.occupants[f"p{i}"].steps[0].end_time for i in range(4)]

        for later_duration in durations[1:]:
            self.assertGreater(later_duration, solo_duration)

        self.assertEqual(durations, sorted(durations))
