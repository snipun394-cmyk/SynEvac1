import unittest

from dataclasses import FrozenInstanceError

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from hazard.edge_state import HazardEdgeState
from hazard.node_state import HazardNodeState
from hazard.severity import HazardSeverity
from hazard.snapshot import HazardSnapshot

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from ai_decision.announcements import DefaultVoiceAnnouncementTemplate
from ai_decision.engine import AIDecisionEngine, DecisionEngine
from ai_decision.priority import SeverityOccupancyPriorityRule
from ai_decision.recommendation import (
    DecisionRecommendation,
    PriorityEvacuationEntry,
    ZoneRecommendation,
)


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


def build_three_zone_building():

    # room --door--> corridor --door--> lobby --exit--> Outside
    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    room = make_zone("Room", x=0.0, y=0.0)
    corridor = make_zone("Corridor", x=10.0, y=0.0)
    lobby = make_zone("Lobby", x=20.0, y=0.0)

    floor.add_zone(room)
    floor.add_zone(corridor)
    floor.add_zone(lobby)

    door1 = Door(name="D1", zone_a_id=room.id, zone_b_id=corridor.id, floor_id=floor.id, width=1.0)
    door2 = Door(name="D2", zone_a_id=corridor.id, zone_b_id=lobby.id, floor_id=floor.id, width=1.0)
    floor.add_door(door1)
    floor.add_door(door2)

    exit_obj = Exit(name="Front Exit", zone_id=lobby.id, floor_id=floor.id)
    floor.add_exit(exit_obj)

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    return building, floor, room, corridor, lobby, door1, door2, exit_obj, engine


class AIDecisionEngineTests(unittest.TestCase):

    def setUp(self):

        (
            self.building, self.floor, self.room, self.corridor, self.lobby,
            self.door1, self.door2, self.exit_obj, self.base_engine,
        ) = build_three_zone_building()

        self.decision_engine = AIDecisionEngine(self.base_engine)

    def test_is_a_decision_engine(self):

        self.assertIsInstance(self.decision_engine, DecisionEngine)

    def test_base_interface_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            DecisionEngine().decide(HazardSnapshot(), OccupancySnapshot(), 0.0)

    def test_clear_building_produces_no_unsafe_zones(self):

        result = self.decision_engine.decide(HazardSnapshot(), OccupancySnapshot(), time=0.0)

        self.assertEqual(result.unsafe_zone_ids, [])
        self.assertEqual(result.blocked_routes, [])
        self.assertEqual(result.voice_announcements, [])
        self.assertEqual(len(result.zone_recommendations), 3)

    def test_every_zone_gets_a_recommended_route_when_clear(self):

        result = self.decision_engine.decide(HazardSnapshot(), OccupancySnapshot(), time=0.0)

        for zone_id in (self.room.id, self.corridor.id, self.lobby.id):

            recommendation = result.zone_recommendation(zone_id)

            self.assertTrue(recommendation.is_reachable)
            self.assertEqual(recommendation.recommended_exit_edge_id, self.exit_obj.id)
            self.assertIsNotNone(recommendation.recommended_route)

    def test_critical_hazard_flags_zone_unsafe_and_announces_it(self):

        hazard_snapshot = HazardSnapshot(
            node_states={self.room.id: HazardNodeState(hazard_score=0.9)},  # CRITICAL
        )

        result = self.decision_engine.decide(hazard_snapshot, OccupancySnapshot(), time=0.0)

        self.assertIn(self.room.id, result.unsafe_zone_ids)

        room_recommendation = result.zone_recommendation(self.room.id)
        self.assertEqual(room_recommendation.severity, HazardSeverity.CRITICAL)

        announcement = next(a for a in result.voice_announcements if a.zone_id == self.room.id)
        self.assertIn("URGENT", announcement.message)
        self.assertEqual(announcement.severity, HazardSeverity.CRITICAL)

    def test_moderate_hazard_below_threshold_is_not_unsafe(self):

        hazard_snapshot = HazardSnapshot(
            node_states={self.room.id: HazardNodeState(hazard_score=0.4)},  # MODERATE
        )

        result = self.decision_engine.decide(hazard_snapshot, OccupancySnapshot(), time=0.0)

        self.assertNotIn(self.room.id, result.unsafe_zone_ids)
        self.assertEqual(result.voice_announcements, [])

    def test_unsafe_threshold_is_configurable(self):

        lenient_engine = AIDecisionEngine(
            self.base_engine, unsafe_severity_threshold=HazardSeverity.MODERATE,
        )
        hazard_snapshot = HazardSnapshot(
            node_states={self.room.id: HazardNodeState(hazard_score=0.4)},  # MODERATE
        )

        result = lenient_engine.decide(hazard_snapshot, OccupancySnapshot(), time=0.0)

        self.assertIn(self.room.id, result.unsafe_zone_ids)

    def test_blocked_door_appears_in_blocked_routes(self):

        graph_door1_edge = next(e for e in self.base_engine.graph.edges if e.id == self.door1.id)

        hazard_snapshot = HazardSnapshot(
            edge_states={self.door1.id: HazardEdgeState(traversable=False)},
        )

        result = self.decision_engine.decide(hazard_snapshot, OccupancySnapshot(), time=0.0)

        blocked_ids = {b.edge_id for b in result.blocked_routes}
        self.assertIn(self.door1.id, blocked_ids)

        blocked = next(b for b in result.blocked_routes if b.edge_id == self.door1.id)
        self.assertEqual(blocked.edge_type, graph_door1_edge.edge_type)

        # Blocking D1 strands Room (its only connection to the rest of
        # the building) -- BuildingAnalysisEngine.critical_connectors()
        # must show up in stranded_zone_ids with zero new graph logic.
        self.assertIn(self.room.id, blocked.stranded_zone_ids)

    def test_room_becomes_unreachable_when_its_only_door_is_blocked(self):

        hazard_snapshot = HazardSnapshot(
            edge_states={self.door1.id: HazardEdgeState(traversable=False)},
        )

        result = self.decision_engine.decide(hazard_snapshot, OccupancySnapshot(), time=0.0)

        room_recommendation = result.zone_recommendation(self.room.id)
        self.assertFalse(room_recommendation.is_reachable)
        self.assertIsNone(room_recommendation.recommended_route)

        announcement = next(a for a in result.voice_announcements if a.zone_id == self.room.id)
        self.assertIn("shelter in place", announcement.message)

    def test_priority_order_ranks_higher_severity_first(self):

        hazard_snapshot = HazardSnapshot(
            node_states={
                self.room.id: HazardNodeState(hazard_score=0.9),      # CRITICAL
                self.corridor.id: HazardNodeState(hazard_score=0.65),  # HIGH
            },
        )

        result = self.decision_engine.decide(hazard_snapshot, OccupancySnapshot(), time=0.0)

        ranked_ids = [entry.zone_id for entry in result.priority_evacuation_order]
        self.assertEqual(ranked_ids.index(self.room.id), 0)
        self.assertLess(ranked_ids.index(self.room.id), ranked_ids.index(self.corridor.id))

    def test_priority_order_breaks_ties_by_occupant_count(self):

        hazard_snapshot = HazardSnapshot(
            node_states={
                self.room.id: HazardNodeState(hazard_score=0.9),
                self.corridor.id: HazardNodeState(hazard_score=0.9),
            },
        )
        occupancy_snapshot = OccupancySnapshot(
            observations={
                self.room.id: OccupancyObservation(occupant_count=2.0),
                self.corridor.id: OccupancyObservation(occupant_count=8.0),
            },
        )

        result = self.decision_engine.decide(hazard_snapshot, occupancy_snapshot, time=0.0)

        ranked_ids = [entry.zone_id for entry in result.priority_evacuation_order]
        self.assertLess(ranked_ids.index(self.corridor.id), ranked_ids.index(self.room.id))

    def test_safe_occupied_zone_is_included_in_priority_order_but_not_unsafe(self):

        occupancy_snapshot = OccupancySnapshot(
            observations={self.lobby.id: OccupancyObservation(occupant_count=5.0)},
        )

        result = self.decision_engine.decide(HazardSnapshot(), occupancy_snapshot, time=0.0)

        self.assertNotIn(self.lobby.id, result.unsafe_zone_ids)
        ranked_ids = [entry.zone_id for entry in result.priority_evacuation_order]
        self.assertIn(self.lobby.id, ranked_ids)

    def test_firefighter_priorities_cover_unsafe_zones_and_critical_blocked_connectors(self):

        hazard_snapshot = HazardSnapshot(
            node_states={self.room.id: HazardNodeState(hazard_score=0.9)},
            edge_states={self.door1.id: HazardEdgeState(traversable=False)},
        )

        result = self.decision_engine.decide(hazard_snapshot, OccupancySnapshot(), time=0.0)

        target_types = {p.target_type for p in result.firefighter_priorities}
        self.assertIn("zone", target_types)
        self.assertIn("connector", target_types)

        zone_entry = next(p for p in result.firefighter_priorities if p.target_type == "zone")
        self.assertEqual(zone_entry.target_id, self.room.id)

        connector_entry = next(p for p in result.firefighter_priorities if p.target_type == "connector")
        self.assertEqual(connector_entry.target_id, self.door1.id)

    def test_recommendation_is_frozen_and_zone_recommendations_are_read_only(self):

        result = self.decision_engine.decide(HazardSnapshot(), OccupancySnapshot(), time=0.0)

        with self.assertRaises(FrozenInstanceError):
            result.timestamp = 5.0

        with self.assertRaises(TypeError):
            result.zone_recommendations[self.room.id] = None

    def test_zone_recommendation_missing_id_returns_none(self):

        result = self.decision_engine.decide(HazardSnapshot(), OccupancySnapshot(), time=0.0)
        self.assertIsNone(result.zone_recommendation("not-a-real-zone"))

    def test_timestamp_and_recommendation_id_are_set(self):

        result = self.decision_engine.decide(HazardSnapshot(), OccupancySnapshot(), time=42.0)

        self.assertEqual(result.timestamp, 42.0)
        self.assertTrue(result.recommendation_id)

    def test_custom_priority_rule_and_announcement_template_are_used(self):

        class StubPriorityRule:
            def rank(self, zone_recommendations):
                return [PriorityEvacuationEntry(rank=1, zone_id="stub", reason="stub reason")]

        class StubAnnouncementTemplate:
            def announce(self, zone_recommendation, zone_name, exit_name):
                return "stub announcement" if zone_recommendation.is_unsafe else None

        engine = AIDecisionEngine(
            self.base_engine,
            priority_rule=StubPriorityRule(),
            announcement_template=StubAnnouncementTemplate(),
        )

        hazard_snapshot = HazardSnapshot(
            node_states={self.room.id: HazardNodeState(hazard_score=0.9)},
        )
        result = engine.decide(hazard_snapshot, OccupancySnapshot(), time=0.0)

        self.assertEqual(result.priority_evacuation_order, [
            PriorityEvacuationEntry(rank=1, zone_id="stub", reason="stub reason"),
        ])
        announcement = next(a for a in result.voice_announcements if a.zone_id == self.room.id)
        self.assertEqual(announcement.message, "stub announcement")

    def test_default_policies_are_used_when_not_supplied(self):

        self.assertIsInstance(self.decision_engine.priority_rule, SeverityOccupancyPriorityRule)
        self.assertIsInstance(self.decision_engine.announcement_template, DefaultVoiceAnnouncementTemplate)


class SeverityOccupancyPriorityRuleTests(unittest.TestCase):

    def test_rank_orders_by_severity_then_occupancy_then_zone_id(self):

        rule = SeverityOccupancyPriorityRule()

        recommendations = [
            ZoneRecommendation(
                zone_id="low", severity=HazardSeverity.LOW, occupant_count=100.0,
                is_unsafe=False, is_reachable=True,
            ),
            ZoneRecommendation(
                zone_id="critical", severity=HazardSeverity.CRITICAL, occupant_count=1.0,
                is_unsafe=True, is_reachable=True,
            ),
        ]

        ranked = rule.rank(recommendations)

        self.assertEqual(ranked[0].zone_id, "critical")
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[1].zone_id, "low")

    def test_none_occupant_count_sorts_as_zero(self):

        rule = SeverityOccupancyPriorityRule()

        recommendations = [
            ZoneRecommendation(
                zone_id="known", severity=HazardSeverity.HIGH, occupant_count=1.0,
                is_unsafe=True, is_reachable=True,
            ),
            ZoneRecommendation(
                zone_id="unknown", severity=HazardSeverity.HIGH, occupant_count=None,
                is_unsafe=True, is_reachable=True,
            ),
        ]

        ranked = rule.rank(recommendations)

        self.assertEqual(ranked[0].zone_id, "known")
        self.assertEqual(ranked[1].zone_id, "unknown")


class DefaultVoiceAnnouncementTemplateTests(unittest.TestCase):

    def setUp(self):
        self.template = DefaultVoiceAnnouncementTemplate()

    def test_safe_zone_gets_no_announcement(self):

        recommendation = ZoneRecommendation(
            zone_id="z1", severity=HazardSeverity.NONE, occupant_count=None,
            is_unsafe=False, is_reachable=True,
        )
        self.assertIsNone(self.template.announce(recommendation, "Room", "Front Exit"))

    def test_unreachable_zone_gets_shelter_in_place_message(self):

        recommendation = ZoneRecommendation(
            zone_id="z1", severity=HazardSeverity.CRITICAL, occupant_count=None,
            is_unsafe=True, is_reachable=False,
        )
        message = self.template.announce(recommendation, "Room", None)
        self.assertIn("shelter in place", message)

    def test_high_severity_evacuate_message_names_the_exit(self):

        recommendation = ZoneRecommendation(
            zone_id="z1", severity=HazardSeverity.HIGH, occupant_count=None,
            is_unsafe=True, is_reachable=True,
        )
        message = self.template.announce(recommendation, "Room", "Front Exit")

        self.assertNotIn("URGENT", message)
        self.assertIn("Room", message)
        self.assertIn("Front Exit", message)


class AIDecisionPackageDependencyDirectionTests(unittest.TestCase):

    def test_package_never_imports_downstream_or_simulation_time_layers(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "ai_decision"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(simulator|behavior|models|designer|hazard_evolution|sensors)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"{path.name} imports a simulation-time or downstream layer directly -- "
                f"AIDecisionEngine must only ever consume the already-frozen "
                f"HazardSnapshot/OccupancySnapshot/Navigation/Pathfinding/Analysis "
                f"interfaces",
            )


if __name__ == "__main__":
    unittest.main()
