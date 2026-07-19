import json
import unittest

from models.assembly_point import AssemblyPoint
from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from scenario.firefighter import ScenarioFirefighter
from scenario.metadata import ScenarioMetadata
from scenario.occupant import ScenarioOccupant
from scenario.scenario import Scenario

from ground_truth.labels import GroundTruth
from decision_policy.policy import DecisionPolicy

from perception.models.human_observation import HumanClassification, HumanObservation, HumanState

from advisory_system.confidence_engine import (
    agreement_confidence, combine_confidence, occupancy_confidence, recommendation_confidence,
)
from advisory_system.explanation_engine import compute_redirect_target, estimate_rset_improvement
from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_history import RecommendationHistory
from advisory_system.recommendation_models import AdvisoryInputs


# =====================================================
# Fixtures -- a small, deterministic 3-zone/2-exit building, mirroring
# the shape every other test_*.py in this codebase builds by hand
# (see tests.test_human_population.make_building(), tests.
# test_campaign_studio.make_building()). Two exits (not one) are
# needed here specifically to exercise the redirect path
# (advisory_system.explanation_engine.compute_redirect_target()).
# =====================================================


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-a", name="Cafeteria", x=0.0, y=0.0, width=10.0, height=8.0, floor_id="floor-1"),
            Zone(id="zone-b", name="Laboratory", x=20.0, y=0.0, width=6.0, height=6.0, floor_id="floor-1"),
            Zone(id="zone-c", name="Zone C", x=40.0, y=0.0, width=6.0, height=6.0, floor_id="floor-1"),
        ],
        doors=[Door(id="door-1", name="D1", zone_a_id="zone-a", zone_b_id="zone-b", floor_id="floor-1")],
        exits=[
            Exit(id="exit-1", name="Exit 1", zone_id="zone-a", floor_id="floor-1"),
            Exit(id="exit-2", name="Exit 2", zone_id="zone-c", floor_id="floor-1"),
        ],
        stairs=[Staircase(id="stair-1", name="Stair 1", from_zone_id="zone-b", to_zone_id="zone-a", to_floor_id="floor-1")],
        assembly_points=[AssemblyPoint(id="ap-1", name="Assembly Area B", floor_id="floor-1")],
    )

    return Building(name="Test Building", id="building-1", floors=[floor])


def make_metadata(scenario_id="scn-1"):

    return ScenarioMetadata(
        scenario_id=scenario_id, definition_id="def-1", definition_content_hash="h",
        generation_version="v1", seed=1, created_at="2026-07-16T00:00:00",
    )


def make_scenario(scenario_id="scn-1"):

    occupants = (
        ScenarioOccupant(
            occupant_id="occ-1", zone_id="zone-a", floor_id="floor-1",
            position=(1.0, 1.0), behaviour_profile_id="Adult_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-2", zone_id="zone-b", floor_id="floor-1",
            position=(1.0, 1.0), behaviour_profile_id="Child_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-3", zone_id="zone-b", floor_id="floor-1",
            position=(1.0, 1.0), behaviour_profile_id="Wheelchair_Default",
        ),
    )
    firefighters = (
        ScenarioFirefighter(
            firefighter_id="ff-1", team_id="team-0", entry_zone_id="zone-a",
            floor_id="floor-1", position=(0.0, 0.0), arrival_time=30.0,
            behaviour_profile_id="Firefighter_Default",
        ),
    )

    return Scenario(metadata=make_metadata(scenario_id), occupants=occupants, firefighters=firefighters)


def make_ground_truth(
    *, maximum_hazard_zone=None, hazard_spread_order=(), zone_risk_scores=None,
    zone_route_stats=None, worst_exit=None, exits_exceeding_capacity=(), exits_underutilized=(),
    stair_risk_scores=None, people_trapped=1, building_cleared=False, recommendations=(),
):

    return GroundTruth(
        scenario_id="scn-1", definition_id="def-1",
        total_evacuation_time=150.0,
        building_cleared=building_cleared,
        reachable_occupants=3, unreachable_occupants=0,
        people_trapped=people_trapped, people_evacuated=3 - people_trapped,
        worst_exit=worst_exit,
        zone_route_stats=zone_route_stats or [],
        maximum_hazard_zone=maximum_hazard_zone,
        hazard_spread_order=hazard_spread_order,
        first_hazardous_zone=maximum_hazard_zone,
        doors_that_became_bottlenecks=(),
        exits_underutilized=exits_underutilized,
        exits_exceeding_capacity=exits_exceeding_capacity,
        stairs_exceeding_capacity=(),
        zone_risk_scores=zone_risk_scores or [],
        stair_risk_scores=stair_risk_scores or [],
        recommendations=recommendations,
        helping_group_count=1, fallen_count=0, possible_injury_count=0,
    )


def make_decision_policy(*, zone_decisions, exit_decisions=(), stair_decisions=(), announcements=(), rescue_order=()):

    return DecisionPolicy(
        scenario_id="scn-1",
        zone_decisions=zone_decisions,
        exit_decisions=exit_decisions,
        stair_decisions=stair_decisions,
        announcements=announcements,
        rescue_priorities=[
            {"zone_id": zd["zone_id"], "rescue_priority": "LOW", "impact_score": 0.0, "occupant_count": 1}
            for zd in zone_decisions
        ],
        rescue_order=rescue_order,
    )


# =====================================================
# Phase 2 -- Civilian Advisory.
# =====================================================


class CivilianAdvisoryTests(unittest.TestCase):

    def test_announcements_are_always_zone_addressed_never_individual(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(
            zone_route_stats=[{"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None, "average_travel_distance": 5.0, "average_travel_time": 10.0}],
        )
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"}],
        )

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        self.assertEqual(len(report.civilian_announcements), 1)
        announcement = report.civilian_announcements[0]

        self.assertTrue(announcement.announcement.startswith("Attention occupants in Cafeteria."))
        for occupant_id in ("occ-1", "occ-2", "occ-3"):
            self.assertNotIn(occupant_id, announcement.announcement)

    def test_closed_exit_never_announced_falls_back_to_shelter_in_place(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(
            zone_route_stats=[{"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None, "average_travel_distance": 5.0, "average_travel_time": 10.0}],
        )
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"}],
            exit_decisions=[{"exit_id": "exit-1", "status": "CLOSE"}],
        )

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        announcement = report.civilian_announcements[0]
        self.assertIn("Remain in place", announcement.announcement)
        self.assertNotIn("exit-1", announcement.announcement.lower())
        self.assertNotIn("Exit 1", announcement.announcement)

    def test_avoided_stair_never_appears_as_a_route_instruction(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(
            zone_route_stats=[{"zone_id": "zone-b", "preferred_exit": "exit-1", "preferred_stair": "stair-1", "average_travel_distance": 5.0, "average_travel_time": 10.0}],
        )
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-b", "recommended_exit": "exit-1", "recommended_stair": "stair-1", "action": "EVACUATE_IMMEDIATELY"}],
            stair_decisions=[{"stair_id": "stair-1", "status": "AVOID"}],
        )

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        announcement = report.civilian_announcements[0]
        self.assertIn("Avoid Stair 1.", announcement.announcement)
        self.assertNotIn("Use Stair 1", announcement.announcement)

    def test_congested_exit_with_underutilized_alternative_redirects_and_reports_rset_improvement(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(
            zone_route_stats=[
                {"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None, "average_travel_distance": 5.0, "average_travel_time": 40.0},
                {"zone_id": "zone-c", "preferred_exit": "exit-2", "preferred_stair": None, "average_travel_distance": 3.0, "average_travel_time": 15.0},
            ],
            worst_exit="exit-1", exits_exceeding_capacity=("exit-1",), exits_underutilized=("exit-2",),
        )
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"}],
            exit_decisions=[
                {"exit_id": "exit-1", "status": "HIGH_CONGESTION"},
                {"exit_id": "exit-2", "status": "KEEP_OPEN"},
            ],
        )

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        announcement = report.civilian_announcements[0]
        self.assertIn("Exit 2", announcement.announcement)
        self.assertNotIn("Proceed to Exit 1.", announcement.announcement)
        self.assertEqual(announcement.predicted_rset_improvement_seconds, 25.0)

    def test_no_genuine_alternative_never_fabricates_an_rset_improvement(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(
            zone_route_stats=[{"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None, "average_travel_distance": 5.0, "average_travel_time": 40.0}],
        )
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"}],
            exit_decisions=[{"exit_id": "exit-1", "status": "KEEP_OPEN"}],
        )

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        self.assertIsNone(report.civilian_announcements[0].predicted_rset_improvement_seconds)

    def test_every_announcement_carries_a_confidence_and_a_reason(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(zone_route_stats=[])
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-c", "recommended_exit": None, "recommended_stair": None, "action": "SHELTER_IN_PLACE"}],
        )

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        announcement = report.civilian_announcements[0]
        self.assertIsNotNone(announcement.confidence)
        self.assertTrue(0.0 <= announcement.confidence <= 1.0)
        self.assertTrue(len(announcement.reason) > 0)


class CivilianConfidenceSourceTests(unittest.TestCase):

    # "Never display a value as 'AI confidence' unless it originates
    # from AI inference" -- confidence_source is the field command_
    # center.recommendation_center reads to decide that; these tests
    # confirm it is populated honestly in both directions (empty when
    # no real AI/RL signal was supplied, non-empty only when one was).

    def _inputs(self, **overrides):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(
            zone_route_stats=[{"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None, "average_travel_distance": 5.0, "average_travel_time": 10.0}],
        )
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"}],
        )

        fields = dict(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        fields.update(overrides)

        return AdvisoryInputs(**fields)

    def test_no_ai_or_rl_signal_yields_empty_confidence_source(self):

        report = AdvisoryOrchestrator().generate_report(self._inputs())

        self.assertEqual(report.civilian_announcements[0].confidence_source, ())

    def test_a_genuine_ai_prediction_for_this_exact_target_is_recorded(self):

        inputs = self._inputs(
            ai_predictions={"bottleneck_location": {"value": "exit-1"}},
            ai_confidence={"bottleneck_location": 0.85},
        )
        report = AdvisoryOrchestrator().generate_report(inputs)

        self.assertIn("ai", report.civilian_announcements[0].confidence_source)

    def test_an_ai_prediction_for_a_different_target_is_not_recorded(self):

        inputs = self._inputs(
            ai_predictions={"bottleneck_location": {"value": "exit-2"}},
            ai_confidence={"bottleneck_location": 0.85},
        )
        report = AdvisoryOrchestrator().generate_report(inputs)

        self.assertNotIn("ai", report.civilian_announcements[0].confidence_source)

    def test_a_genuine_rl_confidence_is_recorded(self):

        inputs = self._inputs(rl_confidence=0.6)
        report = AdvisoryOrchestrator().generate_report(inputs)

        self.assertIn("rl", report.civilian_announcements[0].confidence_source)

    def test_to_dict_round_trips_confidence_source(self):

        inputs = self._inputs(rl_confidence=0.6)
        report = AdvisoryOrchestrator().generate_report(inputs)

        data = report.civilian_announcements[0].to_dict()

        self.assertEqual(data["confidence_source"], ["rl"])


# =====================================================
# Phase 3 -- Firefighter Intelligence: information only, never a
# command.
# =====================================================


class FirefighterIntelligenceTests(unittest.TestCase):

    def test_report_never_carries_a_command_or_mission_field(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(
            zone_risk_scores=[
                {"zone_id": "zone-a", "risk_score": 0.1},
                {"zone_id": "zone-b", "risk_score": 0.6},
                {"zone_id": "zone-c", "risk_score": 0.0},
            ],
        )
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"}],
            rescue_order=("zone-b",),
        )

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        payload = report.firefighter_intelligence.to_dict()
        forbidden_terms = ("command", "mission", "order", "assign", "directive", "instruct")

        for key in payload:
            lowered = key.lower()
            for term in forbidden_terms:
                self.assertNotIn(term, lowered, f"field {key!r} looks like a command, not intelligence")

    def test_reports_children_and_wheelchair_users_from_scenario(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(zone_decisions=[])

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        fi = report.firefighter_intelligence
        self.assertEqual(fi.children_count, 1)
        self.assertEqual(fi.wheelchair_users_count, 1)
        self.assertEqual(fi.occupants_remaining, ground_truth.people_trapped)

    def test_suggested_access_routes_carry_travel_time_when_known(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(
            zone_route_stats=[{"zone_id": "zone-b", "preferred_exit": "exit-1", "preferred_stair": "stair-1", "average_travel_distance": 5.0, "average_travel_time": 22.5}],
        )
        decision_policy = make_decision_policy(zone_decisions=[], rescue_order=("zone-b",))

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        routes = report.firefighter_intelligence.suggested_access_routes
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["estimated_travel_time_seconds"], 22.5)


# =====================================================
# Phase 4 -- Building Advisory: recommend only.
# =====================================================


class BuildingAdvisoryTests(unittest.TestCase):

    def test_no_execute_or_apply_method_exists_anywhere_in_the_package(self):

        import advisory_system

        for module_name in ("advisory_engine", "orchestrator", "recommendation_models"):
            module = getattr(__import__(f"advisory_system.{module_name}", fromlist=[module_name]), "__dict__", {})
            for name in module:
                self.assertNotIn("execute", name.lower())
                self.assertNotIn("apply_control", name.lower())

    def test_deluge_and_smoke_exhaust_recommended_for_hazardous_zone(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(
            maximum_hazard_zone="zone-b", hazard_spread_order=("zone-b",),
            zone_risk_scores=[{"zone_id": "zone-b", "risk_score": 0.8}],
        )
        decision_policy = make_decision_policy(zone_decisions=[])

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        actions = [entry.action for entry in report.building_recommendations]
        self.assertTrue(any(a.startswith("Activate Deluge") for a in actions))
        self.assertTrue(any(a.startswith("Activate Smoke Exhaust") for a in actions))

    def test_every_recommendation_carries_reason_confidence_and_benefit(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(maximum_hazard_zone="zone-b", hazard_spread_order=("zone-b",))
        decision_policy = make_decision_policy(zone_decisions=[])

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        self.assertTrue(len(report.building_recommendations) > 0)
        for entry in report.building_recommendations:
            self.assertTrue(len(entry.reason) > 0)
            self.assertIsNotNone(entry.confidence)
            self.assertTrue(len(entry.expected_engineering_benefit) > 0)


# =====================================================
# Phase 5 -- Incident Commander Dashboard.
# =====================================================


class CommanderDashboardTests(unittest.TestCase):

    def test_occupancy_confidence_is_independent_of_recommendation_confidence(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(
            zone_risk_scores=[{"zone_id": "zone-a", "risk_score": 0.9}],
        )
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"}],
        )
        human_observations = {
            "p1": HumanObservation(person_id="p1", zone_id="zone-a", confidence=0.42, classification=HumanClassification.ADULT, state=HumanState.WALKING),
        }

        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            human_observations=human_observations,
        )
        report = AdvisoryOrchestrator().generate_report(inputs)

        dashboard = report.commander_dashboard
        self.assertEqual(dashboard.occupancy_confidence, 0.42)
        self.assertNotEqual(dashboard.occupancy_confidence, dashboard.recommendation_confidence)

    def test_zone_tiers_are_derived_from_real_risk_scores(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(
            zone_risk_scores=[
                {"zone_id": "zone-a", "risk_score": 0.1},
                {"zone_id": "zone-b", "risk_score": 0.4},
                {"zone_id": "zone-c", "risk_score": 0.9},
            ],
        )
        decision_policy = make_decision_policy(zone_decisions=[])

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        dashboard = report.commander_dashboard
        self.assertIn("zone-a", dashboard.safe_zones)
        self.assertIn("zone-b", dashboard.warning_zones)
        self.assertIn("zone-c", dashboard.critical_zones)

    def test_report_round_trips_through_to_dict_as_json(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(maximum_hazard_zone="zone-b", hazard_spread_order=("zone-b",))
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"}],
        )

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        encoded = json.dumps(report.to_dict())
        self.assertIsInstance(encoded, str)


# =====================================================
# Phase 7 -- Recommendation History.
# =====================================================


class RecommendationHistoryTests(unittest.TestCase):

    def test_no_history_entry_on_first_recommendation(self):

        history = RecommendationHistory()
        change = history.record(timestamp=0.0, zone_id="zone-a", recommendation="A", confidence=0.5)
        self.assertIsNone(change)

    def test_change_recorded_when_recommendation_differs(self):

        history = RecommendationHistory()
        history.record(timestamp=0.0, zone_id="zone-a", recommendation="Use Exit 1", confidence=0.6)
        change = history.record(
            timestamp=5.0, zone_id="zone-a", recommendation="Use Exit 3", confidence=0.9,
            reason_for_change="Smoke spread detected.",
        )

        self.assertIsNotNone(change)
        self.assertEqual(change.previous_recommendation, "Use Exit 1")
        self.assertEqual(change.new_recommendation, "Use Exit 3")
        self.assertEqual(change.reason_for_change, "Smoke spread detected.")
        self.assertEqual(change.confidence_before, 0.6)
        self.assertEqual(change.confidence_after, 0.9)

    def test_reissuing_the_same_recommendation_is_not_a_change(self):

        history = RecommendationHistory()
        history.record(timestamp=0.0, zone_id="zone-a", recommendation="Use Exit 1", confidence=0.6)
        change = history.record(timestamp=5.0, zone_id="zone-a", recommendation="Use Exit 1", confidence=0.6)
        self.assertIsNone(change)

    def test_orchestrator_records_a_change_when_hazard_causes_a_redirect(self):

        building = make_building()
        scenario = make_scenario()
        orchestrator = AdvisoryOrchestrator()

        calm_ground_truth = make_ground_truth(
            zone_route_stats=[{"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None, "average_travel_distance": 5.0, "average_travel_time": 10.0}],
        )
        calm_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"}],
            exit_decisions=[{"exit_id": "exit-1", "status": "KEEP_OPEN"}],
        )
        orchestrator.generate_report(
            AdvisoryInputs(building=building, scenario=scenario, ground_truth=calm_ground_truth, decision_policy=calm_policy, simulation_time=0.0),
        )

        hazardous_ground_truth = make_ground_truth(
            zone_route_stats=[
                {"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None, "average_travel_distance": 5.0, "average_travel_time": 40.0},
                {"zone_id": "zone-c", "preferred_exit": "exit-2", "preferred_stair": None, "average_travel_distance": 3.0, "average_travel_time": 15.0},
            ],
            worst_exit="exit-1", exits_exceeding_capacity=("exit-1",), exits_underutilized=("exit-2",),
        )
        hazardous_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"}],
            exit_decisions=[{"exit_id": "exit-1", "status": "HIGH_CONGESTION"}, {"exit_id": "exit-2", "status": "KEEP_OPEN"}],
        )
        report = orchestrator.generate_report(
            AdvisoryInputs(building=building, scenario=scenario, ground_truth=hazardous_ground_truth, decision_policy=hazardous_policy, simulation_time=120.0),
        )

        self.assertEqual(len(report.recommendation_history), 1)
        change = report.recommendation_history[0]
        self.assertEqual(change.zone_id, "zone-a")
        self.assertEqual(change.timestamp, 120.0)
        self.assertNotEqual(change.previous_recommendation, change.new_recommendation)


# =====================================================
# Confidence Engine.
# =====================================================


class ConfidenceEngineTests(unittest.TestCase):

    def test_combine_confidence_ignores_none_sources(self):

        self.assertEqual(combine_confidence(0.5, None, 0.7), 0.6)

    def test_combine_confidence_is_none_when_every_source_is_none(self):

        self.assertIsNone(combine_confidence(None, None))

    def test_agreement_confidence_excludes_unconsulted_sources(self):

        self.assertEqual(agreement_confidence([True, None, True]), 1.0)
        self.assertEqual(agreement_confidence([True, False]), 0.5)
        self.assertIsNone(agreement_confidence([None, None]))

    def test_occupancy_confidence_reads_perception_confidence_only(self):

        observations = {
            "p1": HumanObservation(person_id="p1", confidence=0.8),
            "p2": HumanObservation(person_id="p2", confidence=0.4),
        }
        self.assertAlmostEqual(occupancy_confidence(observations), 0.6)

    def test_recommendation_confidence_always_returns_a_real_number(self):

        confidence = recommendation_confidence()
        self.assertIsNotNone(confidence)
        self.assertTrue(0.0 <= confidence <= 1.0)


# =====================================================
# Explanation Engine.
# =====================================================


class ExplanationEngineTests(unittest.TestCase):

    def test_redirect_target_is_none_without_a_genuine_alternative(self):

        ground_truth = make_ground_truth(
            zone_route_stats=[{"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None, "average_travel_distance": 5.0, "average_travel_time": 10.0}],
        )
        self.assertIsNone(compute_redirect_target(ground_truth, "zone-a"))

    def test_redirect_target_found_when_preferred_exit_overloaded(self):

        ground_truth = make_ground_truth(
            zone_route_stats=[{"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None, "average_travel_distance": 5.0, "average_travel_time": 10.0}],
            exits_exceeding_capacity=("exit-1",), exits_underutilized=("exit-2",),
        )
        self.assertEqual(compute_redirect_target(ground_truth, "zone-a"), "exit-2")

    def test_rset_improvement_is_none_without_an_alternative(self):

        ground_truth = make_ground_truth(
            zone_route_stats=[{"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None, "average_travel_distance": 5.0, "average_travel_time": 10.0}],
        )
        self.assertIsNone(estimate_rset_improvement(ground_truth, "zone-a", None))


if __name__ == "__main__":
    unittest.main()
