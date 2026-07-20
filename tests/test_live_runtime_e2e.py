import unittest

from ground_truth.labels import GroundTruth
from decision_policy.policy import DecisionPolicy

from scenario.scenario import Scenario

from tests.test_advisory_system import make_metadata

from live_system.live_advisory_gateway import ReplayCompatibleAdvisoryGateway

from voice_evacuation.models import BroadcastStatus

from building_control.types import RequestStatus

from command_center.live_operator_action_gateway import (
    PROVIDER_CAPABILITY_SIMULATION,
    VOICE_STATUS_SENT,
)

from live_runtime.factory import build_offline_demo_runtime

from tests.live_runtime_fixtures import (
    make_demo_building,
    make_offline_frame_sources,
    make_offline_human_detector,
    make_offline_identity_resolver,
)


# =====================================================
# Production Live Runtime Composition Root milestone, Phase 7 -- one
# deterministic, fully offline integration demonstration of the entire
# chain:
#
# Building / Digital Twin Assets -> CameraManager -> Replay camera
# frames -> Detection -> Multi-Camera Fusion -> Sensor observations ->
# FACP -> BuildingState -> Live AI (unconfigured -- optional, see
# tests/test_live_runtime_failure_modes.py for AI-specific coverage) ->
# AdvisoryReport -> Live Command Center snapshot -> Operator reviews
# recommendation -> Operator approves ONE voice action ->
# VoiceEvacuationController -> SimulationVoiceOutputProvider, PLUS one
# building-control approval through SimulationControlProvider.
#
# Zero physical CCTV, zero network access, zero hardware anywhere in
# this file.
# =====================================================


def _make_advisory_gateway(building):

    # A hand-built Scenario/GroundTruth/DecisionPolicy referencing this
    # SAME demo building's own zone/door ids -- the same "advisory needs
    # its own caller-supplied Building/Scenario/GroundTruth, no live
    # equivalent exists" constraint docs/architecture/
    # ai_augmented_advisory_integration.md §1 already established, not
    # something this milestone redesigns. No AI/RL training of any kind
    # occurs here -- this is plain, deterministic fixture data.

    scenario = Scenario(metadata=make_metadata("demo-scn"), occupants=(), firefighters=())

    ground_truth = GroundTruth(
        scenario_id="demo-scn", definition_id="demo-def",
        total_evacuation_time=120.0, building_cleared=False,
        reachable_occupants=2, unreachable_occupants=0,
        people_trapped=0, people_evacuated=2,
        worst_exit=None, zone_route_stats=[], maximum_hazard_zone="zone-cafeteria",
        hazard_spread_order=("zone-cafeteria",), first_hazardous_zone="zone-cafeteria",
        doors_that_became_bottlenecks=(), exits_underutilized=(), exits_exceeding_capacity=(),
        stairs_exceeding_capacity=(), zone_risk_scores=[], stair_risk_scores=[],
        recommendations=[
            {"target_type": "door", "target_id": "DOOR-1", "action": "Open Door",
             "reason": "Improves egress flow away from the hazardous zone."},
        ],
        helping_group_count=0, fallen_count=0, possible_injury_count=0,
    )

    decision_policy = DecisionPolicy(
        scenario_id="demo-scn",
        zone_decisions=[
            {"zone_id": "zone-cafeteria", "recommended_exit": "EXIT-1", "recommended_stair": None,
             "action": "EVACUATE_IMMEDIATELY"},
            {"zone_id": "zone-lobby", "recommended_exit": "EXIT-1", "recommended_stair": None,
             "action": "WAIT"},
        ],
        exit_decisions=(), stair_decisions=(), announcements=(),
        rescue_priorities=[
            {"zone_id": "zone-cafeteria", "rescue_priority": "LOW", "impact_score": 0.0, "occupant_count": 1},
        ],
        rescue_order=(),
    )

    return ReplayCompatibleAdvisoryGateway(
        building=building, scenario=scenario, ground_truth=ground_truth,
        decision_policy_provider=lambda time: decision_policy,
    )


class OfflineFullSystemDemonstrationTests(unittest.TestCase):

    def setUp(self):

        self.building = make_demo_building()

        self.runtime = build_offline_demo_runtime(
            self.building,
            frame_sources=make_offline_frame_sources(self.building),
            human_detector=make_offline_human_detector(),
            identity_resolver=make_offline_identity_resolver(self.building),
            live_advisory_gateway=_make_advisory_gateway(self.building),
        )

    def tearDown(self):

        self.runtime.stop()

    def test_full_chain_reaches_a_live_advisory_report_with_zero_automatic_execution(self):

        self.runtime.start()
        self.runtime.run_cycle(1.0)

        snapshot = self.runtime.command_center_data_source.current_snapshot()

        # Building -> Camera -> Detection -> Fusion -> BuildingState.
        self.assertIsNotNone(snapshot.building_state)
        self.assertEqual(len(snapshot.building_state.occupant_tracks), 2)
        self.assertEqual(set(snapshot.building_state.camera_observations.keys()), {"CAM-LOBBY", "CAM-CAFETERIA"})
        self.assertIn("SMOKE-LOBBY", snapshot.building_state.smoke_detector_states)
        self.assertIn("HEAT-CAFETERIA", snapshot.building_state.heat_detector_states)

        # BuildingState -> AdvisoryReport (Live AI unconfigured in this
        # demo -- deterministic safety rules alone still produce a
        # report, exactly as advisory_system's own rule-based path is
        # designed to when AI/RL are absent).
        report = snapshot.advisory_report
        self.assertIsNotNone(report)
        self.assertGreater(len(report.civilian_announcements), 0)
        self.assertGreater(len(report.building_recommendations), 0)

        # ZERO automatic execution before any operator action.
        self.assertEqual(len(self.runtime.voice_evacuation_controller.broadcast_log.all_instructions()), 0)
        self.assertEqual(self.runtime.building_control_controller.all_requests(), ())

    def test_operator_approves_one_voice_action(self):

        self.runtime.start()
        self.runtime.run_cycle(1.0)

        report = self.runtime.command_center_data_source.current_snapshot().advisory_report
        announcement = next(a for a in report.civilian_announcements if a.zone_id == "zone-cafeteria")

        gateway = self.runtime.operator_action_gateway
        self.assertEqual(gateway.voice_capability, PROVIDER_CAPABILITY_SIMULATION)

        instructions = gateway.approve_voice_message(announcement, report.simulation_time)

        self.assertGreater(len(instructions), 0)
        self.assertEqual(instructions[0].target_zone_id, "zone-cafeteria")
        self.assertEqual(instructions[0].status, BroadcastStatus.BROADCAST)  # SPK-CAFETERIA exists in the demo building
        self.assertEqual(gateway.voice_recommendation_status(announcement), VOICE_STATUS_SENT)

        # Verified through the SAME shared controller the runtime owns.
        self.assertEqual(len(self.runtime.voice_evacuation_controller.broadcast_log.all_instructions()), 1)

    def test_operator_approves_one_building_control_action(self):

        self.runtime.start()
        self.runtime.run_cycle(1.0)

        report = self.runtime.command_center_data_source.current_snapshot().advisory_report
        door_recommendation = next(
            r for r in report.building_recommendations if r.target_id == "DOOR-1"
        )

        gateway = self.runtime.operator_action_gateway
        self.assertEqual(gateway.control_capability, PROVIDER_CAPABILITY_SIMULATION)

        from advisory_system.recommendation_models import AdvisoryReport as _AdvisoryReport

        submit_only_report = _AdvisoryReport(
            scenario_id=report.scenario_id, simulation_time=report.simulation_time,
            civilian_announcements=(), firefighter_intelligence=None,
            building_recommendations=(door_recommendation,), commander_dashboard=None,
            recommendation_history=(),
        )

        request = gateway.ingest_control_recommendations(submit_only_report)[0]
        approved = gateway.approve_control_request(request.request_id)

        # No ActionExecutor configured -- honest FAILED, never a
        # fabricated CONFIRMED. Verified through the SAME shared
        # controller the runtime owns.
        self.assertEqual(
            self.runtime.building_control_controller.status_of(request.request_id), RequestStatus.FAILED,
        )
        self.assertEqual(self.runtime.building_control_controller.snapshot().entries, ())

    def test_zero_network_and_zero_hardware_access_anywhere_in_this_demo(self):

        # Structural proof, not a runtime assertion: every collaborator
        # this test builds is an offline/simulation type
        # (ReplayFrameSource, MockHumanDetector, MappingIdentityResolver,
        # SimulationVoiceOutputProvider, SimulationControlProvider) --
        # confirmed by class name, since none of them import a
        # networking/CV library at all (tests/test_no_cv_dependencies.py
        # already enforces this across every package involved).

        for source in self.runtime.frame_sources.values():
            self.assertEqual(type(source).__name__, "ReplayFrameSource")

        self.assertEqual(type(self.runtime.voice_evacuation_controller.provider).__name__, "SimulationVoiceOutputProvider")
        self.assertEqual(type(self.runtime.building_control_controller.provider).__name__, "SimulationControlProvider")


if __name__ == "__main__":
    unittest.main()
