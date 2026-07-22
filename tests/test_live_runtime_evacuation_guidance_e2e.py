import unittest

from models.building import Building
from models.camera import Camera
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.speaker import Speaker
from models.staircase import Staircase
from models.zone import Zone

from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from voice_evacuation.provider import SimulationVoiceOutputProvider

from command_center.live_operator_action_gateway import LiveOperatorActionGateway

from live_runtime.factory import build_live_runtime

from evacuation_guidance.models import RouteStatus

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone, Phase 26
# -- deterministic offline end-to-end proof, driven through the
# COMPLETE production chain (ReplayFrameSource -> Fake YOLO -> Tracker
# -> World Projection -> Behavior -> LiveOccupants -> BuildingState ->
# Crowd Intelligence -> Evacuation Progress -> Trajectory Intelligence
# -> Emergency Response -> Live AI -> Evacuation Recommendation ->
# Evacuation Guidance -> Advisory -> Command Center -> Operator
# Approval -> SimulationVoiceOutputProvider).
#
# Topology: Floor 2 has Z2 (occupants) -- D2 -- Z2B, which itself has
# TWO stairs down to Floor 1: S1 -> Z1 (EXIT-1, shorter) and S2 -> Z4
# (EXIT-2, longer). Initial recommendation is EXIT-1 (shortest); EXIT-1
# then becomes hazardous, migrating the recommendation -- and therefore
# the guidance route -- to EXIT-2.
# =====================================================


CAMERA_Z2 = "CAM-Z2"


def make_building():

    upper = Floor(
        id="f2", name="Floor 2",
        zones=[
            Zone(id="z2", name="Z2", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f2"),
            Zone(id="z2b", name="Z2B", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f2"),
        ],
        cameras=[Camera(id=CAMERA_Z2, name="Z2 Camera", floor_id="f2", zone_ids=("z2",), position=(5.0, 5.0), mount_height=3.0)],
        doors=[Door(id="D2", name="D2", floor_id="f2", zone_a_id="z2", zone_b_id="z2b")],
        stairs=[
            Staircase(id="S1", name="S1", from_zone_id="z2b", to_zone_id="z1", to_floor_id="f1"),
            Staircase(id="S2", name="S2", from_zone_id="z2b", to_zone_id="z4", to_floor_id="f1"),
        ],
        speakers=[Speaker(id="SPK-Z2", name="Speaker Z2", floor_id="f2", zone_ids=("z2",))],
    )

    lower = Floor(
        id="f1", name="Floor 1",
        zones=[
            Zone(id="z1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            Zone(id="z4", name="Z4", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
        ],
        exits=[
            Exit(id="EXIT-1", name="EXIT-1", floor_id="f1", zone_id="z1"),
            Exit(id="EXIT-2", name="EXIT-2", floor_id="f1", zone_id="z4"),
        ],
    )

    return Building(id="guidance-e2e-building", name="Guidance E2E Building", floors=[upper, lower])


def make_world_projector():

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)

    zone = Zone(name="z2", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f2")
    zone.id = "z2"

    calibrations = {
        CAMERA_Z2: CalibrationProfile(
            camera_id=CAMERA_Z2, floor_id="f2", intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(position=(5.0, 5.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0),
        ),
    }

    return WorldProjector(calibrations=calibrations, zones_by_floor={"f2": [zone]})


_CENTER_BOX = (310.0, 230.0, 330.0, 250.0)


class GuidanceMigrationEndToEndTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()

        backend = FakeYOLOBackend()
        for _ in range(4):
            backend.queue_result(person(confidence=0.9, box=_CENTER_BOX))

        frame_sources = {
            CAMERA_Z2: ReplayFrameSource(
                camera_id=CAMERA_Z2, frames=[(0.0, "f0"), (1.0, "f1"), (2.0, "f2"), (3.0, "f3")],
            ),
        }
        for source in frame_sources.values():
            source.start()

        identity_resolver = MappingIdentityResolver({(CAMERA_Z2, f"{CAMERA_Z2}-T1"): "OCC-1"})

        self.provider = SimulationVoiceOutputProvider()

        self.runtime = build_live_runtime(
            self.building,
            frame_sources=frame_sources,
            human_detector=YOLOHumanDetector(backend),
            identity_resolver=identity_resolver,
            tracker=SimpleSingleCameraTracker(),
            behavior_recognizer=RuleBasedBehaviorRecognizer(),
            world_projector=make_world_projector(),
            voice_output_provider=self.provider,
        )

        self.gateway = LiveOperatorActionGateway(voice_controller=self.runtime.voice_evacuation_controller)

        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_full_migration_scenario(self):

        # --- Cycle 0: baseline, EXIT-1 is the shortest safe exit ---
        self.runtime.run_cycle(0.0)

        recommendation = self.runtime.orchestrator.latest_evacuation_recommendation.zone("z2")
        self.assertEqual(recommendation.recommended_exit_id, "EXIT-1")

        guidance_snapshot = self.runtime.orchestrator.latest_evacuation_guidance
        plan = guidance_snapshot.zone("z2")

        self.assertEqual(plan.route_status, RouteStatus.ROUTE_AVAILABLE)
        self.assertEqual(plan.recommended_exit_id, "EXIT-1")
        self.assertEqual(plan.ordered_door_ids, ("D2",))
        self.assertEqual(plan.ordered_stair_ids, ("S1",))

        voice_plan = guidance_snapshot.voice_plan("z2")
        self.assertIsNotNone(voice_plan)
        self.assertEqual(voice_plan.speaker_ids, ("SPK-Z2",))

        # Nothing was broadcast automatically.
        self.assertEqual(self.provider.sent_instructions(), ())

        # --- Operator approves -- the provider receives EXACTLY this message ---
        self.gateway.approve_guidance_message(voice_plan, 0.5)

        sent = self.provider.sent_instructions()
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].target_zone_id, "z2")
        self.assertEqual(sent[0].message.message_text, voice_plan.message_text)
        self.assertIn("EXIT-1", sent[0].message.message_text)

        first_revision = plan.revision

        # --- EXIT-1 becomes hazardous ---
        def hazard_snapshot_provider(time):

            if time < 2.0:
                return HazardSnapshot()

            return HazardSnapshot(node_states={"z1": HazardNodeState(hazard_score=0.9)})

        self.runtime.orchestrator.building_state_gateway._hazard_snapshot_provider = hazard_snapshot_provider

        self.runtime.run_cycle(1.0)
        self.runtime.run_cycle(2.0)
        self.runtime.run_cycle(3.0)

        new_recommendation = self.runtime.orchestrator.latest_evacuation_recommendation.zone("z2")
        self.assertEqual(new_recommendation.recommended_exit_id, "EXIT-2")

        new_guidance_snapshot = self.runtime.orchestrator.latest_evacuation_guidance
        new_plan = new_guidance_snapshot.zone("z2")

        self.assertEqual(new_plan.route_status, RouteStatus.ROUTE_AVAILABLE)
        self.assertEqual(new_plan.recommended_exit_id, "EXIT-2")
        self.assertEqual(new_plan.ordered_stair_ids, ("S2",))
        self.assertGreater(new_plan.revision, first_revision)

        # --- Old guidance remains historical; nothing new was auto-sent ---
        still_sent = self.provider.sent_instructions()
        self.assertEqual(len(still_sent), 1)  # still only the ORIGINAL EXIT-1 message
        self.assertIn("EXIT-1", still_sent[0].message.message_text)

        # --- New guidance requires its own, fresh operator approval ---
        new_voice_plan = new_guidance_snapshot.voice_plan("z2")
        self.assertIsNotNone(new_voice_plan)
        self.assertNotEqual(new_voice_plan.guidance_revision, voice_plan.guidance_revision)

        status = self.gateway.guidance_recommendation_status(new_voice_plan)
        self.assertEqual(status, "RECOMMENDED")  # not auto-approved, not auto-sent

        self.gateway.approve_guidance_message(new_voice_plan, 3.5)

        final_sent = self.provider.sent_instructions()

        # The controller's own pre-existing supersession logic (unchanged
        # by this milestone) logs a synthetic SUPERSEDED marker for the
        # old EXIT-1 message alongside the new EXIT-2 broadcast -- three
        # provider.send() calls total (original EXIT-1 send, EXIT-1's own
        # retroactive supersession marker, new EXIT-2 send), never a
        # silent, un-logged replacement.
        self.assertEqual(len(final_sent), 3)
        self.assertIn("EXIT-2", final_sent[-1].message.message_text)
        self.assertEqual(final_sent[-1].status.name, "BROADCAST")

        # The stale EXIT-1 instruction is still in history, never erased.
        self.assertIn("EXIT-1", final_sent[0].message.message_text)
        self.assertEqual(final_sent[0].status.name, "BROADCAST")
        self.assertEqual(final_sent[1].status.name, "SUPERSEDED")

    def test_no_automatic_execution_anywhere(self):

        self.runtime.run_cycle(0.0)

        self.assertIsNone(self.runtime.building_control_controller)
        self.assertIsNone(self.runtime.facp)
        self.assertEqual(self.provider.sent_instructions(), ())


if __name__ == "__main__":
    unittest.main()
