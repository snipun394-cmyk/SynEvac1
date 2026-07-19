import os
import tempfile
import unittest
from dataclasses import fields

from models.building import Building
from models.speaker import Speaker

from serialization.serializer import Serializer

from speaker_manager.manager import SpeakerManager

from advisory_system.recommendation_models import CivilianAnnouncement

from voice_evacuation.adapter import (
    civilian_announcement_to_voice_message,
    civilian_announcements_to_voice_messages,
)
from voice_evacuation.broadcast_log import BroadcastLog
from voice_evacuation.controller import VoiceEvacuationController
from voice_evacuation.models import (
    BroadcastInstruction,
    BroadcastStatus,
    VoiceMessage,
    VoiceMessageType,
    priority_for_message_type,
)
from voice_evacuation.provider import SimulationVoiceOutputProvider, VoiceOutputProvider


def make_speaker(speaker_id, floor_id="floor1", zone_ids=(), active=True):

    return Speaker(id=speaker_id, floor_id=floor_id, zone_ids=tuple(zone_ids), active=active)


def make_building(*speakers):

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    for speaker in speakers:
        floor.add_speaker(speaker)

    return building, floor


def make_controller(*speakers):

    building, _floor = make_building(*speakers)
    manager = SpeakerManager()
    manager.discover_speakers(building)

    return VoiceEvacuationController(manager, SimulationVoiceOutputProvider()), manager


def make_message(zone_ids, text="Attention.", priority=50, message_type=None, timestamp=0.0):

    return VoiceMessage(
        timestamp=timestamp, target_zone_ids=tuple(zone_ids), message_text=text,
        message_type=message_type, priority=priority,
    )


def make_announcement(zone_id, text, confidence=0.8):

    return CivilianAnnouncement(
        zone_id=zone_id, zone_name=zone_id, announcement=text, reason="r",
        confidence=confidence, predicted_rset_improvement_seconds=None,
    )


# =====================================================
# Phase 13 validation
# =====================================================


class ZoneToSpeakerRoutingTests(unittest.TestCase):

    def test_one_zone_one_speaker(self):

        controller, _manager = make_controller(make_speaker("SPK-1", zone_ids=("zone-a",)))

        instructions = controller.broadcast(make_message(["zone-a"]), time=1.0)

        self.assertEqual(len(instructions), 1)
        self.assertEqual(instructions[0].status, BroadcastStatus.BROADCAST)
        self.assertEqual(instructions[0].speaker_ids, ("SPK-1",))

    def test_one_zone_multiple_speakers(self):

        controller, _manager = make_controller(
            make_speaker("SPK-1", zone_ids=("zone-a",)),
            make_speaker("SPK-2", zone_ids=("zone-a",)),
        )

        instructions = controller.broadcast(make_message(["zone-a"]), time=1.0)

        self.assertEqual(instructions[0].speaker_ids, ("SPK-1", "SPK-2"))

    def test_one_speaker_multiple_zones(self):

        controller, _manager = make_controller(
            make_speaker("SPK-1", zone_ids=("zone-a", "zone-b")),
        )

        instructions = controller.broadcast(make_message(["zone-a", "zone-b"]), time=1.0)

        self.assertEqual(len(instructions), 2)
        for instruction in instructions:
            self.assertIn("SPK-1", instruction.speaker_ids)

    def test_disabled_speaker_is_not_selected(self):

        controller, _manager = make_controller(
            make_speaker("SPK-1", zone_ids=("zone-a",), active=False),
        )

        instructions = controller.broadcast(make_message(["zone-a"]), time=1.0)

        self.assertEqual(instructions[0].status, BroadcastStatus.NO_SPEAKERS_AVAILABLE)
        self.assertEqual(instructions[0].speaker_ids, ())

    def test_zone_with_no_speaker_at_all_reports_unavailable_honestly(self):

        controller, _manager = make_controller(make_speaker("SPK-1", zone_ids=("zone-b",)))

        instructions = controller.broadcast(make_message(["zone-a"]), time=1.0)

        self.assertEqual(instructions[0].status, BroadcastStatus.NO_SPEAKERS_AVAILABLE)


class MultiZoneIsolationTests(unittest.TestCase):

    def test_two_zones_receive_different_recommendations_simultaneously(self):

        controller, _manager = make_controller(
            make_speaker("SPK-A", zone_ids=("zone-a",)),
            make_speaker("SPK-B", zone_ids=("zone-b",)),
        )

        message_a = make_message(["zone-a"], text="Use Exit 3.")
        message_b = make_message(["zone-b"], text="Avoid Exit 3. Use Exit 5.")

        controller.broadcast(message_a, time=1.0)
        controller.broadcast(message_b, time=1.0)

        self.assertEqual(controller.active_message_for_zone("zone-a").message_text, "Use Exit 3.")
        self.assertEqual(controller.active_message_for_zone("zone-b").message_text, "Avoid Exit 3. Use Exit 5.")

    def test_contradictory_recommendations_in_different_zones_remain_isolated(self):

        # A lower-priority message for zone-b must never be affected by
        # a higher-priority message active in zone-a -- each zone's
        # priority/supersession state is completely independent.
        controller, _manager = make_controller(
            make_speaker("SPK-A", zone_ids=("zone-a",)),
            make_speaker("SPK-B", zone_ids=("zone-b",)),
        )

        controller.broadcast(
            make_message(["zone-a"], text="SHELTER", priority=priority_for_message_type(VoiceMessageType.SHELTER_IN_PLACE)),
            time=1.0,
        )
        controller.broadcast(
            make_message(["zone-b"], text="ROUTE", priority=priority_for_message_type(VoiceMessageType.ROUTE_GUIDANCE)),
            time=1.0,
        )

        self.assertEqual(controller.active_message_for_zone("zone-a").message_text, "SHELTER")
        self.assertEqual(controller.active_message_for_zone("zone-b").message_text, "ROUTE")

    def test_multi_zone_broadcast_produces_one_instruction_per_zone(self):

        controller, _manager = make_controller(
            make_speaker("SPK-A", zone_ids=("zone-a",)),
            make_speaker("SPK-B", zone_ids=("zone-b",)),
        )

        instructions = controller.broadcast(make_message(["zone-a", "zone-b"]), time=1.0)

        self.assertEqual({i.target_zone_id for i in instructions}, {"zone-a", "zone-b"})

    def test_building_wide_broadcast_reaches_every_zone_with_a_speaker(self):

        controller, _manager = make_controller(
            make_speaker("SPK-A", zone_ids=("zone-a",)),
            make_speaker("SPK-B", zone_ids=("zone-b",)),
        )

        instructions = controller.broadcast_building_wide(
            make_message([], text="All Clear.", priority=priority_for_message_type(VoiceMessageType.ALL_CLEAR)),
            time=5.0,
        )

        self.assertEqual({i.target_zone_id for i in instructions}, {"zone-a", "zone-b"})
        self.assertTrue(all(i.status == BroadcastStatus.BROADCAST for i in instructions))


class PriorityAndSupersessionTests(unittest.TestCase):

    def test_message_priority_ladder_is_fixed_and_documented(self):

        self.assertGreater(
            priority_for_message_type(VoiceMessageType.SHELTER_IN_PLACE),
            priority_for_message_type(VoiceMessageType.EVACUATE),
        )
        self.assertGreater(
            priority_for_message_type(VoiceMessageType.EVACUATE),
            priority_for_message_type(VoiceMessageType.ROUTE_GUIDANCE),
        )
        self.assertGreater(
            priority_for_message_type(VoiceMessageType.ROUTE_GUIDANCE),
            priority_for_message_type(VoiceMessageType.ALL_CLEAR),
        )
        # Unclassified must never be assumed the more urgent type.
        self.assertEqual(
            priority_for_message_type(None), priority_for_message_type(VoiceMessageType.ROUTE_GUIDANCE),
        )

    def test_new_recommendation_supersedes_old_in_the_same_zone(self):

        controller, _manager = make_controller(make_speaker("SPK-1", zone_ids=("zone-a",)))

        low = make_message(["zone-a"], text="low", priority=10, timestamp=1.0)
        high = make_message(["zone-a"], text="high", priority=90, timestamp=2.0)

        controller.broadcast(low, time=1.0)
        controller.broadcast(high, time=2.0)

        self.assertEqual(controller.active_message_for_zone("zone-a").message_text, "high")

        statuses = [i.status for i in controller.broadcast_log.by_zone("zone-a")]
        self.assertIn(BroadcastStatus.SUPERSEDED, statuses)
        self.assertIn(BroadcastStatus.BROADCAST, statuses)

    def test_lower_priority_message_does_not_supersede_a_higher_priority_active_one(self):

        controller, _manager = make_controller(make_speaker("SPK-1", zone_ids=("zone-a",)))

        high = make_message(["zone-a"], text="high", priority=90, timestamp=1.0)
        low = make_message(["zone-a"], text="low", priority=10, timestamp=2.0)

        controller.broadcast(high, time=1.0)
        instructions = controller.broadcast(low, time=2.0)

        self.assertEqual(controller.active_message_for_zone("zone-a").message_text, "high")
        self.assertEqual(instructions[0].status, BroadcastStatus.SUPERSEDED)

    def test_message_cancellation(self):

        controller, _manager = make_controller(make_speaker("SPK-1", zone_ids=("zone-a",)))

        controller.broadcast(make_message(["zone-a"]), time=1.0)
        self.assertIsNotNone(controller.active_message_for_zone("zone-a"))

        cancelled = controller.cancel("zone-a", time=2.0)

        self.assertIsNone(controller.active_message_for_zone("zone-a"))
        self.assertEqual(cancelled.status, BroadcastStatus.CANCELLED)

    def test_cancelling_a_zone_with_no_active_message_is_a_no_op(self):

        controller, _manager = make_controller(make_speaker("SPK-1", zone_ids=("zone-a",)))

        self.assertIsNone(controller.cancel("zone-a", time=1.0))


class DeterministicOrderingTests(unittest.TestCase):

    def test_broadcast_ordering_is_sorted_by_zone_id_regardless_of_input_order(self):

        controller, _manager = make_controller(
            make_speaker("SPK-A", zone_ids=("zone-a",)),
            make_speaker("SPK-B", zone_ids=("zone-b",)),
            make_speaker("SPK-C", zone_ids=("zone-c",)),
        )

        message = make_message(["zone-c", "zone-a", "zone-b"])
        instructions = controller.broadcast(message, time=1.0)

        self.assertEqual([i.target_zone_id for i in instructions], ["zone-a", "zone-b", "zone-c"])

    def test_repeated_runs_produce_the_same_sequence(self):

        message = make_message(["zone-b", "zone-a"])

        controller_a, _ = make_controller(
            make_speaker("SPK-A", zone_ids=("zone-a",)), make_speaker("SPK-B", zone_ids=("zone-b",)),
        )
        controller_b, _ = make_controller(
            make_speaker("SPK-A", zone_ids=("zone-a",)), make_speaker("SPK-B", zone_ids=("zone-b",)),
        )

        result_a = [(i.target_zone_id, i.status) for i in controller_a.broadcast(message, time=1.0)]
        result_b = [(i.target_zone_id, i.status) for i in controller_b.broadcast(message, time=1.0)]

        self.assertEqual(result_a, result_b)


class SpeakerIdentityTests(unittest.TestCase):

    def test_stable_speaker_ids_across_rediscovery(self):

        building, floor = make_building(make_speaker("SPK-001", zone_ids=("zone-a",)))

        manager = SpeakerManager()
        manager.discover_speakers(building)
        manager.discover_speakers(building)

        self.assertEqual(manager.speaker_status("SPK-001").speaker_id, "SPK-001")
        self.assertEqual(len(manager.all_speakers()), 1)

    def test_save_reload_speaker_assets(self):

        building, floor = make_building(
            Speaker(
                id="SPK-CAF-01", name="Cafeteria Speaker", floor_id="floor1",
                zone_ids=("zone-a",), speaker_type="Horn", volume_level=90.0,
            ),
        )

        from models.project import Project

        project = Project(name="Speaker Save Test")
        project.set_building(building)

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = os.path.join(tmp_dir, "roundtrip.syn")
            Serializer.save(project, path)
            reloaded_project = Serializer.load(path)

        reloaded_speaker = reloaded_project.building.floors[0].speakers[0]

        self.assertEqual(reloaded_speaker.id, "SPK-CAF-01")
        self.assertEqual(reloaded_speaker.zone_ids, ("zone-a",))
        self.assertEqual(reloaded_speaker.speaker_type, "Horn")
        self.assertEqual(reloaded_speaker.volume_level, 90.0)

        manager = SpeakerManager()
        manager.discover_speakers(reloaded_project.building)

        self.assertEqual(manager.speaker_status("SPK-CAF-01").speaker_id, "SPK-CAF-01")

    def test_legacy_project_with_no_speakers_key_loads_without_error(self):

        from models.floor import Floor

        legacy_floor_data = {
            "id": "floor-legacy", "name": "Ground Floor", "display_order": 0, "height": 3.0,
            "floor_plan": "", "visible": True, "locked": False,
            "zones": [], "exits": [], "stairs": [], "elevators": [], "cameras": [], "detectors": [],
            "smoke_detectors": [], "heat_detectors": [],
            # Deliberately no "speakers" key at all -- simulates a
            # project saved before this milestone existed.
            "assembly_points": [], "obstacles": [], "doors": [],
        }

        floor = Floor.from_dict(legacy_floor_data)

        self.assertEqual(floor.speakers, [])
        self.assertEqual(floor.speaker_count, 0)


class CivilianAnnouncementAdapterTests(unittest.TestCase):

    def test_civilian_announcement_to_voice_message(self):

        announcement = make_announcement("zone-a", "Attention occupants in Cafeteria. Proceed to Exit 3.")

        message = civilian_announcement_to_voice_message(announcement, timestamp=32.0)

        self.assertEqual(message.target_zone_ids, ("zone-a",))
        self.assertEqual(message.message_text, announcement.announcement)
        self.assertEqual(message.confidence, 0.8)
        self.assertIsNone(message.message_type)
        self.assertEqual(message.priority, priority_for_message_type(None))

    def test_zone_action_classifies_message_type(self):

        announcement = make_announcement("zone-a", "Attention occupants in Cafeteria. Remain in place.")

        message = civilian_announcement_to_voice_message(
            announcement, timestamp=32.0, zone_action="SHELTER_IN_PLACE",
        )

        self.assertEqual(message.message_type, VoiceMessageType.SHELTER_IN_PLACE)
        self.assertEqual(message.priority, priority_for_message_type(VoiceMessageType.SHELTER_IN_PLACE))

    def test_batch_adapter_produces_one_message_per_announcement(self):

        announcements = (
            make_announcement("zone-a", "Attention occupants in Cafeteria. Use Exit 3."),
            make_announcement("zone-b", "Attention occupants in Laboratory Wing. Avoid Exit 3. Use Exit 5."),
        )

        messages = civilian_announcements_to_voice_messages(announcements, timestamp=32.0)

        self.assertEqual(len(messages), 2)
        self.assertEqual({m.target_zone_ids[0] for m in messages}, {"zone-a", "zone-b"})


class BroadcastLogQueryTests(unittest.TestCase):

    def test_query_by_zone_status_and_time(self):

        log = BroadcastLog()

        instruction_a = BroadcastInstruction(timestamp=1.0, target_zone_id="zone-a", status=BroadcastStatus.BROADCAST)
        instruction_b = BroadcastInstruction(timestamp=5.0, target_zone_id="zone-b", status=BroadcastStatus.NO_SPEAKERS_AVAILABLE)

        log.append(instruction_a)
        log.append(instruction_b)

        self.assertEqual(log.by_zone("zone-a"), (instruction_a,))
        self.assertEqual(log.by_status(BroadcastStatus.NO_SPEAKERS_AVAILABLE), (instruction_b,))
        self.assertEqual(log.between(0.0, 2.0), (instruction_a,))
        self.assertEqual(len(log), 2)


# =====================================================
# End-to-end: AdvisoryReport -> CivilianAnnouncement -> VoiceMessage ->
# VoiceEvacuationController -> SpeakerManager -> SimulationVoiceOutputProvider
# -> Broadcast History. Two zones, two different recommendations, each
# zone must receive only its own intended message.
# =====================================================


class EndToEndVoiceEvacuationPipelineTests(unittest.TestCase):

    def test_two_zones_each_receive_only_their_own_message(self):

        building, floor = make_building(
            Speaker(id="SPK-CAF-01", floor_id="floor1", zone_ids=("cafeteria",)),
            Speaker(id="SPK-CAF-02", floor_id="floor1", zone_ids=("cafeteria",)),
            Speaker(id="SPK-LAB-01", floor_id="floor1", zone_ids=("lab",)),
        )

        speaker_manager = SpeakerManager()
        speaker_manager.discover_speakers(building)

        provider = SimulationVoiceOutputProvider()
        controller = VoiceEvacuationController(speaker_manager, provider)

        cafeteria_announcement = make_announcement(
            "cafeteria", "Attention occupants in Cafeteria. Proceed to Exit 3.", confidence=0.9,
        )
        lab_announcement = make_announcement(
            "lab", "Attention occupants in Laboratory Wing. Remain in place. Await further instructions.",
            confidence=0.7,
        )

        messages = civilian_announcements_to_voice_messages(
            (cafeteria_announcement, lab_announcement), timestamp=32.0,
            zone_actions={"cafeteria": "EVACUATE_IMMEDIATELY", "lab": "SHELTER_IN_PLACE"},
        )

        all_instructions = []
        for message in messages:
            all_instructions.extend(controller.broadcast(message, time=32.0))

        cafeteria_instruction = next(i for i in all_instructions if i.target_zone_id == "cafeteria")
        lab_instruction = next(i for i in all_instructions if i.target_zone_id == "lab")

        self.assertEqual(cafeteria_instruction.status, BroadcastStatus.BROADCAST)
        self.assertEqual(set(cafeteria_instruction.speaker_ids), {"SPK-CAF-01", "SPK-CAF-02"})
        self.assertIn("Exit 3", cafeteria_instruction.message.message_text)
        self.assertEqual(cafeteria_instruction.message.message_type, VoiceMessageType.EVACUATE)

        self.assertEqual(lab_instruction.status, BroadcastStatus.BROADCAST)
        self.assertEqual(lab_instruction.speaker_ids, ("SPK-LAB-01",))
        self.assertIn("Remain in place", lab_instruction.message.message_text)
        self.assertEqual(lab_instruction.message.message_type, VoiceMessageType.SHELTER_IN_PLACE)

        # Neither zone's speakers appear in the other's instruction.
        self.assertNotIn("SPK-LAB-01", cafeteria_instruction.speaker_ids)
        self.assertNotIn("SPK-CAF-01", lab_instruction.speaker_ids)
        self.assertNotIn("SPK-CAF-02", lab_instruction.speaker_ids)

        # The provider's own record matches the broadcast log exactly --
        # a deterministic, inspectable simulation record (Phase 8).
        self.assertEqual(
            {i.instruction_id for i in provider.sent_instructions()},
            {i.instruction_id for i in controller.broadcast_log.all_instructions()},
        )


# =====================================================
# Phase 14 architecture guards
# =====================================================


class ArchitectureGuardTests(unittest.TestCase):

    def test_voice_message_cannot_carry_an_occupant_id(self):

        field_names = {f.name for f in fields(VoiceMessage)}

        for name in field_names:
            self.assertNotIn("occupant", name.lower())

    def test_broadcast_instruction_cannot_carry_an_occupant_id(self):

        field_names = {f.name for f in fields(BroadcastInstruction)}

        for name in field_names:
            self.assertNotIn("occupant", name.lower())

    def test_voice_evacuation_never_imports_ai_or_rl_training_or_hazard_physics(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "voice_evacuation"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(ai_training|rl_training|ai_decision|ai_inference|hazard|hazard_evolution|"
            r"fire_growth|smoke_propagation|facp|simulator|sandbox|designer|"
            r"gymnasium|gym|numpy|torch)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"voice_evacuation/{path.name} imports a forbidden module -- Voice Evacuation "
                f"must never generate evacuation decisions, calculate hazard physics, or "
                f"couple to FACP/AI/RL training.",
            )

    def test_voice_evacuation_has_no_vendor_hardware_protocol_dependencies(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "voice_evacuation"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(pymodbus|modbus|bacpypes|bacnet|paho|mqtt|sip|pjsua|dante|onvif|"
            r"pyserial|serial|socket|requests)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE | re.IGNORECASE),
                f"voice_evacuation/{path.name} imports a vendor PA/voice-alarm protocol "
                f"library -- real hardware communication must not be implemented yet.",
            )

    def test_speaker_manager_does_not_perform_recommendation_reasoning(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "speaker_manager"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(advisory_system|decision_policy|ai_decision|voice_evacuation|"
            r"camera_manager|sensor_manager|facp|hazard|ai_training|rl_training)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"speaker_manager/{path.name} imports a recommendation/reasoning or "
                f"sibling-manager module -- SpeakerManager must only manage Speaker assets.",
            )

    def test_facp_and_voice_evacuation_remain_separate(self):

        import pathlib
        import re

        repo_root = pathlib.Path(__file__).resolve().parent.parent

        facp_text = "".join(p.read_text() for p in (repo_root / "facp").glob("*.py"))
        voice_text = "".join(p.read_text() for p in (repo_root / "voice_evacuation").glob("*.py"))

        self.assertIsNone(re.search(r"^\s*(from|import)\s+voice_evacuation\b", facp_text, re.MULTILINE))
        self.assertIsNone(re.search(r"^\s*(from|import)\s+facp\b", voice_text, re.MULTILINE))

    def test_controller_never_generates_message_text_of_its_own(self):

        # The controller only ever routes whatever VoiceMessage.message_text
        # it was given -- it must never format/construct a new string
        # itself. A crude but effective guard: the module source contains
        # no f-string/format() literal producing announcement-shaped text.
        import pathlib

        text = (pathlib.Path(__file__).resolve().parent.parent / "voice_evacuation" / "controller.py").read_text()

        self.assertNotIn("Attention occupants", text)
        self.assertNotIn(".format(", text)

    def test_simulation_provider_never_imports_an_audio_or_tts_library(self):

        import pathlib
        import re

        text = (pathlib.Path(__file__).resolve().parent.parent / "voice_evacuation" / "provider.py").read_text()

        forbidden = r"^\s*(from|import)\s+(pyttsx3|gtts|pyaudio|sounddevice|wave|pydub)\b"

        self.assertIsNone(re.search(forbidden, text, re.MULTILINE | re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
