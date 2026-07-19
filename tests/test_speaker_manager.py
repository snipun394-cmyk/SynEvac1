import unittest

from models.building import Building
from models.speaker import Speaker

from speaker_manager.manager import SpeakerManager
from speaker_manager.status import SpeakerStatus


def make_speaker(name, floor_id, zone_ids=(), **overrides):

    fields = dict(name=name, floor_id=floor_id, zone_ids=tuple(zone_ids))
    fields.update(overrides)

    return Speaker(**fields)


class DiscoveryTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor_1 = self.building.create_floor(name="Ground Floor")
        self.floor_2 = self.building.create_floor(name="Floor 1", height=3.0)

        self.speaker_1 = make_speaker("Speaker 1", self.floor_1.id, zone_ids=("zone-a",))
        self.speaker_2 = make_speaker("Speaker 2", self.floor_2.id, zone_ids=("zone-b",))

        self.floor_1.add_speaker(self.speaker_1)
        self.floor_2.add_speaker(self.speaker_2)

        self.manager = SpeakerManager()

    def test_discover_speakers_finds_every_floor(self):

        discovered = self.manager.discover_speakers(self.building)

        self.assertEqual(
            {speaker.id for speaker in discovered}, {self.speaker_1.id, self.speaker_2.id},
        )
        self.assertEqual(len(self.manager.all_speakers()), 2)

    def test_discover_with_no_building_clears_the_registry(self):

        self.manager.discover_speakers(self.building)
        self.assertEqual(len(self.manager.all_speakers()), 2)

        result = self.manager.discover_speakers(None)

        self.assertEqual(result, ())
        self.assertEqual(self.manager.all_speakers(), ())

    def test_rediscovering_picks_up_a_newly_added_speaker(self):

        self.manager.discover_speakers(self.building)

        speaker_3 = make_speaker("Speaker 3", self.floor_1.id)
        self.floor_1.add_speaker(speaker_3)

        self.manager.discover_speakers(self.building)

        self.assertEqual(len(self.manager.all_speakers()), 3)
        self.assertIs(self.manager.get_speaker(speaker_3.id), speaker_3)

    def test_rediscovering_drops_a_removed_speaker(self):

        self.manager.discover_speakers(self.building)

        self.floor_1.remove_speaker(self.speaker_1)

        self.manager.discover_speakers(self.building)

        self.assertIsNone(self.manager.get_speaker(self.speaker_1.id))
        self.assertEqual(len(self.manager.all_speakers()), 1)


class LookupTests(unittest.TestCase):

    def setUp(self):

        self.manager = SpeakerManager()

        self.speaker_a = make_speaker("A", "floor-1", zone_ids=("zone-1",))
        self.speaker_b = make_speaker("B", "floor-1", zone_ids=("zone-2",))
        self.speaker_c = make_speaker("C", "floor-2", zone_ids=("zone-1",))

        for speaker in (self.speaker_a, self.speaker_b, self.speaker_c):
            self.manager.register_speaker(speaker)

    def test_get_speaker(self):

        self.assertIs(self.manager.get_speaker(self.speaker_a.id), self.speaker_a)
        self.assertIsNone(self.manager.get_speaker("no-such-id"))

    def test_speakers_on_floor(self):

        ids_on_floor_1 = {s.id for s in self.manager.speakers_on_floor("floor-1")}
        self.assertEqual(ids_on_floor_1, {self.speaker_a.id, self.speaker_b.id})

    def test_speakers_in_zone(self):

        ids_in_zone_1 = {s.id for s in self.manager.speakers_in_zone("zone-1")}
        self.assertEqual(ids_in_zone_1, {self.speaker_a.id, self.speaker_c.id})
        self.assertEqual(self.manager.speakers_in_zone("no-such-zone"), ())

    def test_active_speakers_in_zone_excludes_disabled(self):

        self.manager.disable_speaker(self.speaker_a.id)

        active_ids = {s.id for s in self.manager.active_speakers_in_zone("zone-1")}
        self.assertEqual(active_ids, {self.speaker_c.id})


class EnableDisableTests(unittest.TestCase):

    def test_disable_then_enable(self):

        manager = SpeakerManager()
        speaker = make_speaker("Speaker", "floor-1", active=True)
        manager.register_speaker(speaker)

        manager.disable_speaker(speaker.id)
        self.assertFalse(speaker.active)

        manager.enable_speaker(speaker.id)
        self.assertTrue(speaker.active)

    def test_enable_unknown_speaker_raises(self):

        manager = SpeakerManager()

        with self.assertRaises(KeyError):
            manager.enable_speaker("no-such-speaker")

    def test_disable_unknown_speaker_raises(self):

        manager = SpeakerManager()

        with self.assertRaises(KeyError):
            manager.disable_speaker("no-such-speaker")


class StatusTests(unittest.TestCase):

    def test_speaker_status_reflects_current_fields(self):

        manager = SpeakerManager()
        speaker = make_speaker("Speaker", "floor-1", zone_ids=("zone-1",), active=True)
        manager.register_speaker(speaker)

        status = manager.speaker_status(speaker.id)

        self.assertIsInstance(status, SpeakerStatus)
        self.assertEqual(status.speaker_id, speaker.id)
        self.assertEqual(status.floor_id, "floor-1")
        self.assertEqual(status.zone_ids, ("zone-1",))
        self.assertTrue(status.active)

    def test_status_for_unknown_speaker_raises(self):

        manager = SpeakerManager()

        with self.assertRaises(KeyError):
            manager.speaker_status("no-such-speaker")

    def test_all_statuses_covers_every_speaker(self):

        manager = SpeakerManager()
        speaker_a = make_speaker("A", "floor-1")
        speaker_b = make_speaker("B", "floor-1")
        manager.register_speaker(speaker_a)
        manager.register_speaker(speaker_b)

        statuses = manager.all_statuses()

        self.assertEqual({s.speaker_id for s in statuses}, {speaker_a.id, speaker_b.id})


class SpeakerManagerPackageDependencyDirectionTests(unittest.TestCase):

    # Same regex-scan-the-source-files convention every other package
    # boundary in this codebase enforces (sensor_manager/camera_manager's
    # own identical tests) -- SpeakerManager must manage Speaker assets
    # only, never couple to a sibling manager or perform recommendation
    # reasoning/broadcasting itself.

    def test_never_imports_sibling_managers_or_reasoning_packages(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "speaker_manager"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(camera_manager|sensor_manager|virtual_camera|visibility|multi_camera_fusion|"
            r"camera_validation|facp|voice_evacuation|advisory_system|decision_policy|"
            r"ai_decision|simulator|ground_truth|behavior|behavior_library|"
            r"behaviour_profile_resolver|simulation_runtime|hazard|hazard_evolution|"
            r"ai_training|rl_training|advisory_system|command_center|designer)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"speaker_manager/{path.name} imports a sibling-manager, reasoning, or "
                f"simulation module -- it must only manage Speaker assets.",
            )


if __name__ == "__main__":
    unittest.main()
