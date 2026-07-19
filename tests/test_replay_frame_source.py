import unittest
from pathlib import Path

from live_camera_pipeline.replay_frame_source import ReplayFrameSource


class ReplayFrameSourceDeterministicPlaybackTests(unittest.TestCase):

    # CCTV Pipeline End-to-End Offline Validation, Phase 2/9 items 1-2:
    # a concrete, production ReplayFrameSource -- not a test double --
    # must play back a fixed, ordered sequence of frames deterministically,
    # preserving camera_id on every single frame it produces.

    def setUp(self):

        self.source = ReplayFrameSource(
            camera_id="CAM-001",
            frames=[
                (0.0, [{"local_track_id": "1"}]),
                (1.0, [{"local_track_id": "1"}]),
                (2.0, None),
            ],
        )

    def test_no_frames_before_start(self):

        self.assertFalse(self.source.is_running)
        self.assertIsNone(self.source.read_frame())

    def test_deterministic_ordered_playback(self):

        self.source.start()

        first = self.source.read_frame()
        second = self.source.read_frame()
        third = self.source.read_frame()
        fourth = self.source.read_frame()

        self.assertEqual(first.timestamp, 0.0)
        self.assertEqual(first.frame_sequence, 0)

        self.assertEqual(second.timestamp, 1.0)
        self.assertEqual(second.frame_sequence, 1)

        self.assertEqual(third.timestamp, 2.0)
        self.assertEqual(third.frame_sequence, 2)
        self.assertIsNone(third.payload_ref)

        # Exhausted -- honest None, not a crash, not a repeat.
        self.assertIsNone(fourth)

    def test_camera_id_preserved_on_every_frame(self):

        self.source.start()

        for _ in range(3):

            frame = self.source.read_frame()
            self.assertEqual(frame.camera_id, "CAM-001")

    def test_stop_then_read_returns_none(self):

        self.source.start()
        self.source.read_frame()
        self.source.stop()

        self.assertFalse(self.source.is_running)
        self.assertIsNone(self.source.read_frame())

    def test_reset_replays_the_same_sequence(self):

        self.source.start()

        first_pass = [self.source.read_frame() for _ in range(3)]

        self.source.reset()

        second_pass = [self.source.read_frame() for _ in range(3)]

        self.assertEqual(
            [f.timestamp for f in first_pass], [f.timestamp for f in second_pass],
        )
        self.assertEqual(
            [f.frame_sequence for f in first_pass], [f.frame_sequence for f in second_pass],
        )


class ReplayFrameSourceMissingSourceTests(unittest.TestCase):

    # Phase 9 item 12: a Replay source with nothing to play must fail
    # gracefully -- read_frame() returns None forever, never raises,
    # and is_source_available honestly reports the gap so a caller (the
    # Camera Manager panel, Phase 8) can surface "Source Missing"
    # instead of a silent, indistinguishable-from-empty stream.

    def test_empty_source_is_not_available(self):

        source = ReplayFrameSource(camera_id="CAM-002")

        self.assertFalse(source.is_source_available)

        source.start()
        self.assertIsNone(source.read_frame())

    def test_nonexistent_source_path_is_not_available(self):

        source = ReplayFrameSource(
            camera_id="CAM-002",
            source_path=Path("C:/definitely/does/not/exist/replay.mp4"),
        )

        self.assertFalse(source.is_source_available)

        source.start()
        self.assertIsNone(source.read_frame())

    def test_in_memory_frames_are_available_without_a_source_path(self):

        source = ReplayFrameSource(
            camera_id="CAM-002", frames=[(0.0, None)],
        )

        self.assertTrue(source.is_source_available)

    def test_existing_source_path_is_available(self):

        # The path is never opened/decoded by this class (Phase 2's own
        # "no cv2 in this package" constraint) -- only checked for
        # existence.
        source = ReplayFrameSource(
            camera_id="CAM-002", source_path=Path(__file__),
        )

        self.assertTrue(source.is_source_available)


if __name__ == "__main__":
    unittest.main()
