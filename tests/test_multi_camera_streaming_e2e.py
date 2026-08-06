import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from models.building import Building
from models.floor import Floor
from models.camera import Camera
from models.engineering_asset import ConnectionInfo, DeviceMode

from camera_manager.manager import CameraManager

from live_runtime_launcher.rtsp_camera_sources import build_rtsp_frame_sources
from live_runtime.factory import build_live_runtime

from live_camera_pipeline.pipeline import LiveCameraPipeline
from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver

from command_center.live_camera_view_gateway import LiveCameraViewGateway
from command_center.live_camera_grid_panel import LiveCameraGridPanel

from tests.live_camera_pipeline_fixtures import FakeRTSPBackend, MockHumanDetector


# =====================================================
# Multi-Camera Streaming Architecture milestone -- the milestone's own
# explicit requirement: "The architecture must scale naturally from 1
# camera -> 4 -> 16 -> 32 -> 100+ cameras without redesign." This test
# proves the FULL production spine end to end, at every one of those
# scale points, with zero LAN/real network/real decoder dependency
# (FakeRTSPBackend/MockHumanDetector, the same offline doubles
# tests/test_rtsp_camera_sources.py and tests/test_live_camera_pipeline.py
# already use):
#
#   Building (N LIVE cameras)
#     -> build_rtsp_frame_sources()   -- exactly N RTSPFrameSource objects
#     -> build_live_runtime()         -- ONE shared LiveCameraPipeline,
#                                         ONE CameraManager holding all N,
#                                         ONE LiveCameraViewGateway
#     -> LiveCameraViewGateway.camera_tiles()  -- exactly N tiles
#     -> LiveCameraGridPanel.show_live()       -- exactly N grid tiles
#
# No RTSPFrameSource.start()/read_frame() is ever called here -- this
# test proves CONSTRUCTION/OWNERSHIP/COUNTING (Investigation questions
# 1-6 of this milestone), never a real stream, matching the milestone's
# own "No LAN verification required today" scope.
# =====================================================


def _building_with_n_live_cameras(n: int) -> Building:

    cameras = [
        Camera(
            id=f"CAM-{i}", name=f"Camera {i}", floor_id="floor-1", mode=DeviceMode.LIVE,
            connection=ConnectionInfo(
                rtsp_address=f"rtsp://fake-nvr/cam/{i}", credential_ref=None,
            ),
        )
        for i in range(n)
    ]

    floor = Floor(id="floor-1", name="Ground Floor", cameras=cameras)
    return Building(id="b1", name="Test Building", floors=[floor])


class MultiCameraStreamingScaleTests(unittest.TestCase):

    def _build_full_stack(self, n: int):

        building = _building_with_n_live_cameras(n)
        camera_manager = CameraManager()

        frame_sources = build_rtsp_frame_sources(
            building, camera_manager, credential_store=None,
            decoder_backend_factory=FakeRTSPBackend,
        )

        runtime = build_live_runtime(
            building,
            frame_sources=frame_sources,
            human_detector=MockHumanDetector(),
            identity_resolver=SimulationIdentityResolver(),
            camera_manager=camera_manager,
        )

        return building, camera_manager, frame_sources, runtime

    def test_scales_from_one_to_over_one_hundred_cameras_without_redesign(self):

        for n in (1, 4, 16, 32, 100):

            with self.subTest(camera_count=n):

                building, camera_manager, frame_sources, runtime = self._build_full_stack(n)

                # Q1/Q6 -- exactly one RTSPFrameSource per configured camera,
                # every one a genuinely distinct object with its own decoder
                # (never a shared/batched connection).
                self.assertEqual(len(frame_sources), n)
                self.assertEqual(set(frame_sources.keys()), {f"CAM-{i}" for i in range(n)})
                self.assertEqual(len({id(source) for source in frame_sources.values()}), n)
                self.assertEqual(len({id(source._decoder) for source in frame_sources.values()}), n)
                for source in frame_sources.values():
                    self.assertIsInstance(source, RTSPFrameSource)

                # Q2 -- exactly ONE shared LiveCameraPipeline for every
                # camera, never one per camera.
                self.assertIsInstance(runtime.camera_pipeline, LiveCameraPipeline)
                self.assertEqual(len(runtime.camera_pipeline.frame_sources), n)

                # CameraManager holds all N Camera assets, one gateway
                # shared across all of them.
                self.assertEqual(len(camera_manager.all_cameras()), n)
                self.assertIsInstance(runtime.camera_view_gateway, LiveCameraViewGateway)

                # Q6 -- the gateway produces exactly one tile per camera.
                tiles = runtime.camera_view_gateway.camera_tiles()
                self.assertEqual(len(tiles), n)
                self.assertEqual({tile.camera_id for tile in tiles}, {f"CAM-{i}" for i in range(n)})
                self.assertTrue(all(tile.configured for tile in tiles))

                # Live CCTV Grid -- builds exactly N tiles, dynamically,
                # no hardcoded ceiling, scrolls cleanly (QScrollArea
                # already proven separately at 32 in
                # tests/test_command_center_live_camera_grid_panel.py;
                # this proves the panel accepts the REAL gateway's own
                # output, not just a hand-built fake tile list).
                panel = LiveCameraGridPanel()
                panel.show_live(runtime.camera_view_gateway)

                self.assertEqual(len(panel._tiles), n)
                self.assertEqual(panel._grid_layout.count(), n)
                self.assertFalse(panel._scroll_area.isHidden())

    def test_no_frame_source_is_ever_shared_across_two_cameras(self):

        # A direct regression guard for "no duplicate stream ownership" --
        # even camera_id collision-adjacent ids (CAM-1 vs CAM-10 vs
        # CAM-100) must never resolve to the same RTSPFrameSource object.

        _, _, frame_sources, _ = self._build_full_stack(16)

        seen_ids = set()

        for camera_id, source in frame_sources.items():

            self.assertEqual(source.camera_id, camera_id)
            self.assertNotIn(id(source), seen_ids)
            seen_ids.add(id(source))

    def test_removing_a_camera_from_the_building_never_leaves_a_stale_frame_source(self):

        # Q7 -- confirms the count genuinely tracks the Building's own
        # configured cameras each time the stack is rebuilt (the honest
        # "rebuild rather than incrementally patch" convention this
        # codebase already establishes for CameraManager.discover_cameras()),
        # not some accumulated/leaked state across builds.

        _, _, frame_sources_before, _ = self._build_full_stack(4)
        self.assertEqual(len(frame_sources_before), 4)

        smaller_building = _building_with_n_live_cameras(2)
        camera_manager = CameraManager()

        frame_sources_after = build_rtsp_frame_sources(
            smaller_building, camera_manager, credential_store=None,
            decoder_backend_factory=FakeRTSPBackend,
        )

        self.assertEqual(len(frame_sources_after), 2)
        self.assertEqual(set(frame_sources_after.keys()), {"CAM-0", "CAM-1"})


if __name__ == "__main__":
    unittest.main()
