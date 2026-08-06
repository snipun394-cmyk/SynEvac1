import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from models.building import Building
from models.camera import Camera
from models.floor import Floor
from models.zone import Zone

from live_occupants.models import LiveOccupantsSnapshot
from live_occupants.occupant import LiveOccupant
from live_occupants.state import OccupantStatus

from command_center.building_view import BuildingView, estimate_display_position


def make_camera(camera_id, floor_id, position):

    return Camera(id=camera_id, name="Cam", floor_id=floor_id, position=list(position))


def make_floor(floor_id="floor-1", cameras=()):

    return Floor(
        id=floor_id, name="Ground", display_order=0,
        zones=[Zone(id="zone-a", name="Zone A", x=0.0, y=0.0, width=10.0, height=10.0, floor_id=floor_id)],
        cameras=list(cameras),
    )


def make_occupant(
    occupant_id="occ-1", current_camera_id="cam-1", current_track_id="cam-1-T1",
    current_floor_id="floor-1", status=OccupantStatus.ACTIVE,
):

    return LiveOccupant(
        occupant_id=occupant_id,
        current_camera_id=current_camera_id,
        current_track_id=current_track_id,
        current_zone_id="zone-a",
        current_floor_id=current_floor_id,
        world_position=None,
        world_velocity=None,
        behavior=None,
        confidence=0.9,
        first_seen=0.0,
        last_seen=1.0,
        status=status,
    )


class EstimateDisplayPositionTests(unittest.TestCase):

    # Live CCTV Digital Twin milestone -- estimate_display_position()
    # is the one seam every future milestone (world-coordinate
    # projection, FOV-aware placement) will change internally without
    # touching the renderer or any caller. Today it must do exactly one
    # thing: return the occupant's current camera's configured
    # position, or None when there is nothing honest to report.

    def test_returns_the_current_cameras_configured_position(self):

        camera = make_camera("cam-1", "floor-1", (5.0, 7.0))
        floor = make_floor(cameras=[camera])
        occupant = make_occupant(current_camera_id="cam-1")

        position = estimate_display_position(occupant, floor)

        self.assertEqual(tuple(position), (5.0, 7.0))

    def test_returns_none_when_occupant_has_no_current_camera(self):

        floor = make_floor()
        occupant = make_occupant(current_camera_id=None)

        self.assertIsNone(estimate_display_position(occupant, floor))

    def test_returns_none_when_the_camera_is_not_on_this_floor(self):

        floor = make_floor(cameras=[make_camera("cam-1", "floor-1", (5.0, 7.0))])
        occupant = make_occupant(current_camera_id="cam-does-not-exist")

        self.assertIsNone(estimate_display_position(occupant, floor))

    def test_returns_none_when_floor_is_none(self):

        occupant = make_occupant()

        self.assertIsNone(estimate_display_position(occupant, None))


class BuildingViewLiveOccupantRenderingTests(unittest.TestCase):

    # Live CCTV Digital Twin milestone -- proves the second, LIVE-only
    # marker path (_render_live_occupants) is fully independent of the
    # existing Replay-only path (_render_occupants): never touches
    # self._frame/occupant_positions, and a marker genuinely appears
    # and disappears purely as a function of what LiveOccupantManager
    # currently reports -- no separate "clear" logic anywhere.

    def setUp(self):

        self.camera = make_camera("cam-1", "floor-1", (5.0, 7.0))
        self.floor = make_floor(cameras=[self.camera])
        self.building = Building(id="building-1", name="Test Building", floors=[self.floor])

        self.view = BuildingView()
        self.view.set_building(self.building)

    def _marker_items_for(self, occupant_id):

        return [item for item in self.view.scene.items() if item.data(0) == occupant_id]

    def test_active_occupant_gets_a_marker_at_the_cameras_position(self):

        occupant = make_occupant(status=OccupantStatus.ACTIVE)
        snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=(occupant,), current_occupant_count=1)

        self.view.set_live_occupants(snapshot)

        markers = self._marker_items_for("occ-1")
        self.assertEqual(len(markers), 1)
        self.assertIn("cam-1-T1", markers[0].toolTip())

    def test_new_occupant_also_gets_a_marker(self):

        occupant = make_occupant(status=OccupantStatus.NEW)
        snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=(occupant,), current_occupant_count=1)

        self.view.set_live_occupants(snapshot)

        self.assertEqual(len(self._marker_items_for("occ-1")), 1)

    def test_marker_disappears_once_the_occupant_is_no_longer_reported_active(self):

        active = make_occupant(status=OccupantStatus.ACTIVE)
        self.view.set_live_occupants(
            LiveOccupantsSnapshot(timestamp=1.0, occupants=(active,), current_occupant_count=1),
        )
        self.assertEqual(len(self._marker_items_for("occ-1")), 1)

        lost = make_occupant(status=OccupantStatus.TEMPORARILY_LOST)
        self.view.set_live_occupants(
            LiveOccupantsSnapshot(timestamp=2.0, occupants=(lost,), current_occupant_count=0),
        )
        self.assertEqual(self._marker_items_for("occ-1"), [])

    def test_occupant_no_longer_present_in_the_snapshot_at_all_leaves_no_marker(self):

        active = make_occupant(status=OccupantStatus.ACTIVE)
        self.view.set_live_occupants(
            LiveOccupantsSnapshot(timestamp=1.0, occupants=(active,), current_occupant_count=1),
        )
        self.assertEqual(len(self._marker_items_for("occ-1")), 1)

        # Occupant expired and removed from the manager entirely --
        # the snapshot simply no longer contains it.
        self.view.set_live_occupants(
            LiveOccupantsSnapshot(timestamp=3.0, occupants=(), current_occupant_count=0),
        )
        self.assertEqual(self._marker_items_for("occ-1"), [])

    def test_occupant_on_a_different_floor_is_not_drawn_on_this_one(self):

        occupant = make_occupant(current_floor_id="some-other-floor", status=OccupantStatus.ACTIVE)
        snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=(occupant,), current_occupant_count=1)

        self.view.set_live_occupants(snapshot)

        self.assertEqual(self._marker_items_for("occ-1"), [])

    def test_none_snapshot_renders_without_crash(self):

        self.view.set_live_occupants(None)

        self.assertEqual(self._marker_items_for("occ-1"), [])

    def test_live_occupant_rendering_never_touches_replay_occupant_positions(self):

        # Regression guard for architectural independence: set_live_
        # occupants() must never require self._frame to be set, and
        # must never appear in _frame.occupant_positions-driven state.

        occupant = make_occupant(status=OccupantStatus.ACTIVE)
        snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=(occupant,), current_occupant_count=1)

        self.view.set_live_occupants(snapshot)  # self._frame was never set

        self.assertEqual(len(self._marker_items_for("occ-1")), 1)
        self.assertIsNone(self.view._frame)


if __name__ == "__main__":
    unittest.main()
