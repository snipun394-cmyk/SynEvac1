import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from live_occupants.models import LiveOccupantsSnapshot
from live_occupants.occupant import LiveOccupant
from live_occupants.state import OccupantStatus

from command_center.live_occupant_panel import LiveOccupantPanel


# =====================================================
# Expose Real Live Camera Occupant State In SynEvac UI milestone --
# proves the panel's own composition seam: it renders exactly what a
# LiveOccupantsSnapshot already contains, never re-deriving headcount
# or lifecycle-inclusion logic itself. No real NVR/YOLO/credentials
# anywhere -- LiveOccupant objects are built directly.
# =====================================================


def _occupant(occupant_id, camera_id="CAM-1", status=OccupantStatus.ACTIVE, confidence=0.5, zone_id="zone-a", floor_id="floor-1"):

    return LiveOccupant(
        occupant_id=occupant_id, current_camera_id=camera_id, current_track_id=occupant_id,
        current_zone_id=zone_id, current_floor_id=floor_id, world_position=None, world_velocity=None,
        behavior=None, confidence=confidence, first_seen=0.0, last_seen=1.0, status=status,
    )


class LiveOccupantPanelTests(unittest.TestCase):

    def setUp(self):
        self.panel = LiveOccupantPanel()

    def test_no_snapshot_shows_no_phantom_occupant(self):

        self.panel.show_live_occupants(None)

        self.assertEqual(self.panel.occupant_table.rowCount(), 0)
        self.assertIn("-", self.panel.currently_present_label.text())

    def test_empty_snapshot_shows_zero_present(self):

        snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=(), current_occupant_count=0)
        self.panel.show_live_occupants(snapshot)

        self.assertEqual(self.panel.occupant_table.rowCount(), 0)
        self.assertEqual(self.panel.currently_present_label.text(), "Currently Present: 0")

    def test_one_current_occupant_renders_one_row(self):

        occupant = _occupant("CAM-1-T1")
        snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=(occupant,), current_occupant_count=1)
        self.panel.show_live_occupants(snapshot)

        self.assertEqual(self.panel.occupant_table.rowCount(), 1)
        self.assertEqual(self.panel.occupant_table.item(0, 0).text(), "CAM-1-T1")
        self.assertEqual(self.panel.occupant_table.item(0, 1).text(), "CAM-1")
        self.assertEqual(self.panel.occupant_table.item(0, 2).text(), "ACTIVE")
        self.assertEqual(self.panel.occupant_table.item(0, 3).text(), "50%")
        self.assertEqual(self.panel.currently_present_label.text(), "Currently Present: 1")

    def test_two_simultaneous_occupants_render_as_two_distinct_rows(self):

        occ_a = _occupant("CAM-1-T1")
        occ_b = _occupant("CAM-1-T2", zone_id="zone-b")
        snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=(occ_a, occ_b), current_occupant_count=2)
        self.panel.show_live_occupants(snapshot)

        self.assertEqual(self.panel.occupant_table.rowCount(), 2)
        shown_ids = {self.panel.occupant_table.item(row, 0).text() for row in range(2)}
        self.assertEqual(shown_ids, {"CAM-1-T1", "CAM-1-T2"})
        self.assertEqual(self.panel.currently_present_label.text(), "Currently Present: 2")

    def test_temporarily_lost_status_is_rendered_honestly(self):

        occupant = _occupant("CAM-1-T1", status=OccupantStatus.TEMPORARILY_LOST)
        snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=(occupant,), current_occupant_count=0)
        self.panel.show_live_occupants(snapshot)

        self.assertEqual(self.panel.occupant_table.item(0, 2).text(), "TEMPORARILY_LOST")

        # The critical guard: a TEMPORARILY_LOST historical row is still
        # SHOWN (lifecycle visibility) but must NOT be counted as
        # currently present -- current_occupant_count is the one
        # correct source, never len(occupants).
        self.assertEqual(self.panel.currently_present_label.text(), "Currently Present: 0")

    def test_historical_track_churn_does_not_inflate_current_headcount(self):

        # Reproduces this milestone's own predecessor scenario: one
        # physical person, three historical track lifecycles (T1
        # TEMPORARILY_LOST, T2 TEMPORARILY_LOST, T3 currently ACTIVE).
        # all shown for lifecycle visibility; headcount must read 1.
        occupants = (
            _occupant("CAM-1-T1", status=OccupantStatus.TEMPORARILY_LOST),
            _occupant("CAM-1-T2", status=OccupantStatus.TEMPORARILY_LOST),
            _occupant("CAM-1-T3", status=OccupantStatus.ACTIVE),
        )
        snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=occupants, current_occupant_count=1)
        self.panel.show_live_occupants(snapshot)

        self.assertEqual(self.panel.occupant_table.rowCount(), 3)
        self.assertEqual(self.panel.currently_present_label.text(), "Currently Present: 1")

    def test_repeated_updates_replace_rather_than_duplicate_rows(self):

        occupant = _occupant("CAM-1-T1")
        snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=(occupant,), current_occupant_count=1)

        for _ in range(5):
            self.panel.show_live_occupants(snapshot)

        self.assertEqual(self.panel.occupant_table.rowCount(), 1, "repeated identical updates must never accumulate rows")

    def test_occupant_disappearing_removes_its_row_next_update(self):

        # T1 present, then T1 no longer in the snapshot (e.g. expired
        # and removed from LiveOccupantManager) -- the panel must not go
        # on showing a stale row for it.
        occ = _occupant("CAM-1-T1")
        self.panel.show_live_occupants(LiveOccupantsSnapshot(timestamp=1.0, occupants=(occ,), current_occupant_count=1))
        self.assertEqual(self.panel.occupant_table.rowCount(), 1)

        self.panel.show_live_occupants(LiveOccupantsSnapshot(timestamp=2.0, occupants=(), current_occupant_count=0))
        self.assertEqual(self.panel.occupant_table.rowCount(), 0)

    def test_camera_association_is_displayed(self):

        occupant = _occupant("CAM-2-T1", camera_id="CAM-2")
        snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=(occupant,), current_occupant_count=1)
        self.panel.show_live_occupants(snapshot)

        self.assertEqual(self.panel.occupant_table.item(0, 1).text(), "CAM-2")

    def test_note_label_clarifies_anonymous_tracking_not_identity(self):

        # UI semantics requirement -- the panel must never imply
        # biometric identity anywhere in its own static text.
        found = False
        for child in self.panel.findChildren(type(self.panel.currently_present_label)):
            if "anonymous" in child.text().lower():
                found = True
        self.assertTrue(found, "panel must contain an explicit anonymous-tracking disclosure")


if __name__ == "__main__":
    unittest.main()
