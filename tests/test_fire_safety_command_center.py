import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from building_state.models import BuildingState

from command_center.live_status_panel import LiveStatusPanel

from fire_safety_manager.snapshot import FireSafetyStatusSnapshot, PassiveAssetCounts, SprinklerCounts
from fire_safety_manager.status import FireSafetyAssetStatus


# =====================================================
# Fire Suppression & Water-Based Safety Asset Digital Twin milestone,
# Phase 12 -- proves LiveStatusPanel.fire_safety_table (command_center/
# live_status_panel.py) displays Sprinkler/FireExtinguisher/
# FireHydrant/HoseReel status uniformly, in ONE table (no four new
# panels), and independently of whether FACP is configured.
# =====================================================


def _row_values(table, row_index):

    return tuple(table.item(row_index, col).text() for col in range(table.columnCount()))


class FireSafetyTableTests(unittest.TestCase):

    def test_all_four_asset_types_shown_with_name_type_zone_state(self):

        entries = (
            FireSafetyAssetStatus(
                asset_id="SPR-1", asset_type="Sprinkler", name="SPR-1", floor_id="f1",
                zone_ids=("Z-A",), active=True, health_status="OK", state="ACTIVATED",
            ),
            FireSafetyAssetStatus(
                asset_id="FE-1", asset_type="FireExtinguisher", name="FE-1", floor_id="f1",
                zone_ids=("Z-A",), active=True, health_status="OK", state="AVAILABLE",
            ),
            FireSafetyAssetStatus(
                asset_id="HYD-1", asset_type="FireHydrant", name="HYD-1", floor_id="f1",
                zone_ids=("Z-B",), active=True, health_status="OK", state="AVAILABLE",
            ),
            FireSafetyAssetStatus(
                asset_id="HR-1", asset_type="HoseReel", name="HR-1", floor_id="f1",
                zone_ids=("Z-B",), active=False, health_status="OK", state="UNAVAILABLE",
            ),
        )
        snapshot = FireSafetyStatusSnapshot(entries=entries)

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(fire_safety_status=snapshot))

        self.assertEqual(panel.fire_safety_table.rowCount(), 4)

        rows = {panel.fire_safety_table.item(r, 0).text(): _row_values(panel.fire_safety_table, r)
                for r in range(panel.fire_safety_table.rowCount())}

        self.assertEqual(rows["SPR-1"], ("SPR-1", "Sprinkler", "Z-A", "ACTIVATED"))
        self.assertEqual(rows["FE-1"], ("FE-1", "FireExtinguisher", "Z-A", "AVAILABLE"))
        self.assertEqual(rows["HYD-1"], ("HYD-1", "FireHydrant", "Z-B", "AVAILABLE"))
        self.assertEqual(rows["HR-1"], ("HR-1", "HoseReel", "Z-B", "UNAVAILABLE"))

    def test_populated_independently_of_facp_status(self):

        # No FACP configured at all -- the fire safety table must
        # still populate (Sprinkler/FireExtinguisher/FireHydrant/
        # HoseReel are never FACP alarm sources).
        entries = (
            FireSafetyAssetStatus(
                asset_id="SPR-1", asset_type="Sprinkler", name="SPR-1", floor_id="f1",
                zone_ids=("Z-A",), active=True, health_status="OK", state="NORMAL",
            ),
        )
        snapshot = FireSafetyStatusSnapshot(entries=entries)

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(facp_status=None, fire_safety_status=snapshot))

        self.assertEqual(panel.fire_safety_table.rowCount(), 1)
        self.assertEqual(panel.facp_state_table.rowCount(), 0)

    def test_no_fire_safety_status_clears_table(self):

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(fire_safety_status=None))

        self.assertEqual(panel.fire_safety_table.rowCount(), 0)

    def test_no_building_state_clears_table(self):

        panel = LiveStatusPanel()
        panel.show_building_state(None)

        self.assertEqual(panel.fire_safety_table.rowCount(), 0)


if __name__ == "__main__":
    unittest.main()
