import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from building_state.models import BuildingState

from command_center.live_status_panel import LiveStatusPanel

from fire_water_manager.snapshot import FireWaterInfrastructureSnapshot, FireWaterSystemStatus, FireWaterSystemStatusReport
from fire_water_manager.status import FireWaterAssetStatus


# =====================================================
# Fire Water Supply & Suppression Infrastructure milestone, Phase 13 --
# proves LiveStatusPanel.fire_water_asset_table/fire_water_system_table
# (command_center/live_status_panel.py) display Tank/Pump/Jockey Pump/
# Inlet status and FireWaterSystem operational state + degradation
# reasons + dependency tracing, extending the existing fire-safety
# panel rather than creating a new one, independently of FACP.
# =====================================================


def _row_values(table, row_index):

    return tuple(table.item(row_index, col).text() for col in range(table.columnCount()))


class FireWaterAssetTableTests(unittest.TestCase):

    def test_all_four_asset_types_shown(self):

        entries = (
            FireWaterAssetStatus(asset_id="T1", asset_type="FireWaterTank", name="T1", floor_id="f1", zone_ids=("Z-A",), active=True, health_status="OK", state="AVAILABLE"),
            FireWaterAssetStatus(asset_id="P1", asset_type="FirePump", name="P1", floor_id="f1", zone_ids=("Z-A",), active=True, health_status="OK", state="RUNNING"),
            FireWaterAssetStatus(asset_id="J1", asset_type="JockeyPump", name="J1", floor_id="f1", zone_ids=("Z-A",), active=True, health_status="OK", state="STOPPED"),
            FireWaterAssetStatus(asset_id="I1", asset_type="FireServiceInlet", name="I1", floor_id="f1", zone_ids=("Z-A",), active=True, health_status="OK", state="AVAILABLE"),
        )
        snapshot = FireWaterInfrastructureSnapshot(entries=entries)

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(fire_water_status=snapshot))

        self.assertEqual(panel.fire_water_asset_table.rowCount(), 4)

        rows = {panel.fire_water_asset_table.item(r, 0).text(): _row_values(panel.fire_water_asset_table, r)
                for r in range(panel.fire_water_asset_table.rowCount())}

        self.assertEqual(rows["T1"], ("T1", "FireWaterTank", "Z-A", "AVAILABLE"))
        self.assertEqual(rows["P1"], ("P1", "FirePump", "Z-A", "RUNNING"))
        self.assertEqual(rows["J1"], ("J1", "JockeyPump", "Z-A", "STOPPED"))
        self.assertEqual(rows["I1"], ("I1", "FireServiceInlet", "Z-A", "AVAILABLE"))

    def test_populated_independently_of_facp_status(self):

        entries = (
            FireWaterAssetStatus(asset_id="T1", asset_type="FireWaterTank", name="T1", floor_id="f1", zone_ids=(), active=True, health_status="OK", state="AVAILABLE"),
        )
        snapshot = FireWaterInfrastructureSnapshot(entries=entries)

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(facp_status=None, fire_water_status=snapshot))

        self.assertEqual(panel.fire_water_asset_table.rowCount(), 1)
        self.assertEqual(panel.facp_state_table.rowCount(), 0)

    def test_no_fire_water_status_clears_tables(self):

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(fire_water_status=None))

        self.assertEqual(panel.fire_water_asset_table.rowCount(), 0)
        self.assertEqual(panel.fire_water_system_table.rowCount(), 0)

    def test_no_building_state_clears_tables(self):

        panel = LiveStatusPanel()
        panel.show_building_state(None)

        self.assertEqual(panel.fire_water_asset_table.rowCount(), 0)
        self.assertEqual(panel.fire_water_system_table.rowCount(), 0)


class FireWaterSystemTableTests(unittest.TestCase):

    def test_shows_status_supply_dependents_and_reasons(self):

        systems = (
            FireWaterSystemStatusReport(
                system_id="FW-1", name="FW-1", status=FireWaterSystemStatus.SYSTEM_DEGRADED,
                reasons=("pump P1 fault",),
                tank_ids=("T1",), pump_ids=("P1",),
                sprinkler_ids=("SPR-1", "SPR-2"), hydrant_ids=("HYD-1",),
            ),
        )
        snapshot = FireWaterInfrastructureSnapshot(systems=systems)

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(fire_water_status=snapshot))

        self.assertEqual(panel.fire_water_system_table.rowCount(), 1)
        row = _row_values(panel.fire_water_system_table, 0)

        self.assertEqual(row[0], "FW-1")
        self.assertEqual(row[1], "SYSTEM_DEGRADED")
        self.assertIn("T1", row[2])
        self.assertIn("P1", row[2])
        self.assertIn("SPR-1", row[3])
        self.assertIn("HYD-1", row[3])
        self.assertIn("pump P1 fault", row[4])

    def test_healthy_system_shows_dash_for_reasons(self):

        systems = (
            FireWaterSystemStatusReport(system_id="FW-1", name="FW-1", status=FireWaterSystemStatus.SYSTEM_AVAILABLE),
        )
        snapshot = FireWaterInfrastructureSnapshot(systems=systems)

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(fire_water_status=snapshot))

        row = _row_values(panel.fire_water_system_table, 0)
        self.assertEqual(row[4], "-")


if __name__ == "__main__":
    unittest.main()
