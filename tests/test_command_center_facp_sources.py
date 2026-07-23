import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from building_state.models import BuildingState

from command_center.live_status_panel import LiveStatusPanel

from facp.models import FACPSnapshot, PanelEvent, PanelEventType, PanelState


# =====================================================
# Manual Call Points & Emergency Lighting milestone, Step 2 -- proves
# LiveStatusPanel.facp_sources_table (command_center/live_status_panel.py)
# displays a ManualCallPoint alarm source with the correct source id,
# type, zone and state, exactly like a SmokeDetector/HeatDetector alarm
# source, using the SAME table (no new MCP-specific panel).
# =====================================================


def _row_values(table, row_index):

    return tuple(table.item(row_index, col).text() for col in range(table.columnCount()))


class FACPSourcesTableTests(unittest.TestCase):

    def test_mcp_alarm_source_shown_with_id_type_zone_state(self):

        facp = FACPSnapshot(
            panel_id="FACP-1", timestamp=1.0, panel_state=PanelState.ALARM,
            active_alarm_source_ids=("MCP-1",),
            recent_events=(
                PanelEvent(
                    event_type=PanelEventType.MANUAL_ALARM, timestamp=1.0,
                    source_asset_id="MCP-1", source_asset_type="ManualCallPoint",
                    floor_id="f1", zone_ids=("Z-A",), panel_state_after=PanelState.ALARM,
                ),
            ),
        )

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(facp_status=facp))

        self.assertEqual(panel.facp_sources_table.rowCount(), 1)
        self.assertEqual(
            _row_values(panel.facp_sources_table, 0),
            ("MCP-1", "ManualCallPoint", "Z-A", "ALARM"),
        )

    def test_smoke_and_heat_alarm_sources_still_shown_correctly(self):

        facp = FACPSnapshot(
            panel_id="FACP-1", timestamp=1.0, panel_state=PanelState.ALARM,
            active_alarm_source_ids=("SD-1", "HD-1"),
            recent_events=(
                PanelEvent(
                    event_type=PanelEventType.DETECTOR_ALARM, timestamp=1.0,
                    source_asset_id="SD-1", source_asset_type="SmokeDetector",
                    floor_id="f1", zone_ids=("Z-A",), panel_state_after=PanelState.ALARM,
                ),
                PanelEvent(
                    event_type=PanelEventType.DETECTOR_ALARM, timestamp=1.0,
                    source_asset_id="HD-1", source_asset_type="HeatDetector",
                    floor_id="f1", zone_ids=("Z-B",), panel_state_after=PanelState.ALARM,
                ),
            ),
        )

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(facp_status=facp))

        self.assertEqual(panel.facp_sources_table.rowCount(), 2)
        rows = {panel.facp_sources_table.item(r, 0).text(): _row_values(panel.facp_sources_table, r)
                for r in range(panel.facp_sources_table.rowCount())}

        self.assertEqual(rows["SD-1"], ("SD-1", "SmokeDetector", "Z-A", "ALARM"))
        self.assertEqual(rows["HD-1"], ("HD-1", "HeatDetector", "Z-B", "ALARM"))

    def test_mixed_mcp_and_detector_sources_coexist_in_same_table(self):

        facp = FACPSnapshot(
            panel_id="FACP-1", timestamp=2.0, panel_state=PanelState.ALARM,
            active_alarm_source_ids=("SD-1", "MCP-1"),
            active_fault_source_ids=("HD-1",),
            recent_events=(
                PanelEvent(
                    event_type=PanelEventType.DETECTOR_ALARM, timestamp=1.0,
                    source_asset_id="SD-1", source_asset_type="SmokeDetector",
                    floor_id="f1", zone_ids=("Z-A",), panel_state_after=PanelState.ALARM,
                ),
                PanelEvent(
                    event_type=PanelEventType.MANUAL_ALARM, timestamp=2.0,
                    source_asset_id="MCP-1", source_asset_type="ManualCallPoint",
                    floor_id="f1", zone_ids=("Z-A",), panel_state_after=PanelState.ALARM,
                ),
                PanelEvent(
                    event_type=PanelEventType.DETECTOR_FAULT, timestamp=2.0,
                    source_asset_id="HD-1", source_asset_type="HeatDetector",
                    floor_id="f1", zone_ids=("Z-B",), panel_state_after=PanelState.ALARM,
                ),
            ),
        )

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(facp_status=facp))

        self.assertEqual(panel.facp_sources_table.rowCount(), 3)
        rows = {panel.facp_sources_table.item(r, 0).text(): _row_values(panel.facp_sources_table, r)
                for r in range(panel.facp_sources_table.rowCount())}

        self.assertEqual(rows["SD-1"], ("SD-1", "SmokeDetector", "Z-A", "ALARM"))
        self.assertEqual(rows["MCP-1"], ("MCP-1", "ManualCallPoint", "Z-A", "ALARM"))
        self.assertEqual(rows["HD-1"], ("HD-1", "HeatDetector", "Z-B", "FAULT"))

    def test_source_with_no_recent_event_shows_dash_not_fabricated(self):

        facp = FACPSnapshot(
            panel_id="FACP-1", timestamp=3.0, panel_state=PanelState.ALARM,
            active_alarm_source_ids=("MCP-1",),
            recent_events=(),
        )

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(facp_status=facp))

        self.assertEqual(_row_values(panel.facp_sources_table, 0), ("MCP-1", "-", "-", "ALARM"))

    def test_no_facp_clears_sources_table(self):

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(facp_status=None))

        self.assertEqual(panel.facp_sources_table.rowCount(), 0)


if __name__ == "__main__":
    unittest.main()
