import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.widgets.execution_panel import ExecutionPanel

from execution_layer.models import ExecutionCategory, ExecutionRequest, ExecutionSet, ExecutionStatus


def make_request(execution_request_id, category=ExecutionCategory.WARDEN_NOTIFICATION, status=ExecutionStatus.CONFIRMED):

    return ExecutionRequest(
        execution_request_id=execution_request_id, category=category, status=status,
        provider_source="SimulationWardenNotificationProvider", target_description="Warden for zone zone-1",
    )


class ExecutionPanelTests(unittest.TestCase):

    def test_refresh_populates_rows(self):

        panel = ExecutionPanel()

        execution_set = ExecutionSet(timestamp=1.0, requests=(
            make_request("e1", category=ExecutionCategory.WARDEN_NOTIFICATION),
            make_request("e2", category=ExecutionCategory.BUILDING_CONTROL),
        ))

        panel.refresh(execution_set)

        self.assertEqual(panel.execution_table.rowCount(), 2)

    def test_refresh_with_none_produces_empty_table(self):

        panel = ExecutionPanel()

        panel.refresh(None)

        self.assertEqual(panel.execution_table.rowCount(), 0)

    def test_category_filter_narrows_rows(self):

        panel = ExecutionPanel()

        execution_set = ExecutionSet(timestamp=1.0, requests=(
            make_request("e1", category=ExecutionCategory.WARDEN_NOTIFICATION),
            make_request("e2", category=ExecutionCategory.BUILDING_CONTROL),
        ))

        panel.refresh(execution_set)

        index = panel.category_filter.findData(ExecutionCategory.WARDEN_NOTIFICATION)
        panel.category_filter.setCurrentIndex(index)

        self.assertEqual(panel.execution_table.rowCount(), 1)


if __name__ == "__main__":
    unittest.main()
