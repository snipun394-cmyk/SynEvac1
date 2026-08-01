from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from execution_layer.models import ExecutionCategory


class ExecutionPanel(QWidget):

    # The Execution Layer V1 milestone -- the Studio-facing, READ-ONLY
    # view onto execution_layer.layer.ExecutionLayer's own output. Same
    # "dumb widget, MainWindow pushes updates in" convention as
    # RecommendationPanel: the one public entry point is
    # refresh(execution_set), called by MainWindow alongside its own
    # _refresh_recommendation_panel() call. Owns no timer, no LiveRuntime
    # reference of its own -- purely presentational, and purely
    # informational: this panel has no Approve/Reject affordance at all
    # (that stays exactly where it already lives, in Command Center's
    # own Building Controls/Voice Evacuation/Warden Notifications panels
    # -- see docs/architecture/execution_layer.md).
    #
    # The Category filter combo is a fixed, static vocabulary -- same
    # "populated once at construction time, never rebuilt" discipline
    # RecommendationPanel's own Priority/Type filters use.

    def __init__(self):

        super().__init__()

        self._last_execution_set = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        filter_row = QHBoxLayout()

        filter_row.addWidget(QLabel("Category:"))
        self.category_filter = QComboBox()
        self.category_filter.addItem("All Categories", None)
        for category in (
            ExecutionCategory.VOICE_EVACUATION, ExecutionCategory.BUILDING_CONTROL,
            ExecutionCategory.DYNAMIC_SIGNAGE, ExecutionCategory.WARDEN_NOTIFICATION,
        ):
            self.category_filter.addItem(category, category)
        self.category_filter.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.category_filter)

        layout.addLayout(filter_row)

        self.execution_table = QTableWidget()
        self.execution_table.setColumnCount(10)
        self.execution_table.setHorizontalHeaderLabels([
            "Category", "Status", "Target", "Recommendation ID", "Provenance",
            "Created", "Approved", "Dispatched", "Completed", "Result",
        ])
        self.execution_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.execution_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.execution_table.setSortingEnabled(True)
        self.execution_table.setRowCount(0)
        layout.addWidget(self.execution_table)

    # =====================================================
    # Public entry point
    # =====================================================

    def refresh(self, execution_set):

        self._last_execution_set = execution_set

        self._refresh_table()

    # =====================================================
    # Filters
    # =====================================================

    def _on_filter_changed(self, _index):

        self._refresh_table()

    # =====================================================
    # Table
    # =====================================================

    def _refresh_table(self):

        requests = self._filtered_requests()

        self.execution_table.setSortingEnabled(False)
        self.execution_table.setRowCount(len(requests))

        for row, request in enumerate(requests):
            self._populate_row(row, request)

        self.execution_table.setSortingEnabled(True)

    # =====================================================

    def _filtered_requests(self):

        if self._last_execution_set is None:
            return ()

        requests = self._last_execution_set.requests

        category_filter = self.category_filter.currentData()

        if category_filter is not None:
            requests = tuple(r for r in requests if r.category == category_filter)

        return requests

    # =====================================================

    @staticmethod
    def _format_timestamp(value):

        return "-" if value is None else f"{value:.2f}"

    # =====================================================

    def _populate_row(self, row, request):

        category_item = QTableWidgetItem(request.category)
        category_item.setData(Qt.ItemDataRole.UserRole, request.execution_request_id)

        self.execution_table.setItem(row, 0, category_item)
        self.execution_table.setItem(row, 1, QTableWidgetItem(request.status))
        self.execution_table.setItem(row, 2, QTableWidgetItem(request.target_description))
        self.execution_table.setItem(row, 3, QTableWidgetItem(request.originating_recommendation_id or "-"))
        self.execution_table.setItem(row, 4, QTableWidgetItem(request.recommendation_id_provenance))
        self.execution_table.setItem(row, 5, QTableWidgetItem(self._format_timestamp(request.created_at)))
        self.execution_table.setItem(row, 6, QTableWidgetItem(self._format_timestamp(request.approved_at)))
        self.execution_table.setItem(row, 7, QTableWidgetItem(self._format_timestamp(request.dispatched_at)))
        self.execution_table.setItem(row, 8, QTableWidgetItem(self._format_timestamp(request.completed_at)))
        self.execution_table.setItem(row, 9, QTableWidgetItem(request.result_message))
