from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from recommendation_layer.models import RecommendationPriority, RecommendationType


class RecommendationPanel(QWidget):

    # The Recommendation Layer milestone -- the Studio-facing view onto
    # recommendation_layer.layer.RecommendationLayer's own output. Same
    # "dumb widget, MainWindow pushes updates in" convention as
    # PerceptionDebugPanel/CameraManagerPanel: the one public entry
    # point is refresh(recommendation_set), called by MainWindow
    # wherever it already calls the other Live-Runtime-family panels'
    # own refresh() methods. Owns no timer, no LiveRuntime reference of
    # its own -- purely presentational.
    #
    # Unlike CameraManagerPanel's Floor/Zone filter combos (rebuilt
    # every refresh() from Building data), the Priority/Type filter
    # combos here are a fixed, static vocabulary -- populated once at
    # construction time and never rebuilt.

    def __init__(self):

        super().__init__()

        # Notified when the user selects (or deselects) a row -- None
        # (the default) is a valid, guarded state, same convention as
        # CameraManagerPanel.on_camera_changed. Invoked with the
        # selected recommendation_layer.models.Recommendation, or None
        # on empty selection.
        self.on_recommendation_selected = None

        self._last_recommendation_set = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        filter_row = QHBoxLayout()

        filter_row.addWidget(QLabel("Priority:"))
        self.priority_filter = QComboBox()
        self.priority_filter.addItem("All Priorities", None)
        for priority in (
            RecommendationPriority.CRITICAL, RecommendationPriority.HIGH, RecommendationPriority.MEDIUM,
            RecommendationPriority.LOW, RecommendationPriority.INFO,
        ):
            self.priority_filter.addItem(priority, priority)
        self.priority_filter.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.priority_filter)

        filter_row.addWidget(QLabel("Type:"))
        self.type_filter = QComboBox()
        self.type_filter.addItem("All Types", None)
        for type_ in (
            RecommendationType.OCCUPANT_ROUTING, RecommendationType.HAZARD_AVOIDANCE,
            RecommendationType.CONGESTION_MITIGATION, RecommendationType.EXIT_UTILIZATION,
            RecommendationType.WARDEN_DISPATCH, RecommendationType.SYSTEM_WARNING,
        ):
            self.type_filter.addItem(type_, type_)
        self.type_filter.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.type_filter)

        layout.addLayout(filter_row)

        self.recommendation_table = QTableWidget()
        self.recommendation_table.setColumnCount(8)
        self.recommendation_table.setHorizontalHeaderLabels(
            ["Priority", "Type", "Status", "Affected Zones", "Affected Exits", "Confidence", "Trigger Condition", "Updated"],
        )
        self.recommendation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.recommendation_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.recommendation_table.setSortingEnabled(True)
        self.recommendation_table.setRowCount(0)
        self.recommendation_table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.recommendation_table)

        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setMaximumHeight(140)
        layout.addWidget(self.detail_view)

    # =====================================================
    # Public entry point
    # =====================================================

    def refresh(self, recommendation_set):

        self._last_recommendation_set = recommendation_set

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

        recommendations = self._filtered_recommendations()

        self.recommendation_table.setSortingEnabled(False)
        self.recommendation_table.setRowCount(len(recommendations))

        for row, recommendation in enumerate(recommendations):
            self._populate_row(row, recommendation)

        self.recommendation_table.setSortingEnabled(True)
        self._refresh_detail_view()

    # =====================================================

    def _filtered_recommendations(self):

        if self._last_recommendation_set is None:
            return ()

        recommendations = self._last_recommendation_set.recommendations

        priority_filter = self.priority_filter.currentData()
        type_filter = self.type_filter.currentData()

        if priority_filter is not None:
            recommendations = tuple(r for r in recommendations if r.priority == priority_filter)

        if type_filter is not None:
            recommendations = tuple(r for r in recommendations if r.type == type_filter)

        return recommendations

    # =====================================================

    def _populate_row(self, row, recommendation):

        priority_item = QTableWidgetItem(recommendation.priority)
        priority_item.setData(Qt.ItemDataRole.UserRole, recommendation.recommendation_id)

        self.recommendation_table.setItem(row, 0, priority_item)
        self.recommendation_table.setItem(row, 1, QTableWidgetItem(recommendation.type))
        self.recommendation_table.setItem(row, 2, QTableWidgetItem(recommendation.status))
        self.recommendation_table.setItem(row, 3, QTableWidgetItem(", ".join(recommendation.affected_zones)))
        self.recommendation_table.setItem(row, 4, QTableWidgetItem(", ".join(recommendation.affected_exits)))
        self.recommendation_table.setItem(
            row, 5, QTableWidgetItem("-" if recommendation.confidence is None else f"{recommendation.confidence:.2f}"),
        )
        self.recommendation_table.setItem(row, 6, QTableWidgetItem(recommendation.trigger_condition))
        self.recommendation_table.setItem(row, 7, QTableWidgetItem(f"{recommendation.updated_at:.2f}"))

    # =====================================================
    # Selection
    # =====================================================

    def _selected_recommendation_id(self):

        selected_items = self.recommendation_table.selectedItems()

        if not selected_items:
            return None

        row = selected_items[0].row()
        first_column_item = self.recommendation_table.item(row, 0)

        return first_column_item.data(Qt.ItemDataRole.UserRole) if first_column_item is not None else None

    # =====================================================

    def _selected_recommendation(self):

        recommendation_id = self._selected_recommendation_id()

        if recommendation_id is None or self._last_recommendation_set is None:
            return None

        for recommendation in self._last_recommendation_set.recommendations:

            if recommendation.recommendation_id == recommendation_id:
                return recommendation

        return None

    # =====================================================

    def _on_selection_changed(self):

        self._refresh_detail_view()

        if self.on_recommendation_selected is not None:
            self.on_recommendation_selected(self._selected_recommendation())

    # =====================================================

    def _refresh_detail_view(self):

        recommendation = self._selected_recommendation()

        if recommendation is None:

            self.detail_view.setPlainText("-")
            return

        supporting_sources = ", ".join(recommendation.supporting_sources) or "-"

        self.detail_view.setPlainText(
            f"Explanation: {recommendation.explanation or '-'}\n"
            f"Technical Reason: {recommendation.technical_reason or '-'}\n"
            f"Recommended Action: {recommendation.recommended_action or '-'}\n"
            f"Primary Source: {recommendation.primary_source or '-'}\n"
            f"Supporting Sources: {supporting_sources}\n"
            f"Supporting Evidence: {dict(recommendation.supporting_evidence)}"
        )
