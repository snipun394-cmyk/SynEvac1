from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSplitter, QTabWidget, QVBoxLayout, QWidget

from command_center.building_view import BuildingView
from command_center.hazard_panel import HazardPanel
from command_center.human_panel import HumanPanel
from command_center.incident_panel import IncidentPanel
from command_center.incident_status_bar import IncidentStatusBar
from command_center.occupancy_panel import OccupancyPanel
from command_center.recommendation_center import RecommendationCenter
from command_center.recommendation_panel import RecommendationPanel
from command_center.recommendation_timeline_panel import RecommendationTimelinePanel
from command_center.timeline_panel import TimelinePanel


# =====================================================
# Dashboard -- the Command Center's central widget. Integrates
# BuildingView and every panel over one IncidentData, and is the only
# place that wires the TimelinePanel's slider back into "which
# IncidentFrame is currently displayed." MainWindow owns the playback
# clock (a QTimer); Dashboard owns nothing time-related itself beyond
# reacting to whatever frame index it is told to show. Also the one
# place that resolves "which advisory_system.AdvisoryReport belongs to
# the currently displayed frame" (IncidentData.advisory_report_at_index(),
# already computed once per frame during IncidentData.__post_init__) and
# hands both the frame and its report to whichever panels need the
# report (RecommendationCenter/RecommendationTimelinePanel/
# IncidentStatusBar) -- every other, pre-existing panel keeps its
# original frame-only show_frame(frame) signature untouched.
# =====================================================


class Dashboard(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None

        self.status_bar = IncidentStatusBar()

        self.building_view = BuildingView()

        self.incident_panel = IncidentPanel()
        self.occupancy_panel = OccupancyPanel()
        self.hazard_panel = HazardPanel()
        self.recommendation_panel = RecommendationPanel()
        self.human_panel = HumanPanel()
        self.recommendation_center = RecommendationCenter()
        self.recommendation_timeline_panel = RecommendationTimelinePanel()

        self.side_tabs = QTabWidget()
        self.side_tabs.addTab(self.recommendation_center, "Recommendation Center")
        self.side_tabs.addTab(self.recommendation_timeline_panel, "Recommendation Timeline")
        self.side_tabs.addTab(self.incident_panel, "Incident")
        self.side_tabs.addTab(self.occupancy_panel, "Occupancy")
        self.side_tabs.addTab(self.hazard_panel, "Hazard")
        self.side_tabs.addTab(self.recommendation_panel, "Decision Policy (Raw)")
        self.side_tabs.addTab(self.human_panel, "People")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.building_view)
        splitter.addWidget(self.side_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.timeline_panel = TimelinePanel()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.status_bar)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.timeline_panel)

        self.timeline_panel.slider.valueChanged.connect(self._on_frame_index_changed)

    # =====================================================
    # Public API
    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        building = incident_data.building if incident_data is not None else None
        policy = incident_data.decision_policy if incident_data is not None else None

        self.building_view.set_building(building)
        self.building_view.set_decision_policy(policy)

        self.status_bar.set_incident(incident_data)
        self.incident_panel.set_incident(incident_data)
        self.occupancy_panel.set_incident(incident_data)
        self.hazard_panel.set_incident(incident_data)
        self.recommendation_panel.set_incident(incident_data)
        self.human_panel.set_incident(incident_data)
        self.recommendation_center.set_incident(incident_data)
        self.recommendation_timeline_panel.set_incident(incident_data)
        self.timeline_panel.set_incident(incident_data)

        if incident_data is not None:
            self.show_frame(incident_data.frame_at_index(0))

    # =====================================================

    def show_frame(self, frame):

        report = self._incident.advisory_report_at_index(self.frame_index) if self._incident is not None else None

        self.building_view.show_frame(frame)
        self.incident_panel.show_frame(frame)
        self.occupancy_panel.show_frame(frame)
        self.hazard_panel.show_frame(frame)
        self.human_panel.show_frame(frame)
        self.status_bar.show_frame(frame, report)
        self.recommendation_center.show_frame(frame, report)
        self.recommendation_timeline_panel.show_frame(frame)

    # =====================================================

    def set_frame_index(self, index):

        if self._incident is None:
            return

        self.timeline_panel.set_frame_index(index)
        self.show_frame(self._incident.frame_at_index(index))

    # =====================================================

    def set_overlay_mode(self, mode):

        self.building_view.set_overlay_mode(mode)

    # =====================================================

    def set_floor(self, floor):

        self.building_view.set_floor(floor)

    # =====================================================

    @property
    def frame_index(self) -> int:

        return self.timeline_panel.frame_index

    # =====================================================

    @property
    def frame_count(self) -> int:

        return self._incident.frame_count if self._incident is not None else 0

    # =====================================================
    # Internal wiring
    # =====================================================

    def _on_frame_index_changed(self, index):

        if self._incident is None:
            return

        self.timeline_panel.set_frame_index(index)
        self.show_frame(self._incident.frame_at_index(index))
