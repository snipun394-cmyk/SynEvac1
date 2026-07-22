from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSplitter, QTabWidget, QVBoxLayout, QWidget

from command_center.building_view import BuildingView
from command_center.data_source import CommandCenterMode, CommandCenterSnapshot, SnapshotConsistency
from command_center.hazard_panel import HazardPanel
from command_center.human_panel import HumanPanel
from command_center.incident_panel import IncidentPanel
from command_center.incident_status_bar import IncidentStatusBar
from command_center.live_ai_panel import LiveAIPanel
from command_center.live_events_panel import LiveEventsPanel
from command_center.live_evacuation_progress_panel import LiveEvacuationProgressPanel
from command_center.live_emergency_response_panel import LiveEmergencyResponsePanel
from command_center.live_trajectory_intelligence_panel import LiveMovementIntelligencePanel
from command_center.live_evacuation_recommendation_panel import LiveEvacuationRecommendationPanel
from command_center.live_evacuation_guidance_panel import LiveEvacuationGuidancePanel
from command_center.live_status_panel import LiveStatusPanel
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

        # Live Command Center Integration milestone -- additive state.
        # mode starts REPLAY (this widget's own pre-existing behavior is
        # unchanged unless/until a caller explicitly switches it), and
        # _live_building tracks whether building_view/status_bar have
        # already been one-time-configured for the current live Building
        # (re-calling set_building() every ~1Hz tick would rebuild the
        # floor combo box and reset the operator's selected floor).
        self.mode = CommandCenterMode.REPLAY
        self._live_building = None

        # Live Operator Action Routing milestone -- additive. An opaque
        # (never imported/typed here) command_center.live_operator_
        # action_gateway.LiveOperatorActionGateway a caller may supply
        # via set_operator_action_gateway(); None (the default) means
        # every Live panel renders its own honest NO_PROVIDER fallback
        # (see VoiceEvacuationPanel.show_live()/BuildingControlsPanel.
        # show_live()). Dashboard itself never calls a method on it --
        # it only ever forwards the same reference into apply_snapshot()'s
        # existing recommendation_center.show_live() call, exactly the
        # same "hold an opaque reference, forward it, never inspect it"
        # discipline this class already applies to `decision_policy`.
        self._operator_action_gateway = None

        self.status_bar = IncidentStatusBar()

        self.building_view = BuildingView()

        self.incident_panel = IncidentPanel()
        self.occupancy_panel = OccupancyPanel()
        self.hazard_panel = HazardPanel()
        self.recommendation_panel = RecommendationPanel()
        self.human_panel = HumanPanel()
        self.recommendation_center = RecommendationCenter()
        self.recommendation_timeline_panel = RecommendationTimelinePanel()

        # Live-only panels (Phase 8/9/15) -- always constructed (so
        # Dashboard stays one widget tree for both modes, never two
        # parallel applications), but only ever populated via
        # apply_snapshot()'s Live path; harmlessly empty in Replay mode.
        self.live_status_panel = LiveStatusPanel()
        self.live_ai_panel = LiveAIPanel()
        self.live_evacuation_progress_panel = LiveEvacuationProgressPanel()
        self.live_emergency_response_panel = LiveEmergencyResponsePanel()
        self.live_movement_intelligence_panel = LiveMovementIntelligencePanel()
        self.live_evacuation_recommendation_panel = LiveEvacuationRecommendationPanel()
        self.live_evacuation_guidance_panel = LiveEvacuationGuidancePanel()
        self.live_events_panel = LiveEventsPanel()

        self.side_tabs = QTabWidget()
        self.side_tabs.addTab(self.recommendation_center, "Recommendation Center")
        self.side_tabs.addTab(self.recommendation_timeline_panel, "Recommendation Timeline")
        self.side_tabs.addTab(self.incident_panel, "Incident")
        self.side_tabs.addTab(self.occupancy_panel, "Occupancy")
        self.side_tabs.addTab(self.hazard_panel, "Hazard")
        self.side_tabs.addTab(self.recommendation_panel, "Decision Policy (Raw)")
        self.side_tabs.addTab(self.human_panel, "People")
        self.side_tabs.addTab(self.live_status_panel, "Live Status")
        self.side_tabs.addTab(self.live_ai_panel, "Live AI")
        self.side_tabs.addTab(self.live_evacuation_progress_panel, "Live Evacuation Progress")
        self.side_tabs.addTab(self.live_emergency_response_panel, "Live Emergency Response")
        self.side_tabs.addTab(self.live_movement_intelligence_panel, "Live Movement Intelligence")
        self.side_tabs.addTab(self.live_evacuation_recommendation_panel, "Live Evacuation Recommendations")
        self.side_tabs.addTab(self.live_evacuation_guidance_panel, "Live Evacuation Guidance")
        self.side_tabs.addTab(self.live_events_panel, "Live Events")

        # Tabs that only make sense against a completed-run IncidentData
        # (GroundTruth-derived metrics, the raw whole-run DecisionPolicy,
        # the full recommendation timeline) -- hidden, never fed
        # fabricated data, whenever Live mode is active (Phase 5/Phase
        # 20 §"avoid fabricating unavailable live information").
        self._replay_only_tabs = (
            self.recommendation_timeline_panel, self.incident_panel,
            self.occupancy_panel, self.hazard_panel, self.recommendation_panel,
        )
        self._live_only_tabs = (self.live_status_panel, self.live_ai_panel, self.live_events_panel)

        for widget in self._live_only_tabs:
            self.side_tabs.setTabVisible(self.side_tabs.indexOf(widget), False)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.building_view)
        splitter.addWidget(self.side_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.timeline_panel = TimelinePanel()

        # Live Command Center Integration milestone -- Phase 14's own
        # honest consistency banner. Only ever visible in Live mode
        # (set_mode() below toggles it); its text is the one place a
        # STALE/PARTIAL/UNAVAILABLE cycle is disclosed in plain language,
        # rather than silently rendering mismatched-cycle data as if it
        # were synchronized.
        self.live_consistency_banner = QLabel("")
        self.live_consistency_banner.setObjectName("LiveConsistencyBanner")
        self.live_consistency_banner.setWordWrap(True)
        self.live_consistency_banner.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.status_bar)
        layout.addWidget(self.live_consistency_banner)
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
    # Live Command Center Integration milestone -- additive. Neither
    # method above (set_incident/show_frame/set_frame_index) is
    # modified by this milestone; the methods below are a second, fully
    # independent rendering path through the SAME Dashboard instance and
    # the SAME child widgets (Phase 2's own "clean data-source
    # abstraction," not two applications).
    # =====================================================

    def set_operator_action_gateway(self, gateway) -> None:

        self._operator_action_gateway = gateway

    # =====================================================

    def set_mode(self, mode: CommandCenterMode) -> None:

        self.mode = mode

        is_live = mode == CommandCenterMode.LIVE

        for widget in self._live_only_tabs:
            self.side_tabs.setTabVisible(self.side_tabs.indexOf(widget), is_live)

        for widget in self._replay_only_tabs:
            self.side_tabs.setTabVisible(self.side_tabs.indexOf(widget), not is_live)

        # Phase 5 -- timeline scrubbing is honestly disabled in Live
        # mode (no bounded-history buffer exists yet to scrub through;
        # see docs/architecture/live_command_center_integration.md §5).
        self.timeline_panel.setEnabled(not is_live)
        self.live_consistency_banner.setVisible(is_live)

        if is_live:
            self.status_bar.set_mode("LIVE")
            self.recommendation_center.set_live_mode()
        else:
            self.status_bar.set_mode("REPLAY")

    # =====================================================

    def apply_snapshot(self, snapshot: CommandCenterSnapshot) -> None:

        # The one Live-mode render call -- Phase 13's UI refresh timer
        # (MainWindow._on_live_refresh_tick()) calls this with whatever
        # LiveCommandCenterDataSource.current_snapshot() just returned.
        # Never recomputes anything: every value rendered below is read
        # straight off the already-computed snapshot.

        if snapshot.mode != self.mode:
            self.set_mode(snapshot.mode)

        if snapshot.building is not None and snapshot.building is not self._live_building:

            self._live_building = snapshot.building
            self.building_view.set_building(snapshot.building)
            self.building_view.set_decision_policy(snapshot.decision_policy)
            self.status_bar.set_live_building(snapshot.building_name)

        frame = snapshot.frame

        if frame is not None:
            self.building_view.show_frame(frame)

        # Phase 14 -- a STALE cycle (BuildingState/AI/Advisory
        # timestamps disagree) never renders its carried-over AdvisoryReport
        # as though it were current; every AdvisoryReport-driven panel
        # degrades to its own already-established "no report" empty
        # state instead (see RecommendationCenter.show_live(None) /
        # IncidentStatusBar.show_frame(..., None, ...)), and the banner
        # below explains why in plain language.
        is_stale = snapshot.consistency == SnapshotConsistency.STALE
        advisory_for_display = None if is_stale else snapshot.advisory_report

        self.status_bar.show_frame(
            frame, advisory_for_display, live=True, ai_prediction_snapshot=snapshot.ai_prediction_snapshot,
        )
        self.recommendation_center.show_live(advisory_for_display, self._operator_action_gateway)

        self.live_status_panel.show_building_state(snapshot.building_state)
        self.live_ai_panel.show_prediction(snapshot.ai_prediction_snapshot, stale=is_stale)
        self.live_evacuation_progress_panel.show_progress(snapshot.evacuation_progress)
        self.live_emergency_response_panel.show_response(snapshot.emergency_response)
        self.live_movement_intelligence_panel.show_trajectory_intelligence(snapshot.trajectory_intelligence)
        self.live_evacuation_recommendation_panel.show_recommendations(snapshot.evacuation_recommendation)
        self.live_evacuation_guidance_panel.show_guidance(
            snapshot.evacuation_guidance, self._operator_action_gateway, snapshot.timestamp,
        )
        self.live_events_panel.show_recent_events(snapshot.recent_events)

        self.live_consistency_banner.setText(_consistency_banner_text(snapshot))

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


# =====================================================


def _consistency_banner_text(snapshot: CommandCenterSnapshot) -> str:

    # Phase 14's own plain-language disclosure -- CURRENT shows nothing
    # (the ordinary, working case needs no banner); every other value
    # explains exactly what is missing or mismatched, never overstating
    # what this cycle's snapshot actually contains.

    if snapshot.consistency == SnapshotConsistency.UNAVAILABLE:
        return "No live BuildingState available yet -- waiting for the first live cycle."

    if snapshot.consistency == SnapshotConsistency.PARTIAL:
        return (
            "Partial live state -- AI prediction and/or Advisory Report are not yet "
            "available this run (BuildingState is current)."
        )

    if snapshot.consistency == SnapshotConsistency.STALE:
        return (
            "Advisory unavailable for current state -- BuildingState/AI prediction/Advisory "
            "Report timestamps do not match this cycle; the previous Advisory Report is "
            "withheld rather than shown as current."
        )

    return ""
