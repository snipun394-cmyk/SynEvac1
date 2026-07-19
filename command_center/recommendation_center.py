from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from advisory_system.confidence_engine import DETERMINISTIC_RULE_BASE_CONFIDENCE

from voice_evacuation.models import BroadcastStatus

from command_center.building_controls_panel import BuildingControlsPanel


# =====================================================
# RecommendationCenter -- Phase 3's four advisory-facing panels
# (Civilian Voice Announcements / Firefighter Intelligence / Building
# System Recommendations / Incident Commander Summary), Phase 5's
# per-recommendation "why" explainability, and Phase 6's firefighter
# intelligence display, all reading from one already-computed
# advisory_system.AdvisoryReport per tick (command_center.incident_data.
# IncidentData.advisory_report_at_index() -- built once per frame during
# IncidentData.__post_init__, never recomputed here). Same "dumb widget,
# pushed updates only" convention every other Command Center panel
# follows: set_incident() runs once per loaded incident, show_frame(frame,
# report) runs on every playback tick. report is None whenever no
# GroundTruth was loaded for this incident -- every panel below degrades
# to an empty table / "-" labels rather than guessing.
#
# Phase 5 explainability is deliberately conservative about what it
# claims: AdvisoryReport's own dataclasses (recommendation_models.py)
# carry no "similarity to previous simulations" and no per-signal
# confidence breakdown field (only the final blended confidence), and
# command_center.incident_data never retains the AdvisoryInputs used to
# build a report (only the resulting AdvisoryReport) -- so this module
# never reconstructs those numbers by guesswork. It discloses the one
# floor that IS a real, public constant (DETERMINISTIC_RULE_BASE_
# CONFIDENCE), the recommendation's own already-known blended
# confidence, and states plainly when a Phase-5 field has no backend
# source at all, rather than fabricating one.
# =====================================================

_ROW_ID_ROLE = Qt.ItemDataRole.UserRole


def _format_percent(value):

    return "-" if value is None else f"{value:.0%}"


def _format_seconds(value):

    if value is None:
        return "-"

    minutes, seconds = divmod(int(value), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _format_number(value):

    return "-" if value is None else str(value)


def _join_or_dash(values):

    return ", ".join(values) if values else "-"


# =====================================================


class CivilianAnnouncementsPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None
        self._report = None

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Zone", "Announcement", "Confidence", "Timestamp", "Status"],
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, 2)

        detail_group = QGroupBox("Why this recommendation? (select a row)")
        detail_layout = QVBoxLayout(detail_group)
        self.detail_label = QLabel("-")
        self.detail_label.setWordWrap(True)
        detail_layout.addWidget(self.detail_label)
        layout.addWidget(detail_group, 1)

    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        first_report = incident_data.advisory_report_at_index(0) if incident_data is not None else None
        self.show_frame(None, first_report)

    # =====================================================

    def show_frame(self, _frame, report):

        self._report = report

        self.table.setRowCount(0)
        self.detail_label.setText("-")

        if report is None:
            return

        announcements = report.civilian_announcements
        self.table.setRowCount(len(announcements))

        for row_index, entry in enumerate(announcements):

            zone_item = QTableWidgetItem(entry.zone_name)
            zone_item.setData(_ROW_ID_ROLE, row_index)
            self.table.setItem(row_index, 0, zone_item)
            self.table.setItem(row_index, 1, QTableWidgetItem(entry.announcement))
            self.table.setItem(row_index, 2, QTableWidgetItem(_format_percent(entry.confidence)))
            self.table.setItem(row_index, 3, QTableWidgetItem(_format_seconds(report.simulation_time)))
            self.table.setItem(row_index, 4, QTableWidgetItem("Active"))

    # =====================================================

    def _on_selection_changed(self):

        selected_items = self.table.selectedItems()

        if not selected_items or self._report is None:
            self.detail_label.setText("-")
            return

        row_index = self.table.item(selected_items[0].row(), 0).data(_ROW_ID_ROLE)
        entry = self._report.civilian_announcements[row_index]

        self.detail_label.setText(_explain_civilian(self._incident, entry))


def _explain_civilian(incident_data, entry):

    confidence_source = getattr(entry, "confidence_source", ())

    lines = [
        f"Why: {entry.reason}",
        f"{_confidence_label(confidence_source)}: {_format_percent(entry.confidence)}",
        f"Confidence breakdown: Deterministic rule base = "
        f"{DETERMINISTIC_RULE_BASE_CONFIDENCE:.0%}; further per-signal breakdown is not "
        f"separately exposed by this recommendation, only the final blended value above.",
        "Similarity to previous simulations: Not available in this build.",
        _prediction_source_line(confidence_source),
        _rl_influence_line(confidence_source),
        f"Decision Policy influence: {_decision_policy_influence(incident_data, entry.zone_id)}",
    ]

    if entry.predicted_rset_improvement_seconds is not None:
        lines.append(f"Predicted RSET improvement: {entry.predicted_rset_improvement_seconds:.0f}s")

    return "\n".join(lines)


# =====================================================
# Confidence provenance labeling -- the one place allowed to say "AI
# confidence"/"RL influence: deployed," and only when advisory_engine.
# py's own CivilianAnnouncement.confidence_source actually recorded that
# a real, non-None ai_confidence/rl_confidence contributed to THIS
# recommendation. Never a blanket claim: every current run has
# confidence_source == () (no AI Inference/RL gateway configured, see
# designer/campaign/campaign_worker.py's own comment), so today every
# label below reads as plain, honest, rule-based -- but the exact same
# code becomes accurate again, with no further change here, the moment
# a real AI/RL signal is supplied for a recommendation.
# =====================================================


def _confidence_label(confidence_source):

    if "ai" in confidence_source:
        return "AI-augmented confidence (blended)"

    return "Recommendation confidence (blended, rule-based)"


def _prediction_source_line(confidence_source):

    if not confidence_source:
        return "Prediction source: Deterministic Decision Policy (no AI/RL signal supplied for this recommendation)."

    contributors = []
    if "ai" in confidence_source:
        contributors.append("an AI prediction")
    if "rl" in confidence_source:
        contributors.append("an RL policy")

    return (
        f"Prediction source: Deterministic Decision Policy, augmented by "
        f"{' and '.join(contributors)} signal for this recommendation."
    )


def _rl_influence_line(confidence_source):

    if "rl" in confidence_source:
        return "RL influence: Contributed to this recommendation's confidence."

    return "RL influence: Not deployed."


def _decision_policy_influence(incident_data, zone_id):

    policy = incident_data.decision_policy if incident_data is not None else None

    if policy is None:
        return "-"

    for entry in policy.zone_decisions:

        if entry["zone_id"] != zone_id:
            continue

        name = incident_data.name_for
        action = entry["action"].replace("_", " ").title()
        exit_name = name(entry.get("recommended_exit")) if entry.get("recommended_exit") else "-"
        stair_name = name(entry.get("recommended_stair")) if entry.get("recommended_stair") else "-"

        return f"Decision Policy action = {action} (exit: {exit_name}, stair: {stair_name})"

    return "-"


# =====================================================


class VoiceEvacuationPanel(QWidget):

    # Zoned Voice Evacuation & Speaker Network Framework -- displays
    # IncidentData.active_voice_messages_at_index()/voice_broadcast_history
    # (command_center/incident_data.py), the deterministic, already-
    # computed replay of AdvisoryReport.civilian_announcements routed
    # through voice_evacuation.controller.VoiceEvacuationController.
    # Same "dumb widget, pushed updates only" convention as
    # CivilianAnnouncementsPanel above: set_incident() runs once per
    # loaded incident, show_frame(frame, report) runs on every playback
    # tick.
    #
    # Explicitly distinguishes the RECOMMENDATION (what CivilianAnnouncementsPanel
    # already shows) from the BROADCAST STATUS (whether/how it actually
    # reached speakers) -- Phase 11's own explicit requirement. Never
    # claims a physical announcement was successfully played: every
    # status shown here is exactly whatever BroadcastStatus the
    # SimulationVoiceOutputProvider recorded, and this Designer/Command
    # Center integration is Simulation-mode only -- labeled as such.

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None

        layout = QVBoxLayout(self)

        mode_label = QLabel(
            "Simulation output (deterministic replay) -- no physical speaker hardware "
            "is involved. \"Broadcast\" below means the Simulation Voice Output Provider "
            "recorded the instruction, never that audio was actually played."
        )
        mode_label.setWordWrap(True)
        layout.addWidget(mode_label)

        layout.addWidget(QLabel("Active Zone Messages"))
        self.active_table = QTableWidget(0, 5)
        self.active_table.setHorizontalHeaderLabels(
            ["Zone", "Message", "Confidence", "Priority", "Message Type"],
        )
        self.active_table.horizontalHeader().setStretchLastSection(True)
        self.active_table.verticalHeader().setVisible(False)
        layout.addWidget(self.active_table, 1)

        layout.addWidget(QLabel("Broadcast History"))
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(
            ["Time", "Zone", "Speakers", "Status", "Message"],
        )
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.verticalHeader().setVisible(False)
        layout.addWidget(self.history_table, 1)

    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        first_frame = incident_data.frame_at_index(0) if incident_data is not None else None
        self.show_frame(first_frame, None)

    # =====================================================

    def show_frame(self, frame, _report):

        self.active_table.setRowCount(0)
        self.history_table.setRowCount(0)

        if self._incident is None or frame is None:
            return

        active_messages = self._incident.active_voice_messages_at_time(frame.time)
        name = self._incident.name_for

        zone_ids = sorted(active_messages.keys(), key=name)
        self.active_table.setRowCount(len(zone_ids))

        for row_index, zone_id in enumerate(zone_ids):

            message = active_messages[zone_id]

            self.active_table.setItem(row_index, 0, QTableWidgetItem(name(zone_id)))
            self.active_table.setItem(row_index, 1, QTableWidgetItem(message.message_text))
            self.active_table.setItem(row_index, 2, QTableWidgetItem(_format_percent(message.confidence)))
            self.active_table.setItem(row_index, 3, QTableWidgetItem(str(message.priority)))
            self.active_table.setItem(
                row_index, 4,
                QTableWidgetItem(message.message_type.name if message.message_type else "-"),
            )

        history = self._incident.voice_broadcast_history
        self.history_table.setRowCount(len(history))

        for row_index, instruction in enumerate(history):

            speakers = ", ".join(instruction.speaker_ids) or "(none)"
            message_text = instruction.message.message_text if instruction.message else "-"

            self.history_table.setItem(row_index, 0, QTableWidgetItem(_format_seconds(instruction.timestamp)))
            self.history_table.setItem(row_index, 1, QTableWidgetItem(name(instruction.target_zone_id)))
            self.history_table.setItem(row_index, 2, QTableWidgetItem(speakers))
            self.history_table.setItem(row_index, 3, QTableWidgetItem(_status_label(instruction.status)))
            self.history_table.setItem(row_index, 4, QTableWidgetItem(message_text))


_STATUS_LABELS = {
    BroadcastStatus.BROADCAST: "Broadcast (simulated)",
    BroadcastStatus.NO_SPEAKERS_AVAILABLE: "Unavailable -- no active speaker in zone",
    BroadcastStatus.SUPERSEDED: "Superseded",
    BroadcastStatus.CANCELLED: "Cancelled",
}


def _status_label(status):

    return _STATUS_LABELS.get(status, status.name)


# =====================================================


class BuildingRecommendationsPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None
        self._report = None

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Action", "Target", "Confidence", "Timestamp", "Status", "Expected Benefit"],
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, 2)

        detail_group = QGroupBox("Why this recommendation? (select a row)")
        detail_layout = QVBoxLayout(detail_group)
        self.detail_label = QLabel("-")
        self.detail_label.setWordWrap(True)
        detail_layout.addWidget(self.detail_label)
        layout.addWidget(detail_group, 1)

    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        first_report = incident_data.advisory_report_at_index(0) if incident_data is not None else None
        self.show_frame(None, first_report)

    # =====================================================

    def show_frame(self, _frame, report):

        self._report = report

        self.table.setRowCount(0)
        self.detail_label.setText("-")

        if report is None:
            return

        name = self._incident.name_for if self._incident is not None else (lambda object_id: object_id)
        recommendations = report.building_recommendations
        self.table.setRowCount(len(recommendations))

        for row_index, entry in enumerate(recommendations):

            target = f"{entry.target_type}: {name(entry.target_id)}" if entry.target_id else entry.target_type

            action_item = QTableWidgetItem(entry.action)
            action_item.setData(_ROW_ID_ROLE, row_index)
            self.table.setItem(row_index, 0, action_item)
            self.table.setItem(row_index, 1, QTableWidgetItem(target))
            self.table.setItem(row_index, 2, QTableWidgetItem(_format_percent(entry.confidence)))
            self.table.setItem(row_index, 3, QTableWidgetItem(_format_seconds(report.simulation_time)))
            self.table.setItem(row_index, 4, QTableWidgetItem("Active"))
            self.table.setItem(row_index, 5, QTableWidgetItem(entry.expected_engineering_benefit))

    # =====================================================

    def _on_selection_changed(self):

        selected_items = self.table.selectedItems()

        if not selected_items or self._report is None:
            self.detail_label.setText("-")
            return

        row_index = self.table.item(selected_items[0].row(), 0).data(_ROW_ID_ROLE)
        entry = self._report.building_recommendations[row_index]

        name = self._incident.name_for if self._incident is not None else (lambda object_id: object_id)
        target = f"{entry.target_type} {name(entry.target_id)}" if entry.target_id else entry.target_type

        lines = [
            f"Why: {entry.reason}",
            f"Recommendation confidence (blended, rule-based): {_format_percent(entry.confidence)}",
            f"Confidence breakdown: Deterministic rule base = "
            f"{DETERMINISTIC_RULE_BASE_CONFIDENCE:.0%}; further per-signal breakdown is not "
            f"separately exposed by this recommendation, only the final blended value above.",
            "Similarity to previous simulations: Not available in this build.",
            # Unlike Civilian Announcements, Building Recommendations
            # (advisory_engine.py's own building-recommendation call
            # sites) never pass ai_confidence/rl_confidence into
            # recommendation_confidence() at all -- this is not a
            # per-run state, it is how this category of recommendation
            # is structurally computed, so the line below is always
            # accurate rather than conditional.
            "Prediction source: Deterministic Decision Policy risk thresholds (this recommendation "
            "category is never AI/RL-augmented).",
            "RL influence: Not deployed.",
            f"Decision Policy influence: target = {target}, action = {entry.action}",
        ]

        self.detail_label.setText("\n".join(lines))


# =====================================================


class FirefighterIntelligencePanel(QWidget):

    # Phase 3 + Phase 6 -- intelligence only, never a directive: every
    # field read below comes straight off FirefighterIntelligenceReport
    # (advisory_system.recommendation_models), which itself carries no
    # "assigned_task"/"order" field by design (see that dataclass's own
    # docstring) -- this panel never renders zone_decisions' imperative
    # action verbs, only already-neutral intelligence.

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None

        layout = QVBoxLayout(self)

        status_group = QGroupBox("Building Status")
        status_layout = QVBoxLayout(status_group)
        self.building_status_label = QLabel("-")
        self.rset_label = QLabel("Predicted RSET: -")
        self.confidence_label = QLabel("Confidence: -")
        for label in (self.building_status_label, self.rset_label, self.confidence_label):
            status_layout.addWidget(label)
        layout.addWidget(status_group)

        counts_group = QGroupBox("Occupants Remaining")
        counts_layout = QHBoxLayout(counts_group)
        self.remaining_value, remaining_tile = self._stat_tile("Remaining")
        self.children_value, children_tile = self._stat_tile("Children")
        self.wheelchair_value, wheelchair_tile = self._stat_tile("Wheelchair Users")
        self.injury_value, injury_tile = self._stat_tile("Possible Injuries")
        self.fallen_value, fallen_tile = self._stat_tile("Fallen")
        self.helping_value, helping_tile = self._stat_tile("Helping Groups")
        for tile in (remaining_tile, children_tile, wheelchair_tile, injury_tile, fallen_tile, helping_tile):
            counts_layout.addWidget(tile)
        layout.addWidget(counts_group)

        zones_group = QGroupBox("Occupants / Hazard / Smoke by Zone")
        zones_layout = QVBoxLayout(zones_group)
        self.zone_table = QTableWidget(0, 4)
        self.zone_table.setHorizontalHeaderLabels(["Zone", "Occupants", "Hazard Severity", "Smoke"])
        self.zone_table.horizontalHeader().setStretchLastSection(True)
        self.zone_table.verticalHeader().setVisible(False)
        zones_layout.addWidget(self.zone_table)
        layout.addWidget(zones_group)

        routes_group = QGroupBox("Route Conditions")
        routes_layout = QVBoxLayout(routes_group)
        self.available_routes_label = QLabel("Available access routes: -")
        self.available_routes_label.setWordWrap(True)
        self.blocked_routes_label = QLabel("Blocked routes: -")
        self.blocked_routes_label.setWordWrap(True)
        routes_layout.addWidget(self.available_routes_label)
        routes_layout.addWidget(self.blocked_routes_label)
        self.access_route_table = QTableWidget(0, 4)
        self.access_route_table.setHorizontalHeaderLabels(
            ["Priority Zone", "Via Exit", "Via Stair", "Est. Travel Time"],
        )
        self.access_route_table.horizontalHeader().setStretchLastSection(True)
        self.access_route_table.verticalHeader().setVisible(False)
        routes_layout.addWidget(self.access_route_table)
        layout.addWidget(routes_group)

        systems_group = QGroupBox("Building System Status")
        systems_layout = QVBoxLayout(systems_group)
        self.systems_table = QTableWidget(0, 2)
        self.systems_table.setHorizontalHeaderLabels(["System / Object", "State"])
        self.systems_table.horizontalHeader().setStretchLastSection(True)
        self.systems_table.verticalHeader().setVisible(False)
        systems_layout.addWidget(self.systems_table)
        layout.addWidget(systems_group)

        rescue_group = QGroupBox("Priority Rescue Areas")
        rescue_layout = QVBoxLayout(rescue_group)
        self.rescue_table = QTableWidget(0, 3)
        self.rescue_table.setHorizontalHeaderLabels(["Zone", "Priority", "Occupants"])
        self.rescue_table.horizontalHeader().setStretchLastSection(True)
        self.rescue_table.verticalHeader().setVisible(False)
        rescue_layout.addWidget(self.rescue_table)
        layout.addWidget(rescue_group)

    # =====================================================

    @staticmethod
    def _stat_tile(title):

        tile = QWidget()
        tile_layout = QVBoxLayout(tile)

        value_label = QLabel("-")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = value_label.font()
        font.setPointSizeF(font.pointSizeF() * 1.6)
        font.setBold(True)
        value_label.setFont(font)

        caption_label = QLabel(title)
        caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption_label.setWordWrap(True)

        tile_layout.addWidget(value_label)
        tile_layout.addWidget(caption_label)

        return value_label, tile

    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        first_report = incident_data.advisory_report_at_index(0) if incident_data is not None else None
        self.show_frame(None, first_report)

    # =====================================================

    def show_frame(self, _frame, report):

        intel = report.firefighter_intelligence if report is not None else None

        if intel is None:

            self.building_status_label.setText("-")
            self.rset_label.setText("Predicted RSET: -")
            self.confidence_label.setText("Confidence: -")

            for value_label in (
                self.remaining_value, self.children_value, self.wheelchair_value,
                self.injury_value, self.fallen_value, self.helping_value,
            ):
                value_label.setText("-")

            for table in (self.zone_table, self.access_route_table, self.systems_table, self.rescue_table):
                table.setRowCount(0)

            self.available_routes_label.setText("Available access routes: -")
            self.blocked_routes_label.setText("Blocked routes: -")
            return

        name = self._incident.name_for if self._incident is not None else (lambda object_id: object_id)

        self.building_status_label.setText(intel.building_status)
        self.rset_label.setText(f"Predicted RSET: {_format_seconds(intel.predicted_rset_seconds)}")
        self.confidence_label.setText(f"Confidence: {_format_percent(intel.confidence)}")

        self.remaining_value.setText(_format_number(intel.occupants_remaining))
        self.children_value.setText(_format_number(intel.children_count))
        self.wheelchair_value.setText(_format_number(intel.wheelchair_users_count))
        self.injury_value.setText(_format_number(intel.possible_injury_count))
        self.fallen_value.setText(_format_number(intel.fallen_count))
        self.helping_value.setText(_format_number(intel.helping_groups_count))

        zone_ids = sorted(
            set(intel.occupants_by_zone) | set(intel.hazard_severity_by_zone) | set(intel.smoke_conditions_by_zone),
        )
        self.zone_table.setRowCount(len(zone_ids))
        for row_index, zone_id in enumerate(zone_ids):

            hazard = intel.hazard_severity_by_zone.get(zone_id)
            smoke = intel.smoke_conditions_by_zone.get(zone_id)

            self.zone_table.setItem(row_index, 0, QTableWidgetItem(name(zone_id)))
            self.zone_table.setItem(
                row_index, 1, QTableWidgetItem(_format_number(intel.occupants_by_zone.get(zone_id))),
            )
            self.zone_table.setItem(
                row_index, 2, QTableWidgetItem(f"{hazard:.2f}" if hazard is not None else "-"),
            )
            self.zone_table.setItem(
                row_index, 3, QTableWidgetItem(f"{smoke:.2f}" if smoke is not None else "-"),
            )

        self.available_routes_label.setText(
            f"Available access routes: {_join_or_dash(intel.available_routes)}",
        )
        self.blocked_routes_label.setText(f"Blocked routes: {_join_or_dash(intel.blocked_routes)}")

        self.access_route_table.setRowCount(len(intel.suggested_access_routes))
        for row_index, entry in enumerate(intel.suggested_access_routes):
            self.access_route_table.setItem(row_index, 0, QTableWidgetItem(name(entry.get("zone_id"))))
            self.access_route_table.setItem(row_index, 1, QTableWidgetItem(name(entry.get("via_exit"))))
            self.access_route_table.setItem(row_index, 2, QTableWidgetItem(name(entry.get("via_stair"))))
            travel_time = entry.get("estimated_travel_time_seconds")
            self.access_route_table.setItem(
                row_index, 3, QTableWidgetItem(f"{travel_time:.0f}s" if travel_time is not None else "-"),
            )

        system_ids = sorted(intel.building_system_status)
        self.systems_table.setRowCount(len(system_ids))
        for row_index, object_id in enumerate(system_ids):
            state = intel.building_system_status[object_id].get("state", "-")
            self.systems_table.setItem(row_index, 0, QTableWidgetItem(name(object_id)))
            self.systems_table.setItem(row_index, 1, QTableWidgetItem(str(state)))

        priorities = [entry for entry in intel.rescue_priority_areas if entry.get("rescue_priority") != "NONE"]
        self.rescue_table.setRowCount(len(priorities))
        for row_index, entry in enumerate(priorities):
            self.rescue_table.setItem(row_index, 0, QTableWidgetItem(name(entry.get("zone_id"))))
            self.rescue_table.setItem(row_index, 1, QTableWidgetItem(str(entry.get("rescue_priority"))))
            self.rescue_table.setItem(row_index, 2, QTableWidgetItem(_format_number(entry.get("occupant_count"))))


# =====================================================


class CommanderSummaryPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None

        layout = QVBoxLayout(self)

        stats_group = QGroupBox("Incident Commander Summary")
        stats_layout = QHBoxLayout(stats_group)
        self.severity_value, severity_tile = self._stat_tile("Overall Severity")
        self.remaining_value, remaining_tile = self._stat_tile("Occupants Remaining")
        self.rset_value, rset_tile = self._stat_tile("Predicted RSET")
        self.occupancy_confidence_value, occupancy_confidence_tile = self._stat_tile("Occupancy Confidence")
        self.recommendation_confidence_value, recommendation_confidence_tile = self._stat_tile(
            "Recommendation Confidence",
        )
        for tile in (
            severity_tile, remaining_tile, rset_tile,
            occupancy_confidence_tile, recommendation_confidence_tile,
        ):
            stats_layout.addWidget(tile)
        layout.addWidget(stats_group)

        zones_group = QGroupBox("Zone Status")
        zones_layout = QVBoxLayout(zones_group)
        self.safe_zones_label = QLabel("Safe zones: -")
        self.safe_zones_label.setWordWrap(True)
        self.warning_zones_label = QLabel("Warning zones: -")
        self.warning_zones_label.setWordWrap(True)
        self.critical_zones_label = QLabel("Critical zones: -")
        self.critical_zones_label.setWordWrap(True)
        for label in (self.safe_zones_label, self.warning_zones_label, self.critical_zones_label):
            zones_layout.addWidget(label)
        layout.addWidget(zones_group)

        routes_group = QGroupBox("Routes")
        routes_layout = QVBoxLayout(routes_group)
        self.available_exits_label = QLabel("Available exits: -")
        self.available_exits_label.setWordWrap(True)
        self.blocked_routes_label = QLabel("Blocked routes: -")
        self.blocked_routes_label.setWordWrap(True)
        self.bottlenecks_label = QLabel("Predicted bottlenecks: -")
        self.bottlenecks_label.setWordWrap(True)
        for label in (self.available_exits_label, self.blocked_routes_label, self.bottlenecks_label):
            routes_layout.addWidget(label)
        layout.addWidget(routes_group)

        rescue_group = QGroupBox("Highest Priority Rescue Areas")
        rescue_layout = QVBoxLayout(rescue_group)
        self.priority_areas_label = QLabel("-")
        self.priority_areas_label.setWordWrap(True)
        rescue_layout.addWidget(self.priority_areas_label)
        layout.addWidget(rescue_group)

        occupants_group = QGroupBox("Occupants by Zone")
        occupants_layout = QVBoxLayout(occupants_group)
        self.occupants_table = QTableWidget(0, 2)
        self.occupants_table.setHorizontalHeaderLabels(["Zone", "Occupants"])
        self.occupants_table.horizontalHeader().setStretchLastSection(True)
        self.occupants_table.verticalHeader().setVisible(False)
        occupants_layout.addWidget(self.occupants_table)
        layout.addWidget(occupants_group)

        layout.addStretch(1)

    # =====================================================

    @staticmethod
    def _stat_tile(title):

        tile = QWidget()
        tile_layout = QVBoxLayout(tile)

        value_label = QLabel("-")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = value_label.font()
        font.setPointSizeF(font.pointSizeF() * 1.6)
        font.setBold(True)
        value_label.setFont(font)

        caption_label = QLabel(title)
        caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption_label.setWordWrap(True)

        tile_layout.addWidget(value_label)
        tile_layout.addWidget(caption_label)

        return value_label, tile

    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        first_report = incident_data.advisory_report_at_index(0) if incident_data is not None else None
        self.show_frame(None, first_report)

    # =====================================================

    def show_frame(self, _frame, report):

        dashboard = report.commander_dashboard if report is not None else None

        if dashboard is None:

            self.severity_value.setText("-")
            self.remaining_value.setText("-")
            self.rset_value.setText("-")
            self.occupancy_confidence_value.setText("-")
            self.recommendation_confidence_value.setText("-")
            self.safe_zones_label.setText("Safe zones: -")
            self.warning_zones_label.setText("Warning zones: -")
            self.critical_zones_label.setText("Critical zones: -")
            self.available_exits_label.setText("Available exits: -")
            self.blocked_routes_label.setText("Blocked routes: -")
            self.bottlenecks_label.setText("Predicted bottlenecks: -")
            self.priority_areas_label.setText("-")
            self.occupants_table.setRowCount(0)
            return

        name = self._incident.name_for if self._incident is not None else (lambda object_id: object_id)

        self.severity_value.setText(dashboard.overall_incident_severity or "-")
        self.remaining_value.setText(_format_number(dashboard.occupants_remaining))
        self.rset_value.setText(_format_seconds(dashboard.predicted_rset_seconds))
        self.occupancy_confidence_value.setText(_format_percent(dashboard.occupancy_confidence))
        self.recommendation_confidence_value.setText(_format_percent(dashboard.recommendation_confidence))

        self.safe_zones_label.setText(f"Safe zones: {_join_or_dash([name(z) for z in dashboard.safe_zones])}")
        self.warning_zones_label.setText(
            f"Warning zones: {_join_or_dash([name(z) for z in dashboard.warning_zones])}",
        )
        self.critical_zones_label.setText(
            f"Critical zones: {_join_or_dash([name(z) for z in dashboard.critical_zones])}",
        )

        self.available_exits_label.setText(
            f"Available exits: {_join_or_dash([name(e) for e in dashboard.available_exits])}",
        )
        self.blocked_routes_label.setText(f"Blocked routes: {_join_or_dash(dashboard.blocked_routes)}")
        self.bottlenecks_label.setText(
            f"Predicted bottlenecks: {_join_or_dash([name(b) for b in dashboard.predicted_bottlenecks])}",
        )

        self.priority_areas_label.setText(
            _join_or_dash([name(z) for z in dashboard.highest_priority_rescue_areas]),
        )

        zone_ids = sorted(dashboard.occupants_by_zone)
        self.occupants_table.setRowCount(len(zone_ids))
        for row_index, zone_id in enumerate(zone_ids):
            self.occupants_table.setItem(row_index, 0, QTableWidgetItem(name(zone_id)))
            self.occupants_table.setItem(
                row_index, 1, QTableWidgetItem(_format_number(dashboard.occupants_by_zone[zone_id])),
            )


# =====================================================


class RecommendationCenter(QTabWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.civilian_panel = CivilianAnnouncementsPanel()
        self.firefighter_panel = FirefighterIntelligencePanel()
        self.building_panel = BuildingRecommendationsPanel()
        self.commander_panel = CommanderSummaryPanel()
        self.voice_evacuation_panel = VoiceEvacuationPanel()
        self.building_controls_panel = BuildingControlsPanel()

        self.addTab(self.civilian_panel, "Civilian Voice Announcements")
        self.addTab(self.voice_evacuation_panel, "Voice Evacuation / Broadcast")
        self.addTab(self.firefighter_panel, "Firefighter Intelligence")
        self.addTab(self.building_panel, "Building System Recommendations")
        self.addTab(self.building_controls_panel, "Building Controls")
        self.addTab(self.commander_panel, "Incident Commander Summary")

    # =====================================================

    def set_incident(self, incident_data):

        self.civilian_panel.set_incident(incident_data)
        self.firefighter_panel.set_incident(incident_data)
        self.building_panel.set_incident(incident_data)
        self.commander_panel.set_incident(incident_data)
        self.voice_evacuation_panel.set_incident(incident_data)
        self.building_controls_panel.set_incident(incident_data)

    # =====================================================

    def show_frame(self, frame, report):

        self.civilian_panel.show_frame(frame, report)
        self.firefighter_panel.show_frame(frame, report)
        self.building_panel.show_frame(frame, report)
        self.commander_panel.show_frame(frame, report)
        self.voice_evacuation_panel.show_frame(frame, report)
        self.building_controls_panel.show_frame(frame, report)
