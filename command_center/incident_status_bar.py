from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


# =====================================================
# IncidentStatusBar -- Phase 1's Live Building Dashboard. A row of
# stat tiles summarizing the whole incident at the currently displayed
# instant. Dumb widget, same convention every other Command Center
# panel follows: set_incident() runs once per loaded incident (Building
# name + a zone_id -> floor_name lookup built once from Building),
# show_frame(frame, report) runs on every playback tick. Every value is
# read straight off IncidentFrame (per-tick simulated facts) or the
# current tick's advisory_system.AdvisoryReport.commander_dashboard
# (already-computed severity/RSET/confidence fields) -- nothing here
# recomputes a severity, a confidence, or an RSET of its own. report is
# None whenever no GroundTruth was loaded for this incident (see
# IncidentData.advisory_report_at_index()'s own docstring); every tile
# that depends on it degrades to "-" rather than guessing.
# =====================================================

_SEVERITY_STYLES = {
    "RESOLVED": "color: #6bbf8a;",
    "STABLE": "color: #6bbf8a;",
    "ELEVATED": "color: #e0b04a;",
    "CRITICAL": "color: #d9534f;",
}


def _format_seconds(value):

    if value is None:
        return "-"

    minutes, seconds = divmod(int(value), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _format_percent(value):

    return "-" if value is None else f"{value:.0%}"


def _format_number(value):

    return "-" if value is None else str(value)


class IncidentStatusBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None
        self._zone_floor_names = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)

        self.building_name_value, building_name_tile = self._tile("Building")
        self.incident_status_value, incident_status_tile = self._tile("Incident Status")
        self.incident_severity_value, incident_severity_tile = self._tile("Severity")
        self.rset_value, rset_tile = self._tile("Predicted RSET")
        self.remaining_value, remaining_tile = self._tile("Remaining")
        self.evacuated_value, evacuated_tile = self._tile("Evacuated")
        self.occupancy_confidence_value, occupancy_confidence_tile = self._tile("Occupancy Confidence")
        self.recommendation_confidence_value, recommendation_confidence_tile = self._tile(
            "Recommendation Confidence",
        )
        self.time_since_alarm_value, time_since_alarm_tile = self._tile("Time Since Alarm")
        self.priority_zone_value, priority_zone_tile = self._tile("Highest Priority Zone")
        self.fire_floor_value, fire_floor_tile = self._tile("Current Fire Floor")

        # Live Command Center Integration milestone -- additive tiles.
        # Mode is always shown (REPLAY today, LIVE once a live data
        # source is attached); AI Bottleneck Probability is kept
        # structurally separate from Recommendation Confidence and
        # Occupancy Confidence above -- three genuinely different
        # quantities (see docs/architecture/ai_augmented_advisory_
        # integration.md §10), never conflated into one tile.
        self.mode_value, mode_tile = self._tile("Mode")
        self.mode_value.setText("REPLAY")
        self.ai_probability_value, ai_probability_tile = self._tile("AI Bottleneck Probability")

        self._tiles = (
            mode_tile, building_name_tile, incident_status_tile, incident_severity_tile, rset_tile,
            remaining_tile, evacuated_tile, occupancy_confidence_tile, recommendation_confidence_tile,
            ai_probability_tile, time_since_alarm_tile, priority_zone_tile, fire_floor_tile,
        )

        for index, tile in enumerate(self._tiles):
            layout.addWidget(tile, 1)
            if index < len(self._tiles) - 1:
                layout.addWidget(self._separator())

    # =====================================================

    @staticmethod
    def _tile(title):

        tile = QFrame()
        tile.setObjectName("StatTile")
        tile_layout = QVBoxLayout(tile)
        tile_layout.setContentsMargins(10, 4, 10, 4)

        value_label = QLabel("-")
        value_label.setObjectName("StatTileValue")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setWordWrap(True)
        font = value_label.font()
        font.setPointSizeF(font.pointSizeF() * 1.3)
        font.setBold(True)
        value_label.setFont(font)

        caption_label = QLabel(title)
        caption_label.setObjectName("StatTileCaption")
        caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        tile_layout.addWidget(value_label)
        tile_layout.addWidget(caption_label)

        return value_label, tile

    # =====================================================

    @staticmethod
    def _separator():

        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    # =====================================================
    # Public API -- pushed updates only.
    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data
        self._zone_floor_names = _zone_floor_name_lookup(incident_data)

        self.building_name_value.setText(
            incident_data.building.name if incident_data is not None else "-",
        )

        first_frame = incident_data.frame_at_index(0) if incident_data is not None else None
        first_report = incident_data.advisory_report_at_index(0) if incident_data is not None else None
        self.show_frame(first_frame, first_report)

    # =====================================================

    def set_mode(self, mode_label: str) -> None:

        # Live Command Center Integration milestone -- Phase 5's "Mode:
        # [Replay | Live]" requirement, applied to the one status-bar
        # tile that always shows it. Additive -- set_incident() above is
        # unmodified and continues to reset this tile to "REPLAY" for
        # every existing Replay caller.

        self.mode_value.setText(mode_label)

    # =====================================================

    def set_live_building(self, building_name) -> None:

        # Live Command Center Integration milestone -- the Live-mode
        # equivalent of set_incident()'s one-time building-name setup,
        # for a caller with no IncidentData to hand this widget.

        self.building_name_value.setText(building_name if building_name else "-")

    # =====================================================

    def show_frame(self, frame, report, *, live=False, ai_prediction_snapshot=None):

        # live/ai_prediction_snapshot are additive, defaulted kwargs --
        # every existing Replay caller (Dashboard.show_frame()) keeps
        # calling this positionally with exactly (frame, report), so
        # this signature change is backward compatible (Phase 3's own
        # "existing tests pass unmodified" requirement).

        commander_dashboard = report.commander_dashboard if report is not None else None

        if frame is None:
            self.remaining_value.setText("-")
            self.evacuated_value.setText("-")
            self.time_since_alarm_value.setText("-")
            self.fire_floor_value.setText("-")
        else:
            self.remaining_value.setText(_format_number(frame.people_remaining))
            self.evacuated_value.setText(_format_number(frame.people_evacuated))
            self.time_since_alarm_value.setText(_format_seconds(frame.time))
            self.fire_floor_value.setText(self._current_fire_floors(frame))

        # Live-only AI Bottleneck Probability -- kept structurally
        # separate from occupancy_confidence/recommendation_confidence
        # below (Phase 10's own confidence-separation requirement,
        # restated for this tile). "-" (never a fabricated 0%) whenever
        # no AI evidence is available this cycle.
        bottleneck = getattr(ai_prediction_snapshot, "bottleneck", None) if ai_prediction_snapshot is not None else None
        self.ai_probability_value.setText(_format_percent(bottleneck.probability if bottleneck is not None else None))

        if commander_dashboard is None:

            self.incident_status_value.setText("-")
            self.incident_severity_value.setText("-")
            self.incident_severity_value.setStyleSheet("")
            self.rset_value.setText("-")
            self.occupancy_confidence_value.setText("-")
            self.recommendation_confidence_value.setText("-")
            self.priority_zone_value.setText("-")
            return

        severity = commander_dashboard.overall_incident_severity
        self.incident_status_value.setText("Resolved" if severity == "RESOLVED" else "Active")
        self.incident_severity_value.setText(severity or "-")
        self.incident_severity_value.setStyleSheet(_SEVERITY_STYLES.get(severity, ""))

        # Predicted RSET is deliberately blanked in Live mode -- the
        # value behind commander_dashboard.predicted_rset_seconds comes
        # from GroundTruth.total_evacuation_time (see AdvisoryInputs),
        # which a live deployment can only supply via a caller-provided,
        # non-live GroundTruth artifact (ReplayCompatibleAdvisoryGateway
        # -- see docs/architecture/ai_augmented_advisory_integration.md
        # §12). Presenting it as a live estimate here would overclaim
        # exactly what Phase 6 forbids. The Live AI panel shows the
        # genuinely-live (but EXPERIMENTAL, separately labeled) evacuation-
        # time estimate instead, never this tile.
        self.rset_value.setText("-" if live else _format_seconds(commander_dashboard.predicted_rset_seconds))

        self.occupancy_confidence_value.setText(_format_percent(commander_dashboard.occupancy_confidence))
        self.recommendation_confidence_value.setText(
            _format_percent(commander_dashboard.recommendation_confidence),
        )

        priority_zones = commander_dashboard.highest_priority_rescue_areas
        name = self._incident.name_for if self._incident is not None else (lambda zone_id: zone_id)
        self.priority_zone_value.setText(name(priority_zones[0]) if priority_zones else "-")

    # =====================================================

    def _current_fire_floors(self, frame):

        floor_names = sorted({
            self._zone_floor_names[zone_id]
            for zone_id in frame.current_fire_zone_ids
            if zone_id in self._zone_floor_names
        })

        return ", ".join(floor_names) if floor_names else "None"


# =====================================================


def _zone_floor_name_lookup(incident_data):

    if incident_data is None:
        return {}

    lookup = {}
    for floor in incident_data.building.ordered_floors():
        for zone in floor.zones:
            lookup[zone.id] = floor.name

    return lookup
