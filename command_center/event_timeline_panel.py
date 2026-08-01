from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


# =====================================================
# EventTimelinePanel -- Simulation Replay Studio V1's unified,
# chronological Event Timeline. Every entry is either read directly off
# an already-recorded artifact or an honestly DERIVED computation over
# consecutive, already-computed IncidentFrames (a door/exit/stair/
# detector state diff) -- nothing here is fabricated or re-simulated.
# Sources, merged and sorted by time:
#
#   - Occupant depart/arrive -- simulation_recording.occupant_routes.
#     OccupantRouteRecord.depart_time/arrival_time (from
#     IncidentData.occupant_routes).
#   - Engineering state changes -- a diff between consecutive
#     IncidentFrame.door_states/exit_states/stair_states/detector_states
#     (nothing else in this codebase already exposes discrete "this
#     door just changed" events with a timestamp -- see this milestone's
#     own architectural inventory).
#   - Decision events -- human_decision_engine.events.DecisionEvent.
#     to_dict()'s own shape (IncidentData.decision_events). Most of
#     these are registration-time (pre-departure) decisions with no
#     simulation-clock timestamp of their own -- shown as "Pre-departure"
#     rather than a fabricated t=0.0s, except the two post-hoc event
#     types (Rescue_Completed/Group_Dissolved) whose own metadata
#     already carries a real arrival_time/dissolution_time.
#   - Recommendation changes -- IncidentData.recommendation_history
#     (advisory_system.RecommendationHistory, already computed).
#   - Voice broadcasts -- IncidentData.voice_broadcast_history
#     (voice_evacuation.BroadcastLog, already computed).
#
# Same convention as every other Command Center panel: set_incident()
# builds the whole (static, whole-run) list once; show_frame() only
# updates which row is highlighted as "current", never rebuilds rows.
# =====================================================

_PRE_DEPARTURE_SENTINEL = -1.0


class EventTimelinePanel(QWidget):

    # Emitted when the operator double-clicks a row -- Dashboard wires
    # this straight into the same slider-driven jump path TimelinePanel's
    # own "Jump to" control already uses (see command_center/dashboard.py).
    jump_to_time = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None
        self._entries = []  # list of (time, category, description)

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Time", "Category", "Event"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table)

    # =====================================================
    # Public API -- pushed updates only.
    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data
        self._entries = []

        if incident_data is not None:

            self._entries.extend(self._occupant_entries(incident_data))
            self._entries.extend(self._engineering_entries(incident_data))
            self._entries.extend(self._decision_entries(incident_data))
            self._entries.extend(self._recommendation_entries(incident_data))
            self._entries.extend(self._voice_entries(incident_data))

            self._entries.sort(key=lambda entry: entry[0])

        self._populate_table()

    # =====================================================

    def show_frame(self, frame):

        # No row content ever changes per frame (this is a whole-run,
        # static timeline) -- nothing to do; the highlighted-row concept
        # is intentionally left to a future pass rather than guessed at
        # here (see this milestone's own final report for this
        # disclosed limitation).
        return

    # =====================================================
    # Entry sources
    # =====================================================

    def _occupant_entries(self, incident_data):

        entries = []

        for record in incident_data.occupant_routes:

            if record.hops:
                entries.append((record.depart_time, "Occupant", f"{record.occupant_id} starts moving"))

            if record.arrival_time is not None:
                entries.append((record.arrival_time, "Occupant", f"{record.occupant_id} arrives"))

        return entries

    # =====================================================

    def _engineering_entries(self, incident_data):

        entries = []
        frames = incident_data.frames

        state_maps = ("door_states", "exit_states", "stair_states", "detector_states")

        for previous_frame, frame in zip(frames, frames[1:]):

            for attribute in state_maps:

                previous_states = getattr(previous_frame, attribute)
                current_states = getattr(frame, attribute)

                for object_id, state in current_states.items():

                    if previous_states.get(object_id) != state:

                        name = incident_data.name_for(object_id)
                        entries.append((frame.time, "Engineering", f"{name} changed to {state}"))

        return entries

    # =====================================================

    def _decision_entries(self, incident_data):

        entries = []

        for event in incident_data.decision_events:

            metadata = event.get("metadata", {}) or {}
            time = metadata.get("arrival_time", metadata.get("dissolution_time", _PRE_DEPARTURE_SENTINEL))

            occupant_name = incident_data.name_for(event.get("occupant_id"))
            related_id = event.get("related_occupant_id")
            related_text = f" -> {incident_data.name_for(related_id)}" if related_id else ""
            reason = event.get("reason", "")

            description = f"{event.get('event_type', 'Decision')}: {occupant_name}{related_text}"
            if reason:
                description += f" ({reason})"

            entries.append((time, "Decision", description))

        return entries

    # =====================================================

    def _recommendation_entries(self, incident_data):

        entries = []

        for change in incident_data.recommendation_history:

            zone_name = incident_data.name_for(change.zone_id)
            description = (
                f"Zone {zone_name}: {change.previous_recommendation} -> {change.new_recommendation}"
            )
            if change.reason_for_change:
                description += f" ({change.reason_for_change})"

            entries.append((change.timestamp, "Recommendation", description))

        return entries

    # =====================================================

    def _voice_entries(self, incident_data):

        entries = []

        for instruction in incident_data.voice_broadcast_history:

            zone_name = incident_data.name_for(instruction.target_zone_id)
            text = instruction.message.message_text if instruction.message is not None else ""

            entries.append((instruction.timestamp, "Voice", f"{zone_name}: {text}"))

        return entries

    # =====================================================
    # Rendering
    # =====================================================

    def _populate_table(self):

        self.table.setRowCount(len(self._entries))

        for row, (time, category, description) in enumerate(self._entries):

            time_text = "Pre-departure" if time == _PRE_DEPARTURE_SENTINEL else f"{time:.1f}s"

            self.table.setItem(row, 0, QTableWidgetItem(time_text))
            self.table.setItem(row, 1, QTableWidgetItem(category))
            self.table.setItem(row, 2, QTableWidgetItem(description))

    # =====================================================

    def _on_row_double_clicked(self, row, _column):

        if row < 0 or row >= len(self._entries):
            return

        time, _category, _description = self._entries[row]

        self.jump_to_time.emit(max(time, 0.0))
