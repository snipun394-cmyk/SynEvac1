from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from perception.human_inference import derive_inference_flags


# =====================================================
# HumanPanel -- Phase 7's own operator-facing requirement: let an
# operator inspect a detected person (id, classification, current
# state, behavior events, inference flags, confidence). Dumb widget,
# same convention every other Command Center panel already follows:
# set_incident() runs once per loaded incident, show_frame() on every
# playback/live tick, and no domain logic beyond formatting lives here
# -- classification/state/behavior_events are read straight off
# HumanObservation, and inference flags are computed by calling
# perception.human_inference.derive_inference_flags() (never
# reimplemented here), the one function Phase 5 designates as the only
# place allowed to produce one.
# =====================================================


def _format_confidence(confidence):

    return "-" if confidence is None else f"{confidence:.0%}"


def _format_enum(value):

    return "-" if value is None else value.name


def _format_enum_set(values):

    if not values:
        return "-"

    return ", ".join(sorted(value.name for value in values))


def _format_decision_field(decision_info, key):

    # Human Decision Engine Phase 7 -- decision_info is the plain dict
    # IncidentFrame.human_decisions carries for one person (see that
    # field's own docstring); "-" for a key this person's frame never
    # supplied, exactly the same "-" placeholder every other optional
    # field in this panel already uses (_format_confidence/_format_enum).

    if not decision_info:
        return "-"

    value = decision_info.get(key)

    if value is None or value == "":
        return "-"

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


class HumanPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None
        self._frame = None

        layout = QVBoxLayout(self)

        table_group = QGroupBox("Detected People")
        table_layout = QVBoxLayout(table_group)
        self.people_table = QTableWidget(0, 6)
        self.people_table.setHorizontalHeaderLabels(
            ["Person", "Classification", "State", "Zone", "Confidence", "Decision"],
        )
        self.people_table.horizontalHeader().setStretchLastSection(True)
        self.people_table.verticalHeader().setVisible(False)
        self.people_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.people_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.people_table.itemSelectionChanged.connect(self._on_selection_changed)
        table_layout.addWidget(self.people_table)
        layout.addWidget(table_group)

        detail_group = QGroupBox("Selected Person")
        detail_layout = QVBoxLayout(detail_group)
        self.person_id_label = self._label("Person: -")
        self.classification_label = self._label("Classification: -")
        self.state_label = self._label("State: -")
        self.behavior_events_label = self._label("Behavior events: -")
        self.inference_flags_label = self._label("Inference flags: -")
        self.confidence_label = self._label("Confidence: -")
        # Human Decision Engine Phase 7 -- Current Decision/Goal/Group/
        # Assistance Status/Rescue Priority/Firefighter Current Task.
        self.decision_label = self._label("Decision: -")
        self.goal_label = self._label("Goal: -")
        self.group_label = self._label("Group: -")
        self.assistance_status_label = self._label("Assistance status: -")
        self.rescue_priority_label = self._label("Rescue priority: -")
        self.firefighter_task_label = self._label("Firefighter task: -")
        for label in (
            self.person_id_label,
            self.classification_label,
            self.state_label,
            self.behavior_events_label,
            self.inference_flags_label,
            self.confidence_label,
            self.decision_label,
            self.goal_label,
            self.group_label,
            self.assistance_status_label,
            self.rescue_priority_label,
            self.firefighter_task_label,
        ):
            detail_layout.addWidget(label)
        layout.addWidget(detail_group)

        layout.addStretch(1)

    # =====================================================

    @staticmethod
    def _label(text):

        label = QLabel(text)
        label.setWordWrap(True)
        return label

    # =====================================================
    # Public API -- pushed updates only.
    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        first_frame = incident_data.frame_at_index(0) if incident_data is not None else None
        self.show_frame(first_frame)

    # =====================================================

    def show_frame(self, frame):

        self._frame = frame

        self.people_table.setRowCount(0)
        self._clear_detail()

        if frame is None:
            return

        # Human Decision Engine Phase 7 -- a dynamic-registration-only
        # run (behaviour_profile_resolver.dynamic_registrar) has no
        # perception layer involved at all, so a person may appear in
        # frame.human_decisions with no matching HumanObservation (and
        # vice versa, for an ordinary perception-only run predating
        # this phase) -- the row set is the union of both, never just
        # one.
        person_ids = sorted(set(frame.human_observations) | set(frame.human_decisions))

        for row, person_id in enumerate(person_ids):

            observation = frame.human_observations.get(person_id)
            decision_info = frame.human_decisions.get(person_id, {})

            self.people_table.insertRow(row)

            person_item = QTableWidgetItem(self._person_label(person_id))
            person_item.setData(Qt.ItemDataRole.UserRole, person_id)
            self.people_table.setItem(row, 0, person_item)

            if observation is not None:
                self.people_table.setItem(row, 1, QTableWidgetItem(observation.classification.name))
                self.people_table.setItem(row, 2, QTableWidgetItem(_format_enum(observation.state)))
                self.people_table.setItem(row, 3, QTableWidgetItem(self._zone_name(observation.zone_id)))
                self.people_table.setItem(
                    row, 4, QTableWidgetItem(_format_confidence(observation.confidence)),
                )
            else:
                for column in (1, 2, 3, 4):
                    self.people_table.setItem(row, column, QTableWidgetItem("-"))

            self.people_table.setItem(
                row, 5, QTableWidgetItem(_format_decision_field(decision_info, "decision")),
            )

    # =====================================================
    # Internal wiring
    # =====================================================

    def _on_selection_changed(self):

        selected_items = self.people_table.selectedItems()

        if not selected_items or self._frame is None:
            self._clear_detail()
            return

        person_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        observation = self._frame.human_observations.get(person_id)
        decision_info = self._frame.human_decisions.get(person_id, {})

        if observation is None and not decision_info:
            self._clear_detail()
            return

        self.person_id_label.setText(f"Person: {self._person_label(person_id)}")

        if observation is not None:
            flags = derive_inference_flags(observation)
            self.classification_label.setText(f"Classification: {observation.classification.name}")
            self.state_label.setText(f"State: {_format_enum(observation.state)}")
            self.behavior_events_label.setText(
                f"Behavior events: {_format_enum_set(observation.behavior_events)}",
            )
            self.inference_flags_label.setText(f"Inference flags: {_format_enum_set(flags)}")
            self.confidence_label.setText(f"Confidence: {_format_confidence(observation.confidence)}")
        else:
            self.classification_label.setText("Classification: -")
            self.state_label.setText("State: -")
            self.behavior_events_label.setText("Behavior events: -")
            self.inference_flags_label.setText("Inference flags: -")
            self.confidence_label.setText("Confidence: -")

        self.decision_label.setText(f"Decision: {_format_decision_field(decision_info, 'decision')}")
        self.goal_label.setText(f"Goal: {_format_decision_field(decision_info, 'goal')}")
        self.group_label.setText(f"Group: {_format_decision_field(decision_info, 'group')}")
        self.assistance_status_label.setText(
            f"Assistance status: {_format_decision_field(decision_info, 'assistance_status')}",
        )
        self.rescue_priority_label.setText(
            f"Rescue priority: {_format_decision_field(decision_info, 'rescue_priority')}",
        )
        self.firefighter_task_label.setText(
            f"Firefighter task: {_format_decision_field(decision_info, 'firefighter_task')}",
        )

    # =====================================================

    def _clear_detail(self):

        self.person_id_label.setText("Person: -")
        self.classification_label.setText("Classification: -")
        self.state_label.setText("State: -")
        self.behavior_events_label.setText("Behavior events: -")
        self.inference_flags_label.setText("Inference flags: -")
        self.confidence_label.setText("Confidence: -")
        self.decision_label.setText("Decision: -")
        self.goal_label.setText("Goal: -")
        self.group_label.setText("Group: -")
        self.assistance_status_label.setText("Assistance status: -")
        self.rescue_priority_label.setText("Rescue priority: -")
        self.firefighter_task_label.setText("Firefighter task: -")

    # =====================================================

    def _person_label(self, person_id):

        return person_id

    # =====================================================

    def _zone_name(self, zone_id):

        if zone_id is None:
            return "-"

        if self._incident is None:
            return zone_id

        return self._incident.name_for(zone_id)
