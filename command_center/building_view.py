from collections import Counter

from PyQt6.QtCore import QEvent, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hazard.severity import HazardSeverity

from decision_policy.exit_policy import CLOSE as EXIT_CLOSE
from decision_policy.exit_policy import HIGH_CONGESTION
from decision_policy.exit_policy import KEEP_OPEN
from decision_policy.exit_policy import OPEN as EXIT_OPEN
from decision_policy.stair_policy import AVOID, CONGESTED, USE
from decision_policy.zone_policy import EVACUATE_IMMEDIATELY, SHELTER_IN_PLACE, WAIT

from perception.models.human_observation import BehaviorEvent, HumanClassification, HumanState

from simulation_recording.occupant_position import BuildingPositionIndex


# =====================================================
# BuildingView -- a read-only floor-plan rendering of one Building
# floor, color-coded by whichever overlay mode is active. Same "dumb
# widget, pushed updates only" convention every other Command Center
# panel follows (see designer/widgets/simulation_panel.py's own
# documented convention): this widget owns no domain logic and no
# timer, it never mutates Building/Scenario/IncidentData, and it never
# reruns anything -- MainWindow/Dashboard push a Building, a Floor, an
# IncidentFrame, and (optionally) a DecisionPolicy into it, and it
# repaints its QGraphicsScene accordingly.
# =====================================================

OVERLAY_HAZARD = "Hazard"
OVERLAY_OCCUPANCY = "Occupancy"
OVERLAY_CONGESTION = "Congestion"
OVERLAY_RECOMMENDATION = "Recommendation"

OVERLAY_MODES = (OVERLAY_HAZARD, OVERLAY_OCCUPANCY, OVERLAY_CONGESTION, OVERLAY_RECOMMENDATION)

_NO_DATA_COLOR = QColor(90, 95, 100)
_SAFE_COLOR = QColor(66, 133, 99)
_MODERATE_COLOR = QColor(224, 156, 45)
_ELEVATED_COLOR = QColor(214, 91, 45)
_CRITICAL_COLOR = QColor(176, 33, 33)
_INACTIVE_COLOR = QColor(130, 130, 130)
_DEVICE_COLOR = QColor(90, 140, 200)

_SEVERITY_COLORS = {
    HazardSeverity.NONE: _SAFE_COLOR,
    HazardSeverity.LOW: QColor(178, 186, 66),
    HazardSeverity.MODERATE: _MODERATE_COLOR,
    HazardSeverity.HIGH: _ELEVATED_COLOR,
    HazardSeverity.CRITICAL: _CRITICAL_COLOR,
}

_ZONE_ACTION_COLORS = {
    SHELTER_IN_PLACE: _CRITICAL_COLOR,
    EVACUATE_IMMEDIATELY: _ELEVATED_COLOR,
    WAIT: QColor(210, 180, 45),
}

_EXIT_STATUS_COLORS = {
    KEEP_OPEN: _SAFE_COLOR,
    EXIT_OPEN: _SAFE_COLOR,
    HIGH_CONGESTION: _MODERATE_COLOR,
    EXIT_CLOSE: _CRITICAL_COLOR,
}

_STAIR_STATUS_COLORS = {
    USE: _SAFE_COLOR,
    CONGESTED: _MODERATE_COLOR,
    AVOID: _CRITICAL_COLOR,
}

# Simulation Replay Studio V1 -- occupant marker coloring by their own
# live OccupantState name (simulation_recording.occupant_position.
# OccupantPosition.state, already an OccupantState.name string -- see
# that module's own _live_state_for reasoning for why this is never
# the run's whole-run-final state for a still-moving occupant).
_OCCUPANT_STATE_COLORS = {
    "PENDING": _NO_DATA_COLOR,
    "AT_NODE": _MODERATE_COLOR,
    "QUEUED": _ELEVATED_COLOR,
    "TRAVERSING": _DEVICE_COLOR,
    "ARRIVED": _SAFE_COLOR,
    "UNREACHABLE": _CRITICAL_COLOR,
    "STATIONARY": _INACTIVE_COLOR,
}

_OCCUPANT_MARKER_SIZE = 0.35
_SELECTED_OCCUPANT_MARKER_SIZE = 0.55
_SELECTED_OCCUPANT_COLOR = QColor(240, 220, 60)
_ROUTE_HIGHLIGHT_COLOR = QColor(240, 220, 60)


def _occupancy_color(count, capacity):

    if not count:
        return _NO_DATA_COLOR

    # Zones without a modeled max_occupancy still deserve a visual
    # signal -- shade by absolute headcount against a generous display
    # cap instead, rather than showing "no data" for every uncapped
    # zone.
    ratio = min(count / capacity, 1.0) if capacity else min(count / 10.0, 1.0)

    if ratio >= 0.85:
        return _CRITICAL_COLOR
    if ratio >= 0.6:
        return _ELEVATED_COLOR
    if ratio >= 0.3:
        return _MODERATE_COLOR

    return _SAFE_COLOR


def _smoke_severity(smoke):

    # smoke_level is a producer-defined magnitude, not guaranteed
    # normalized to [0, 1] the way hazard_score is -- these buckets are
    # a display-only convenience, distinct from HazardSeverity.
    # from_score()'s own engineering classification.

    if smoke is None:
        return HazardSeverity.NONE
    if smoke >= 0.85:
        return HazardSeverity.CRITICAL
    if smoke >= 0.6:
        return HazardSeverity.HIGH
    if smoke >= 0.3:
        return HazardSeverity.MODERATE
    if smoke > 0:
        return HazardSeverity.LOW

    return HazardSeverity.NONE


def _zone_hazard_severity(zone_id, frame):

    if frame is None:
        return None

    if zone_id == frame.ignition_zone_id or zone_id in frame.current_fire_zone_ids:
        return HazardSeverity.CRITICAL

    return _smoke_severity(frame.zone_smoke.get(zone_id))


def _congestion_color(value):

    if value is None:
        return _NO_DATA_COLOR

    if value >= 8:
        return _CRITICAL_COLOR
    if value >= 4:
        return _ELEVATED_COLOR
    if value >= 1:
        return _MODERATE_COLOR

    return _SAFE_COLOR


class BuildingView(QWidget):

    # Simulation Replay Studio V1 -- emitted when the operator clicks an
    # occupant marker directly on the floor plan (see eventFilter()
    # below); occupant_id is the same id every other occupant-keyed
    # frame field (IncidentFrame.occupant_positions/human_observations/
    # human_decisions) already uses. Purely additive: nothing before
    # this milestone ever connected to this signal, so no existing
    # behavior changes.
    occupant_clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view.viewport().installEventFilter(self)

        self.floor_combo = QComboBox()
        self.floor_combo.currentIndexChanged.connect(self._on_floor_combo_changed)

        self.overlay_combo = QComboBox()
        self.overlay_combo.addItems(OVERLAY_MODES)
        self.overlay_combo.currentTextChanged.connect(self._on_overlay_combo_changed)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Floor:"))
        top_bar.addWidget(self.floor_combo)
        top_bar.addSpacing(16)
        top_bar.addWidget(QLabel("Overlay:"))
        top_bar.addWidget(self.overlay_combo)
        top_bar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(top_bar)
        layout.addWidget(self._view, 1)

        self._building = None
        self._floors = []
        self._current_floor = None
        self._frame = None
        self._decision_policy = None
        self._overlay_mode = OVERLAY_HAZARD

        # Simulation Replay Studio V1 -- additive occupant/route state.
        self._position_index = None
        self._occupant_routes_by_id = {}
        self._selected_occupant_id = None

    # =====================================================
    # Public API -- pushed updates only.
    # =====================================================

    def set_building(self, building):

        self._building = building
        self._floors = list(building.ordered_floors()) if building is not None else []
        self._position_index = BuildingPositionIndex(building) if building is not None else None

        self.floor_combo.blockSignals(True)
        self.floor_combo.clear()
        for floor in self._floors:
            self.floor_combo.addItem(floor.name)
        self.floor_combo.blockSignals(False)

        self._current_floor = self._floors[0] if self._floors else None

        if self._floors:
            self.floor_combo.setCurrentIndex(0)

        self._redraw()

    # =====================================================

    def set_floor(self, floor):

        self._current_floor = floor

        if floor in self._floors:
            self.floor_combo.blockSignals(True)
            self.floor_combo.setCurrentIndex(self._floors.index(floor))
            self.floor_combo.blockSignals(False)

        self._redraw()

    # =====================================================

    def set_decision_policy(self, decision_policy):

        self._decision_policy = decision_policy
        self._redraw()

    # =====================================================

    def set_overlay_mode(self, mode):

        if mode not in OVERLAY_MODES:
            raise ValueError(f"Unknown overlay mode: {mode!r}")

        self._overlay_mode = mode

        index = self.overlay_combo.findText(mode)
        if index >= 0:
            self.overlay_combo.blockSignals(True)
            self.overlay_combo.setCurrentIndex(index)
            self.overlay_combo.blockSignals(False)

        self._redraw()

    # =====================================================

    def show_frame(self, frame):

        self._frame = frame
        self._redraw()

    # =====================================================

    def set_occupant_routes(self, records):

        # Simulation Replay Studio V1 -- records is an iterable of
        # simulation_recording.occupant_routes.OccupantRouteRecord (one
        # per occupant/firefighter); () when no occupant_routes artifact
        # was loaded for this incident, in which case highlight_route()
        # simply has nothing to draw for any occupant_id -- the same
        # "empty means artifact absent" convention this whole milestone
        # uses throughout.

        self._occupant_routes_by_id = {record.occupant_id: record for record in records}
        self._redraw()

    # =====================================================

    def select_occupant(self, occupant_id):

        self._selected_occupant_id = occupant_id
        self._redraw()

    # =====================================================

    @property
    def selected_occupant_id(self):
        return self._selected_occupant_id

    # =====================================================

    def eventFilter(self, watched, event):

        # Hit-tests whatever occupant marker (if any) sits under a click
        # on the floor-plan viewport -- occupant markers are the only
        # items this view ever tags with QGraphicsItem.setData(0, ...)
        # (see _render_occupants() below), so a hit on any other item
        # (a Zone rect, a Door/Exit line, ...) simply yields None and is
        # ignored, exactly like clicking empty canvas.

        if watched is self._view.viewport() and event.type() == QEvent.Type.MouseButtonPress:

            scene_pos = self._view.mapToScene(event.pos())
            item = self._scene.itemAt(scene_pos, self._view.transform())
            occupant_id = item.data(0) if item is not None else None

            if occupant_id:
                self.select_occupant(occupant_id)
                self.occupant_clicked.emit(occupant_id)

        return super().eventFilter(watched, event)

    # =====================================================

    @property
    def current_floor(self):
        return self._current_floor

    @property
    def overlay_mode(self):
        return self._overlay_mode

    @property
    def scene(self):
        return self._scene

    # =====================================================
    # Internal wiring
    # =====================================================

    def _on_floor_combo_changed(self, index):

        if 0 <= index < len(self._floors):
            self._current_floor = self._floors[index]
            self._redraw()

    # =====================================================

    def _on_overlay_combo_changed(self, text):

        if text in OVERLAY_MODES:
            self._overlay_mode = text
            self._redraw()

    # =====================================================
    # Rendering -- rebuilt from scratch on every push. Simplicity over
    # incremental diffing: this is a visualization layer over an
    # already-completed run, not a live high-frequency renderer.
    # =====================================================

    def _redraw(self):

        self._scene.clear()

        floor = self._current_floor
        if floor is None:
            return

        frame = self._frame

        for zone in floor.zones:

            rect_item = QGraphicsRectItem(QRectF(zone.x, zone.y, zone.width, zone.height))
            rect_item.setBrush(QBrush(self._zone_color(zone, frame)))
            rect_item.setPen(QPen(QColor(25, 25, 25), 0.05))
            rect_item.setToolTip(self._zone_tooltip(zone, frame))
            self._scene.addItem(rect_item)

            label = QGraphicsSimpleTextItem(zone.name)
            label.setBrush(QBrush(QColor(240, 240, 240)))
            label.setPos(zone.x + 0.1, zone.y + 0.1)
            label.setScale(0.25)
            self._scene.addItem(label)

        for door in floor.doors:
            self._add_line(door.start_point, door.end_point, self._door_color(door, frame), 0.15)

        for exit_obj in floor.exits:
            self._add_line(
                exit_obj.start_point, exit_obj.end_point, self._exit_color(exit_obj, frame), 0.25,
            )

        for stair in floor.stairs:
            self._add_marker(stair.from_position, self._stair_color(stair, frame), 0.5, rectangular=True)

        for camera in floor.cameras:
            self._add_marker(
                camera.position, self._device_color(camera.id, camera.active, frame, "camera"), 0.25,
            )

        for detector in floor.detectors:
            self._add_marker(
                detector.position,
                self._device_color(detector.id, detector.active, frame, "detector"),
                0.25,
            )

        bounds = self._scene.itemsBoundingRect()
        if not bounds.isNull():
            self._view.setSceneRect(bounds.adjusted(-2, -2, 2, 2))
            self._view.fitInView(self._view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Simulation Replay Studio V1 -- drawn after the view is fitted
        # to the building's own static geometry above, so a moving
        # occupant marker never causes the floor plan itself to rescale
        # or pan from frame to frame.
        self._render_selected_route(floor)
        self._render_occupants(floor)

    # =====================================================

    def _add_line(self, start_point, end_point, color, width):

        line_item = QGraphicsLineItem(start_point[0], start_point[1], end_point[0], end_point[1])
        line_item.setPen(QPen(color, width))
        self._scene.addItem(line_item)

    # =====================================================

    def _add_marker(self, position, color, size, rectangular=False, object_id=None):

        x, y = position

        if rectangular:
            item = QGraphicsRectItem(QRectF(x - size / 2, y - size / 2, size, size))
        else:
            item = QGraphicsEllipseItem(QRectF(x - size / 2, y - size / 2, size, size))

        item.setBrush(QBrush(color))
        item.setPen(QPen(QColor(20, 20, 20), 0.03))

        # Simulation Replay Studio V1 -- object_id (only ever an
        # occupant_id today) tags this item for eventFilter()'s own
        # click hit-test; None (every pre-existing caller) leaves the
        # item exactly as untagged as before.
        if object_id is not None:
            item.setData(0, object_id)

        self._scene.addItem(item)
        return item

    # =====================================================

    def _render_occupants(self, floor):

        # Simulation Replay Studio V1 -- one marker per occupant whose
        # interpolated position (IncidentFrame.occupant_positions,
        # already computed by IncidentData -- never recomputed here)
        # falls on the currently displayed floor. An "outside" position
        # (floor_id=None) or one with no resolvable (x, y) at all draws
        # nothing, rather than fabricating a location.

        if self._frame is None or floor is None:
            return

        for occupant_id, position in self._frame.occupant_positions.items():

            if position.floor_id != floor.id or position.x is None or position.y is None:
                continue

            is_selected = occupant_id == self._selected_occupant_id

            color = _SELECTED_OCCUPANT_COLOR if is_selected else _OCCUPANT_STATE_COLORS.get(
                position.state, _NO_DATA_COLOR,
            )
            size = _SELECTED_OCCUPANT_MARKER_SIZE if is_selected else _OCCUPANT_MARKER_SIZE

            item = self._add_marker((position.x, position.y), color, size, object_id=occupant_id)
            item.setToolTip(self._occupant_tooltip(occupant_id, position))

    # =====================================================

    def _occupant_tooltip(self, occupant_id, position):

        lines = [
            occupant_id,
            f"State: {position.state.replace('_', ' ').title()}",
        ]

        if position.zone_id is not None:
            lines.append(f"Zone: {self._object_name(position.zone_id)}")

        if position.current_stair_id is not None:
            lines.append(f"Stair: {self._object_name(position.current_stair_id)}")

        if position.speed is not None:
            lines.append(f"Speed: {position.speed:.2f} m/s")

        return "\n".join(lines)

    # =====================================================

    def _object_name(self, object_id):

        # BuildingView renders straight off Building/IncidentFrame and
        # has no id -> name lookup of its own (that already lives on
        # IncidentData, one layer up) -- falls back to the bare id, the
        # same honest "no name known" default every tooltip already uses
        # elsewhere in this widget.
        return object_id

    # =====================================================

    def _render_selected_route(self, floor):

        # Simulation Replay Studio V1 -- draws the selected occupant's
        # own actually-travelled route (OccupantRouteRecord.hops, already
        # recorded, never re-planned) as a polyline of already-completed
        # or in-progress hops, restricted to this floor (a Stair hop's
        # own two endpoints live in two different, non-comparable
        # floors' coordinate spaces -- see simulation_recording.
        # occupant_position's own docstring -- so it is skipped here
        # rather than drawn as a misleading straight line).

        if self._position_index is None or floor is None:
            return

        record = self._occupant_routes_by_id.get(self._selected_occupant_id)
        if record is None:
            return

        for hop in record.hops:

            if hop.edge_type == "Stair":
                continue

            from_position, from_floor_id = self._position_index.node_position(hop.from_node_id)
            to_position, _to_floor_id = self._position_index.node_position(
                hop.to_node_id, arriving_edge_id=hop.edge_id,
            )

            if from_position is None or to_position is None or from_floor_id != floor.id:
                continue

            self._add_line(from_position, to_position, _ROUTE_HIGHLIGHT_COLOR, 0.08)

    # =====================================================
    # Per-object coloring
    # =====================================================

    def _zone_color(self, zone, frame):

        mode = self._overlay_mode

        if mode == OVERLAY_HAZARD:
            return self._hazard_color(zone.id, frame)

        if mode == OVERLAY_OCCUPANCY:
            count = frame.zone_occupancy.get(zone.id) if frame else None
            return _occupancy_color(count, zone.max_occupancy)

        if mode == OVERLAY_CONGESTION:
            return _congestion_color(frame.current_congestion if frame else None)

        if mode == OVERLAY_RECOMMENDATION:
            return self._zone_recommendation_color(zone.id)

        return _NO_DATA_COLOR

    # =====================================================

    def _hazard_color(self, zone_id, frame):

        severity = _zone_hazard_severity(zone_id, frame)

        if severity is None:
            return _NO_DATA_COLOR

        return _SEVERITY_COLORS[severity]

    # =====================================================

    def _zone_action(self, zone_id):

        policy = self._decision_policy
        if policy is None:
            return None

        for entry in policy.zone_decisions:
            if entry["zone_id"] == zone_id:
                return entry["action"]

        return None

    # =====================================================

    def _zone_recommendation_color(self, zone_id):

        action = self._zone_action(zone_id)

        if action is None:
            return _NO_DATA_COLOR

        return _ZONE_ACTION_COLORS.get(action, _NO_DATA_COLOR)

    # =====================================================
    # Phase 2 -- hover tooltip. Every line is read straight off this
    # zone's own already-computed IncidentFrame/DecisionPolicy fields
    # (the same ones _zone_color()/_zone_action() already read for
    # coloring) -- no new per-zone signal is invented for the tooltip
    # that the overlay itself does not already have access to.
    # =====================================================

    def _zone_tooltip(self, zone, frame):

        lines = [zone.name]

        if frame is None:
            lines.append("No incident data loaded.")
            return "\n".join(lines)

        occupancy = frame.zone_occupancy.get(zone.id)
        lines.append(f"Occupancy: {occupancy if occupancy is not None else '-'}")

        smoke = frame.zone_smoke.get(zone.id)
        lines.append(f"Smoke: {smoke:.2f}" if smoke is not None else "Smoke: -")

        on_fire = zone.id == frame.ignition_zone_id or zone.id in frame.current_fire_zone_ids
        lines.append(f"Fire: {'Yes' if on_fire else 'No'}")

        severity = _zone_hazard_severity(zone.id, frame)
        lines.append(f"Hazard intensity: {severity.name.title() if severity else '-'}")

        action = self._zone_action(zone.id)
        lines.append(f"Recommendation: {action.replace('_', ' ').title() if action else '-'}")

        people = [
            observation for observation in frame.human_observations.values()
            if observation.zone_id == zone.id
        ]
        classification_counts = Counter(observation.classification for observation in people)

        # POSSIBLE_INJURY has no observationally distinct HumanState of
        # its own -- command_center.incident_data's own human
        # observation mapping already collapses it into FALLEN (a
        # possible injury always implies a fall already happened), so
        # this tooltip reports the same single, honest count rather
        # than fabricating two independently-tracked numbers.
        fallen_count = sum(1 for observation in people if observation.state == HumanState.FALLEN)
        helping_count = sum(
            1 for observation in people
            if BehaviorEvent.HELPING_ANOTHER_PERSON in observation.behavior_events
        )

        lines.append(f"Wheelchair users: {classification_counts.get(HumanClassification.WHEELCHAIR_USER, 0)}")
        lines.append(f"Children: {classification_counts.get(HumanClassification.CHILD, 0)}")
        lines.append(f"Elderly: {classification_counts.get(HumanClassification.ELDERLY, 0)}")
        lines.append(f"Fallen / possible injury: {fallen_count}")
        lines.append(f"Helping groups: {helping_count}")
        lines.append(f"Firefighters present: {classification_counts.get(HumanClassification.FIREFIGHTER, 0)}")

        return "\n".join(lines)

    # =====================================================

    def _door_color(self, door, frame):

        state = frame.door_states.get(door.id) if frame else None

        if state == "LOCKED":
            return _CRITICAL_COLOR
        if state == "CLOSED":
            return _INACTIVE_COLOR
        if state == "OPEN":
            return _SAFE_COLOR

        if door.locked:
            return _CRITICAL_COLOR
        if not door.active:
            return _INACTIVE_COLOR

        return _SAFE_COLOR

    # =====================================================

    def _exit_color(self, exit_obj, frame):

        if self._overlay_mode == OVERLAY_RECOMMENDATION and self._decision_policy is not None:

            for entry in self._decision_policy.exit_decisions:
                if entry["exit_id"] == exit_obj.id:
                    return _EXIT_STATUS_COLORS.get(entry["status"], _NO_DATA_COLOR)

        state = frame.exit_states.get(exit_obj.id) if frame else None

        if state == "OPEN":
            return _SAFE_COLOR
        if state == "CLOSED":
            return _INACTIVE_COLOR

        return _INACTIVE_COLOR if exit_obj.is_blocked else _SAFE_COLOR

    # =====================================================

    def _stair_color(self, stair, frame):

        if self._overlay_mode == OVERLAY_RECOMMENDATION and self._decision_policy is not None:

            for entry in self._decision_policy.stair_decisions:
                if entry["stair_id"] == stair.id:
                    return _STAIR_STATUS_COLORS.get(entry["status"], _NO_DATA_COLOR)

        state = frame.stair_states.get(stair.id) if frame else None

        if state == "CLOSED":
            return _INACTIVE_COLOR

        return _SAFE_COLOR

    # =====================================================

    def _device_color(self, object_id, active_baseline, frame, kind):

        if frame is None:
            return _DEVICE_COLOR if active_baseline else _INACTIVE_COLOR

        states_map = frame.camera_states if kind == "camera" else frame.detector_states
        state = states_map.get(object_id)

        if state == "FAILED":
            return _INACTIVE_COLOR
        if state == "AVAILABLE":
            return _DEVICE_COLOR

        return _DEVICE_COLOR if active_baseline else _INACTIVE_COLOR
