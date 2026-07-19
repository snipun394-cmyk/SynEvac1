from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from perception.fusion.occupancy_estimation import EstimatedOccupancy
from perception.models.building_observation import ObservationState, ObservedNodeState, ObservedOccupancy

from designer.perception_debug_runner import PerceptionDebugRunner


_NOT_EXPOSED = "N/A (not exposed by this reading type)"

_STATUS_OK = QColor("#2e7d32")
_STATUS_WARN = QColor("#f9a825")
_STATUS_BAD = QColor("#c62828")
_STATUS_NEUTRAL = QColor("#455a64")


class PerceptionDebugPanel(QWidget):

    # Verification/visualization-only -- this widget never calls into
    # any perception algorithm directly. It owns one PerceptionDebugRunner
    # (Designer glue, see designer/perception_debug_runner.py) and only
    # ever reads what that runner already computed by driving the
    # unmodified Phase 1-4 classes. Follows this codebase's established
    # "dumb widget, MainWindow pushes updates in" convention (see
    # BottomInfoBar/PropertyPanel): the only public entry point is
    # refresh(building, sandbox_manager, time), called by MainWindow
    # exactly where it already calls self.property_panel.refresh().

    def __init__(self):

        super().__init__()

        self._runner = PerceptionDebugRunner()

        self._last_building = None
        self._last_sandbox_manager = None
        self._last_time = 0.0
        self._last_snapshot = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_ground_truth_tab()
        self._build_camera_tab()
        self._build_smoke_tab()
        self._build_heat_tab()
        self._build_occupancy_estimation_tab()
        self._build_sensor_fusion_tab()
        self._build_differences_tab()

    # =====================================================
    # Tab construction
    # =====================================================

    def _build_ground_truth_tab(self):

        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        note = QLabel(
            "Simulator-only information, for comparison against Perception's output "
            "below. Never reaches the Rule-Based Engine or a future RL Agent -- this "
            "tab exists purely so a developer can see what Perception *should* be "
            "limited to.\n"
            "Note: Hazard Score/Severity reflect HazardNodeState.hazard_score only, "
            "which -- consistent with every HazardSource in this codebase (fire_growth, "
            "smoke_propagation) -- is never derived from temperature. A zone with only a "
            "heat reading set below can correctly show Severity = NONE here even though "
            "its Temperature column is non-zero; that is not a bug in this panel."
        )
        note.setWordWrap(True)
        container_layout.addWidget(note)

        self.ground_truth_table = self._make_table(
            ["Zone", "Floor", "Occupants (actual)", "Smoke Level", "Temperature", "Hazard Score", "Severity"],
        )
        container_layout.addWidget(self.ground_truth_table)

        container_layout.addWidget(self._build_hazard_injector_box())

        self.tabs.addTab(container, "1. Ground Truth")

    # =====================================================

    def _build_hazard_injector_box(self) -> QGroupBox:

        # A manual, hand-set hazard override -- no time evolution, no
        # physics, exactly ManualHazardProvider's own role. This exists
        # only because no fire/smoke simulation is wired into the
        # Designer UI at all yet (see PerceptionDebugRunner's module
        # docstring); it is explicitly not a Scenario Engine.

        box = QGroupBox("Inject Test Hazard (debug only -- no fire/smoke physics)")
        box_layout = QHBoxLayout()
        box.setLayout(box_layout)

        self.hazard_zone_combo = QComboBox()
        box_layout.addWidget(QLabel("Zone:"))
        box_layout.addWidget(self.hazard_zone_combo)

        self.smoke_enable_checkbox = QCheckBox("Set Smoke")
        box_layout.addWidget(self.smoke_enable_checkbox)

        self.smoke_level_spin = QDoubleSpinBox()
        self.smoke_level_spin.setRange(0.0, 1.0)
        self.smoke_level_spin.setSingleStep(0.05)
        box_layout.addWidget(self.smoke_level_spin)

        self.temperature_enable_checkbox = QCheckBox("Set Temperature")
        box_layout.addWidget(self.temperature_enable_checkbox)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 500.0)
        self.temperature_spin.setSingleStep(5.0)
        self.temperature_spin.setValue(20.0)
        box_layout.addWidget(self.temperature_spin)

        apply_button = QPushButton("Apply to Zone")
        apply_button.clicked.connect(self._on_apply_hazard)
        box_layout.addWidget(apply_button)

        clear_zone_button = QPushButton("Clear Zone")
        clear_zone_button.clicked.connect(self._on_clear_zone_hazard)
        box_layout.addWidget(clear_zone_button)

        clear_all_button = QPushButton("Clear All")
        clear_all_button.clicked.connect(self._on_clear_all_hazard)
        box_layout.addWidget(clear_all_button)

        return box

    # =====================================================

    def _build_camera_tab(self):

        self.camera_table = self._make_table(
            ["Camera ID", "Covered Zone(s)", "Estimated Occupants", "Confidence", "Visibility Category"],
        )
        self.tabs.addTab(self._wrap(self.camera_table), "2. Camera Observations")

    # =====================================================

    def _build_smoke_tab(self):

        self.smoke_table = self._make_table(
            ["Detector ID", "Zone", "Alarm State", "Smoke Detected", "Estimated Smoke Level", "Confidence"],
        )
        self.tabs.addTab(self._wrap(self.smoke_table), "3. Smoke Detector Observations")

    # =====================================================

    def _build_heat_tab(self):

        self.heat_table = self._make_table(
            ["Detector ID", "Zone", "Temperature", "Alarm State", "Confidence"],
        )
        self.tabs.addTab(self._wrap(self.heat_table), "4. Heat Detector Observations")

    # =====================================================

    def _build_occupancy_estimation_tab(self):

        self.occupancy_estimation_table = self._make_table(
            ["Zone", "Estimated Occupancy", "Confidence", "Observation State", "Cameras Contributing"],
        )
        self.tabs.addTab(self._wrap(self.occupancy_estimation_table), "5. Occupancy Estimation")

    # =====================================================

    def _build_sensor_fusion_tab(self):

        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        container_layout.addWidget(QLabel("Node Observations (hazard side)"))
        self.fusion_node_table = self._make_table(
            ["Zone", "Observation State", "Alarm Active", "Alarm Sources", "Visibility", "Severity", "Last Observed"],
        )
        container_layout.addWidget(self.fusion_node_table)

        container_layout.addWidget(QLabel("Occupancy Observations"))
        self.fusion_occupancy_table = self._make_table(["Zone", "Estimated Count", "Confidence"])
        container_layout.addWidget(self.fusion_occupancy_table)

        container_layout.addWidget(QLabel("Edge Observations"))
        self.fusion_edge_table = self._make_table(["Edge", "Blocked Estimate"])
        container_layout.addWidget(self.fusion_edge_table)

        container_layout.addWidget(QLabel("System Status"))
        self.fusion_status_table = self._make_table(
            ["Panel Communication OK", "Active Cameras", "Active Detectors"],
        )
        self.fusion_status_table.setMaximumHeight(60)
        container_layout.addWidget(self.fusion_status_table)

        self.tabs.addTab(container, "6. Sensor Fusion")

    # =====================================================

    def _build_differences_tab(self):

        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        legend = QLabel(
            "Green = consistent. Blue = Perception correctly reports UNKNOWN where it has no "
            "coverage (expected, not a bug). Amber = expected estimation limitation (e.g. below "
            "detector threshold, or a duplicate-resolution/multi-zone-camera artifact). "
            "Red = Perception disagrees with Ground Truth in a way that should not happen."
        )
        legend.setWordWrap(True)
        container_layout.addWidget(legend)

        self.differences_table = self._make_table(
            [
                "Zone", "GT Occupants", "Perception Occupancy", "Occupancy Status",
                "GT Hazard", "Perception Hazard", "Hazard Status",
            ],
        )
        container_layout.addWidget(self.differences_table)

        self.tabs.addTab(container, "7. Differences (Validation)")

    # =====================================================
    # Helpers
    # =====================================================

    def _make_table(self, headers) -> QTableWidget:

        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setRowCount(0)

        return table

    # =====================================================

    def _wrap(self, table) -> QWidget:

        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)
        container_layout.addWidget(table)

        return container

    # =====================================================

    def _set_row(self, table, row, values, color=None):

        for column, value in enumerate(values):

            item = QTableWidgetItem(str(value))

            if color is not None:
                item.setForeground(color)

            table.setItem(row, column, item)

    # =====================================================
    # Public entry point -- called by MainWindow after every
    # simulation tick/step/reset, exactly where it already calls
    # self.property_panel.refresh().
    # =====================================================

    def refresh(self, building, sandbox_manager, time: float):

        self._last_building = building
        self._last_sandbox_manager = sandbox_manager
        self._last_time = time

        if building is None:
            return

        snapshot = self._runner.run(building, sandbox_manager, time)
        self._last_snapshot = snapshot

        self._refresh_zone_combo(snapshot)
        self._refresh_ground_truth_table(snapshot)
        self._refresh_camera_table(snapshot)
        self._refresh_smoke_table(snapshot)
        self._refresh_heat_table(snapshot)
        self._refresh_occupancy_estimation_table(snapshot)
        self._refresh_sensor_fusion_tables(snapshot)
        self._refresh_differences_table(snapshot)

    # =====================================================

    def _refresh_zone_combo(self, snapshot):

        current_zone_id = self.hazard_zone_combo.currentData()

        self.hazard_zone_combo.blockSignals(True)
        self.hazard_zone_combo.clear()

        for zone_id, name in sorted(snapshot.zone_names.items(), key=lambda pair: pair[1]):
            self.hazard_zone_combo.addItem(name, zone_id)

        if current_zone_id is not None:

            index = self.hazard_zone_combo.findData(current_zone_id)

            if index >= 0:
                self.hazard_zone_combo.setCurrentIndex(index)

        self.hazard_zone_combo.blockSignals(False)

    # =====================================================

    def _refresh_ground_truth_table(self, snapshot):

        readings = sorted(snapshot.ground_truth.values(), key=lambda reading: reading.zone_name)

        self.ground_truth_table.setRowCount(len(readings))

        for row, reading in enumerate(readings):

            self._set_row(
                self.ground_truth_table, row,
                [
                    reading.zone_name,
                    reading.floor_name,
                    f"{reading.occupant_count:g}",
                    "-" if reading.smoke_level is None else f"{reading.smoke_level:.2f}",
                    "-" if reading.temperature is None else f"{reading.temperature:.1f}",
                    f"{reading.hazard_score:.2f}",
                    reading.severity.name,
                ],
            )

    # =====================================================

    def _refresh_camera_table(self, snapshot):

        self.camera_table.setRowCount(len(snapshot.camera_observations))

        for row, observation in enumerate(snapshot.camera_observations):

            covered_zone_ids = snapshot.camera_covered_zone_ids.get(observation.camera_id, [])
            covered_names = ", ".join(
                snapshot.zone_names.get(zone_id, zone_id) for zone_id in covered_zone_ids
            ) or "(none)"

            self._set_row(
                self.camera_table, row,
                [
                    observation.camera_id,
                    covered_names,
                    "UNKNOWN" if observation.estimated_occupant_count is None
                    else f"{observation.estimated_occupant_count:g}",
                    "-" if observation.confidence is None else f"{observation.confidence:.2f}",
                    observation.visibility_estimate or _NOT_EXPOSED,
                ],
            )

    # =====================================================

    def _refresh_smoke_table(self, snapshot):

        self.smoke_table.setRowCount(len(snapshot.smoke_detector_readings))

        for row, reading in enumerate(snapshot.smoke_detector_readings):

            zone_id = snapshot.smoke_detector_zone_id.get(reading.detector_id)
            zone_name = snapshot.zone_names.get(zone_id, "-") if zone_id else "-"

            self._set_row(
                self.smoke_table, row,
                [
                    reading.detector_id,
                    zone_name,
                    "ALARM" if reading.alarm_active else "clear",
                    "yes" if reading.alarm_active else "no",
                    _NOT_EXPOSED,
                    "-" if reading.confidence is None else f"{reading.confidence:.2f}",
                ],
            )

    # =====================================================

    def _refresh_heat_table(self, snapshot):

        self.heat_table.setRowCount(len(snapshot.heat_detector_readings))

        for row, reading in enumerate(snapshot.heat_detector_readings):

            zone_id = snapshot.heat_detector_zone_id.get(reading.detector_id)
            zone_name = snapshot.zone_names.get(zone_id, "-") if zone_id else "-"

            self._set_row(
                self.heat_table, row,
                [
                    reading.detector_id,
                    zone_name,
                    _NOT_EXPOSED,
                    "ALARM" if reading.alarm_active else "clear",
                    "-" if reading.confidence is None else f"{reading.confidence:.2f}",
                ],
            )

    # =====================================================

    def _refresh_occupancy_estimation_table(self, snapshot):

        # Iterates every zone in the Building, not just
        # snapshot.occupancy_estimates's own keys -- that mapping is
        # deliberately sparse (OccupancyEstimator.estimate() only
        # returns entries for zones its own camera topology knows
        # about, see perception/fusion/occupancy_estimation.py), which
        # would otherwise make a zone with zero camera coverage
        # silently *absent* from this table rather than visibly marked
        # UNOBSERVED. For a validation tool whose whole purpose is
        # confirming UNKNOWN propagates correctly, an absent row is the
        # wrong way to show that -- a developer needs to positively see
        # "yes, this zone reads UNOBSERVED", not infer it from a zone
        # missing off the list entirely.

        all_zone_ids = sorted(snapshot.zone_names.keys(), key=lambda zone_id: snapshot.zone_names[zone_id])

        self.occupancy_estimation_table.setRowCount(len(all_zone_ids))

        for row, zone_id in enumerate(all_zone_ids):

            estimate = snapshot.occupancy_estimates.get(zone_id, EstimatedOccupancy())

            self._set_row(
                self.occupancy_estimation_table, row,
                [
                    snapshot.zone_names.get(zone_id, zone_id),
                    "UNKNOWN" if estimate.estimated_count is None else f"{estimate.estimated_count:g}",
                    "-" if estimate.confidence is None else f"{estimate.confidence:.2f}",
                    estimate.observation_state.name,
                    ", ".join(estimate.contributing_camera_ids) or "(none)",
                ],
            )

    # =====================================================

    def _refresh_sensor_fusion_tables(self, snapshot):

        # Same reasoning as _refresh_occupancy_estimation_table above:
        # BuildingObservation.node_observations/occupancy_observations
        # are deliberately sparse (only touched zones get an entry --
        # see perception/models/building_observation.py), so this
        # iterates every zone in the Building and falls back to each
        # type's own documented UNOBSERVED/no-reading default
        # (ObservedNodeState()/ObservedOccupancy()) rather than letting
        # an untouched zone go missing from the table entirely.

        observation = snapshot.building_observation

        all_zone_ids = sorted(snapshot.zone_names.keys(), key=lambda zone_id: snapshot.zone_names[zone_id])

        self.fusion_node_table.setRowCount(len(all_zone_ids))

        for row, zone_id in enumerate(all_zone_ids):

            node = observation.node_observations.get(zone_id, ObservedNodeState())

            self._set_row(
                self.fusion_node_table, row,
                [
                    snapshot.zone_names.get(zone_id, zone_id),
                    node.observation_state.name,
                    "-" if node.alarm_active is None else ("yes" if node.alarm_active else "no"),
                    ", ".join(node.alarm_source_types) or "(none)",
                    node.visibility_estimate or "-",
                    node.estimated_severity.name if node.estimated_severity else "-",
                    "-" if node.last_observed_time is None else f"{node.last_observed_time:.1f}",
                ],
            )

        self.fusion_occupancy_table.setRowCount(len(all_zone_ids))

        for row, zone_id in enumerate(all_zone_ids):

            occupancy = observation.occupancy_observations.get(zone_id, ObservedOccupancy())

            self._set_row(
                self.fusion_occupancy_table, row,
                [
                    snapshot.zone_names.get(zone_id, zone_id),
                    "-" if occupancy.estimated_count is None else f"{occupancy.estimated_count:g}",
                    "-" if occupancy.confidence is None else f"{occupancy.confidence:.2f}",
                ],
            )

        edge_items = list(observation.edge_observations.items())
        self.fusion_edge_table.setRowCount(len(edge_items))

        for row, (edge_id, edge) in enumerate(edge_items):

            self._set_row(
                self.fusion_edge_table, row,
                [
                    edge_id,
                    "-" if edge.blocked_estimate is None
                    else ("BLOCKED" if edge.blocked_estimate else "clear"),
                ],
            )

        status = observation.system_status
        self.fusion_status_table.setRowCount(1)
        self._set_row(
            self.fusion_status_table, 0,
            [
                "yes" if status.panel_communication_ok else "no",
                status.active_camera_count,
                status.active_detector_count,
            ],
        )

    # =====================================================

    def _refresh_differences_table(self, snapshot):

        zone_ids = sorted(snapshot.ground_truth.keys(), key=lambda zone_id: snapshot.zone_names.get(zone_id, zone_id))

        self.differences_table.setRowCount(len(zone_ids))

        for row, zone_id in enumerate(zone_ids):

            ground_truth = snapshot.ground_truth[zone_id]
            occupancy = snapshot.building_observation.occupancy_observation(zone_id)
            node = snapshot.building_observation.node_observation(zone_id)

            occupancy_status, occupancy_color = self._occupancy_status(ground_truth, occupancy)
            hazard_status, hazard_color = self._hazard_status(ground_truth, node)

            row_color = _STATUS_BAD if occupancy_color == _STATUS_BAD or hazard_color == _STATUS_BAD else None

            self._set_row(
                self.differences_table, row,
                [
                    ground_truth.zone_name,
                    f"{ground_truth.occupant_count:g}",
                    "UNKNOWN" if occupancy.estimated_count is None else f"{occupancy.estimated_count:g}",
                    occupancy_status,
                    f"{ground_truth.severity.name} (score={ground_truth.hazard_score:.2f})",
                    "UNOBSERVED" if node.observation_state == ObservationState.UNOBSERVED
                    else (node.estimated_severity.name if node.estimated_severity else "-"),
                    hazard_status,
                ],
                color=row_color,
            )

    # =====================================================

    def _occupancy_status(self, ground_truth, occupancy):

        if occupancy.estimated_count is None:

            if ground_truth.occupant_count > 0:
                return "UNKNOWN (no camera coverage -- expected)", _STATUS_NEUTRAL

            return "UNKNOWN (none present anyway)", _STATUS_NEUTRAL

        if occupancy.estimated_count == ground_truth.occupant_count:
            return "MATCH", _STATUS_OK

        if occupancy.estimated_count < ground_truth.occupant_count:
            # Never expected from the current GroundTruthCameraProvider/
            # OccupancyEstimator pass-through design (no noise model) --
            # a real mismatch here means either a multi-zone-camera
            # routing gap or a genuine bug. See the panel's own reported
            # findings.
            return "UNDER-COUNT (investigate)", _STATUS_BAD

        return "OVER-COUNT (investigate)", _STATUS_BAD

    # =====================================================

    def _hazard_status(self, ground_truth, node):

        has_ground_truth_smoke_or_heat = (
            (ground_truth.smoke_level or 0.0) > 0.0 or (ground_truth.temperature or 0.0) > 0.0
        )

        if node.observation_state == ObservationState.UNOBSERVED:

            if has_ground_truth_smoke_or_heat:
                return "UNKNOWN (no detector coverage -- expected)", _STATUS_NEUTRAL

            return "UNKNOWN (nothing to detect anyway)", _STATUS_NEUTRAL

        if node.alarm_active and not has_ground_truth_smoke_or_heat:
            return "FALSE ALARM (investigate)", _STATUS_BAD

        if not node.alarm_active and has_ground_truth_smoke_or_heat:
            return "BELOW THRESHOLD (expected -- detectors are threshold devices)", _STATUS_WARN

        if node.alarm_active and has_ground_truth_smoke_or_heat:
            return "CONSISTENT (alarm confirmed)", _STATUS_OK

        return "CONSISTENT (clear)", _STATUS_OK

    # =====================================================
    # Hazard injector handlers
    # =====================================================

    def _on_apply_hazard(self):

        zone_id = self.hazard_zone_combo.currentData()

        if zone_id is None:
            return

        smoke_level = self.smoke_level_spin.value() if self.smoke_enable_checkbox.isChecked() else None
        temperature = self.temperature_spin.value() if self.temperature_enable_checkbox.isChecked() else None

        self._runner.set_zone_hazard(zone_id, smoke_level, temperature)

        self._rerun_last()

    # =====================================================

    def _on_clear_zone_hazard(self):

        zone_id = self.hazard_zone_combo.currentData()

        if zone_id is None:
            return

        self._runner.clear_zone_hazard(zone_id)

        self._rerun_last()

    # =====================================================

    def _on_clear_all_hazard(self):

        self._runner.clear_all_hazard_overrides()

        self._rerun_last()

    # =====================================================

    def _rerun_last(self):

        if self._last_building is None:
            return

        self.refresh(self._last_building, self._last_sandbox_manager, self._last_time)
