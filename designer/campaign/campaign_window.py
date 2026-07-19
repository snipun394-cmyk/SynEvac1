from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scenario_definition import (
    DeviceAvailability,
    DoorState,
    EngineeringConstraints,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    PresenceState,
    ScenarioDefinition,
    StairAvailability,
    UniformRange,
    WeightedOptions,
)

from designer.campaign.engineering_category_editor import EngineeringCategoryEditor
from designer.campaign.population_profile_editor import PopulationProfileEditor


# The graphical orchestration layer's own UI, and only its UI --
# docs handed down for this phase: "Every option should simply
# configure the existing ScenarioDefinition." build_scenario_definition()
# below is the one place that translates widget values into
# ScenarioDefinition/Distribution/FireDefinition/OccupantDefinition/
# EngineeringConstraints construction calls -- every one of those
# types is imported, never redefined, and this method sample()s
# nothing itself (no random import anywhere in this file) -- it only
# ever builds the *rule*, exactly as scenario_definition's own
# "declares what may be sampled, samples nothing" ethos requires of
# any caller constructing one.
#
# V1 scope, stated plainly rather than left implicit: every engineering
# rule still applies uniformly across every matching object on the
# Building (every door the same door rule, ...), and occupancy count is
# still one rule for every zone -- not per-object-instance authoring.
# ScenarioDefinition's own Mapping[id, Distribution] shape already
# supports full per-object granularity (scenario_generator reads it
# that way); this window simply does not expose that granularity yet
# for those two categories. A future revision could add a per-object
# table for either without any backend change at all -- occupant
# behaviour profile is the one category that already got exactly that
# treatment (see PopulationProfileEditor / population_profile_editor.py):
# each zone carries its own independently-configured weighted mix,
# translated into a real WeightedOptions per zone in
# build_scenario_definition() below -- not a single building-wide rule.
#
# Dataset Intent: only "Custom" is wired to anything -- the other five
# named intents (docs/architecture/dataset_intent.md) are shown,
# disabled, with an explanatory tooltip, the same "future fields
# present but disabled" convention designer/widgets/
# occupant_generation_dialog.py already established for its own
# not-yet-implemented fields. dataset_intent/ has no code anywhere in
# this repository (architecture-only, and explicitly not in this
# phase's list of production-ready packages) -- wiring a real
# selection to it would mean fabricating a backend this phase was
# told not to build.


DATASET_INTENTS = (
    "Custom",
    "Standard Evacuation",
    "Mixed Realistic Incidents",
    "Rescue Training",
    "Firefighter Training",
    "RL Training",
)

DOOR_STATE_NAMES = tuple(member.name for member in DoorState)
STAIR_STATE_NAMES = tuple(member.name for member in StairAvailability)
OBSTACLE_STATE_NAMES = tuple(member.name for member in PresenceState)
DEVICE_STATE_NAMES = tuple(member.name for member in DeviceAvailability)

# Exits are bool-shaped in EngineeringConstraints (exit_state_distribution
# values are True/False, "is open"), not backed by an enum -- these two
# labels are the Campaign Studio-only vocabulary mapped to that
# boolean, same as the pre-existing exit combo already used.
EXIT_STATE_NAMES = ("OPEN", "CLOSED")


class CampaignWindow(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Scenario Campaign Studio")
        self.resize(1100, 750)

        self._zone_ids = []
        self._floor_ids = []
        self._door_ids = []
        self._exit_ids = []
        self._stair_ids = []
        self._obstacle_ids = []
        self._camera_ids = []
        self._detector_ids = []

        root_layout = QHBoxLayout()
        self.setLayout(root_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_execution_panel())

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        self.set_running_state(False)

    # =====================================================
    # Config panel -- Campaign + Scenario Definition, scrollable
    # =====================================================

    def _build_config_panel(self):

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        layout.addWidget(self._build_campaign_group())
        layout.addWidget(self._build_occupants_group())

        self.population_profile_editor = PopulationProfileEditor()
        layout.addWidget(self.population_profile_editor)

        layout.addWidget(self._build_fire_group())
        layout.addWidget(self._build_engineering_group())
        layout.addWidget(self._build_engineering_summary_group())
        layout.addStretch(1)

        scroll.setWidget(container)

        self._refresh_engineering_summary()

        return scroll

    # =====================================================

    def _build_campaign_group(self):

        group = QGroupBox("Campaign")
        form = QFormLayout()
        group.setLayout(form)

        self.name_edit = QLineEdit("Untitled Campaign")
        form.addRow("Campaign Name", self.name_edit)

        self.dataset_intent_combo = QComboBox()

        for name in DATASET_INTENTS:
            self.dataset_intent_combo.addItem(name)

        model = self.dataset_intent_combo.model()

        for index in range(1, len(DATASET_INTENTS)):

            item = model.item(index)
            item.setEnabled(False)
            item.setToolTip(
                "Requires the dataset_intent package (architecture-only, "
                "not yet implemented). Use Custom and configure Scenario "
                "Definition directly below."
            )

        self.dataset_intent_combo.setCurrentIndex(0)
        form.addRow("Dataset Intent", self.dataset_intent_combo)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 1_000_000)
        self.count_spin.setValue(100)
        form.addRow("Number of Scenarios", self.count_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)
        form.addRow("Master Random Seed", self.seed_spin)

        self.max_attempts_spin = QSpinBox()
        self.max_attempts_spin.setRange(1, 1000)
        self.max_attempts_spin.setValue(10)
        form.addRow("Maximum Generation Attempts", self.max_attempts_spin)

        self.diagnostics_checkbox = QCheckBox(
            "Enable Diagnostics Mode (capture every rejection reason)"
        )
        self.diagnostics_checkbox.setChecked(True)
        self.diagnostics_checkbox.setToolTip(
            "Captures every ScenarioValidationReport during generation and "
            "shows rejection reasons in the Validation Report panel. "
            "Negligible overhead -- recommended, especially the first time "
            "you run a new Scenario Definition against a Building."
        )
        form.addRow(self.diagnostics_checkbox)

        output_row = QHBoxLayout()
        self.output_directory_edit = QLineEdit("")
        self.browse_output_button = QPushButton("Browse...")
        self.browse_output_button.clicked.connect(self._on_browse_output_directory)
        output_row.addWidget(self.output_directory_edit)
        output_row.addWidget(self.browse_output_button)
        form.addRow("Output Directory", output_row)

        return group

    # =====================================================

    def _build_occupants_group(self):

        group = QGroupBox("Scenario Definition — Occupants")
        form = QFormLayout()
        group.setLayout(form)

        self.occupancy_mode_combo = QComboBox()
        self.occupancy_mode_combo.addItems(["Fixed", "Range"])
        self.occupancy_mode_combo.currentTextChanged.connect(self._on_occupancy_mode_changed)
        form.addRow("Occupancy Mode", self.occupancy_mode_combo)

        self.occupancy_fixed_spin = QSpinBox()
        self.occupancy_fixed_spin.setRange(0, 10_000)
        self.occupancy_fixed_spin.setValue(4)
        form.addRow("Fixed Occupant Count (per zone)", self.occupancy_fixed_spin)

        self.occupancy_min_spin = QSpinBox()
        self.occupancy_min_spin.setRange(0, 10_000)
        self.occupancy_min_spin.setValue(1)
        self.occupancy_min_spin.setEnabled(False)
        form.addRow("Minimum (per zone)", self.occupancy_min_spin)

        self.occupancy_max_spin = QSpinBox()
        self.occupancy_max_spin.setRange(0, 10_000)
        self.occupancy_max_spin.setValue(8)
        self.occupancy_max_spin.setEnabled(False)
        form.addRow("Maximum (per zone)", self.occupancy_max_spin)

        return group

    # =====================================================

    def _on_occupancy_mode_changed(self, mode):

        is_fixed = mode == "Fixed"

        self.occupancy_fixed_spin.setEnabled(is_fixed)
        self.occupancy_min_spin.setEnabled(not is_fixed)
        self.occupancy_max_spin.setEnabled(not is_fixed)

    # =====================================================

    def _build_fire_group(self):

        group = QGroupBox("Scenario Definition — Fire")
        layout = QVBoxLayout()
        group.setLayout(layout)

        form = QFormLayout()
        layout.addLayout(form)

        self.fire_profile_combo = QComboBox()
        self.fire_profile_combo.setEditable(True)
        self.fire_profile_combo.addItems(["Electrical", "Flaming", "Smoldering"])
        form.addRow("Fire Profile", self.fire_profile_combo)

        self.growth_mode_combo = QComboBox()
        self.growth_mode_combo.addItems(["Fixed", "Range"])
        self.growth_mode_combo.currentTextChanged.connect(self._on_growth_mode_changed)
        form.addRow("Growth Time Mode", self.growth_mode_combo)

        self.growth_fixed_spin = QDoubleSpinBox()
        self.growth_fixed_spin.setRange(1.0, 100_000.0)
        self.growth_fixed_spin.setValue(200.0)
        self.growth_fixed_spin.setSuffix(" s")
        form.addRow("Fixed Growth Time", self.growth_fixed_spin)

        self.growth_min_spin = QDoubleSpinBox()
        self.growth_min_spin.setRange(1.0, 100_000.0)
        self.growth_min_spin.setValue(100.0)
        self.growth_min_spin.setSuffix(" s")
        self.growth_min_spin.setEnabled(False)
        form.addRow("Minimum Growth Time", self.growth_min_spin)

        self.growth_max_spin = QDoubleSpinBox()
        self.growth_max_spin.setRange(1.0, 100_000.0)
        self.growth_max_spin.setValue(400.0)
        self.growth_max_spin.setSuffix(" s")
        self.growth_max_spin.setEnabled(False)
        form.addRow("Maximum Growth Time", self.growth_max_spin)

        layout.addWidget(QLabel("Allowed Ignition Zones (none checked = any zone)"))
        self.fire_allowed_zones_list = QListWidget()
        self.fire_allowed_zones_list.setMaximumHeight(90)
        layout.addWidget(self.fire_allowed_zones_list)

        layout.addWidget(QLabel("Forbidden Ignition Zones"))
        self.fire_forbidden_zones_list = QListWidget()
        self.fire_forbidden_zones_list.setMaximumHeight(90)
        layout.addWidget(self.fire_forbidden_zones_list)

        layout.addWidget(QLabel("Allowed Ignition Floors (none checked = any floor)"))
        self.fire_allowed_floors_list = QListWidget()
        self.fire_allowed_floors_list.setMaximumHeight(70)
        layout.addWidget(self.fire_allowed_floors_list)

        return group

    # =====================================================

    def _on_growth_mode_changed(self, mode):

        is_fixed = mode == "Fixed"

        self.growth_fixed_spin.setEnabled(is_fixed)
        self.growth_min_spin.setEnabled(not is_fixed)
        self.growth_max_spin.setEnabled(not is_fixed)

    # =====================================================

    def _build_engineering_group(self):

        group = QGroupBox("Scenario Definition — Engineering Objects")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.door_editor = EngineeringCategoryEditor("Doors", DOOR_STATE_NAMES)
        layout.addWidget(self.door_editor)

        self.exit_editor = EngineeringCategoryEditor("Exits", EXIT_STATE_NAMES)
        layout.addWidget(self.exit_editor)

        min_open_form = QFormLayout()
        self.min_open_exits_spin = QSpinBox()
        self.min_open_exits_spin.setRange(0, 1000)
        self.min_open_exits_spin.setValue(0)
        min_open_form.addRow("Minimum Open Exits", self.min_open_exits_spin)
        layout.addLayout(min_open_form)

        self.stair_editor = EngineeringCategoryEditor("Stairs", STAIR_STATE_NAMES)
        layout.addWidget(self.stair_editor)

        self.obstacle_editor = EngineeringCategoryEditor("Obstacles", OBSTACLE_STATE_NAMES)
        layout.addWidget(self.obstacle_editor)

        self.camera_editor = EngineeringCategoryEditor("Cameras", DEVICE_STATE_NAMES)
        layout.addWidget(self.camera_editor)

        self.detector_editor = EngineeringCategoryEditor("Detectors", DEVICE_STATE_NAMES)
        layout.addWidget(self.detector_editor)

        for _, editor in self._engineering_editors():
            editor.changed.connect(self._refresh_engineering_summary)

        self.min_open_exits_spin.valueChanged.connect(self._refresh_engineering_summary)

        return group

    # =====================================================

    def _build_engineering_summary_group(self):

        # Requirement: "Before starting generation, show a concise
        # summary ... This lets the user verify campaign settings
        # before generating thousands of scenarios." Kept always
        # visible and always current (refreshed on every editor
        # change) rather than a dialog the user must remember to open.

        group = QGroupBox("Campaign Summary — Engineering Objects")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.engineering_summary_label = QLabel()
        self.engineering_summary_label.setWordWrap(True)
        layout.addWidget(self.engineering_summary_label)

        self.engineering_variation_warning_label = QLabel()
        self.engineering_variation_warning_label.setWordWrap(True)
        self.engineering_variation_warning_label.setStyleSheet(
            "color: #a86b00; font-weight: bold;"
        )
        self.engineering_variation_warning_label.setVisible(False)
        layout.addWidget(self.engineering_variation_warning_label)

        return group

    # =====================================================

    def _engineering_editors(self):

        return (
            ("Doors", self.door_editor),
            ("Exits", self.exit_editor),
            ("Stairs", self.stair_editor),
            ("Obstacles", self.obstacle_editor),
            ("Cameras", self.camera_editor),
            ("Detectors", self.detector_editor),
        )

    # =====================================================

    def all_engineering_defaults(self) -> bool:

        return all(
            editor.mode() == EngineeringCategoryEditor.MODE_BUILDING_DEFAULT
            for _, editor in self._engineering_editors()
        )

    # =====================================================

    def validate_engineering_setup(self) -> list:

        # Requirement: "The probabilities must total 100%." Returns a
        # list of human-readable messages (empty when everything is
        # valid) -- CampaignController.build_config() raises on a
        # non-empty list, the same "raise ValueError, window shows it"
        # convention already used for the no-Building/no-output-
        # directory checks.

        issues = []

        for label, editor in self._engineering_editors():

            if not editor.is_valid():

                issues.append(
                    f"{label}: Random Distribution percentages must total 100% "
                    f"(currently {editor.total_percentage()}%)."
                )

        return issues

    # =====================================================

    def validate_population_setup(self) -> list:

        # Same shape/placement as validate_engineering_setup() above --
        # CampaignController.build_config() checks both before starting
        # a campaign.

        return self.population_profile_editor.validation_messages()

    # =====================================================

    def _refresh_engineering_summary(self):

        lines = []

        for label, editor in self._engineering_editors():

            lines.append(f"{label}:")
            lines.extend(f"  {line}" for line in editor.summary_lines())

            if editor is self.exit_editor and self.min_open_exits_spin.value() > 0:
                lines.append(f"  Minimum Open Exits: {self.min_open_exits_spin.value()}")

        self.engineering_summary_label.setText("\n".join(lines))

        if self.all_engineering_defaults():

            self.engineering_variation_warning_label.setText(
                "This campaign will produce little or no engineering-state "
                "variation. This is acceptable for deterministic studies but "
                "may reduce dataset diversity for machine learning."
            )
            self.engineering_variation_warning_label.setVisible(True)

        else:

            self.engineering_variation_warning_label.setVisible(False)

    # =====================================================
    # Execution panel -- controls, live progress, results
    # =====================================================

    def _build_execution_panel(self):

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        button_row = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.cancel_button = QPushButton("Cancel")

        for button in (self.start_button, self.pause_button, self.resume_button, self.cancel_button):
            button_row.addWidget(button)

        layout.addLayout(button_row)

        layout.addWidget(self._build_progress_group())
        layout.addWidget(self._build_results_group())
        layout.addWidget(self._build_validation_report_group())
        layout.addStretch(1)

        scroll.setWidget(container)

        return scroll

    # =====================================================

    def _build_progress_group(self):

        group = QGroupBox("Live Progress")
        form = QFormLayout()
        group.setLayout(form)

        self.completed_label = QLabel("0")
        self.remaining_label = QLabel("0")
        self.rejected_label = QLabel("0")
        self.current_scenario_label = QLabel("-")
        self.average_simulation_time_label = QLabel("-")
        self.generation_speed_label = QLabel("-")
        self.eta_label = QLabel("-")
        self.elapsed_label = QLabel("-")

        form.addRow("Completed Scenarios", self.completed_label)
        form.addRow("Remaining", self.remaining_label)
        form.addRow("Rejected", self.rejected_label)
        form.addRow("Current Scenario", self.current_scenario_label)
        form.addRow("Average Simulation Time", self.average_simulation_time_label)
        form.addRow("Generation Speed", self.generation_speed_label)
        form.addRow("ETA", self.eta_label)
        form.addRow("Elapsed Time", self.elapsed_label)

        return group

    # =====================================================

    def _build_results_group(self):

        group = QGroupBox("Results")
        form = QFormLayout()
        group.setLayout(form)

        self.total_generated_label = QLabel("-")
        self.accepted_label = QLabel("-")
        self.rejected_total_label = QLabel("-")
        self.average_evacuation_time_label = QLabel("-")
        self.average_simulation_duration_label = QLabel("-")
        self.output_folder_label = QLabel("-")

        form.addRow("Total Generated", self.total_generated_label)
        form.addRow("Accepted", self.accepted_label)
        form.addRow("Rejected", self.rejected_total_label)
        form.addRow("Average Evacuation Time", self.average_evacuation_time_label)
        form.addRow("Average Simulation Duration", self.average_simulation_duration_label)
        form.addRow("Output Folder", self.output_folder_label)

        self.open_output_button = QPushButton("Open Output Directory")
        self.open_output_button.setEnabled(False)
        form.addRow(self.open_output_button)

        self.rejection_explanation_label = QLabel("")
        self.rejection_explanation_label.setWordWrap(True)
        self.rejection_explanation_label.setStyleSheet("color: #a83232;")
        self.rejection_explanation_label.setVisible(False)
        form.addRow(self.rejection_explanation_label)

        return group

    # =====================================================
    # Validation Report panel -- Requirements 2/3/4/5/6/7.
    # =====================================================

    def _build_validation_report_group(self):

        group = QGroupBox("Validation Report")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.preflight_issues_label = QLabel("No pre-flight issues.")
        self.preflight_issues_label.setWordWrap(True)
        layout.addWidget(self.preflight_issues_label)

        self.first_rejection_label = QLabel("First rejection: none yet.")
        self.first_rejection_label.setWordWrap(True)
        self.first_rejection_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.first_rejection_label)

        self.category_counts_label = QLabel("Rejection counts by category: none yet.")
        self.category_counts_label.setWordWrap(True)
        layout.addWidget(self.category_counts_label)

        self.validation_table = QTableWidget(0, 4)
        self.validation_table.setHorizontalHeaderLabels(
            ["Failure Category", "Validation Code", "Message", "Count"]
        )
        self.validation_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch,
        )
        self.validation_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.validation_table.setMinimumHeight(160)
        layout.addWidget(self.validation_table)

        return group

    # =====================================================
    # Building wiring -- called once, whenever the Campaign Studio is
    # opened against the Designer's currently loaded Building.
    # =====================================================

    def set_building(self, building):

        self._zone_ids = []
        self._floor_ids = []
        self._door_ids = []
        self._exit_ids = []
        self._stair_ids = []
        self._obstacle_ids = []
        self._camera_ids = []
        self._detector_ids = []

        self.fire_allowed_zones_list.clear()
        self.fire_forbidden_zones_list.clear()
        self.fire_allowed_floors_list.clear()

        if building is None:
            self.population_profile_editor.set_zones({})
            return

        zone_names = {}

        for floor in building.floors:

            self._floor_ids.append(floor.id)
            self._add_checkable_item(self.fire_allowed_floors_list, floor.name, floor.id)

            for zone in floor.zones:

                self._zone_ids.append(zone.id)
                self._add_checkable_item(self.fire_allowed_zones_list, zone.name, zone.id)
                self._add_checkable_item(self.fire_forbidden_zones_list, zone.name, zone.id)
                zone_names[zone.id] = zone.name

            self._door_ids.extend(door.id for door in floor.doors)
            self._exit_ids.extend(exit_obj.id for exit_obj in floor.exits)
            self._stair_ids.extend(stair.id for stair in floor.stairs)
            self._obstacle_ids.extend(obstacle.id for obstacle in floor.obstacles)
            self._camera_ids.extend(camera.id for camera in floor.cameras)
            self._detector_ids.extend(detector.id for detector in floor.detectors)

        self.population_profile_editor.set_zones(zone_names)

    # =====================================================

    def _add_checkable_item(self, list_widget, label, object_id):

        item = QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)
        item.setData(Qt.ItemDataRole.UserRole, object_id)

        list_widget.addItem(item)

    # =====================================================

    def _checked_ids(self, list_widget):

        ids = []

        for index in range(list_widget.count()):

            item = list_widget.item(index)

            if item.checkState() == Qt.CheckState.Checked:
                ids.append(item.data(Qt.ItemDataRole.UserRole))

        return ids

    # =====================================================
    # ScenarioDefinition translation -- the one place widget values
    # become real scenario_definition constructor calls.
    # =====================================================

    def build_scenario_definition(self) -> ScenarioDefinition:

        occupancy_distribution = self._occupancy_distribution()

        occupant = OccupantDefinition(
            occupancy_distribution={
                zone_id: occupancy_distribution for zone_id in self._zone_ids
            },
            behaviour_profile_distribution={
                zone_id: WeightedOptions(self.population_profile_editor.weights_for_zone(zone_id))
                for zone_id in self._zone_ids
            },
        )

        fire = FireDefinition(
            growth_parameter_distribution=self._growth_distribution(),
            allowed_ignition_zone_ids=frozenset(self._checked_ids(self.fire_allowed_zones_list)),
            forbidden_ignition_zone_ids=frozenset(
                self._checked_ids(self.fire_forbidden_zones_list)
            ),
            allowed_ignition_floor_ids=frozenset(
                self._checked_ids(self.fire_allowed_floors_list)
            ),
            allowed_fire_profiles=self._fire_profiles(),
        )

        engineering = EngineeringConstraints(
            door_state_distribution=self._category_distribution_map(
                self._door_ids, self.door_editor,
            ),
            exit_state_distribution=self._exit_distribution_map(),
            min_open_exits=self.min_open_exits_spin.value(),
            stair_state_distribution=self._category_distribution_map(
                self._stair_ids, self.stair_editor,
            ),
            obstacle_state_distribution=self._category_distribution_map(
                self._obstacle_ids, self.obstacle_editor,
            ),
            camera_state_distribution=self._category_distribution_map(
                self._camera_ids, self.camera_editor,
            ),
            detector_state_distribution=self._category_distribution_map(
                self._detector_ids, self.detector_editor,
            ),
        )

        return ScenarioDefinition(
            fire=fire, engineering=engineering, occupant=occupant,
            event_templates=(), seed=self.master_seed(),
        )

    # =====================================================

    def _occupancy_distribution(self):

        if self.occupancy_mode_combo.currentText() == "Fixed":
            return FixedValue(self.occupancy_fixed_spin.value())

        return UniformRange(
            self.occupancy_min_spin.value(), self.occupancy_max_spin.value(), discrete=True,
        )

    # =====================================================

    def _growth_distribution(self):

        if self.growth_mode_combo.currentText() == "Fixed":
            return FixedValue(self.growth_fixed_spin.value())

        return UniformRange(self.growth_min_spin.value(), self.growth_max_spin.value())

    # =====================================================

    def _fire_profiles(self):

        profile = self.fire_profile_combo.currentText().strip()

        return frozenset({profile}) if profile else frozenset()

    # =====================================================

    def _category_distribution_map(self, object_ids, editor):

        if not object_ids or editor.mode() == EngineeringCategoryEditor.MODE_BUILDING_DEFAULT:
            return {}

        if editor.mode() == EngineeringCategoryEditor.MODE_FIXED:
            distribution = FixedValue(editor.fixed_state())
        else:
            distribution = WeightedOptions(editor.weights())

        return {object_id: distribution for object_id in object_ids}

    # =====================================================

    def _exit_distribution_map(self):

        # Exits are bool-shaped in EngineeringConstraints -- the
        # editor's "OPEN"/"CLOSED" state names are mapped to True/False
        # here rather than passed through as strings (see
        # EXIT_STATE_NAMES's comment).

        editor = self.exit_editor

        if not self._exit_ids or editor.mode() == EngineeringCategoryEditor.MODE_BUILDING_DEFAULT:
            return {}

        if editor.mode() == EngineeringCategoryEditor.MODE_FIXED:
            distribution = FixedValue(editor.fixed_state() == "OPEN")
        else:
            distribution = WeightedOptions(
                {state_name == "OPEN": weight for state_name, weight in editor.weights().items()}
            )

        return {exit_id: distribution for exit_id in self._exit_ids}

    # =====================================================
    # Campaign-level field accessors
    # =====================================================

    def campaign_name(self) -> str:

        return self.name_edit.text().strip() or "Untitled Campaign"

    def dataset_intent(self) -> str:

        return self.dataset_intent_combo.currentText()

    def scenario_count(self) -> int:

        return self.count_spin.value()

    def master_seed(self) -> int:

        return self.seed_spin.value()

    def max_attempts(self) -> int:

        return self.max_attempts_spin.value()

    def diagnostics_mode(self) -> bool:

        return self.diagnostics_checkbox.isChecked()

    def output_directory(self) -> str:

        return self.output_directory_edit.text().strip()

    # =====================================================

    def _on_browse_output_directory(self):

        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")

        if directory:
            self.output_directory_edit.setText(directory)

    # =====================================================
    # Live state display -- a dumb widget, pushed updates only, same
    # convention SimulationPanel/PerceptionDebugPanel already follow.
    # =====================================================

    def set_running_state(self, running: bool):

        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.resume_button.setEnabled(False)
        self.cancel_button.setEnabled(running)

        if running:
            self.reset_validation_report()

    # =====================================================

    def set_paused_state(self, paused: bool):

        self.pause_button.setEnabled(not paused)
        self.resume_button.setEnabled(paused)

    # =====================================================

    def update_progress(self, progress):

        self.completed_label.setText(str(progress.accepted))
        self.remaining_label.setText(str(progress.remaining))
        self.rejected_label.setText(str(progress.rejected))
        self.current_scenario_label.setText(progress.current_scenario_label)
        self.average_simulation_time_label.setText(f"{progress.average_simulation_time:.3f} s")
        self.generation_speed_label.setText(f"{progress.generation_speed:.2f} scenarios/s")
        self.eta_label.setText(self._format_duration(progress.eta_seconds))
        self.elapsed_label.setText(self._format_duration(progress.elapsed_seconds))

    # =====================================================

    def show_summary(self, summary):

        self.total_generated_label.setText(str(summary.total_generated))
        self.accepted_label.setText(str(summary.accepted))
        self.rejected_total_label.setText(str(summary.rejected))

        self.average_evacuation_time_label.setText(
            f"{summary.average_evacuation_time:.2f} s"
            if summary.average_evacuation_time is not None
            else "N/A"
        )
        self.average_simulation_duration_label.setText(
            f"{summary.average_simulation_duration:.3f} s"
        )
        self.output_folder_label.setText(summary.output_directory)

        self.open_output_button.setEnabled(True)

        # Requirement 7 -- only ever shown when the worker actually
        # produced an explanation (accepted == 0 with diagnostics data
        # collected); hidden otherwise, never left over from a prior
        # campaign that did produce accepted scenarios.
        if summary.rejection_explanation:

            self.rejection_explanation_label.setText(summary.rejection_explanation)
            self.rejection_explanation_label.setVisible(True)

        else:

            self.rejection_explanation_label.setVisible(False)

    # =====================================================

    def show_error(self, message: str):

        QMessageBox.critical(self, "Campaign Error", message)

    # =====================================================
    # Validation Report panel -- pushed updates only, same "dumb
    # widget" convention as update_progress()/show_summary() above.
    # =====================================================

    def reset_validation_report(self):

        self.preflight_issues_label.setText("No pre-flight issues.")
        self.first_rejection_label.setText("First rejection: none yet.")
        self.category_counts_label.setText("Rejection counts by category: none yet.")
        self.validation_table.setRowCount(0)
        self.rejection_explanation_label.setVisible(False)

    # =====================================================

    def show_preflight(self, preflight):

        # Requirements 5/6 -- the Building/Definition pre-flight
        # verdict, shown as soon as it's available (before, or instead
        # of, any generation attempt).

        if not preflight.has_errors:

            self.preflight_issues_label.setText(
                "Pre-flight checks passed -- Building and Scenario Definition "
                "are both structurally valid."
            )
            return

        lines = ["Pre-flight checks FAILED:"]

        for row in preflight.building_issues:
            lines.append(f"  [BUILDING/{row.code}] {row.message}")

        for row in preflight.definition_issues:
            lines.append(f"  [DEFINITION/{row.code}] {row.message}")

        self.preflight_issues_label.setText("\n".join(lines))

    # =====================================================

    def show_first_rejection(self, row):

        # Requirement 2 -- displayed immediately, the first time it
        # fires, via a direct connection to
        # CampaignWorker.first_rejection_detected rather than waiting
        # for the next periodic diagnostics_updated snapshot.

        self.first_rejection_label.setText(
            f"First rejection: [{row.source}/{row.code}] {row.message}"
        )

    # =====================================================

    def update_diagnostics(self, summary):

        # Requirements 3/4 -- rejection counts grouped by
        # FailureCategory, and the full per-(category, code) breakdown
        # table.

        if summary.category_counts:

            parts = [
                f"{category}: {count}"
                for category, count in sorted(summary.category_counts.items())
            ]
            self.category_counts_label.setText(
                "Rejection counts by category: " + ", ".join(parts)
            )

        else:

            self.category_counts_label.setText("Rejection counts by category: none yet.")

        self.validation_table.setRowCount(len(summary.rows))

        for row_index, row in enumerate(summary.rows):

            self.validation_table.setItem(row_index, 0, QTableWidgetItem(row.source))
            self.validation_table.setItem(row_index, 1, QTableWidgetItem(row.code))
            self.validation_table.setItem(row_index, 2, QTableWidgetItem(row.message))
            self.validation_table.setItem(row_index, 3, QTableWidgetItem(str(row.count)))

    # =====================================================

    def _format_duration(self, seconds: float) -> str:

        seconds = max(0, int(seconds))

        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
