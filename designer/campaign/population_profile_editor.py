from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from serialization.json_reader import JsonReader
from serialization.json_writer import JsonWriter


# =====================================================
# PopulationProfileEditor -- Campaign Studio's per-zone Occupant
# Population Mix authoring widget. Replaces the single, building-wide
# `behaviour_profile_combo` (campaign_window.py's own investigated root
# cause of the population-homogeneity bug: one QComboBox selection
# wrapped in FixedValue(...) and applied identically to every zone) with
# a genuinely per-zone weighted mix over the five civilian behaviour
# profile ids behaviour_profile_resolver/category.py's own
# _CATEGORY_BY_PROFILE_ID table already classifies as ADULT/CHILD/
# ELDERLY/WHEELCHAIR_USER/VISITOR.
#
# This widget produces only plain percentages/weights -- it never
# imports random, never samples anything, and never constructs a
# WeightedOptions itself (campaign_window.build_scenario_definition()
# is the one place that does, exactly as it already does for every
# other Distribution). Same "declares the rule, samples nothing"
# boundary scenario_definition/ and campaign_window.py's own module
# docstring already establish for every other editor in this window.
#
# Structurally mirrors designer/campaign/engineering_category_editor.py
# (EngineeringCategoryEditor)'s percentage-spin / live-total /
# weights()-drops-zero / is_valid() shape, but is deliberately NOT a
# subclass of it: that widget applies one rule uniformly across every
# object in a category, while this one must hold independent state per
# zone -- a genuinely different shape that would strain, not simplify,
# a shared base.
# =====================================================


POPULATION_PROFILE_LABELS = {
    "Adult_Default": "Adults",
    "Child_Default": "Children",
    "Elderly_Default": "Elderly",
    "Wheelchair_Default": "Wheelchair Users",
    "Visitor_Default": "Visitors",
}

_PROFILE_IDS = tuple(POPULATION_PROFILE_LABELS.keys())


# Illustrative starting points, not measured real-world data -- an
# author is always free to edit every percentage after applying one.
# Each sums to exactly 100 over (Adult, Child, Elderly, Wheelchair,
# Visitor), in that order.
BUILT_IN_POPULATION_PRESETS = {
    "Office": {
        "Adult_Default": 80, "Child_Default": 0, "Elderly_Default": 5,
        "Wheelchair_Default": 5, "Visitor_Default": 10,
    },
    "School": {
        "Adult_Default": 25, "Child_Default": 65, "Elderly_Default": 0,
        "Wheelchair_Default": 5, "Visitor_Default": 5,
    },
    "Hospital": {
        "Adult_Default": 25, "Child_Default": 5, "Elderly_Default": 30,
        "Wheelchair_Default": 25, "Visitor_Default": 15,
    },
    "Shopping Mall": {
        "Adult_Default": 50, "Child_Default": 15, "Elderly_Default": 15,
        "Wheelchair_Default": 5, "Visitor_Default": 15,
    },
    "Restaurant": {
        "Adult_Default": 60, "Child_Default": 10, "Elderly_Default": 15,
        "Wheelchair_Default": 5, "Visitor_Default": 10,
    },
    "Library": {
        "Adult_Default": 45, "Child_Default": 20, "Elderly_Default": 25,
        "Wheelchair_Default": 5, "Visitor_Default": 5,
    },
    "Airport": {
        "Adult_Default": 55, "Child_Default": 10, "Elderly_Default": 15,
        "Wheelchair_Default": 5, "Visitor_Default": 15,
    },
    "Hotel": {
        "Adult_Default": 50, "Child_Default": 5, "Elderly_Default": 20,
        "Wheelchair_Default": 10, "Visitor_Default": 15,
    },
    "Residential": {
        "Adult_Default": 50, "Child_Default": 20, "Elderly_Default": 20,
        "Wheelchair_Default": 10, "Visitor_Default": 0,
    },
}


def _even_split_weights():

    even_share, remainder = divmod(100, len(_PROFILE_IDS))

    return {
        profile_id: even_share + (remainder if index == len(_PROFILE_IDS) - 1 else 0)
        for index, profile_id in enumerate(_PROFILE_IDS)
    }


class PopulationProfileEditor(QGroupBox):

    def __init__(self, parent=None):
        super().__init__("Scenario Definition — Occupant Population Mix", parent)

        self._zone_names = {}
        self._zone_weights = {}
        self._current_zone_id = None
        self._custom_presets = {}

        layout = QVBoxLayout()
        self.setLayout(layout)

        zone_form = QFormLayout()
        self.zone_combo = QComboBox()
        self.zone_combo.currentIndexChanged.connect(self._on_zone_combo_changed)
        zone_form.addRow("Editing Zone", self.zone_combo)
        layout.addLayout(zone_form)

        percentage_form = QFormLayout()

        self.percentage_spins = {}

        for profile_id in _PROFILE_IDS:

            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix("%")
            spin.valueChanged.connect(self._on_percentage_changed)

            self.percentage_spins[profile_id] = spin
            percentage_form.addRow(POPULATION_PROFILE_LABELS[profile_id], spin)

        self.total_label = QLabel()
        percentage_form.addRow("Total", self.total_label)

        layout.addLayout(percentage_form)

        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        for name in sorted(BUILT_IN_POPULATION_PRESETS):
            self.preset_combo.addItem(name)
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self.preset_combo, 1)

        self.apply_preset_button = QPushButton("Apply to Current Zone")
        self.apply_preset_button.clicked.connect(self._on_apply_preset)
        preset_row.addWidget(self.apply_preset_button)

        self.save_preset_button = QPushButton("Save As Preset...")
        self.save_preset_button.clicked.connect(self._on_save_preset)
        preset_row.addWidget(self.save_preset_button)

        self.load_preset_button = QPushButton("Load Preset...")
        self.load_preset_button.clicked.connect(self._on_load_preset)
        preset_row.addWidget(self.load_preset_button)

        layout.addLayout(preset_row)

        copy_group = QGroupBox("Copy Current Zone's Mix to Other Zones")
        copy_layout = QVBoxLayout()
        copy_group.setLayout(copy_layout)

        self.copy_zone_list = QListWidget()
        self.copy_zone_list.setMaximumHeight(90)
        copy_layout.addWidget(self.copy_zone_list)

        self.copy_button = QPushButton("Copy to Checked Zones")
        self.copy_button.clicked.connect(self._on_copy_to_zones)
        copy_layout.addWidget(self.copy_button)

        layout.addWidget(copy_group)

        self._on_percentage_changed()

    # =====================================================
    # Public API
    # =====================================================

    def set_zones(self, zone_names: dict):

        # Resync, never blindly reset -- CampaignController.build_config()
        # calls set_building() (which calls this) again on every "Start"
        # click, against what is normally the same Building. A naive
        # reset here would silently discard every mix an author already
        # configured on each run; only zones that are genuinely new (or
        # genuinely gone) change.

        self._zone_names = dict(zone_names)

        for zone_id in list(self._zone_weights):
            if zone_id not in self._zone_names:
                del self._zone_weights[zone_id]

        for zone_id in self._zone_names:
            if zone_id not in self._zone_weights:
                self._zone_weights[zone_id] = _even_split_weights()

        previous_zone_id = self._current_zone_id

        self.zone_combo.blockSignals(True)
        self.zone_combo.clear()

        for zone_id, name in self._zone_names.items():

            item_index = self.zone_combo.count()
            self.zone_combo.addItem(name)
            self.zone_combo.setItemData(item_index, zone_id, Qt.ItemDataRole.UserRole)

        self.zone_combo.blockSignals(False)

        self.copy_zone_list.clear()
        for zone_id, name in self._zone_names.items():
            self._add_checkable_item(self.copy_zone_list, name, zone_id)

        if not self._zone_names:
            self._current_zone_id = None
            self._refresh_zone_combo_labels()
            self._on_percentage_changed()
            return

        restore_index = 0
        if previous_zone_id in self._zone_names:
            restore_index = list(self._zone_names.keys()).index(previous_zone_id)

        self.zone_combo.setCurrentIndex(restore_index)
        self._current_zone_id = self.zone_combo.itemData(restore_index, Qt.ItemDataRole.UserRole)

        self._load_zone_into_spins(self._current_zone_id)
        self._refresh_zone_combo_labels()

    # =====================================================

    def weights_for_zone(self, zone_id: str) -> dict:

        percentages = self._zone_weights.get(zone_id) or _even_split_weights()

        return {
            profile_id: value / 100.0
            for profile_id, value in percentages.items()
            if value > 0
        }

    # =====================================================

    def validation_messages(self) -> list:

        messages = []

        for zone_id in sorted(self._zone_weights):

            total = sum(self._zone_weights[zone_id].values())

            if total != 100:

                zone_name = self._zone_names.get(zone_id, zone_id)
                messages.append(
                    f"Occupant Population Mix — {zone_name}: percentages must total 100% "
                    f"(currently {total}%)."
                )

        return messages

    # =====================================================

    def is_valid(self) -> bool:

        return not self.validation_messages()

    # =====================================================
    # Internal wiring
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

    def _on_zone_combo_changed(self, index):

        if index < 0:
            return

        zone_id = self.zone_combo.itemData(index, Qt.ItemDataRole.UserRole)

        if zone_id is None:
            return

        self._current_zone_id = zone_id
        self._load_zone_into_spins(zone_id)

    # =====================================================

    def _load_zone_into_spins(self, zone_id):

        weights = self._zone_weights.get(zone_id) or _even_split_weights()

        for profile_id, spin in self.percentage_spins.items():

            spin.blockSignals(True)
            spin.setValue(weights.get(profile_id, 0))
            spin.blockSignals(False)

        self._on_percentage_changed()

    # =====================================================

    def _on_percentage_changed(self, *_):

        total = self.total_percentage()

        self.total_label.setText(f"{total}%")
        self.total_label.setStyleSheet(
            "color: #2f7d32; font-weight: bold;" if total == 100
            else "color: #a83232; font-weight: bold;"
        )

        if self._current_zone_id is not None:
            self._zone_weights[self._current_zone_id] = self.percentages()

        self._refresh_zone_combo_labels()

    # =====================================================

    def _refresh_zone_combo_labels(self):

        self.zone_combo.blockSignals(True)

        for index in range(self.zone_combo.count()):

            zone_id = self.zone_combo.itemData(index, Qt.ItemDataRole.UserRole)
            name = self._zone_names.get(zone_id, zone_id)
            total = sum(self._zone_weights.get(zone_id, {}).values())

            label = f"{name} — {total}%" if total == 100 else f"⚠ {name} — {total}%"
            self.zone_combo.setItemText(index, label)

        self.zone_combo.blockSignals(False)

    # =====================================================

    def _on_apply_preset(self):

        name = self.preset_combo.currentText()
        weights = BUILT_IN_POPULATION_PRESETS.get(name) or self._custom_presets.get(name)

        if weights is None or self._current_zone_id is None:
            return

        self.set_percentages(weights)

    # =====================================================

    def _on_save_preset(self):

        if self._current_zone_id is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Population Profile Preset", "", "Population Profile Preset (*.json)",
        )

        if not filename:
            return

        if not filename.endswith(".json"):
            filename += ".json"

        preset_name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if preset_name.endswith(".json"):
            preset_name = preset_name[: -len(".json")]

        JsonWriter.write(filename, {"name": preset_name, "weights": self.percentages()})

    # =====================================================

    def _on_load_preset(self):

        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Population Profile Preset", "", "Population Profile Preset (*.json)",
        )

        if not filename:
            return

        try:
            data = JsonReader.read(filename)
        except (OSError, ValueError):
            QMessageBox.critical(self, "Load Preset", f"Could not read preset file:\n{filename}")
            return

        raw_weights = data.get("weights", {})
        weights = {
            profile_id: int(raw_weights[profile_id])
            for profile_id in _PROFILE_IDS
            if profile_id in raw_weights
        }

        if not weights:
            QMessageBox.critical(
                self, "Load Preset", "This file has no recognized population profile weights.",
            )
            return

        preset_name = data.get("name") or filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        self._custom_presets[preset_name] = weights

        if self.preset_combo.findText(preset_name) < 0:
            self.preset_combo.addItem(preset_name)
        self.preset_combo.setCurrentText(preset_name)

        if self._current_zone_id is not None:
            self.set_percentages(weights)

    # =====================================================

    def _on_copy_to_zones(self):

        if self._current_zone_id is None:
            return

        current_weights = dict(self.percentages())

        for zone_id in self._checked_ids(self.copy_zone_list):

            if zone_id == self._current_zone_id:
                continue

            self._zone_weights[zone_id] = dict(current_weights)

        self._refresh_zone_combo_labels()

    # =====================================================
    # Accessors -- same shape as EngineeringCategoryEditor's own
    # percentages()/set_percentages()/total_percentage().
    # =====================================================

    def percentages(self) -> dict:

        return {profile_id: spin.value() for profile_id, spin in self.percentage_spins.items()}

    def set_percentages(self, values: dict):

        for profile_id, spin in self.percentage_spins.items():

            spin.blockSignals(True)
            spin.setValue(values.get(profile_id, 0))
            spin.blockSignals(False)

        self._on_percentage_changed()

    def total_percentage(self) -> int:

        return sum(spin.value() for spin in self.percentage_spins.values())
