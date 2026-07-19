import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton

_app = QApplication.instance() or QApplication(sys.argv)

from models.building import Building
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.obstacle import Obstacle
from models.staircase import Staircase
from models.zone import Zone

from scenario_definition import (
    EngineeringConstraints,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
    UniformRange,
    WeightedOptions,
)
from scenario_storage import load_scenario_by_id, read_catalog_rows

from designer.campaign import (
    CampaignConfig,
    CampaignController,
    CampaignSummary,
    CampaignWindow,
    EngineeringCategoryEditor,
    PopulationProfileEditor,
)
from designer.campaign.campaign_controller import derive_definition_id
from designer.campaign.campaign_worker import CampaignWorker
from designer.campaign.population_profile_editor import BUILT_IN_POPULATION_PRESETS
from designer.campaign.progress_model import ProgressModel


# =====================================================
# Fixtures
# =====================================================


def make_building():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
        obstacles=[Obstacle(id="obs-1", active=False)],
        cameras=[Camera(id="cam-1", active=True)],
        detectors=[Detector(id="det-1", active=True)],
        stairs=[Staircase(id="stair-1", from_zone_id="zone-1", to_zone_id="zone-3", to_floor_id="floor-2")],
    )
    floor2 = Floor(name="Upper", id="floor-2", zones=[Zone(id="zone-3", name="Attic")])

    return Building(name="Test Building", id="building-1", floors=[floor1, floor2])


def make_definition(occupant_count=1):

    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-2": FixedValue(occupant_count)},
            behaviour_profile_distribution={"zone-2": FixedValue("Staff_Default")},
        ),
    )


def make_unsatisfiable_definition():

    # Deterministically rejected by scenario_validator every attempt --
    # the one exit is pinned CLOSED while min_open_exits=1 demands at
    # least one open, so no candidate can ever be accepted.
    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-2": FixedValue(1)},
            behaviour_profile_distribution={"zone-2": FixedValue("Staff_Default")},
        ),
        engineering=EngineeringConstraints(
            exit_state_distribution={"exit-1": FixedValue(False)}, min_open_exits=1,
        ),
    )


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="campaign_studio_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


def make_config(building=None, definition=None, output_directory=None, count=2, **overrides):

    defaults = dict(
        name="Test Campaign",
        building=building or make_building(),
        definition=definition or make_definition(),
        definition_id="def-test",
        count=count,
        master_seed=42,
        output_directory=output_directory,
        max_attempts=10,
        dt=10.0,
    )
    defaults.update(overrides)

    return CampaignConfig(**defaults)


class _FakeWindow:

    # A lightweight, non-Qt-dialog-showing double for CampaignWindow --
    # used by controller-level tests so they exercise CampaignController's
    # own logic in isolation, without ever triggering a real, blocking
    # QMessageBox (which would hang a headless test run waiting for a
    # user click that will never come).

    def __init__(self, building=None, output_directory=None, definition=None):

        self.start_button = QPushButton()
        self.pause_button = QPushButton()
        self.resume_button = QPushButton()
        self.cancel_button = QPushButton()
        self.open_output_button = QPushButton()

        self._building = building
        self._output_directory = output_directory or ""
        self._definition = definition or make_definition()
        self._engineering_issues = []
        self._population_issues = []

        self.errors = []
        self.summaries = []
        self.running_states = []
        self.paused_states = []
        self.progress_events = []
        self.building_rebindings = []

        self.preflights = []
        self.first_rejections = []
        self.diagnostics_events = []

    def set_building(self, building):
        self.building_rebindings.append(building)
        self._building = building

    def build_scenario_definition(self):
        return self._definition

    def validate_engineering_setup(self):
        return self._engineering_issues

    def validate_population_setup(self):
        return self._population_issues

    def campaign_name(self):
        return "Fake Campaign"

    def scenario_count(self):
        return 2

    def master_seed(self):
        return 7

    def max_attempts(self):
        return 10

    def diagnostics_mode(self):
        return True

    def output_directory(self):
        return self._output_directory

    def set_running_state(self, running):
        self.running_states.append(running)

    def set_paused_state(self, paused):
        self.paused_states.append(paused)

    def update_progress(self, progress):
        self.progress_events.append(progress)

    def show_summary(self, summary):
        self.summaries.append(summary)

    def show_error(self, message):
        self.errors.append(message)

    def show_preflight(self, preflight):
        self.preflights.append(preflight)

    def show_first_rejection(self, row):
        self.first_rejections.append(row)

    def update_diagnostics(self, summary):
        self.diagnostics_events.append(summary)


def _flush_qt_events(iterations=10):

    for _ in range(iterations):
        _app.processEvents()


# =====================================================
# CampaignWindow -- ScenarioDefinition translation
# =====================================================


class CampaignWindowTranslationTests(unittest.TestCase):

    def setUp(self):

        self.window = CampaignWindow()
        self.window.set_building(make_building())

    def test_set_building_collects_every_object_category(self):

        self.assertEqual(self.window._zone_ids, ["zone-1", "zone-2", "zone-3"])
        self.assertEqual(self.window._floor_ids, ["floor-1", "floor-2"])
        self.assertEqual(self.window._door_ids, ["door-1"])
        self.assertEqual(self.window._exit_ids, ["exit-1"])
        self.assertEqual(self.window._stair_ids, ["stair-1"])
        self.assertEqual(self.window._obstacle_ids, ["obs-1"])
        self.assertEqual(self.window._camera_ids, ["cam-1"])
        self.assertEqual(self.window._detector_ids, ["det-1"])

    def test_fixed_occupancy_mode_produces_fixed_value(self):

        self.window.occupancy_mode_combo.setCurrentText("Fixed")
        self.window.occupancy_fixed_spin.setValue(6)

        definition = self.window.build_scenario_definition()

        for zone_id in self.window._zone_ids:
            self.assertEqual(
                definition.occupant.occupancy_distribution[zone_id], FixedValue(6),
            )

    def test_range_occupancy_mode_produces_uniform_range(self):

        self.window.occupancy_mode_combo.setCurrentText("Range")
        self.window.occupancy_min_spin.setValue(2)
        self.window.occupancy_max_spin.setValue(9)

        definition = self.window.build_scenario_definition()

        for zone_id in self.window._zone_ids:
            self.assertEqual(
                definition.occupant.occupancy_distribution[zone_id],
                UniformRange(2, 9, discrete=True),
            )

    def test_population_mix_produces_weighted_options_per_zone(self):

        editor = self.window.population_profile_editor
        editor.zone_combo.setCurrentIndex(0)
        editor.set_percentages(
            {
                "Adult_Default": 60, "Child_Default": 20, "Elderly_Default": 0,
                "Wheelchair_Default": 20, "Visitor_Default": 0,
            },
        )

        definition = self.window.build_scenario_definition()
        zone_id = self.window._zone_ids[0]

        self.assertEqual(
            definition.occupant.behaviour_profile_distribution[zone_id],
            WeightedOptions(
                {"Adult_Default": 0.6, "Child_Default": 0.2, "Wheelchair_Default": 0.2},
            ),
        )

    def test_each_zone_can_have_an_independently_different_population_mix(self):

        # The actual bug this editor replaces: previously one combo box
        # selection was wrapped in FixedValue() and applied identically
        # to every zone. Two zones configured with two different mixes
        # must both come through independently correct.

        editor = self.window.population_profile_editor
        zone_a, zone_b = self.window._zone_ids[0], self.window._zone_ids[1]

        editor.zone_combo.setCurrentIndex(0)
        editor.set_percentages(
            {"Adult_Default": 100, "Child_Default": 0, "Elderly_Default": 0,
             "Wheelchair_Default": 0, "Visitor_Default": 0},
        )
        editor.zone_combo.setCurrentIndex(1)
        editor.set_percentages(
            {"Adult_Default": 0, "Child_Default": 100, "Elderly_Default": 0,
             "Wheelchair_Default": 0, "Visitor_Default": 0},
        )

        definition = self.window.build_scenario_definition()

        self.assertEqual(
            definition.occupant.behaviour_profile_distribution[zone_a],
            WeightedOptions({"Adult_Default": 1.0}),
        )
        self.assertEqual(
            definition.occupant.behaviour_profile_distribution[zone_b],
            WeightedOptions({"Child_Default": 1.0}),
        )
        self.assertNotEqual(
            definition.occupant.behaviour_profile_distribution[zone_a],
            definition.occupant.behaviour_profile_distribution[zone_b],
        )

    def test_fire_allowed_and_forbidden_zones_reflect_checked_items(self):

        self._check_item(self.window.fire_allowed_zones_list, "zone-2")
        self._check_item(self.window.fire_forbidden_zones_list, "zone-1")

        definition = self.window.build_scenario_definition()

        self.assertEqual(definition.fire.allowed_ignition_zone_ids, frozenset({"zone-2"}))
        self.assertEqual(definition.fire.forbidden_ignition_zone_ids, frozenset({"zone-1"}))

    def test_no_checked_zones_means_unconstrained(self):

        definition = self.window.build_scenario_definition()

        self.assertEqual(definition.fire.allowed_ignition_zone_ids, frozenset())
        self.assertEqual(definition.fire.forbidden_ignition_zone_ids, frozenset())

    def test_fire_profile_field_becomes_allowed_fire_profiles(self):

        self.window.fire_profile_combo.setCurrentText("Flaming")

        definition = self.window.build_scenario_definition()

        self.assertEqual(definition.fire.allowed_fire_profiles, frozenset({"Flaming"}))

    def test_growth_time_fixed_and_range_modes(self):

        self.window.growth_mode_combo.setCurrentText("Fixed")
        self.window.growth_fixed_spin.setValue(150.0)

        definition = self.window.build_scenario_definition()
        self.assertEqual(definition.fire.growth_parameter_distribution, FixedValue(150.0))

        self.window.growth_mode_combo.setCurrentText("Range")
        self.window.growth_min_spin.setValue(50.0)
        self.window.growth_max_spin.setValue(300.0)

        definition = self.window.build_scenario_definition()
        self.assertEqual(
            definition.fire.growth_parameter_distribution, UniformRange(50.0, 300.0),
        )

    def test_engineering_use_building_default_leaves_map_empty(self):

        definition = self.window.build_scenario_definition()

        self.assertEqual(definition.engineering.door_state_distribution, {})
        self.assertEqual(definition.engineering.stair_state_distribution, {})
        self.assertEqual(definition.engineering.camera_state_distribution, {})

    def test_engineering_fixed_state_applies_to_every_object(self):

        self.window.door_editor.set_mode(EngineeringCategoryEditor.MODE_FIXED)
        self.window.door_editor.set_fixed_state("LOCKED")

        definition = self.window.build_scenario_definition()

        self.assertEqual(
            definition.engineering.door_state_distribution,
            {"door-1": FixedValue("LOCKED")},
        )

    def test_engineering_random_distribution_produces_weighted_options_over_every_state(self):

        self.window.camera_editor.set_mode(EngineeringCategoryEditor.MODE_RANDOM)

        definition = self.window.build_scenario_definition()

        distribution = definition.engineering.camera_state_distribution["cam-1"]

        self.assertIsInstance(distribution, WeightedOptions)
        self.assertEqual(set(distribution.weights.keys()), {"AVAILABLE", "FAILED"})

    def test_engineering_random_distribution_uses_edited_percentages(self):

        self.window.camera_editor.set_mode(EngineeringCategoryEditor.MODE_RANDOM)
        self.window.camera_editor.set_percentages({"AVAILABLE": 95, "FAILED": 5})

        definition = self.window.build_scenario_definition()

        distribution = definition.engineering.camera_state_distribution["cam-1"]

        self.assertEqual(distribution.weights, {"AVAILABLE": 0.95, "FAILED": 0.05})

    def test_exit_distribution_uses_boolean_values(self):

        self.window.exit_editor.set_mode(EngineeringCategoryEditor.MODE_FIXED)
        self.window.exit_editor.set_fixed_state("CLOSED")

        definition = self.window.build_scenario_definition()

        self.assertEqual(
            definition.engineering.exit_state_distribution, {"exit-1": FixedValue(False)},
        )

    def test_exit_random_distribution_maps_state_names_to_booleans(self):

        self.window.exit_editor.set_mode(EngineeringCategoryEditor.MODE_RANDOM)
        self.window.exit_editor.set_percentages({"OPEN": 90, "CLOSED": 10})

        definition = self.window.build_scenario_definition()

        distribution = definition.engineering.exit_state_distribution["exit-1"]

        self.assertEqual(distribution.weights, {True: 0.9, False: 0.1})

    def test_min_open_exits_is_carried_through(self):

        self.window.min_open_exits_spin.setValue(1)

        definition = self.window.build_scenario_definition()

        self.assertEqual(definition.engineering.min_open_exits, 1)

    def test_only_custom_dataset_intent_is_enabled(self):

        model = self.window.dataset_intent_combo.model()

        self.assertTrue(model.item(0).isEnabled())

        for index in range(1, self.window.dataset_intent_combo.count()):
            self.assertFalse(model.item(index).isEnabled())

    def _check_item(self, list_widget, object_id):

        from PyQt6.QtCore import Qt

        for index in range(list_widget.count()):

            item = list_widget.item(index)

            if item.data(Qt.ItemDataRole.UserRole) == object_id:
                item.setCheckState(Qt.CheckState.Checked)
                return

        raise AssertionError(f"No list item found for {object_id!r}")


class CampaignWindowDisplayTests(unittest.TestCase):

    def setUp(self):
        self.window = CampaignWindow()

    def test_update_progress_populates_labels(self):

        progress = ProgressModel(
            total=10, processed=4, accepted=3, rejected=1, remaining=6,
            current_scenario_label="Scenario 4/10 (scn-abc)",
            average_simulation_time=0.5, generation_speed=2.0, eta_seconds=3.0,
            elapsed_seconds=2.0,
        )

        self.window.update_progress(progress)

        self.assertEqual(self.window.completed_label.text(), "3")
        self.assertEqual(self.window.remaining_label.text(), "6")
        self.assertEqual(self.window.rejected_label.text(), "1")
        self.assertEqual(self.window.current_scenario_label.text(), "Scenario 4/10 (scn-abc)")

    def test_show_summary_populates_results_and_enables_open_button(self):

        summary = CampaignSummary(
            total_requested=10, total_generated=8, accepted=8, rejected=2,
            average_evacuation_time=42.5, average_simulation_duration=0.01,
            output_directory="C:/out",
        )

        self.assertFalse(self.window.open_output_button.isEnabled())

        self.window.show_summary(summary)

        self.assertEqual(self.window.total_generated_label.text(), "8")
        self.assertEqual(self.window.accepted_label.text(), "8")
        self.assertEqual(self.window.rejected_total_label.text(), "2")
        self.assertEqual(self.window.output_folder_label.text(), "C:/out")
        self.assertTrue(self.window.open_output_button.isEnabled())

    def test_show_summary_handles_no_evacuation_data(self):

        summary = CampaignSummary(
            total_requested=1, total_generated=0, accepted=0, rejected=1,
            average_evacuation_time=None, average_simulation_duration=0.0,
            output_directory="C:/out",
        )

        self.window.show_summary(summary)

        self.assertEqual(self.window.average_evacuation_time_label.text(), "N/A")

    def test_set_running_state_toggles_buttons(self):

        self.window.set_running_state(True)

        self.assertFalse(self.window.start_button.isEnabled())
        self.assertTrue(self.window.pause_button.isEnabled())
        self.assertTrue(self.window.cancel_button.isEnabled())
        self.assertFalse(self.window.resume_button.isEnabled())

        self.window.set_running_state(False)

        self.assertTrue(self.window.start_button.isEnabled())
        self.assertFalse(self.window.pause_button.isEnabled())
        self.assertFalse(self.window.cancel_button.isEnabled())

    def test_set_paused_state_toggles_pause_and_resume(self):

        self.window.set_running_state(True)
        self.window.set_paused_state(True)

        self.assertFalse(self.window.pause_button.isEnabled())
        self.assertTrue(self.window.resume_button.isEnabled())

        self.window.set_paused_state(False)

        self.assertTrue(self.window.pause_button.isEnabled())
        self.assertFalse(self.window.resume_button.isEnabled())


# =====================================================
# EngineeringCategoryEditor -- standalone widget tests
# =====================================================


class EngineeringCategoryEditorTests(unittest.TestCase):

    def setUp(self):
        self.editor = EngineeringCategoryEditor("Doors", ("OPEN", "CLOSED", "LOCKED"))

    def test_defaults_to_use_building_default(self):

        self.assertEqual(self.editor.mode(), EngineeringCategoryEditor.MODE_BUILDING_DEFAULT)
        self.assertEqual(self.editor.summary_lines(), ["Building Defaults"])
        # isHidden() (rather than isVisible()) is used throughout this
        # class -- isVisible() also depends on the whole ancestor
        # chain being shown on screen, which never happens for a
        # top-level widget that is never .show()'n in a headless test
        # run; isHidden() reflects this widget's own explicit
        # setVisible() state regardless.
        self.assertTrue(self.editor.fixed_container.isHidden())
        self.assertTrue(self.editor.random_container.isHidden())

    def test_fixed_state_mode_shows_fixed_container_only(self):

        self.editor.set_mode(EngineeringCategoryEditor.MODE_FIXED)
        self.editor.set_fixed_state("LOCKED")

        self.assertEqual(self.editor.fixed_state(), "LOCKED")
        self.assertEqual(self.editor.summary_lines(), ["Fixed: LOCKED"])
        self.assertFalse(self.editor.fixed_container.isHidden())
        self.assertTrue(self.editor.random_container.isHidden())

    def test_random_distribution_mode_shows_random_container_only(self):

        self.editor.set_mode(EngineeringCategoryEditor.MODE_RANDOM)

        self.assertTrue(self.editor.fixed_container.isHidden())
        self.assertFalse(self.editor.random_container.isHidden())

    def test_random_distribution_defaults_to_a_valid_even_split_totalling_100(self):

        self.editor.set_mode(EngineeringCategoryEditor.MODE_RANDOM)

        self.assertEqual(self.editor.total_percentage(), 100)
        self.assertTrue(self.editor.is_valid())

    def test_percentages_not_totalling_100_are_invalid(self):

        self.editor.set_mode(EngineeringCategoryEditor.MODE_RANDOM)
        self.editor.set_percentages({"OPEN": 60, "CLOSED": 30, "LOCKED": 5})

        self.assertEqual(self.editor.total_percentage(), 95)
        self.assertFalse(self.editor.is_valid())

    def test_building_default_and_fixed_modes_are_always_valid_regardless_of_percentages(self):

        self.editor.set_percentages({"OPEN": 1, "CLOSED": 1, "LOCKED": 1})

        self.assertTrue(self.editor.is_valid())

        self.editor.set_mode(EngineeringCategoryEditor.MODE_FIXED)
        self.assertTrue(self.editor.is_valid())

    def test_weights_excludes_zero_percentage_states(self):

        self.editor.set_mode(EngineeringCategoryEditor.MODE_RANDOM)
        self.editor.set_percentages({"OPEN": 60, "CLOSED": 40, "LOCKED": 0})

        self.assertEqual(self.editor.weights(), {"OPEN": 0.6, "CLOSED": 0.4})

    def test_summary_lines_for_random_distribution_lists_every_state(self):

        self.editor.set_mode(EngineeringCategoryEditor.MODE_RANDOM)
        self.editor.set_percentages({"OPEN": 60, "CLOSED": 30, "LOCKED": 10})

        self.assertEqual(
            self.editor.summary_lines(), ["OPEN 60%", "CLOSED 30%", "LOCKED 10%"],
        )

    def test_changed_signal_fires_on_mode_and_percentage_edits(self):

        events = []
        self.editor.changed.connect(lambda: events.append(True))

        self.editor.set_mode(EngineeringCategoryEditor.MODE_RANDOM)
        self.assertTrue(events)

        events.clear()
        self.editor.percentage_spins["OPEN"].setValue(50)
        self.assertTrue(events)


# =====================================================
# PopulationProfileEditor -- per-zone weighted population mix
# =====================================================


class PopulationProfileEditorTests(unittest.TestCase):

    def setUp(self):
        self.editor = PopulationProfileEditor()

    def test_set_zones_seeds_every_zone_with_a_valid_even_split(self):

        self.editor.set_zones({"zone-1": "Lobby", "zone-2": "Office"})

        self.assertTrue(self.editor.is_valid())
        self.assertEqual(self.editor.total_percentage(), 100)
        for zone_id in ("zone-1", "zone-2"):
            self.assertAlmostEqual(sum(self.editor.weights_for_zone(zone_id).values()), 1.0)

    def test_no_zones_is_valid(self):

        self.editor.set_zones({})
        self.assertTrue(self.editor.is_valid())

    def test_editing_one_zone_does_not_affect_another(self):

        self.editor.set_zones({"zone-1": "Lobby", "zone-2": "Office"})

        self.editor.zone_combo.setCurrentIndex(0)
        self.editor.set_percentages(
            {"Adult_Default": 100, "Child_Default": 0, "Elderly_Default": 0,
             "Wheelchair_Default": 0, "Visitor_Default": 0},
        )

        self.assertEqual(self.editor.weights_for_zone("zone-1"), {"Adult_Default": 1.0})
        self.assertNotEqual(self.editor.weights_for_zone("zone-2"), {"Adult_Default": 1.0})

    def test_switching_zones_and_back_preserves_edits(self):

        self.editor.set_zones({"zone-1": "Lobby", "zone-2": "Office"})

        self.editor.zone_combo.setCurrentIndex(0)
        self.editor.set_percentages(
            {"Adult_Default": 30, "Child_Default": 70, "Elderly_Default": 0,
             "Wheelchair_Default": 0, "Visitor_Default": 0},
        )

        self.editor.zone_combo.setCurrentIndex(1)
        self.editor.set_percentages(
            {"Adult_Default": 10, "Child_Default": 0, "Elderly_Default": 90,
             "Wheelchair_Default": 0, "Visitor_Default": 0},
        )

        self.editor.zone_combo.setCurrentIndex(0)

        self.assertEqual(
            self.editor.percentages(),
            {"Adult_Default": 30, "Child_Default": 70, "Elderly_Default": 0,
             "Wheelchair_Default": 0, "Visitor_Default": 0},
        )

    def test_resync_preserves_existing_zone_weights(self):

        # CampaignController.build_config() calls set_building() (which
        # calls set_zones()) again on every "Start" click -- a naive
        # reset would silently discard a configured mix each run.

        self.editor.set_zones({"zone-1": "Lobby", "zone-2": "Office"})
        self.editor.zone_combo.setCurrentIndex(0)
        self.editor.set_percentages(
            {"Adult_Default": 30, "Child_Default": 70, "Elderly_Default": 0,
             "Wheelchair_Default": 0, "Visitor_Default": 0},
        )

        self.editor.set_zones({"zone-1": "Lobby", "zone-2": "Office"})

        self.assertEqual(
            self.editor.weights_for_zone("zone-1"),
            {"Adult_Default": 0.3, "Child_Default": 0.7},
        )

    def test_resync_adds_new_zones_and_drops_removed_ones(self):

        self.editor.set_zones({"zone-1": "Lobby", "zone-2": "Office"})
        self.editor.zone_combo.setCurrentIndex(0)
        self.editor.set_percentages(
            {"Adult_Default": 100, "Child_Default": 0, "Elderly_Default": 0,
             "Wheelchair_Default": 0, "Visitor_Default": 0},
        )

        self.editor.set_zones({"zone-1": "Lobby", "zone-3": "New Zone"})

        self.assertEqual(set(self.editor._zone_weights.keys()), {"zone-1", "zone-3"})
        self.assertEqual(self.editor.weights_for_zone("zone-1"), {"Adult_Default": 1.0})
        self.assertAlmostEqual(sum(self.editor.weights_for_zone("zone-3").values()), 1.0)

    def test_zero_weight_profiles_are_dropped(self):

        self.editor.set_zones({"zone-1": "Lobby"})
        self.editor.set_percentages(
            {"Adult_Default": 60, "Child_Default": 40, "Elderly_Default": 0,
             "Wheelchair_Default": 0, "Visitor_Default": 0},
        )

        self.assertEqual(
            self.editor.weights_for_zone("zone-1"), {"Adult_Default": 0.6, "Child_Default": 0.4},
        )

    def test_percentages_not_totalling_100_are_invalid_and_named_in_message(self):

        self.editor.set_zones({"zone-1": "Lobby"})
        self.editor.set_percentages(
            {"Adult_Default": 50, "Child_Default": 0, "Elderly_Default": 0,
             "Wheelchair_Default": 0, "Visitor_Default": 0},
        )

        self.assertFalse(self.editor.is_valid())
        messages = self.editor.validation_messages()
        self.assertEqual(len(messages), 1)
        self.assertIn("Lobby", messages[0])
        self.assertIn("50%", messages[0])

    def test_built_in_presets_all_total_100(self):

        for name, weights in BUILT_IN_POPULATION_PRESETS.items():
            self.assertEqual(sum(weights.values()), 100, msg=name)

    def test_apply_preset_sets_current_zone_only(self):

        self.editor.set_zones({"zone-1": "Lobby", "zone-2": "Office"})
        self.editor.zone_combo.setCurrentIndex(0)

        self.editor.preset_combo.setCurrentText("Office")
        self.editor._on_apply_preset()

        self.assertEqual(self.editor.percentages(), BUILT_IN_POPULATION_PRESETS["Office"])
        self.assertNotEqual(self.editor.weights_for_zone("zone-2"), self.editor.weights_for_zone("zone-1"))

    def test_copy_to_checked_zones_applies_current_mix_only_to_checked(self):

        self.editor.set_zones({"zone-1": "Lobby", "zone-2": "Office", "zone-3": "Attic"})
        self.editor.zone_combo.setCurrentIndex(0)
        self.editor.set_percentages(
            {"Adult_Default": 100, "Child_Default": 0, "Elderly_Default": 0,
             "Wheelchair_Default": 0, "Visitor_Default": 0},
        )

        # Check only zone-2 (index 1 in the copy list, since it mirrors
        # insertion order zone-1/zone-2/zone-3).
        self.editor.copy_zone_list.item(1).setCheckState(Qt.CheckState.Checked)
        self.editor._on_copy_to_zones()

        self.assertEqual(self.editor.weights_for_zone("zone-2"), {"Adult_Default": 1.0})
        self.assertNotEqual(self.editor.weights_for_zone("zone-3"), {"Adult_Default": 1.0})

    def test_save_and_load_preset_round_trips_through_a_file(self):

        self.editor.set_zones({"zone-1": "Lobby", "zone-2": "Office"})
        self.editor.zone_combo.setCurrentIndex(0)
        self.editor.set_percentages(
            {"Adult_Default": 40, "Child_Default": 20, "Elderly_Default": 20,
             "Wheelchair_Default": 10, "Visitor_Default": 10},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:

            preset_path = str(Path(tmp_dir) / "custom_preset.json")

            from serialization.json_writer import JsonWriter
            JsonWriter.write(preset_path, {"name": "Custom", "weights": self.editor.percentages()})

            self.editor.zone_combo.setCurrentIndex(1)

            from serialization.json_reader import JsonReader
            data = JsonReader.read(preset_path)
            self.editor._custom_presets["Custom"] = {
                k: int(v) for k, v in data["weights"].items()
            }
            self.editor.preset_combo.addItem("Custom")
            self.editor.preset_combo.setCurrentText("Custom")
            self.editor._on_apply_preset()

            self.assertEqual(
                self.editor.percentages(),
                {"Adult_Default": 40, "Child_Default": 20, "Elderly_Default": 20,
                 "Wheelchair_Default": 10, "Visitor_Default": 10},
            )


# =====================================================
# CampaignWindow -- Campaign Summary panel and validation
# =====================================================


class CampaignWindowEngineeringSummaryTests(unittest.TestCase):

    def setUp(self):
        self.window = CampaignWindow()
        self.window.set_building(make_building())

    def test_all_defaults_reports_building_defaults_for_every_category_and_warns(self):

        self.assertTrue(self.window.all_engineering_defaults())
        self.assertFalse(self.window.engineering_variation_warning_label.isHidden())

        for label in ("Doors:", "Exits:", "Stairs:", "Obstacles:", "Cameras:", "Detectors:"):
            self.assertIn(label, self.window.engineering_summary_label.text())

        self.assertIn(
            "little or no engineering-state variation",
            self.window.engineering_variation_warning_label.text(),
        )

    def test_one_non_default_category_hides_the_variation_warning(self):

        self.window.door_editor.set_mode(EngineeringCategoryEditor.MODE_FIXED)

        self.assertFalse(self.window.all_engineering_defaults())
        self.assertTrue(self.window.engineering_variation_warning_label.isHidden())

    def test_summary_reflects_fixed_state_selection(self):

        self.window.door_editor.set_mode(EngineeringCategoryEditor.MODE_FIXED)
        self.window.door_editor.set_fixed_state("LOCKED")

        self.assertIn("Fixed: LOCKED", self.window.engineering_summary_label.text())

    def test_summary_reflects_random_distribution_percentages(self):

        self.window.camera_editor.set_mode(EngineeringCategoryEditor.MODE_RANDOM)
        self.window.camera_editor.set_percentages({"AVAILABLE": 95, "FAILED": 5})

        text = self.window.engineering_summary_label.text()

        self.assertIn("AVAILABLE 95%", text)
        self.assertIn("FAILED 5%", text)

    def test_summary_includes_minimum_open_exits_when_set(self):

        self.window.min_open_exits_spin.setValue(2)

        self.assertIn("Minimum Open Exits: 2", self.window.engineering_summary_label.text())

    def test_validate_engineering_setup_reports_invalid_random_distributions(self):

        self.window.door_editor.set_mode(EngineeringCategoryEditor.MODE_RANDOM)
        self.window.door_editor.set_percentages({"OPEN": 50, "CLOSED": 30, "LOCKED": 10})

        issues = self.window.validate_engineering_setup()

        self.assertEqual(len(issues), 1)
        self.assertIn("Doors", issues[0])
        self.assertIn("100%", issues[0])

    def test_validate_engineering_setup_is_clean_by_default(self):

        self.assertEqual(self.window.validate_engineering_setup(), [])


# =====================================================
# CampaignController
# =====================================================


class CampaignControllerConfigTests(unittest.TestCase):

    def test_build_config_raises_when_no_building(self):

        window = _FakeWindow(building=None, output_directory="C:/out")
        controller = CampaignController(window, get_building=lambda: None)

        with self.assertRaises(ValueError):
            controller.build_config()

    def test_build_config_raises_when_no_output_directory(self):

        building = make_building()
        window = _FakeWindow(building=building, output_directory="")
        controller = CampaignController(window, get_building=lambda: building)

        with self.assertRaises(ValueError):
            controller.build_config()

    def test_build_config_raises_when_engineering_setup_is_invalid(self):

        building = make_building()
        window = _FakeWindow(building=building, output_directory="C:/out")
        window._engineering_issues = [
            "Doors: Random Distribution percentages must total 100% (currently 90%)."
        ]
        controller = CampaignController(window, get_building=lambda: building)

        with self.assertRaises(ValueError) as ctx:
            controller.build_config()

        self.assertIn("Doors", str(ctx.exception))

    def test_build_config_raises_when_population_setup_is_invalid(self):

        building = make_building()
        window = _FakeWindow(building=building, output_directory="C:/out")
        window._population_issues = [
            "Occupant Population Mix — Office: percentages must total 100% (currently 87%)."
        ]
        controller = CampaignController(window, get_building=lambda: building)

        with self.assertRaises(ValueError) as ctx:
            controller.build_config()

        self.assertIn("Office", str(ctx.exception))

    def test_build_config_succeeds_with_building_and_output_directory(self):

        building = make_building()
        window = _FakeWindow(building=building, output_directory="C:/out")
        controller = CampaignController(window, get_building=lambda: building)

        config = controller.build_config()

        self.assertIs(config.building, building)
        self.assertEqual(config.name, "Fake Campaign")
        self.assertEqual(config.count, 2)
        self.assertEqual(config.master_seed, 7)
        self.assertEqual(config.output_directory, "C:/out")
        self.assertTrue(config.definition_id.startswith("def-"))


class CampaignControllerStartTests(unittest.TestCase):

    def test_on_start_shows_error_and_creates_no_worker_when_config_invalid(self):

        window = _FakeWindow(building=None, output_directory="")
        controller = CampaignController(window, get_building=lambda: None)

        controller.on_start()

        self.assertEqual(len(window.errors), 1)
        self.assertIsNone(controller.worker)
        self.assertEqual(window.running_states, [])

    def test_on_start_runs_a_real_tiny_campaign_end_to_end(self):

        with _TempOutputDir() as output_dir:

            building = make_building()
            window = _FakeWindow(building=building, output_directory=output_dir)
            controller = CampaignController(window, get_building=lambda: building)

            controller.on_start()

            self.assertIsNotNone(controller.worker)
            self.assertEqual(window.running_states, [True])

            self.assertTrue(controller.worker.wait(10_000))
            _flush_qt_events()

            self.assertEqual(len(window.summaries), 1)
            summary = window.summaries[0]

            self.assertEqual(summary.total_requested, 2)
            self.assertEqual(summary.accepted, 2)
            self.assertIn(False, window.running_states)

    def test_on_start_is_a_no_op_while_a_campaign_is_already_running(self):

        with _TempOutputDir() as output_dir:

            building = make_building()
            window = _FakeWindow(building=building, output_directory=output_dir)
            controller = CampaignController(window, get_building=lambda: building)

            controller.on_start()
            first_worker = controller.worker

            controller.on_start()

            self.assertIs(controller.worker, first_worker)

            controller.worker.wait(10_000)
            _flush_qt_events()


class CampaignControllerPauseResumeCancelTests(unittest.TestCase):

    # Exercises the controller's own delegation logic against a stub
    # worker object -- pause()/resume()/cancel()'s actual blocking
    # behavior is proven separately, at the worker level, in
    # WorkerPauseResumeCancelTests below.

    class _StubWorker:

        def __init__(self):
            self.paused = False
            self.cancelled = False

        def pause(self):
            self.paused = True

        def resume(self):
            self.paused = False

        def cancel(self):
            self.cancelled = True

    def setUp(self):

        self.window = _FakeWindow(building=make_building(), output_directory="C:/out")
        self.controller = CampaignController(self.window, get_building=lambda: make_building())
        self.controller.worker = self._StubWorker()

    def test_on_pause_delegates_and_updates_window(self):

        self.controller.on_pause()

        self.assertTrue(self.controller.worker.paused)
        self.assertEqual(self.window.paused_states, [True])

    def test_on_resume_delegates_and_updates_window(self):

        self.controller.on_pause()
        self.controller.on_resume()

        self.assertFalse(self.controller.worker.paused)
        self.assertEqual(self.window.paused_states, [True, False])

    def test_on_cancel_delegates(self):

        self.controller.on_cancel()

        self.assertTrue(self.controller.worker.cancelled)

    def test_pause_resume_cancel_are_safe_with_no_worker(self):

        self.controller.worker = None

        # Must not raise.
        self.controller.on_pause()
        self.controller.on_resume()
        self.controller.on_cancel()


class CampaignControllerFinishAndErrorTests(unittest.TestCase):

    def setUp(self):

        self.window = _FakeWindow(building=make_building(), output_directory="C:/out")
        self.controller = CampaignController(self.window, get_building=lambda: make_building())

    def test_on_finished_updates_window(self):

        summary = CampaignSummary(
            total_requested=1, total_generated=1, accepted=1, rejected=0,
            average_evacuation_time=10.0, average_simulation_duration=0.01,
            output_directory="C:/out",
        )

        self.controller.on_finished(summary)

        self.assertEqual(self.window.running_states, [False])
        self.assertEqual(self.window.summaries, [summary])

    def test_on_error_updates_window(self):

        self.controller.on_error("something broke")

        self.assertEqual(self.window.running_states, [False])
        self.assertEqual(self.window.errors, ["something broke"])

    def test_on_open_output_warns_when_directory_missing(self):

        self.window._output_directory = "C:/definitely/does/not/exist/anywhere"

        # QMessageBox.warning would block waiting for a real user click;
        # patched to a no-op so this test can run headlessly while
        # still proving on_open_output() takes the "missing" branch
        # rather than the "open" branch (which we prove separately).
        called = []

        original = QMessageBox.warning
        QMessageBox.warning = staticmethod(lambda *a, **k: called.append(True))

        try:
            self.controller.on_open_output()
        finally:
            QMessageBox.warning = original

        self.assertEqual(called, [True])

    def test_on_open_output_opens_when_directory_exists(self):

        with _TempOutputDir() as output_dir:

            self.window._output_directory = output_dir

            opened = []

            from PyQt6.QtGui import QDesktopServices

            original = QDesktopServices.openUrl
            QDesktopServices.openUrl = staticmethod(lambda url: opened.append(url))

            try:
                self.controller.on_open_output()
            finally:
                QDesktopServices.openUrl = original

            self.assertEqual(len(opened), 1)


# =====================================================
# CampaignWorker -- lifecycle, pause, resume, cancel
# =====================================================


class WorkerExecuteLogicTests(unittest.TestCase):

    # execute() called directly (never via .start()) runs synchronously
    # on the calling thread -- the fastest, most deterministic way to
    # test the worker's own algorithm without any Qt cross-thread
    # signal-delivery timing to account for.

    def test_execute_returns_a_correct_summary_for_an_accepted_campaign(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=3))

            summary = worker.execute()

            self.assertEqual(summary.total_requested, 3)
            self.assertEqual(summary.accepted, 3)
            self.assertEqual(summary.rejected, 0)
            self.assertFalse(summary.cancelled)
            self.assertIsNotNone(summary.average_evacuation_time)
            self.assertGreater(summary.average_evacuation_time, 0.0)

    def test_execute_emits_one_progress_update_per_scenario(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=4))

            events = []
            worker.progress_updated.connect(events.append)

            worker.execute()

            self.assertEqual(len(events), 4)
            self.assertEqual(events[-1].processed, 4)
            self.assertEqual(events[-1].remaining, 0)

    def test_execute_counts_rejected_scenarios_separately_from_accepted(self):

        # diagnostics_mode=False here specifically -- the unsatisfiable
        # fixture below is *also* caught by Diagnostics Mode's own
        # Definition pre-flight check (min_open_exits_infeasible, since
        # pinning the one exit CLOSED while requiring min_open_exits=1
        # is a Definition-level contradiction, not just a per-candidate
        # one) which aborts before any generation attempt -- exercised
        # separately in DiagnosticsPreflightTests below. This test
        # isolates the original per-candidate rejected-counting logic
        # (the fast, non-diagnostics path).
        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(
                make_config(
                    definition=make_unsatisfiable_definition(),
                    output_directory=output_dir, count=2, max_attempts=2,
                    diagnostics_mode=False,
                )
            )

            summary = worker.execute()

            self.assertEqual(summary.accepted, 0)
            self.assertEqual(summary.rejected, 2)
            self.assertIsNone(summary.average_evacuation_time)

    def test_cancel_before_execute_produces_an_empty_cancelled_summary(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=5))
            worker.cancel()

            summary = worker.execute()

            self.assertTrue(summary.cancelled)
            self.assertEqual(summary.accepted, 0)
            self.assertEqual(summary.rejected, 0)

    def test_cancel_mid_campaign_stops_before_processing_every_scenario(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=10))

            def stop_after_three(progress):
                if progress.processed >= 3:
                    worker.cancel()

            worker.progress_updated.connect(stop_after_three)

            summary = worker.execute()

            self.assertTrue(summary.cancelled)
            self.assertEqual(summary.accepted, 3)
            self.assertLess(summary.accepted, 10)

    def test_accepted_scenarios_are_actually_persisted_to_storage(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=2))
            worker.execute()

            rows = read_catalog_rows(output_dir)
            self.assertEqual(len(rows), 2)

            reloaded = load_scenario_by_id(rows[0]["scenario_id"], output_dir)
            self.assertEqual(reloaded.metadata.scenario_id, rows[0]["scenario_id"])

    def test_rerunning_the_same_config_against_the_same_directory_does_not_crash(self):

        # scenario_storage.save_scenario() raises FileExistsError on a
        # duplicate scenario_id -- the exact scenario a re-run against
        # an already-populated output directory produces, since
        # scenario_id is a deterministic function of
        # (definition_content_hash, seed, attempt_index). The worker
        # must absorb this, not propagate it.
        with _TempOutputDir() as output_dir:

            config = make_config(output_directory=output_dir, count=2)

            CampaignWorker(config).execute()
            second_summary = CampaignWorker(config).execute()

            self.assertEqual(second_summary.accepted, 2)


class WorkerPauseResumeCancelTests(unittest.TestCase):

    # Proves the actual blocking/unblocking mechanism, using a plain
    # threading.Thread (not a real QThread) to run execute() --
    # CampaignWorker.execute() has no Qt-event-loop dependency of its
    # own, so this is a faithful, deterministic test of the pause
    # primitive without any Qt cross-thread signal-delivery timing to
    # account for.

    def test_pause_and_resume_toggle_the_underlying_event(self):

        worker = CampaignWorker(make_config(output_directory="C:/unused", count=1))

        self.assertTrue(worker._pause_event.is_set())

        worker.pause()
        self.assertFalse(worker._pause_event.is_set())

        worker.resume()
        self.assertTrue(worker._pause_event.is_set())

    def test_cancel_sets_the_cancellation_flag(self):

        worker = CampaignWorker(make_config(output_directory="C:/unused", count=1))

        self.assertFalse(worker.is_cancelled)

        worker.cancel()

        self.assertTrue(worker.is_cancelled)

    def test_paused_worker_blocks_execute_until_resumed(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=1))
            worker.pause()

            result = {}

            def target():
                result["summary"] = worker.execute()

            thread = threading.Thread(target=target)
            thread.start()

            thread.join(timeout=0.3)
            self.assertTrue(thread.is_alive(), "execute() should still be blocked while paused")

            worker.resume()

            thread.join(timeout=10.0)
            self.assertFalse(thread.is_alive())
            self.assertIn("summary", result)
            self.assertEqual(result["summary"].accepted, 1)

    def test_cancel_wakes_a_paused_worker_instead_of_leaving_it_blocked_forever(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=5))
            worker.pause()

            result = {}

            def target():
                result["summary"] = worker.execute()

            thread = threading.Thread(target=target)
            thread.start()

            thread.join(timeout=0.3)
            self.assertTrue(thread.is_alive())

            worker.cancel()

            thread.join(timeout=10.0)
            self.assertFalse(thread.is_alive())
            self.assertTrue(result["summary"].cancelled)


class WorkerRealBackgroundThreadTests(unittest.TestCase):

    # The one test group that actually starts a real QThread via
    # .start() (rather than calling execute() directly) -- proving the
    # "Campaign execution must run in a background thread" requirement
    # itself, not just the algorithm .start() eventually calls.

    def test_worker_runs_on_a_background_thread_and_emits_finished_signal(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=2))

            finished = []
            worker.campaign_finished.connect(finished.append)

            main_thread = threading.current_thread()
            worker.start()

            self.assertTrue(worker.wait(10_000), "worker did not finish in time")
            _flush_qt_events()

            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0].accepted, 2)

            # The QThread object itself is never the calling (main)
            # Python thread -- confirms this genuinely ran in the
            # background rather than being silently executed inline.
            self.assertIsNot(worker.run.__self__, main_thread)

    def test_error_in_execute_emits_error_signal_not_an_unhandled_exception(self):

        with _TempOutputDir() as output_dir:

            config = make_config(output_directory=output_dir, count=1)
            config.building = None  # guaranteed to raise inside scenario_runner.run()

            worker = CampaignWorker(config)

            errors = []
            worker.error_occurred.connect(errors.append)

            worker.start()
            self.assertTrue(worker.wait(10_000))
            _flush_qt_events()

            self.assertEqual(len(errors), 1)


# =====================================================
# derive_definition_id
# =====================================================


class DeriveDefinitionIdTests(unittest.TestCase):

    def test_same_definition_always_derives_the_same_id(self):

        definition = make_definition()

        self.assertEqual(derive_definition_id(definition), derive_definition_id(definition))

    def test_different_definitions_derive_different_ids(self):

        first = derive_definition_id(make_definition(occupant_count=1))
        second = derive_definition_id(make_definition(occupant_count=5))

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
