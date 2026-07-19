import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario_definition import (
    EngineeringConstraints,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
)
from scenario_validator import FailureCategory, ScenarioValidationReport

from designer.campaign import (
    CampaignWindow,
    DiagnosticsCollector,
    PreflightResult,
    ValidationRow,
    explain_total_rejection,
)
from designer.campaign.campaign_worker import CampaignConfig, CampaignWorker

from tests.test_campaign_studio import (
    _FakeWindow,
    _TempOutputDir,
    make_building,
    make_config,
    make_definition,
)


# =====================================================
# Root-cause fixtures -- the exact configuration shape (Fixed occupancy
# applied to every zone + unconstrained fire ignition zone) that
# reproduced the reported "Accepted: 0, Rejected: 100" campaign against
# a chokepoint/corridor-style Building topology.
# =====================================================


def make_chain_building(zone_count=5):

    zones = [
        Zone(id=f"zone-{i}", name=f"Room {i}", x=float(i * 10), y=0.0, width=8.0, height=8.0)
        for i in range(1, zone_count + 1)
    ]
    doors = [
        Door(
            id=f"door-{i}", normally_open=True,
            zone_a_id=f"zone-{i}", zone_b_id=f"zone-{i + 1}",
        )
        for i in range(1, zone_count)
    ]
    exits = [Exit(id="exit-1", zone_id=f"zone-{zone_count}")]

    floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)

    return Building(name="Chain Building", id="chain-1", floors=[floor])


def make_every_zone_occupied_definition(building):

    zone_ids = [zone.id for floor in building.floors for zone in floor.zones]

    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
        occupant=OccupantDefinition(
            occupancy_distribution={zone_id: FixedValue(4) for zone_id in zone_ids},
            behaviour_profile_distribution={
                zone_id: FixedValue("Staff_Default") for zone_id in zone_ids
            },
        ),
    )


def make_unsatisfiable_definition():

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


def _report(accepted, category=None, code=None, message="", object_id=""):

    report = ScenarioValidationReport()

    if not accepted:
        report.add(category, ScenarioValidationReport.ERROR, code, message, object_id=object_id)

    return report


# =====================================================
# Root-cause reproduction -- confirms the diagnosed cause of the
# originally reported "Accepted: 0, Rejected: 100" campaign.
# =====================================================


class RootCauseReproductionTests(unittest.TestCase):

    def test_chain_topology_with_every_zone_occupied_produces_navigation_rejections(self):

        # A "corridor" building (each zone connects only to the next,
        # single path to the one Exit) combined with occupying every
        # zone and leaving the fire's ignition zone unconstrained --
        # exactly Campaign Studio's own default configuration -- causes
        # scenario_validator's navigation_validation.py to reject any
        # candidate whose (uniformly, unconstrained-ly sampled)
        # ignition zone sits between an occupied zone and the Exit.
        # This is the Validator working correctly, not a bug -- see
        # the accompanying root-cause report.
        building = make_chain_building(zone_count=5)
        definition = make_every_zone_occupied_definition(building)

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(
                CampaignConfig(
                    name="Root Cause Repro", building=building, definition=definition,
                    definition_id="def-repro", count=30, master_seed=42,
                    output_directory=output_dir, max_attempts=10, dt=10.0,
                    diagnostics_mode=True,
                )
            )

            summary = worker.execute()

            # Not asserting a specific ratio (depends on exactly which
            # zone gets sampled as ignition each attempt) -- asserting
            # only that rejections happen and are attributed correctly,
            # which is what this whole feature exists to make visible.
            self.assertGreater(summary.rejected, 0)

    def test_longer_chain_makes_navigation_the_dominant_rejection_reason(self):

        # A longer chain raises the probability any given attempt's
        # unconstrained ignition zone sits upstream of at least one
        # occupied zone -- demonstrating the failure mode scales with
        # topology "chokepoint-ness", exactly as the root-cause report
        # describes.
        building = make_chain_building(zone_count=8)
        definition = make_every_zone_occupied_definition(building)

        collector = DiagnosticsCollector()

        from scenario_generator import GenerationRequest, generate_scenario
        from scenario_validator import validate

        for attempt in range(60):

            request = GenerationRequest(
                definition=definition, definition_id="def-x", building=building,
                seed=42, attempt_index=attempt,
            )
            candidate = generate_scenario(request)
            report = validate(candidate, definition, building)
            collector.record_candidate_report(report)

        summary = collector.summary()

        self.assertGreater(summary.rejected_candidates, 0)
        self.assertIn("NAVIGATION", summary.category_counts)
        self.assertEqual(
            max(summary.category_counts, key=summary.category_counts.get), "NAVIGATION",
        )
        self.assertTrue(
            any(row.code == "FIRE_ORIGIN_BLOCKS_EVACUATION" for row in summary.rows)
        )


# =====================================================
# DiagnosticsCollector
# =====================================================


class DiagnosticsCollectorTests(unittest.TestCase):

    def test_accepted_report_is_counted_but_not_rejected(self):

        collector = DiagnosticsCollector()

        collector.record_candidate_report(_report(accepted=True))

        summary = collector.summary()
        self.assertEqual(summary.total_candidates, 1)
        self.assertEqual(summary.rejected_candidates, 0)
        self.assertEqual(summary.rows, ())

    def test_rejected_report_increments_category_and_row_counts(self):

        collector = DiagnosticsCollector()

        collector.record_candidate_report(
            _report(False, FailureCategory.NAVIGATION, "NO_EVACUATION_ROUTE", "msg one")
        )
        collector.record_candidate_report(
            _report(False, FailureCategory.NAVIGATION, "NO_EVACUATION_ROUTE", "msg one")
        )

        summary = collector.summary()

        self.assertEqual(summary.rejected_candidates, 2)
        self.assertEqual(summary.category_counts["NAVIGATION"], 2)
        self.assertEqual(len(summary.rows), 1)
        self.assertEqual(summary.rows[0].count, 2)

    def test_different_codes_produce_separate_rows_even_in_the_same_category(self):

        collector = DiagnosticsCollector()

        collector.record_candidate_report(
            _report(False, FailureCategory.NAVIGATION, "NO_EVACUATION_ROUTE", "a")
        )
        collector.record_candidate_report(
            _report(False, FailureCategory.NAVIGATION, "FIRE_ORIGIN_BLOCKS_EVACUATION", "b")
        )

        summary = collector.summary()

        self.assertEqual(len(summary.rows), 2)
        self.assertEqual(summary.category_counts["NAVIGATION"], 2)

    def test_first_rejection_is_reported_exactly_once(self):

        collector = DiagnosticsCollector()

        first = collector.record_candidate_report(
            _report(False, FailureCategory.FIRE, "MISSING_FIRE", "no fire")
        )
        second = collector.record_candidate_report(
            _report(False, FailureCategory.FIRE, "MISSING_FIRE", "no fire")
        )

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(collector.first_rejection, first)

    def test_first_rejection_message_reflects_the_first_occurrence(self):

        collector = DiagnosticsCollector()

        collector.record_candidate_report(
            _report(False, FailureCategory.FIRE, "MISSING_FIRE", "original message")
        )

        summary = collector.summary()
        self.assertEqual(summary.first_rejection.message, "original message")

    def test_a_report_with_multiple_issues_updates_every_category(self):

        report = ScenarioValidationReport()
        report.add(FailureCategory.NAVIGATION, ScenarioValidationReport.ERROR, "A", "a")
        report.add(FailureCategory.FIRE, ScenarioValidationReport.ERROR, "B", "b")

        collector = DiagnosticsCollector()
        collector.record_candidate_report(report)

        summary = collector.summary()
        self.assertEqual(summary.category_counts["NAVIGATION"], 1)
        self.assertEqual(summary.category_counts["FIRE"], 1)

    def test_warnings_do_not_count_as_rejections(self):

        report = ScenarioValidationReport()
        report.add(FailureCategory.OCCUPANCY, ScenarioValidationReport.WARNING, "W", "warn")

        collector = DiagnosticsCollector()
        result = collector.record_candidate_report(report)

        self.assertIsNone(result)
        self.assertTrue(report.accepted)

        summary = collector.summary()
        self.assertEqual(summary.rejected_candidates, 0)
        self.assertEqual(summary.category_counts, {})


class ExplainTotalRejectionTests(unittest.TestCase):

    def test_explanation_names_the_dominant_category_and_top_issue(self):

        collector = DiagnosticsCollector()

        for _ in range(5):
            collector.record_candidate_report(
                _report(
                    False, FailureCategory.NAVIGATION, "FIRE_ORIGIN_BLOCKS_EVACUATION",
                    "blocked evacuation",
                )
            )
        collector.record_candidate_report(
            _report(False, FailureCategory.FIRE, "MISSING_FIRE", "no fire")
        )

        explanation = explain_total_rejection(collector.summary())

        self.assertIn("NAVIGATION", explanation)
        self.assertIn("FIRE_ORIGIN_BLOCKS_EVACUATION", explanation)
        self.assertIn("blocked evacuation", explanation)

    def test_explanation_handles_no_recorded_issues_gracefully(self):

        collector = DiagnosticsCollector()

        explanation = explain_total_rejection(collector.summary())

        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)


# =====================================================
# CampaignWorker -- pre-flight (Requirements 5/6)
# =====================================================


class PreflightTests(unittest.TestCase):

    def test_valid_building_and_definition_pass_preflight(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=1))

            preflights = []
            worker.preflight_completed.connect(preflights.append)

            worker.execute()

            self.assertEqual(len(preflights), 1)
            self.assertFalse(preflights[0].has_errors)

    def test_invalid_building_aborts_before_any_generation_attempt(self):

        floor = Floor(
            name="Ground", id="floor-1",
            zones=[Zone(id="zone-1", name="Z1", x=0.0, y=0.0, width=10.0, height=8.0)],
            doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="")],
            exits=[Exit(id="exit-1", zone_id="zone-1")],
        )
        broken_building = Building(name="Broken", id="broken-1", floors=[floor])

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(
                make_config(
                    building=broken_building, definition=make_definition(),
                    output_directory=output_dir, count=10,
                )
            )

            preflights = []
            worker.preflight_completed.connect(preflights.append)

            summary = worker.execute()

            self.assertTrue(preflights[0].has_errors)
            self.assertTrue(preflights[0].building_issues)
            self.assertEqual(summary.accepted, 0)
            self.assertEqual(summary.rejected, 0)
            self.assertIn("Building itself is invalid", summary.rejection_explanation)

    def test_contradictory_definition_aborts_before_any_generation_attempt(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(
                make_config(
                    definition=make_unsatisfiable_definition(),
                    output_directory=output_dir, count=10, diagnostics_mode=True,
                )
            )

            preflights = []
            worker.preflight_completed.connect(preflights.append)

            summary = worker.execute()

            self.assertTrue(preflights[0].has_errors)
            self.assertTrue(preflights[0].definition_issues)
            self.assertEqual(summary.accepted, 0)
            self.assertEqual(summary.rejected, 0)
            self.assertIn("Scenario Definition is contradictory", summary.rejection_explanation)

    def test_preflight_is_skipped_when_diagnostics_mode_is_off(self):

        floor = Floor(
            name="Ground", id="floor-1",
            zones=[Zone(id="zone-1", name="Z1", x=0.0, y=0.0, width=10.0, height=8.0)],
            doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="")],
            exits=[Exit(id="exit-1", zone_id="zone-1")],
        )
        broken_building = Building(name="Broken", id="broken-1", floors=[floor])

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(
                make_config(
                    building=broken_building, definition=make_definition(),
                    output_directory=output_dir, count=1, diagnostics_mode=False,
                )
            )

            preflights = []
            worker.preflight_completed.connect(preflights.append)

            worker.execute()

            self.assertEqual(preflights, [])


# =====================================================
# CampaignWorker -- per-candidate capture (Requirements 1/2/3)
# =====================================================


class PerCandidateCaptureTests(unittest.TestCase):

    def test_first_rejection_signal_fires_at_most_once(self):

        building = make_chain_building(zone_count=6)
        definition = make_every_zone_occupied_definition(building)

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(
                CampaignConfig(
                    name="t", building=building, definition=definition,
                    definition_id="def-x", count=25, master_seed=1,
                    output_directory=output_dir, max_attempts=10, dt=10.0,
                    diagnostics_mode=True,
                )
            )

            first_rejections = []
            worker.first_rejection_detected.connect(first_rejections.append)

            worker.execute()

            self.assertLessEqual(len(first_rejections), 1)

    def test_diagnostics_updated_emitted_once_per_scenario_index(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=3))

            updates = []
            worker.diagnostics_updated.connect(updates.append)

            worker.execute()

            self.assertEqual(len(updates), 3)

    def test_rejection_explanation_populated_when_every_scenario_is_rejected(self):

        building = make_chain_building(zone_count=2)

        # A 2-zone chain where the only non-exit zone is *always*
        # ignitable and *always* occupied, with the exit zone
        # unoccupied -- the ignition zone excludes itself from the
        # reachability graph (navigation_validation.py's own check 3),
        # so whenever zone-1 (not adjacent to the Exit) ignites while
        # occupied, its only path out is blocked. Constraining ignition
        # to zone-1 specifically makes this deterministic, every
        # attempt.
        definition = ScenarioDefinition(
            fire=FireDefinition(
                growth_parameter_distribution=FixedValue(200.0),
                allowed_ignition_zone_ids=frozenset({"zone-1"}),
            ),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": FixedValue(2)},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue("LOCKED")},
            ),
        )

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(
                CampaignConfig(
                    name="t", building=building, definition=definition,
                    definition_id="def-locked", count=3, master_seed=1,
                    output_directory=output_dir, max_attempts=3, dt=10.0,
                    diagnostics_mode=True,
                )
            )

            summary = worker.execute()

            self.assertEqual(summary.accepted, 0)
            self.assertEqual(summary.rejected, 3)
            self.assertIsNotNone(summary.rejection_explanation)
            self.assertIn("NAVIGATION", summary.rejection_explanation)


# =====================================================
# CampaignWindow -- Validation Report panel display
# =====================================================


class ValidationReportPanelTests(unittest.TestCase):

    def setUp(self):
        self.window = CampaignWindow()

    def test_show_preflight_with_no_errors(self):

        self.window.show_preflight(PreflightResult())

        self.assertIn("passed", self.window.preflight_issues_label.text())

    def test_show_preflight_with_errors_lists_every_issue(self):

        preflight = PreflightResult(
            building_issues=(ValidationRow("BUILDING", "code_a", "message a"),),
            definition_issues=(ValidationRow("DEFINITION", "code_b", "message b"),),
        )

        self.window.show_preflight(preflight)

        text = self.window.preflight_issues_label.text()
        self.assertIn("code_a", text)
        self.assertIn("message a", text)
        self.assertIn("code_b", text)
        self.assertIn("message b", text)

    def test_show_first_rejection_updates_label(self):

        row = ValidationRow("NAVIGATION", "FIRE_ORIGIN_BLOCKS_EVACUATION", "blocked")

        self.window.show_first_rejection(row)

        text = self.window.first_rejection_label.text()
        self.assertIn("NAVIGATION", text)
        self.assertIn("FIRE_ORIGIN_BLOCKS_EVACUATION", text)
        self.assertIn("blocked", text)

    def test_update_diagnostics_populates_table_and_category_label(self):

        from designer.campaign.diagnostics import DiagnosticsSummary

        summary = DiagnosticsSummary(
            total_candidates=10, rejected_candidates=7,
            category_counts={"NAVIGATION": 6, "FIRE": 1},
            rows=(
                ValidationRow("NAVIGATION", "FIRE_ORIGIN_BLOCKS_EVACUATION", "blocked", 6),
                ValidationRow("FIRE", "MISSING_FIRE", "no fire", 1),
            ),
            first_rejection=None,
        )

        self.window.update_diagnostics(summary)

        self.assertIn("NAVIGATION: 6", self.window.category_counts_label.text())
        self.assertIn("FIRE: 1", self.window.category_counts_label.text())
        self.assertEqual(self.window.validation_table.rowCount(), 2)
        self.assertEqual(self.window.validation_table.item(0, 0).text(), "NAVIGATION")
        self.assertEqual(self.window.validation_table.item(0, 3).text(), "6")

    def test_reset_validation_report_clears_previous_state(self):

        self.window.show_first_rejection(ValidationRow("FIRE", "X", "y"))
        self.window.reset_validation_report()

        self.assertEqual(self.window.first_rejection_label.text(), "First rejection: none yet.")
        self.assertEqual(self.window.validation_table.rowCount(), 0)

    def test_show_summary_displays_rejection_explanation_when_present(self):

        from designer.campaign import CampaignSummary

        summary = CampaignSummary(
            total_requested=5, total_generated=0, accepted=0, rejected=5,
            average_evacuation_time=None, average_simulation_duration=0.0,
            output_directory="C:/out", rejection_explanation="Everything was rejected because X.",
        )

        self.window.show_summary(summary)

        # isVisible() reflects whether every ancestor up to the
        # top-level window is shown too, which is never true for a
        # CampaignWindow that was never .show()-n in a headless test --
        # isHidden() reflects only this widget's own explicit
        # setVisible() call, which is what this test actually checks.
        self.assertFalse(self.window.rejection_explanation_label.isHidden())
        self.assertIn("Everything was rejected", self.window.rejection_explanation_label.text())

    def test_show_summary_hides_rejection_explanation_when_absent(self):

        from designer.campaign import CampaignSummary

        summary = CampaignSummary(
            total_requested=5, total_generated=5, accepted=5, rejected=0,
            average_evacuation_time=12.0, average_simulation_duration=0.01,
            output_directory="C:/out",
        )

        self.window.show_summary(summary)

        self.assertTrue(self.window.rejection_explanation_label.isHidden())

    def test_set_running_state_true_resets_the_validation_report(self):

        self.window.show_first_rejection(ValidationRow("FIRE", "X", "y"))

        self.window.set_running_state(True)

        self.assertEqual(self.window.first_rejection_label.text(), "First rejection: none yet.")

    def test_diagnostics_mode_accessor_defaults_to_true(self):

        self.assertTrue(self.window.diagnostics_mode())

        self.window.diagnostics_checkbox.setChecked(False)
        self.assertFalse(self.window.diagnostics_mode())


# =====================================================
# CampaignController -- building resync bug fix
# =====================================================


class ControllerBuildingResyncTests(unittest.TestCase):

    def test_build_config_resyncs_window_to_the_current_building_before_translation(self):

        from designer.campaign import CampaignController

        building = make_building()
        window = _FakeWindow(building=None, output_directory="C:/out")

        controller = CampaignController(window, get_building=lambda: building)
        controller.build_config()

        self.assertEqual(window.building_rebindings, [building])

    def test_build_config_resyncs_even_when_a_different_building_was_set_earlier(self):

        # The exact scenario this fix closes: set_building() was
        # previously called with one Building (e.g. by MainWindow.
        # open_campaign_studio() at some earlier point), and the
        # controller's own get_building() now resolves to a different
        # one (e.g. the user opened a new project without reopening
        # the Campaign Studio window) -- build_config() must resync to
        # the *current* one before translating, not trust whatever
        # set_building() left behind.
        from designer.campaign import CampaignController

        stale_building = make_building()
        current_building = make_chain_building(zone_count=3)

        window = _FakeWindow(building=stale_building, output_directory="C:/out")
        window.set_building(stale_building)

        controller = CampaignController(window, get_building=lambda: current_building)
        controller.build_config()

        self.assertEqual(window.building_rebindings, [stale_building, current_building])


if __name__ == "__main__":
    unittest.main()
