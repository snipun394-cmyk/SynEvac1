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
    WeightedOptions,
)

from campaign_feasibility import (
    CampaignFeasibilityReport,
    ZoneFeasibilityResult,
    ZoneFeasibilityStatus,
    analyze_campaign_feasibility,
)

from designer.campaign.campaign_worker import CampaignConfig, CampaignWorker

from tests.test_campaign_studio import _TempOutputDir


# =====================================================
# Fixtures
# =====================================================


def _two_zone_building():

    # zone-1 --door-1-- zone-2 --exit-1--> Outside.
    door = Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")
    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Room 1", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-2", name="Room 2", x=10.0, y=0.0, width=8.0, height=8.0),
        ],
        doors=[door],
        exits=[Exit(id="exit-1", zone_id="zone-2")],
    )
    return Building(name="Two Zone", id="b-1", floors=[floor])


def _three_zone_branching_building():

    # zone-1 --door-1-- zone-2 --exit-1--> Outside (zone-1's only route).
    # zone-3 is a disconnected, unrelated zone with no doors -- used
    # only as a fire-eligible id that can never affect zone-1's own
    # reachability, giving a clean SAFE/LETHAL split for probability
    # tests.
    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Room 1", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-2", name="Room 2", x=10.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-3", name="Room 3", x=20.0, y=0.0, width=8.0, height=8.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-2")],
    )
    return Building(name="Branching", id="b-2", floors=[floor])


def _occupant_only(zone_id, count=1):

    return OccupantDefinition(
        occupancy_distribution={zone_id: FixedValue(count)},
        behaviour_profile_distribution={zone_id: FixedValue("Staff_Default")},
    )


def _fire(**overrides):

    defaults = dict(growth_parameter_distribution=FixedValue(200.0))
    defaults.update(overrides)
    return FireDefinition(**defaults)


# =====================================================
# Part C/D -- optimistic/pessimistic bounds
# =====================================================


class OptimisticBoundTests(unittest.TestCase):

    def test_zone_unreachable_even_optimistically_is_blocking_error(self):

        # Part J.1 -- door-1 is pinned LOCKED (FixedValue, zero
        # sampling uncertainty) -- no combination of sampled states can
        # ever open it, so zone-1 must be a proven, blocking
        # ZERO-FEASIBILITY finding.
        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue("LOCKED")},
            ),
        )

        report = analyze_campaign_feasibility(building, definition)

        self.assertTrue(report.has_errors)
        self.assertEqual(len(report.zone_results), 1)

        result = report.zone_results[0]
        self.assertEqual(result.occupied_zone_id, "zone-1")
        self.assertEqual(result.status, ZoneFeasibilityStatus.ERROR)
        self.assertFalse(result.optimistic_reachable)
        self.assertFalse(result.pessimistic_reachable)
        self.assertIn("zone-1", result.explanation)
        self.assertIn("favorable", result.explanation.lower())

    def test_reachable_zone_that_stays_reachable_pessimistically_passes_that_dimension(self):

        # Part J.2 -- door-1 has no distribution at all (falls back to
        # the Building's own default, normally_open=True) -- guaranteed
        # traversable under every bound, so zone-1 must be reachable
        # both optimistically and pessimistically. Fire is constrained
        # to zone-1 itself (excluded from its own fire-cut analysis, see
        # analysis.py's own documented reasoning) so this test isolates
        # the engineering-state dimension only, unaffected by the fire
        # dimension's own default (every-zone-eligible) behavior.
        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=_occupant_only("zone-1"),
        )

        report = analyze_campaign_feasibility(building, definition)
        result = report.zone_results[0]

        self.assertTrue(result.optimistic_reachable)
        self.assertTrue(result.pessimistic_reachable)
        self.assertFalse(report.has_errors)

    def test_zone_reachable_optimistically_but_not_pessimistically_is_a_warning_not_a_block(self):

        # Part J.3 -- door-1's WeightedOptions assigns positive weight
        # to both OPEN (traversable) and LOCKED (not) -- optimistically
        # traversable (some weight on OPEN), but NOT pessimistically
        # (LOCKED has positive weight too, so it is not GUARANTEED
        # traversable). This must surface as a non-blocking WARNING,
        # never an ERROR -- Phase 1 must not fabricate a proof it
        # cannot make. Fire constrained to zone-1 itself for the same
        # isolation reason as the test above.
        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(
                door_state_distribution={
                    "door-1": WeightedOptions({"OPEN": 0.6, "LOCKED": 0.4}),
                },
            ),
        )

        report = analyze_campaign_feasibility(building, definition)
        result = report.zone_results[0]

        self.assertTrue(result.optimistic_reachable)
        self.assertFalse(result.pessimistic_reachable)
        self.assertEqual(result.status, ZoneFeasibilityStatus.WARNING)
        self.assertFalse(report.has_errors)
        self.assertTrue(report.has_warnings)


# =====================================================
# Part E -- fire-origin cut analysis (SAFE/LETHAL)
# =====================================================


class FireCutAnalysisTests(unittest.TestCase):

    def test_safe_fire_origin_does_not_disconnect_the_occupied_zone(self):

        # Part J.4 -- zone-3 is a disconnected, unrelated zone; igniting
        # it can never affect zone-1's reachability at all.
        building = _three_zone_branching_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-3"})),
            occupant=_occupant_only("zone-1"),
        )

        report = analyze_campaign_feasibility(building, definition)
        result = report.zone_results[0]

        self.assertIn("zone-3", result.safe_fire_zone_ids)
        self.assertNotIn("zone-3", result.lethal_fire_zone_ids)
        self.assertEqual(result.lethal_fire_probability, 0.0)

    def test_lethal_fire_origin_disconnects_the_occupied_zone(self):

        # Part J.5 -- zone-2 is zone-1's ONLY route to Outside; igniting
        # it must classify as LETHAL.
        building = _three_zone_branching_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-2"})),
            occupant=_occupant_only("zone-1"),
        )

        report = analyze_campaign_feasibility(building, definition)
        result = report.zone_results[0]

        self.assertIn("zone-2", result.lethal_fire_zone_ids)
        self.assertNotIn("zone-2", result.safe_fire_zone_ids)
        self.assertEqual(result.lethal_fire_probability, 1.0)
        self.assertEqual(result.status, ZoneFeasibilityStatus.ERROR)


# =====================================================
# Part F -- analytical probability
# =====================================================


class FireProbabilityTests(unittest.TestCase):

    def test_uniform_fire_distribution_with_mixed_safe_and_lethal_origins(self):

        # Part J.6 -- no ignition_zone_preference stated -> uniform
        # over the eligible set {zone-2 (LETHAL), zone-3 (SAFE)} ->
        # P(lethal) = 1/2 exactly.
        building = _three_zone_branching_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-2", "zone-3"})),
            occupant=_occupant_only("zone-1"),
        )

        report = analyze_campaign_feasibility(building, definition)
        result = report.zone_results[0]

        self.assertEqual(result.lethal_fire_zone_ids, frozenset({"zone-2"}))
        self.assertEqual(result.safe_fire_zone_ids, frozenset({"zone-3"}))
        self.assertAlmostEqual(result.lethal_fire_probability, 0.5, places=9)
        self.assertEqual(result.status, ZoneFeasibilityStatus.WARNING)

    def test_weighted_fire_distribution_with_mixed_safe_and_lethal_origins(self):

        # Part J.7 -- an explicit ignition_zone_preference is sampled
        # from DIRECTLY (generator.py's own documented behavior, not
        # intersected with the eligible set) -> P(lethal) must equal
        # weight(zone-2) / (weight(zone-2) + weight(zone-3)) = 0.3/1.0.
        building = _three_zone_branching_building()
        definition = ScenarioDefinition(
            fire=_fire(
                ignition_zone_preference=WeightedOptions({"zone-2": 0.3, "zone-3": 0.7}),
            ),
            occupant=_occupant_only("zone-1"),
        )

        report = analyze_campaign_feasibility(building, definition)
        result = report.zone_results[0]

        self.assertAlmostEqual(result.lethal_fire_probability, 0.3, places=9)
        self.assertEqual(result.status, ZoneFeasibilityStatus.WARNING)

    def test_100_percent_lethal_probability_is_a_blocking_error(self):

        # Part J.8 -- both eligible zones LETHAL -> P=1.0 -> must be a
        # blocking ERROR (Part H, Case 2), not merely a WARNING.
        building = _three_zone_branching_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-2"})),
            occupant=_occupant_only("zone-1"),
        )

        report = analyze_campaign_feasibility(building, definition)

        self.assertTrue(report.has_errors)
        self.assertEqual(report.zone_results[0].status, ZoneFeasibilityStatus.ERROR)
        self.assertEqual(report.zone_results[0].lethal_fire_probability, 1.0)


# =====================================================
# Part B -- potentially occupied zones (precision)
# =====================================================


class PotentiallyOccupiedZoneTests(unittest.TestCase):

    def test_zone_with_zero_fixed_occupancy_is_excluded_from_analysis(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(), occupant=_occupant_only("zone-1", count=0),
        )

        report = analyze_campaign_feasibility(building, definition)

        self.assertEqual(report.zone_results, ())

    def test_zone_not_present_in_occupancy_distribution_is_excluded(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(fire=_fire(), occupant=OccupantDefinition())

        report = analyze_campaign_feasibility(building, definition)

        self.assertEqual(report.zone_results, ())

    def test_occupancy_distribution_referencing_a_nonexistent_zone_is_skipped_not_crashed(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(), occupant=_occupant_only("zone-does-not-exist"),
        )

        report = analyze_campaign_feasibility(building, definition)

        self.assertEqual(report.zone_results, ())

    def test_weighted_occupancy_distribution_with_only_zero_weight_is_excluded(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": WeightedOptions({0: 1.0})},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
        )

        report = analyze_campaign_feasibility(building, definition)

        self.assertEqual(report.zone_results, ())

    def test_weighted_occupancy_distribution_with_any_positive_count_is_included(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": WeightedOptions({0: 0.9, 2: 0.1})},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
        )

        report = analyze_campaign_feasibility(building, definition)

        self.assertEqual(len(report.zone_results), 1)
        self.assertEqual(report.zone_results[0].occupied_zone_id, "zone-1")


# =====================================================
# Part G -- result model
# =====================================================


class ResultModelTests(unittest.TestCase):

    def test_empty_report_has_no_errors_or_warnings(self):

        report = CampaignFeasibilityReport()

        self.assertFalse(report.has_errors)
        self.assertFalse(report.has_warnings)
        self.assertEqual(report.error_results, ())
        self.assertEqual(report.warning_results, ())

    def test_error_and_warning_results_are_correctly_partitioned(self):

        error_result = ZoneFeasibilityResult(
            occupied_zone_id="z1", optimistic_reachable=False, pessimistic_reachable=False,
            status=ZoneFeasibilityStatus.ERROR, explanation="e",
        )
        warning_result = ZoneFeasibilityResult(
            occupied_zone_id="z2", optimistic_reachable=True, pessimistic_reachable=False,
            status=ZoneFeasibilityStatus.WARNING, explanation="w",
        )
        ok_result = ZoneFeasibilityResult(
            occupied_zone_id="z3", optimistic_reachable=True, pessimistic_reachable=True,
            status=ZoneFeasibilityStatus.OK, explanation="ok",
        )

        report = CampaignFeasibilityReport(zone_results=(error_result, warning_result, ok_result))

        self.assertTrue(report.has_errors)
        self.assertTrue(report.has_warnings)
        self.assertEqual(report.error_results, (error_result,))
        self.assertEqual(report.warning_results, (warning_result,))

    def test_analyze_with_no_building_returns_an_empty_report(self):

        definition = ScenarioDefinition(fire=_fire(), occupant=_occupant_only("zone-1"))

        report = analyze_campaign_feasibility(None, definition)

        self.assertEqual(report.zone_results, ())
        self.assertFalse(report.has_errors)


# =====================================================
# Part J.9 -- reproduces the exact previously-observed failure class
# =====================================================


class PreviousFailureClassReproductionTests(unittest.TestCase):

    def test_chain_topology_fire_only_blocking_is_caught_before_generation(self):

        # The exact structural shape traced in the zero-generation
        # investigation: a corridor building, every zone occupied, fire
        # ignition zone unconstrained (defaults to every zone) --
        # zone-1 (farthest from the Exit) has every OTHER zone on its
        # only path out, so its entire fire-eligible set is LETHAL.
        from tests.test_campaign_diagnostics import (
            make_chain_building, make_every_zone_occupied_definition,
        )

        building = make_chain_building(zone_count=5)
        definition = make_every_zone_occupied_definition(building)

        report = analyze_campaign_feasibility(building, definition)

        zone_1_result = next(r for r in report.zone_results if r.occupied_zone_id == "zone-1")

        self.assertEqual(zone_1_result.status, ZoneFeasibilityStatus.ERROR)
        self.assertEqual(zone_1_result.lethal_fire_probability, 1.0)
        self.assertTrue(report.has_errors)


# =====================================================
# Part J.10 -- scenario_generator remains connectivity-blind
# =====================================================


class GeneratorUnaffectedTests(unittest.TestCase):

    def test_scenario_generator_module_does_not_import_navigation_or_feasibility_code(self):

        # Confirms this phase never inserted reachability logic into
        # the Generator -- a structural, source-level check, not merely
        # "the tests still pass" (which could be true even if an unused
        # import were accidentally added).
        import inspect

        import scenario_generator.generator as generator_module

        source = inspect.getsource(generator_module)

        self.assertNotIn("navigation", source)
        self.assertNotIn("campaign_feasibility", source)
        self.assertNotIn("bfs_reachable", source)

    def test_generate_scenario_output_is_unaffected_by_the_new_package_existing(self):

        # A deterministic, pre-existing-shape smoke check: generating a
        # candidate against a Definition/Building combination this
        # phase's own tests already reason about produces the exact
        # same kind of Scenario object as before, with no new fields,
        # no altered door/exit/stair state resolution.
        from scenario_generator import GenerationRequest, generate_scenario

        building = _two_zone_building()
        definition = ScenarioDefinition(fire=_fire(), occupant=_occupant_only("zone-1"))

        request = GenerationRequest(
            definition=definition, definition_id="def-x", building=building,
            seed=1, attempt_index=0,
        )

        candidate = generate_scenario(request)

        self.assertEqual(len(candidate.occupants), 1)
        self.assertEqual(candidate.occupants[0].zone_id, "zone-1")


# =====================================================
# Part J.8/J.9 -- CampaignWorker integration: blocked before any
# generation attempt
# =====================================================


class CampaignIntegrationTests(unittest.TestCase):

    def test_100_percent_lethal_fire_probability_blocks_the_whole_campaign(self):

        # Part J.8, at the CampaignWorker level (not just the analysis
        # module in isolation): a 100%-lethal fire-origin distribution
        # must produce a blocking pre-flight ERROR and an empty
        # CampaignSummary, exactly like an invalid Building/Definition
        # already does.
        building = _three_zone_branching_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-2"})),
            occupant=_occupant_only("zone-1"),
        )

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(
                CampaignConfig(
                    name="t", building=building, definition=definition,
                    definition_id="def-lethal", count=5, master_seed=1,
                    output_directory=output_dir, max_attempts=5, dt=10.0,
                    diagnostics_mode=True,
                )
            )

            preflights = []
            worker.preflight_completed.connect(preflights.append)

            summary = worker.execute()

            self.assertTrue(preflights[0].has_errors)
            self.assertTrue(preflights[0].feasibility_issues)
            self.assertEqual(summary.accepted, 0)
            self.assertEqual(summary.rejected, 0)
            self.assertIn("proven infeasible", summary.rejection_explanation)

    def test_zero_feasibility_configuration_never_calls_generate_scenario(self):

        # Part J.9, the decisive proof: replacing scenario_generator.
        # generate_scenario() with a function that raises, then running
        # the full campaign -- if generation were ever attempted even
        # once before the feasibility block took effect, this test
        # would fail with the raised AssertionError instead of passing.
        import designer.campaign.campaign_worker as campaign_worker_module

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue("LOCKED")},
            ),
        )

        def _must_not_be_called(request):
            raise AssertionError(
                "generate_scenario() must not be called when the feasibility "
                "preflight has already proven zero feasible generation space."
            )

        original = campaign_worker_module.generate_scenario
        campaign_worker_module.generate_scenario = _must_not_be_called

        try:

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
                self.assertEqual(summary.rejected, 0)
                self.assertIn("proven infeasible", summary.rejection_explanation)

        finally:

            campaign_worker_module.generate_scenario = original


if __name__ == "__main__":
    unittest.main()
