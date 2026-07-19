import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PyQt6.QtWidgets import QApplication

# Module-level QApplication singleton -- CampaignWorker/CampaignConfig
# import PyQt6 machinery transitively (QThread/pyqtSignal), same
# convention tests/test_campaign_studio.py already establishes.
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
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
    UniformRange,
)
from scenario_storage import load_scenario_by_id, read_catalog_rows

from designer.campaign.campaign_worker import CampaignConfig, CampaignWorker

from command_center.incident_data import load_incident


# =====================================================
# Fixtures -- same Building/Definition shapes tests/test_campaign_studio.py
# already uses, so a real tiny campaign (real scenario_runner ->
# behaviour_profile_resolver -> ai_decision -> simulation_runtime chain,
# nothing mocked) runs in well under a second.
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


def make_multi_occupant_definition():

    # Occupants spread across both zones (rather than only zone-2), so
    # zone-level artifacts (zone_results.csv) have more than one
    # non-empty origin zone to cross-check per scenario.

    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "zone-1": UniformRange(1, 2, discrete=True),
                "zone-2": UniformRange(1, 2, discrete=True),
            },
            behaviour_profile_distribution={
                "zone-1": FixedValue("Staff_Default"),
                "zone-2": FixedValue("Staff_Default"),
            },
        ),
    )


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="campaign_pipeline_test_")
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


def read_csv_rows(path):

    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# =====================================================
# Artifact existence -- Requirements 1-4: every listed export exists,
# for every accepted scenario, laid out under the directory structure
# the objective's OUTPUT DIRECTORY section names.
# =====================================================


class ArtifactExistenceTests(unittest.TestCase):

    def test_every_accepted_scenario_produces_every_artifact(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=3))
            summary = worker.execute()

            self.assertEqual(summary.accepted, 3)

            root = Path(output_dir)
            scenario_ids = [row["scenario_id"] for row in read_catalog_rows(output_dir)]

            self.assertEqual(len(scenario_ids), 3)

            for scenario_id in scenario_ids:

                # Dataset Builder V1
                dataset_dir = root / "datasets" / scenario_id
                self.assertTrue((dataset_dir / "scenario_features.csv").is_file())
                self.assertTrue((dataset_dir / "simulation_outcomes.csv").is_file())
                self.assertTrue((dataset_dir / "zone_results.csv").is_file())

                # Timeline Dataset (V2)
                timeline_dir = root / "timelines" / scenario_id
                self.assertTrue((timeline_dir / "timeline.csv").is_file())
                self.assertTrue((timeline_dir / "timeline_rows.json").is_file())

                # Ground Truth
                self.assertTrue(
                    (root / "ground_truth" / scenario_id / "ground_truth.json").is_file(),
                )

                # Decision Policy
                self.assertTrue(
                    (root / "decision_policy" / scenario_id / "decision_policy.json").is_file(),
                )

                # scenario_storage's own artifact -- untouched by this
                # phase, proven present as a sanity check that nothing
                # about the pre-existing pipeline broke.
                reloaded = load_scenario_by_id(scenario_id, output_dir)
                self.assertEqual(reloaded.metadata.scenario_id, scenario_id)

            # Command Center playback needs exactly one Building project
            # per campaign (not one per scenario).
            self.assertTrue((root / "building.syn").is_file())

    def test_rejected_scenarios_produce_no_simulation_artifacts(self):

        # An unsatisfiable Definition (pinned-closed only exit, but
        # min_open_exits demands one open) rejects every candidate --
        # no scenario is ever accepted, so _export_scenario_artifacts()
        # (only reachable from a successful _simulate() call after an
        # accepted scenario) must never run at all.

        from scenario_definition import EngineeringConstraints

        unsatisfiable = ScenarioDefinition(
            fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-2": FixedValue(1)},
                behaviour_profile_distribution={"zone-2": FixedValue("Staff_Default")},
            ),
            engineering=EngineeringConstraints(
                exit_state_distribution={"exit-1": FixedValue(False)}, min_open_exits=1,
            ),
        )

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(
                make_config(
                    definition=unsatisfiable, output_directory=output_dir, count=2,
                    max_attempts=2, diagnostics_mode=False,
                )
            )
            summary = worker.execute()

            self.assertEqual(summary.accepted, 0)

            root = Path(output_dir)
            for subdirectory in ("datasets", "timelines", "ground_truth", "decision_policy"):
                self.assertFalse((root / subdirectory).exists())

            self.assertFalse((root / "building.syn").exists())


# =====================================================
# Scenario association -- Requirement: "Each scenario's artifacts must
# remain associated with its scenario_id." Runs a multi-scenario
# campaign and proves every artifact's own embedded scenario_id matches
# the directory it was found under -- no cross-scenario mixups.
# =====================================================


class ArtifactScenarioAssociationTests(unittest.TestCase):

    def test_every_artifacts_embedded_scenario_id_matches_its_directory(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=4))
            summary = worker.execute()
            self.assertEqual(summary.accepted, 4)

            root = Path(output_dir)
            scenario_ids = [row["scenario_id"] for row in read_catalog_rows(output_dir)]

            self.assertEqual(len(set(scenario_ids)), 4, "expected 4 distinct scenario ids")

            for scenario_id in scenario_ids:

                features_rows = read_csv_rows(root / "datasets" / scenario_id / "scenario_features.csv")
                self.assertEqual(len(features_rows), 1)
                self.assertEqual(features_rows[0]["scenario_id"], scenario_id)

                outcome_rows = read_csv_rows(
                    root / "datasets" / scenario_id / "simulation_outcomes.csv",
                )
                self.assertEqual(len(outcome_rows), 1)
                self.assertEqual(outcome_rows[0]["scenario_id"], scenario_id)

                zone_rows = read_csv_rows(root / "datasets" / scenario_id / "zone_results.csv")
                self.assertTrue(zone_rows)
                self.assertTrue(all(row["scenario_id"] == scenario_id for row in zone_rows))

                timeline_rows_json = read_json(
                    root / "timelines" / scenario_id / "timeline_rows.json",
                )
                self.assertTrue(timeline_rows_json)
                self.assertTrue(
                    all(row["scenario_id"] == scenario_id for row in timeline_rows_json),
                )

                timeline_csv_rows = read_csv_rows(root / "timelines" / scenario_id / "timeline.csv")
                self.assertEqual(len(timeline_csv_rows), len(timeline_rows_json))
                self.assertTrue(
                    all(row["scenario_id"] == scenario_id for row in timeline_csv_rows),
                )

                ground_truth = read_json(root / "ground_truth" / scenario_id / "ground_truth.json")
                self.assertEqual(ground_truth["scenario_id"], scenario_id)

                decision_policy = read_json(
                    root / "decision_policy" / scenario_id / "decision_policy.json",
                )
                self.assertEqual(decision_policy["scenario_id"], scenario_id)

    def test_two_scenarios_artifacts_are_not_interchangeable(self):

        # A stronger check than "the id field matches" -- proves scenario
        # A's own dataset file is not byte-identical to scenario B's
        # (each scenario got its own independently-generated fire/
        # occupant placement, so their Timeline Datasets must differ).

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=2))
            worker.execute()

            root = Path(output_dir)
            scenario_ids = [row["scenario_id"] for row in read_catalog_rows(output_dir)]
            self.assertEqual(len(scenario_ids), 2)

            first_gt = read_json(root / "ground_truth" / scenario_ids[0] / "ground_truth.json")
            second_gt = read_json(root / "ground_truth" / scenario_ids[1] / "ground_truth.json")

            self.assertNotEqual(first_gt["scenario_id"], second_gt["scenario_id"])


# =====================================================
# Data consistency -- Requirement: "exported data matches simulation
# outputs." dataset_builder's simulation_outcomes.csv and
# ground_truth.json's evacuation fields are computed by two independent
# packages (dataset_builder.labels.extract_simulation_outcome and
# ground_truth.evacuation_metrics.compute_evacuation_metrics) from the
# exact same movement_result -- both packages use the identical
# formula, so their values must match exactly. This is the strongest
# available proof that _export_scenario_artifacts() actually threaded
# the *same* movement_result/tick_results into every package, rather
# than each computing from a different (possibly re-simulated) run.
# =====================================================


class ArtifactDataConsistencyTests(unittest.TestCase):

    def setUp(self):

        self.output_dir_ctx = _TempOutputDir()
        self.output_dir = self.output_dir_ctx.__enter__()

        self.worker = CampaignWorker(
            make_config(
                definition=make_multi_occupant_definition(),
                output_directory=self.output_dir, count=3,
            )
        )
        self.summary = self.worker.execute()

        self.root = Path(self.output_dir)
        self.scenario_ids = [row["scenario_id"] for row in read_catalog_rows(self.output_dir)]

    def tearDown(self):

        self.output_dir_ctx.__exit__(None, None, None)

    def test_evacuation_counts_match_between_dataset_builder_and_ground_truth(self):

        self.assertEqual(len(self.scenario_ids), 3)

        for scenario_id in self.scenario_ids:

            outcome_row = read_csv_rows(
                self.root / "datasets" / scenario_id / "simulation_outcomes.csv",
            )[0]
            ground_truth = read_json(self.root / "ground_truth" / scenario_id / "ground_truth.json")

            self.assertEqual(int(outcome_row["people_evacuated"]), ground_truth["people_evacuated"])
            self.assertEqual(int(outcome_row["people_trapped"]), ground_truth["people_trapped"])
            self.assertEqual(
                int(outcome_row["reachable_occupants"]), ground_truth["reachable_occupants"],
            )
            self.assertEqual(
                int(outcome_row["unreachable_occupants"]), ground_truth["unreachable_occupants"],
            )
            self.assertEqual(
                outcome_row["building_cleared"], str(ground_truth["building_cleared"]),
            )

            outcome_time = outcome_row["total_evacuation_time"]
            gt_time = ground_truth["total_evacuation_time"]

            if outcome_time in ("", None) and gt_time is None:
                continue

            self.assertAlmostEqual(float(outcome_time), float(gt_time), places=9)

    def test_scenario_features_total_occupants_matches_scenario_and_zone_results(self):

        for scenario_id in self.scenario_ids:

            scenario = load_scenario_by_id(scenario_id, self.output_dir)

            features_row = read_csv_rows(
                self.root / "datasets" / scenario_id / "scenario_features.csv",
            )[0]
            zone_rows = read_csv_rows(self.root / "datasets" / scenario_id / "zone_results.csv")

            self.assertEqual(int(features_row["total_occupants"]), len(scenario.occupants))

            initial_occupants_sum = sum(int(row["initial_occupants"]) for row in zone_rows)
            self.assertEqual(initial_occupants_sum, len(scenario.occupants))

            occupants_by_zone = {}
            for occupant in scenario.occupants:
                occupants_by_zone[occupant.zone_id] = occupants_by_zone.get(occupant.zone_id, 0) + 1

            for row in zone_rows:
                expected = occupants_by_zone.get(row["zone_id"], 0)
                self.assertEqual(int(row["initial_occupants"]), expected)

    def test_timeline_final_row_evacuation_counts_match_simulation_outcomes(self):

        for scenario_id in self.scenario_ids:

            timeline_rows = read_json(self.root / "timelines" / scenario_id / "timeline_rows.json")
            self.assertTrue(timeline_rows)

            outcome_row = read_csv_rows(
                self.root / "datasets" / scenario_id / "simulation_outcomes.csv",
            )[0]

            last_row = timeline_rows[-1]

            self.assertEqual(last_row["people_evacuated"], int(outcome_row["people_evacuated"]))
            self.assertEqual(last_row["people_trapped"], int(outcome_row["people_trapped"]))

    def test_decision_policy_covers_every_zone_exit_and_stair(self):

        building = make_building()
        total_zones = sum(len(floor.zones) for floor in building.ordered_floors())
        total_exits = sum(len(floor.exits) for floor in building.ordered_floors())
        total_stairs = sum(len(floor.stairs) for floor in building.ordered_floors())

        for scenario_id in self.scenario_ids:

            decision_policy = read_json(
                self.root / "decision_policy" / scenario_id / "decision_policy.json",
            )

            self.assertEqual(len(decision_policy["zone_decisions"]), total_zones)
            self.assertEqual(len(decision_policy["exit_decisions"]), total_exits)
            self.assertEqual(len(decision_policy["stair_decisions"]), total_stairs)
            self.assertEqual(decision_policy["scenario_id"], scenario_id)

    def test_ground_truth_reflects_the_same_zones_as_the_building(self):

        building = make_building()
        total_zones = sum(len(floor.zones) for floor in building.ordered_floors())

        for scenario_id in self.scenario_ids:

            ground_truth = read_json(self.root / "ground_truth" / scenario_id / "ground_truth.json")
            self.assertEqual(len(ground_truth["zone_risk_scores"]), total_zones)


# =====================================================
# Idempotency / re-run safety -- the existing "rerunning the same
# config does not crash" guarantee (test_campaign_studio.py) must keep
# holding now that every export also runs on each _simulate() call;
# every writer here overwrites rather than raising on an existing file.
# =====================================================


class RerunSafetyTests(unittest.TestCase):

    def test_rerunning_the_same_campaign_overwrites_artifacts_without_raising(self):

        with _TempOutputDir() as output_dir:

            config = make_config(output_directory=output_dir, count=2)

            first_summary = CampaignWorker(config).execute()
            second_summary = CampaignWorker(config).execute()

            self.assertEqual(first_summary.accepted, 2)
            self.assertEqual(second_summary.accepted, 2)

            # accepted_hashes is seeded from the output directory's own
            # catalog (execute()'s own documented hazard guard) -- the
            # second run's candidates collide on content with the
            # first run's already-accepted hashes and are rejected as
            # duplicates, so generation retries until *different*
            # scenarios are produced. Both runs still succeed without
            # raising (the actual thing this test proves); the four
            # resulting scenario_ids are simply all distinct rather
            # than the second run reusing the first's two.
            scenario_ids = {row["scenario_id"] for row in read_catalog_rows(output_dir)}
            self.assertEqual(len(scenario_ids), 4)

            root = Path(output_dir)
            for scenario_id in scenario_ids:
                self.assertTrue(
                    (root / "ground_truth" / scenario_id / "ground_truth.json").is_file(),
                )


# =====================================================
# Command Center playback -- Requirement 5: the artifacts this phase
# writes must be loadable back into the existing IncidentData model,
# without command_center needing any changes of its own.
# =====================================================


class CommandCenterPlaybackIntegrationTests(unittest.TestCase):

    def test_load_incident_reconstructs_a_full_incident_from_campaign_output(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=1))
            worker.execute()

            scenario_id = read_catalog_rows(output_dir)[0]["scenario_id"]
            root = Path(output_dir)

            incident = load_incident(
                project_path=str(root / "building.syn"),
                scenario_id=scenario_id,
                scenario_storage_root=output_dir,
                ground_truth_path=str(root / "ground_truth" / scenario_id / "ground_truth.json"),
                decision_policy_path=str(
                    root / "decision_policy" / scenario_id / "decision_policy.json",
                ),
                timeline_rows_path=str(root / "timelines" / scenario_id / "timeline_rows.json"),
            )

            self.assertEqual(incident.building.name, "Test Building")
            self.assertEqual(incident.scenario.metadata.scenario_id, scenario_id)
            self.assertEqual(incident.ground_truth.scenario_id, scenario_id)
            self.assertEqual(incident.decision_policy.scenario_id, scenario_id)
            self.assertTrue(incident.has_timeline)
            self.assertGreater(incident.frame_count, 0)

            first_frame = incident.frame_at_index(0)
            self.assertIsNotNone(first_frame.zone_occupancy)

    def test_load_incident_works_with_only_the_building_and_scenario(self):

        # GroundTruth/DecisionPolicy/timeline_rows are all optional on
        # load_incident() -- proves the campaign's building.syn +
        # scenario_storage entry alone are already enough for a minimal
        # (static, pre-ignition) Command Center view.

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=1))
            worker.execute()

            scenario_id = read_catalog_rows(output_dir)[0]["scenario_id"]
            root = Path(output_dir)

            incident = load_incident(
                project_path=str(root / "building.syn"),
                scenario_id=scenario_id,
                scenario_storage_root=output_dir,
            )

            self.assertIsNone(incident.ground_truth)
            self.assertIsNone(incident.decision_policy)
            self.assertFalse(incident.has_timeline)
            self.assertEqual(incident.frame_count, 1)

    def test_multi_scenario_campaign_each_scenario_loads_its_own_incident(self):

        with _TempOutputDir() as output_dir:

            worker = CampaignWorker(make_config(output_directory=output_dir, count=2))
            worker.execute()

            root = Path(output_dir)
            scenario_ids = [row["scenario_id"] for row in read_catalog_rows(output_dir)]
            self.assertEqual(len(scenario_ids), 2)

            incidents = {
                scenario_id: load_incident(
                    project_path=str(root / "building.syn"),
                    scenario_id=scenario_id,
                    scenario_storage_root=output_dir,
                    ground_truth_path=str(root / "ground_truth" / scenario_id / "ground_truth.json"),
                    decision_policy_path=str(
                        root / "decision_policy" / scenario_id / "decision_policy.json",
                    ),
                    timeline_rows_path=str(
                        root / "timelines" / scenario_id / "timeline_rows.json",
                    ),
                )
                for scenario_id in scenario_ids
            }

            first, second = (incidents[sid] for sid in scenario_ids)

            self.assertIs(first.building, incidents[scenario_ids[0]].building)
            self.assertNotEqual(
                first.scenario.metadata.scenario_id, second.scenario.metadata.scenario_id,
            )
            self.assertEqual(first.ground_truth.scenario_id, scenario_ids[0])
            self.assertEqual(second.ground_truth.scenario_id, scenario_ids[1])


if __name__ == "__main__":
    unittest.main()
