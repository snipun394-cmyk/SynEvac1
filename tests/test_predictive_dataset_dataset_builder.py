import csv
import tempfile
import unittest
from pathlib import Path

from ai_decision.recommendation import DecisionRecommendation

from hazard.snapshot import HazardSnapshot

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from occupancy.snapshot import OccupancySnapshot

from scenario.fire import ScenarioFire
from scenario.metadata import ScenarioMetadata
from scenario.scenario import Scenario

from simulation_runtime.result import TickResult

from simulator.multi_agent_result import MultiAgentSimulationResult

from dataset_builder.timeline import TimelineRun

from predictive_dataset.candidate import enumerate_candidates
from predictive_dataset.dataset_builder import (
    DEFAULT_HORIZONS, build_candidate_dataset, build_candidate_dataset_rows, export_candidate_dataset_csv,
)


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0)],
        exits=[Exit(id="exit-1", zone_id="zone-1", capacity=2), Exit(id="exit-2", zone_id="zone-1", capacity=2)],
    )

    return Building(name="Dataset Builder Test Building", id="building-1", floors=[floor])


def make_scenario(scenario_id="scn-1"):

    metadata = ScenarioMetadata(
        scenario_id=scenario_id, definition_id="def-1", definition_content_hash="hash-1",
        generation_version="v1", seed=1, created_at="2026-01-01T00:00:00",
    )

    return Scenario(
        metadata=metadata, occupants=(),
        fire=ScenarioFire(ignition_zone_id="zone-1", ignition_floor_id="floor-1", fire_profile="Electrical", growth_parameters={}),
    )


def make_tick(time):

    return TickResult(
        time=time, fired_events=(), hazard_snapshot=HazardSnapshot(),
        occupancy_snapshot=OccupancySnapshot(), decision=DecisionRecommendation(),
    )


def make_run(scenario_id="scn-1", tick_count=2):

    building = make_building()
    scenario = make_scenario(scenario_id)
    tick_results = tuple(make_tick(float(t) * 5.0) for t in range(tick_count))

    return TimelineRun(
        scenario=scenario, building=building,
        movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
        tick_results=tick_results,
    )


class RowAssemblyTests(unittest.TestCase):

    def test_row_count_is_ticks_times_candidates_times_horizons(self):

        run = make_run(tick_count=2)
        candidate_count = len(enumerate_candidates(run.building))

        rows = build_candidate_dataset_rows(run, horizons=DEFAULT_HORIZONS)

        self.assertEqual(len(rows), 2 * candidate_count * len(DEFAULT_HORIZONS))

    def test_every_row_carries_full_identity(self):

        run = make_run(tick_count=1)
        rows = build_candidate_dataset_rows(run, horizons=(30.0,))

        for row in rows:
            self.assertIn(row["candidate_id"], {"exit-1", "exit-2"})
            self.assertEqual(row["candidate_type"], "Exit")
            self.assertEqual(row["scenario_id"], "scn-1")
            self.assertIn("observation_time", row)
            self.assertEqual(row["prediction_horizon"], 30.0)

    def test_deterministic_extraction_repeated_calls_are_identical(self):

        run = make_run(tick_count=3)

        self.assertEqual(build_candidate_dataset_rows(run), build_candidate_dataset_rows(run))


class ScenarioGroupingTests(unittest.TestCase):

    def test_rows_from_different_scenarios_keep_distinct_scenario_ids(self):

        run_a = make_run(scenario_id="scn-a", tick_count=1)
        run_b = make_run(scenario_id="scn-b", tick_count=1)

        rows = build_candidate_dataset(runs=[run_a, run_b], horizons=(30.0,))

        scenario_ids = {row["scenario_id"] for row in rows}

        self.assertEqual(scenario_ids, {"scn-a", "scn-b"})

    def test_every_row_from_one_run_shares_that_runs_scenario_id_no_cross_contamination(self):

        run_a = make_run(scenario_id="scn-a", tick_count=2)
        run_b = make_run(scenario_id="scn-b", tick_count=2)

        rows_a = build_candidate_dataset_rows(run_a)
        rows_b = build_candidate_dataset_rows(run_b)

        self.assertTrue(all(row["scenario_id"] == "scn-a" for row in rows_a))
        self.assertTrue(all(row["scenario_id"] == "scn-b" for row in rows_b))


class CsvExportTests(unittest.TestCase):

    def test_export_round_trips_row_count(self):

        run = make_run(tick_count=2)
        rows = build_candidate_dataset_rows(run, horizons=(30.0,))

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = export_candidate_dataset_csv(rows, str(Path(tmp_dir) / "candidates.csv"))

            with open(path, newline="", encoding="utf-8") as csv_file:
                read_rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(read_rows), len(rows))
            self.assertIn("candidate_id", read_rows[0])
            self.assertIn("target", read_rows[0])


if __name__ == "__main__":
    unittest.main()
