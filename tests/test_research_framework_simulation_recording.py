import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from research_framework.runner import run_scenario_artifacts

from replay_studio.session import resolve_scenario_artifacts

from tests.test_command_center import make_building, make_scenario


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="research_framework_recording_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


# =====================================================
# research_framework.runner.run_scenario_artifacts() is the other
# scenario-artifact producer this milestone wires the same two new
# artifacts into (its own docstring already claims "byte-for-byte the
# same layout" as designer/campaign/campaign_worker.py -- this test
# keeps that claim true for occupant_routes/decision_events too).
# =====================================================


class RunScenarioArtifactsSimulationRecordingTests(unittest.TestCase):

    def test_writes_occupant_routes_and_decision_events(self):

        building = make_building()
        scenario = make_scenario(building)

        with _TempOutputDir() as output_dir:

            evacuation_time = run_scenario_artifacts(
                scenario, building, output_dir, dt=10.0, registration="dynamic",
            )

            self.assertIsNotNone(evacuation_time)

            scenario_id = scenario.metadata.scenario_id
            root = Path(output_dir)

            occupant_routes_path = root / "occupant_routes" / scenario_id / "occupant_routes.json"
            decision_events_path = root / "decision_events" / scenario_id / "decision_events.json"

            self.assertTrue(occupant_routes_path.is_file())
            self.assertTrue(decision_events_path.is_file())

            from simulation_recording.decision_events import load_decision_events
            from simulation_recording.occupant_routes import load_occupant_routes

            records = load_occupant_routes(str(occupant_routes_path))
            self.assertEqual(
                {record.occupant_id for record in records},
                {occupant.occupant_id for occupant in scenario.occupants},
            )

            # decision_events is a valid, loadable JSON list -- may be
            # empty for this tiny scenario (no assistance pairing exists
            # among occupants placed in different zones), never invalid.
            events = load_decision_events(str(decision_events_path))
            self.assertIsInstance(events, tuple)

    # =====================================================

    def test_writes_timeline_rows_json_not_only_csv(self):

        # Simulation Replay Studio V1 end-to-end verification,
        # Verification Task 22 -- this function used to write only
        # timeline.csv, never timeline_rows.json, silently blocking
        # timeline-scrubbed replay for any scenario produced by this
        # (as opposed to designer/campaign/campaign_worker.py's) producer.

        building = make_building()
        scenario = make_scenario(building)

        with _TempOutputDir() as output_dir:

            run_scenario_artifacts(scenario, building, output_dir, dt=10.0, registration="dynamic")

            scenario_id = scenario.metadata.scenario_id
            timeline_rows_path = Path(output_dir) / "timelines" / scenario_id / "timeline_rows.json"

            self.assertTrue(timeline_rows_path.is_file())

            with open(timeline_rows_path, "r", encoding="utf-8") as handle:
                rows = json.load(handle)

            self.assertIsInstance(rows, list)
            self.assertTrue(len(rows) > 0)
            self.assertEqual(rows[0]["scenario_id"], scenario_id)

            # resolve_scenario_artifacts() (replay_studio's own scenario
            # picker) must now find it too.
            artifacts = resolve_scenario_artifacts(output_dir, scenario_id)
            self.assertIsNotNone(artifacts["timeline_rows_path"])


if __name__ == "__main__":
    unittest.main()
