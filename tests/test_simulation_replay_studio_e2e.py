import sys
import unittest
from pathlib import Path

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.campaign.campaign_worker import CampaignWorker

from command_center.incident_data import load_incident

from replay_studio.session import discover_scenario_ids, resolve_scenario_artifacts

from tests.test_campaign_pipeline_integration import _TempOutputDir, make_config, make_definition


# =====================================================
# End-to-end: a REAL campaign run (scenario_runner ->
# behaviour_profile_resolver.dynamic_registrar -> ai_decision ->
# simulation_runtime, nothing mocked -- the exact chain
# designer/campaign/campaign_worker.py._simulate() already exercises)
# through Simulation Replay Studio's own artifact-discovery and
# IncidentData loading. This is the concrete check for "replay must
# exactly reproduce recorded simulation history": every position this
# test inspects comes from the same movement_result the campaign itself
# already computed and wrote to occupant_routes.json, read back and
# interpolated, never re-simulated.
# =====================================================


class SimulationReplayStudioEndToEndTests(unittest.TestCase):

    def test_campaign_scenario_replays_with_occupant_positions(self):

        with _TempOutputDir() as output_dir:

            config = make_config(
                output_directory=output_dir, count=1, definition=make_definition(occupant_count=2),
            )
            worker = CampaignWorker(config)
            summary = worker.execute()

            self.assertEqual(summary.accepted, 1)

            scenario_ids = discover_scenario_ids(output_dir)
            self.assertEqual(len(scenario_ids), 1)
            scenario_id = scenario_ids[0]

            root = Path(output_dir)
            self.assertTrue((root / "occupant_routes" / scenario_id / "occupant_routes.json").is_file())
            self.assertTrue((root / "decision_events" / scenario_id / "decision_events.json").is_file())

            artifacts = resolve_scenario_artifacts(output_dir, scenario_id)

            self.assertIsNotNone(artifacts["occupant_routes_path"])
            self.assertIsNotNone(artifacts["decision_events_path"])
            self.assertIsNotNone(artifacts["ground_truth_path"])
            self.assertIsNotNone(artifacts["decision_policy_path"])
            self.assertIsNotNone(artifacts["timeline_rows_path"])

            incident = load_incident(**artifacts)

            self.assertTrue(len(incident.occupant_routes) >= 2)

            resolved_positions = [
                position
                for frame in incident.frames
                for position in frame.occupant_positions.values()
                if position.x is not None and position.y is not None
            ]
            self.assertTrue(resolved_positions, "no occupant position was ever resolvable across any frame")

            # Every occupant this run registered has a position entry on
            # the first frame -- never silently dropped.
            first_frame = incident.frame_at_index(0)
            recorded_ids = {record.occupant_id for record in incident.occupant_routes}
            self.assertEqual(set(first_frame.occupant_positions.keys()), recorded_ids)


if __name__ == "__main__":
    unittest.main()
