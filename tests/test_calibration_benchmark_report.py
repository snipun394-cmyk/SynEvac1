import unittest

from calibration_benchmark.candidates import WalkingSpeedCandidate
from calibration_benchmark.harness import run_calibration_benchmark
from calibration_benchmark.report import render_markdown_report, save_report

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


class RenderMarkdownReportTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        candidate = WalkingSpeedCandidate("Adult_Default", 0.5, "Julich stair egress dataset", "test rationale")
        cls.result = run_calibration_benchmark(
            candidate, make_building(), make_definition(occupant_count=8), DEFINITION_ID, MASTER_SEED,
            n_scenarios=2, dt=1.0,
        )
        cls.markdown = render_markdown_report(cls.result)

    def test_report_names_the_parameter_and_both_values(self):

        self.assertIn("Adult_Default.walking_speed", self.markdown)
        self.assertIn("1.2", self.markdown)  # current default
        self.assertIn("0.5", self.markdown)  # candidate value

    def test_report_states_no_default_was_changed(self):

        self.assertIn("No SynEvac default was changed", self.markdown)

    def test_report_includes_a_recommendation_section(self):

        self.assertIn("## 5. Recommendation", self.markdown)
        self.assertIn("Overall verdict:", self.markdown)

    def test_report_includes_experiment_settings(self):

        self.assertIn("Scenarios requested:", self.markdown)
        self.assertIn("paired", self.markdown.lower())

    def test_save_report_writes_the_same_content_to_disk(self):

        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = os.path.join(tmp_dir, "report.md")
            written = save_report(self.result, path)

            with open(path, encoding="utf-8") as handle:
                on_disk = handle.read()

            self.assertEqual(written, on_disk)
            self.assertIn("Adult_Default.walking_speed", on_disk)


if __name__ == "__main__":
    unittest.main()
