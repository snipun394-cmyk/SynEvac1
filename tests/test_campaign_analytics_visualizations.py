import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from campaign_analytics.analyzer import analyze_campaign
from campaign_analytics.visualizations import (
    _bar_chart,
    generate_all_charts,
    save_bottleneck_frequency_chart,
)

from tests.training_dataset_fixtures import make_campaign

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="campaign_analytics_visualizations_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


class ChartGenerationTests(unittest.TestCase):

    def test_generate_all_charts_writes_one_valid_png_per_chart(self):

        with _TempOutputDir() as campaign_dir, _TempOutputDir() as reports_dir:

            make_campaign(campaign_dir, count=6, master_seed=41)
            analysis = analyze_campaign(campaign_dir)

            paths = generate_all_charts(analysis, reports_dir)

            expected_keys = {
                "evacuation_time_histogram", "fire_origin_distribution", "bottleneck_frequency",
                "exit_usage", "stair_usage", "recommendation_frequency", "detector_failures",
                "camera_failures",
            }
            self.assertEqual(set(paths.keys()), expected_keys)

            for path in paths.values():

                file_path = Path(path)
                self.assertTrue(file_path.is_file())
                self.assertGreater(file_path.stat().st_size, 1000)

                with open(file_path, "rb") as handle:
                    self.assertEqual(handle.read(8), PNG_MAGIC)

    def test_charts_are_written_under_the_requested_reports_directory(self):

        with _TempOutputDir() as campaign_dir, _TempOutputDir() as reports_root:

            make_campaign(campaign_dir, count=2, master_seed=42)
            analysis = analyze_campaign(campaign_dir)

            reports_dir = str(Path(reports_root) / "reports")
            paths = generate_all_charts(analysis, reports_dir)

            for path in paths.values():
                self.assertTrue(str(Path(path).parent) == reports_dir)


class ChartDeterminismTests(unittest.TestCase):

    def test_the_same_data_produces_a_byte_identical_chart(self):

        data = {"door-1": 4, "door-2": 1, "door-3": 7}

        with _TempOutputDir() as output_dir:

            first_path = str(Path(output_dir) / "first.png")
            second_path = str(Path(output_dir) / "second.png")

            _bar_chart(data, title="T", xlabel="X", ylabel="Y", file_path=first_path)
            _bar_chart(data, title="T", xlabel="X", ylabel="Y", file_path=second_path)

            first_hash = hashlib.sha256(Path(first_path).read_bytes()).hexdigest()
            second_hash = hashlib.sha256(Path(second_path).read_bytes()).hexdigest()

            self.assertEqual(first_hash, second_hash)

    def test_a_full_real_campaign_produces_deterministic_charts_across_two_runs(self):

        with _TempOutputDir() as campaign_dir, _TempOutputDir() as reports_a, _TempOutputDir() as reports_b:

            make_campaign(campaign_dir, count=4, master_seed=43)

            analysis_a = analyze_campaign(campaign_dir)
            analysis_b = analyze_campaign(campaign_dir)

            paths_a = generate_all_charts(analysis_a, reports_a)
            paths_b = generate_all_charts(analysis_b, reports_b)

            for key in paths_a:

                hash_a = hashlib.sha256(Path(paths_a[key]).read_bytes()).hexdigest()
                hash_b = hashlib.sha256(Path(paths_b[key]).read_bytes()).hexdigest()

                self.assertEqual(hash_a, hash_b, f"chart {key!r} was not deterministic")


class EmptyDataChartTests(unittest.TestCase):

    def test_a_chart_with_no_data_does_not_raise(self):

        with _TempOutputDir() as output_dir:

            path = save_bottleneck_frequency_chart(
                {"bottleneck_locations": {}}, str(Path(output_dir) / "empty.png"),
            )

            self.assertTrue(Path(path).is_file())

            with open(path, "rb") as handle:
                self.assertEqual(handle.read(8), PNG_MAGIC)


if __name__ == "__main__":
    unittest.main()
