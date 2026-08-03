import unittest

from calibration_studio.benchmark import (
    BenchmarkType,
    GeometryVersion,
    InvalidBenchmarkDefinitionError,
    PublishedBenchmark,
    PublishedValue,
    ValidationStatus,
)


def _building_benchmark(**overrides):

    defaults = dict(
        title="NIST 10-Story Office Building",
        source_citation="Peacock, Hoskins & Kuligowski (2012), Safety Science 50",
        dataset="NIST",
        benchmark_type=BenchmarkType.BUILDING_RECREATION,
        geometry_reference=GeometryVersion(version="v1", ref="scripts.run_nist_10story_validation.build"),
    )
    defaults.update(overrides)
    return PublishedBenchmark(**defaults)


def _dataset_benchmark(**overrides):

    defaults = dict(
        title="Julich Stair-Egress Walking Speed",
        source_citation="Julich Pedestrian Dynamics Data Archive",
        dataset="Julich",
        benchmark_type=BenchmarkType.DATASET_VALIDATION,
    )
    defaults.update(overrides)
    return PublishedBenchmark(**defaults)


class PublishedValueTests(unittest.TestCase):

    def test_construction_and_to_dict(self):

        value = PublishedValue(value=1022.0, unit="s", uncertainty=5.0)

        self.assertEqual(value.to_dict(), {"value": 1022.0, "unit": "s", "uncertainty": 5.0})

    def test_uncertainty_defaults_to_none(self):

        self.assertIsNone(PublishedValue(value=1.0, unit="m/s").uncertainty)


class UnifiedObjectModelTests(unittest.TestCase):

    # This milestone's own central requirement: ONE class covers both
    # kinds of benchmark, discriminated by benchmark_type, never two
    # separate hierarchies.

    def test_building_recreation_and_dataset_validation_are_the_same_class(self):

        building = _building_benchmark()
        dataset = _dataset_benchmark()

        self.assertIs(type(building), type(dataset))
        self.assertIsInstance(building, PublishedBenchmark)
        self.assertIsInstance(dataset, PublishedBenchmark)

    def test_building_recreation_requires_geometry_reference(self):

        with self.assertRaises(InvalidBenchmarkDefinitionError):
            PublishedBenchmark(
                title="Bad", source_citation="x", dataset="x",
                benchmark_type=BenchmarkType.BUILDING_RECREATION,
            )

    def test_dataset_validation_does_not_require_geometry_reference(self):

        benchmark = _dataset_benchmark()
        self.assertIsNone(benchmark.geometry_reference)

    def test_dataset_validation_may_optionally_carry_geometry_reference(self):

        # Not forbidden -- a future benchmark could plausibly have both
        # a dataset citation and geometry without becoming a different
        # class.
        benchmark = _dataset_benchmark(
            geometry_reference=GeometryVersion(version="v1", ref="x"),
        )
        self.assertIsNotNone(benchmark.geometry_reference)


class CitationFactsAreImmutableTests(unittest.TestCase):

    def test_no_public_setter_exists_for_title(self):

        benchmark = _building_benchmark()
        self.assertFalse(hasattr(benchmark, "set_title"))

    def test_no_public_setter_exists_for_published_values(self):

        benchmark = _building_benchmark()
        self.assertFalse(hasattr(benchmark, "set_published_values"))

    def test_published_values_property_returns_a_copy(self):

        benchmark = _building_benchmark(
            published_values={"evacuation_time_s": PublishedValue(value=1022.0, unit="s")},
        )

        snapshot = benchmark.published_values
        snapshot["injected"] = PublishedValue(value=0.0, unit="x")

        self.assertNotIn("injected", benchmark.published_values)


class EvolvingStateMutatorTests(unittest.TestCase):

    def test_add_tag_is_idempotent_and_bumps_version(self):

        benchmark = _building_benchmark()
        version_before = benchmark.version

        benchmark.add_tag("stair")
        version_after_first = benchmark.version
        benchmark.add_tag("stair")

        self.assertEqual(benchmark.tags, ("stair",))
        self.assertGreater(version_after_first, version_before)
        self.assertEqual(benchmark.version, version_after_first)

    def test_remove_tag(self):

        benchmark = _building_benchmark(tags=("stair", "nist"))
        benchmark.remove_tag("stair")

        self.assertEqual(benchmark.tags, ("nist",))

    def test_set_validation_status(self):

        benchmark = _building_benchmark()
        self.assertEqual(benchmark.validation_status, ValidationStatus.NOT_RUN)

        benchmark.set_validation_status(ValidationStatus.KNOWN_BROKEN)

        self.assertEqual(benchmark.validation_status, ValidationStatus.KNOWN_BROKEN)

    def test_add_calibration_session_is_append_only_and_idempotent(self):

        benchmark = _building_benchmark()

        benchmark.add_calibration_session("sess-1")
        benchmark.add_calibration_session("sess-2")
        benchmark.add_calibration_session("sess-1")

        self.assertEqual(benchmark.calibration_history, ("sess-1", "sess-2"))

    def test_set_current_error(self):

        benchmark = _building_benchmark()
        self.assertIsNone(benchmark.current_error)

        benchmark.set_current_error({"evacuation_time_s": {"ratio": 3.5}})

        self.assertEqual(benchmark.current_error, {"evacuation_time_s": {"ratio": 3.5}})

    def test_set_notes(self):

        benchmark = _building_benchmark()
        benchmark.set_notes("systemic overprediction, root-caused to per-flight admission control")

        self.assertIn("root-caused", benchmark.notes)

    def test_set_geometry_reference(self):

        benchmark = _building_benchmark()
        new_geometry = GeometryVersion(version="v2", ref="new-ref", superseded_by=None)

        benchmark.set_geometry_reference(new_geometry)

        self.assertEqual(benchmark.geometry_reference.version, "v2")

    def test_set_dataset_artifacts(self):

        benchmark = _dataset_benchmark()
        benchmark.set_dataset_artifacts("/data/julich/tu11.txt")

        self.assertEqual(benchmark.dataset_artifacts, "/data/julich/tu11.txt")


if __name__ == "__main__":
    unittest.main()
