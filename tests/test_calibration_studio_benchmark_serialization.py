import unittest

from calibration_studio.benchmark import (
    SCHEMA_VERSION,
    BenchmarkType,
    CorruptedBenchmarkRecordError,
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
        doi=None,
        authors=("Peacock", "Hoskins", "Kuligowski"),
        publication_year=2012,
        venue="Safety Science",
        published_values={"evacuation_time_s": PublishedValue(value=1022.0, unit="s", uncertainty=5.0)},
        assumptions=("Stairs A and B only",),
        tags=("stair", "nist"),
        geometry_reference=GeometryVersion(version="v1", ref="scripts.run_nist_10story_validation.build"),
    )
    defaults.update(overrides)
    return PublishedBenchmark(**defaults)


class RoundTripTests(unittest.TestCase):

    def test_building_benchmark_round_trips_every_field(self):

        benchmark = _building_benchmark()
        benchmark.set_validation_status(ValidationStatus.RUN_WITH_DEFAULTS)
        benchmark.add_calibration_session("sess-1")
        benchmark.set_current_error({"evacuation_time_s": {"ratio": 3.5}})
        benchmark.set_notes("systemic overprediction")

        restored = PublishedBenchmark.from_dict(benchmark.to_dict())

        self.assertEqual(restored.benchmark_id, benchmark.benchmark_id)
        self.assertEqual(restored.title, benchmark.title)
        self.assertEqual(restored.source_citation, benchmark.source_citation)
        self.assertEqual(restored.dataset, benchmark.dataset)
        self.assertEqual(restored.benchmark_type, benchmark.benchmark_type)
        self.assertEqual(restored.authors, benchmark.authors)
        self.assertEqual(restored.publication_year, benchmark.publication_year)
        self.assertEqual(restored.venue, benchmark.venue)
        self.assertEqual(restored.assumptions, benchmark.assumptions)
        self.assertEqual(restored.tags, benchmark.tags)
        self.assertEqual(restored.created_at, benchmark.created_at)
        self.assertEqual(restored.updated_at, benchmark.updated_at)
        self.assertEqual(restored.version, benchmark.version)
        self.assertEqual(restored.geometry_reference, benchmark.geometry_reference)
        self.assertEqual(restored.validation_status, benchmark.validation_status)
        self.assertEqual(restored.calibration_history, benchmark.calibration_history)
        self.assertEqual(restored.current_error, benchmark.current_error)
        self.assertEqual(restored.notes, benchmark.notes)

        restored_values = restored.published_values
        original_values = benchmark.published_values
        self.assertEqual(restored_values.keys(), original_values.keys())
        self.assertEqual(
            restored_values["evacuation_time_s"].to_dict(),
            original_values["evacuation_time_s"].to_dict(),
        )

    def test_dataset_validation_benchmark_round_trips_with_no_geometry(self):

        benchmark = PublishedBenchmark(
            title="Julich Stair-Egress Walking Speed",
            source_citation="Julich Pedestrian Dynamics Data Archive",
            dataset="Julich",
            benchmark_type=BenchmarkType.DATASET_VALIDATION,
            doi="10.34735/ped.2009.5",
            published_values={"walking_speed_ms": PublishedValue(value=0.649, unit="m/s")},
        )

        restored = PublishedBenchmark.from_dict(benchmark.to_dict())

        self.assertEqual(restored.benchmark_type, BenchmarkType.DATASET_VALIDATION)
        self.assertIsNone(restored.geometry_reference)
        self.assertEqual(restored.doi, "10.34735/ped.2009.5")

    def test_extra_round_trips(self):

        benchmark = _building_benchmark(extra={"reviewer": "engineer-1"})

        restored = PublishedBenchmark.from_dict(benchmark.to_dict())

        self.assertEqual(restored.extra, {"reviewer": "engineer-1"})


class ForwardCompatibilityTests(unittest.TestCase):

    def test_missing_optional_fields_use_sensible_defaults(self):

        minimal = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": "bench-minimal",
            "title": "T",
            "source_citation": "C",
            "dataset": "D",
            "benchmark_type": BenchmarkType.DATASET_VALIDATION.value,
        }

        restored = PublishedBenchmark.from_dict(minimal)

        self.assertEqual(restored.benchmark_id, "bench-minimal")
        self.assertEqual(restored.authors, ())
        self.assertIsNone(restored.publication_year)
        self.assertEqual(restored.venue, "")
        self.assertEqual(restored.published_values, {})
        self.assertEqual(restored.assumptions, ())
        self.assertEqual(restored.tags, ())
        self.assertEqual(restored.validation_status, ValidationStatus.NOT_RUN)
        self.assertEqual(restored.calibration_history, ())
        self.assertIsNone(restored.current_error)
        self.assertEqual(restored.notes, "")
        self.assertEqual(restored.version, 1)

    def test_missing_geometry_reference_on_a_building_recreation_record_raises(self):

        # Not a "graceful default" case -- geometry_reference is a
        # genuine domain invariant for BUILDING_RECREATION (this
        # module's own __init__ enforces it unconditionally, including
        # during from_dict()), so a record missing it is honestly
        # corrupted/invalid, not merely old-schema.
        broken = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": "bench-broken",
            "title": "T",
            "source_citation": "C",
            "dataset": "D",
            "benchmark_type": BenchmarkType.BUILDING_RECREATION.value,
        }

        with self.assertRaises(InvalidBenchmarkDefinitionError):
            PublishedBenchmark.from_dict(broken)

    def test_unknown_top_level_field_is_preserved_in_extra_not_dropped(self):

        benchmark = _building_benchmark()
        data = benchmark.to_dict()
        data["a_future_field_this_version_has_never_heard_of"] = {"nested": True}

        restored = PublishedBenchmark.from_dict(data)

        self.assertEqual(
            restored.extra["a_future_field_this_version_has_never_heard_of"], {"nested": True},
        )

    def test_unrecognised_benchmark_type_raises_corrupted_error(self):

        benchmark = _building_benchmark()
        data = benchmark.to_dict()
        data["benchmark_type"] = "NOT_A_REAL_TYPE"

        with self.assertRaises(CorruptedBenchmarkRecordError):
            PublishedBenchmark.from_dict(data)

    def test_unrecognised_validation_status_raises_corrupted_error(self):

        benchmark = _building_benchmark()
        data = benchmark.to_dict()
        data["validation_status"] = "NOT_A_REAL_STATUS"

        with self.assertRaises(CorruptedBenchmarkRecordError):
            PublishedBenchmark.from_dict(data)


if __name__ == "__main__":
    unittest.main()
