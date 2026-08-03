import json
import tempfile
import unittest
from pathlib import Path

import calibration_studio.storage as storage
from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, PublishedValue, ValidationStatus
from calibration_studio.benchmark_library import PublishedBenchmarkLibrary


def _building(**overrides):

    defaults = dict(
        title="NIST 10-Story Office Building",
        source_citation="Peacock, Hoskins & Kuligowski (2012), Safety Science 50",
        dataset="NIST",
        benchmark_type=BenchmarkType.BUILDING_RECREATION,
        authors=("Peacock", "Hoskins", "Kuligowski"),
        publication_year=2012,
        published_values={"evacuation_time_s": PublishedValue(value=1022.0, unit="s")},
        tags=("stair", "nist"),
        geometry_reference=GeometryVersion(version="v1", ref="scripts.run_nist_10story_validation.build"),
    )
    defaults.update(overrides)
    return PublishedBenchmark(**defaults)


def _julich(**overrides):

    defaults = dict(
        title="Julich Stair-Egress Walking Speed",
        source_citation="Julich Pedestrian Dynamics Data Archive",
        dataset="Julich",
        benchmark_type=BenchmarkType.DATASET_VALIDATION,
        doi="10.34735/ped.2009.5",
        published_values={"walking_speed_ms": PublishedValue(value=0.649, unit="m/s")},
        tags=("walking-speed",),
    )
    defaults.update(overrides)
    return PublishedBenchmark(**defaults)


class CalibrationStudioBenchmarkStorageTestCase(unittest.TestCase):

    def setUp(self):

        self._tmpdir = tempfile.TemporaryDirectory()
        self.storage_root = Path(self._tmpdir.name)

    def tearDown(self):

        self._tmpdir.cleanup()


class SaveLoadRoundTripTests(CalibrationStudioBenchmarkStorageTestCase):

    def test_save_then_load_restores_metadata(self):

        benchmark = _building()
        storage.save_benchmark(benchmark, self.storage_root)

        reloaded = storage.load_benchmark(benchmark.benchmark_id, self.storage_root)

        self.assertEqual(reloaded.title, benchmark.title)
        self.assertEqual(reloaded.dataset, "NIST")
        self.assertEqual(reloaded.geometry_reference, benchmark.geometry_reference)

    def test_load_unknown_benchmark_id_raises_file_not_found(self):

        with self.assertRaises(FileNotFoundError):
            storage.load_benchmark("does-not-exist", self.storage_root)

    def test_save_overwrites_in_place_no_duplicate_catalog_row(self):

        benchmark = _building()
        storage.save_benchmark(benchmark, self.storage_root)

        benchmark.set_validation_status(ValidationStatus.RUN_WITH_DEFAULTS)
        storage.save_benchmark(benchmark, self.storage_root)

        reloaded = storage.load_benchmark(benchmark.benchmark_id, self.storage_root)
        self.assertEqual(reloaded.validation_status, ValidationStatus.RUN_WITH_DEFAULTS)
        self.assertEqual(len(storage.list_benchmarks(self.storage_root)), 1)

    def test_dataset_validation_benchmark_round_trips_through_disk(self):

        benchmark = _julich()
        storage.save_benchmark(benchmark, self.storage_root)

        reloaded = storage.load_benchmark(benchmark.benchmark_id, self.storage_root)

        self.assertEqual(reloaded.benchmark_type, BenchmarkType.DATASET_VALIDATION)
        self.assertIsNone(reloaded.geometry_reference)
        self.assertEqual(reloaded.doi, "10.34735/ped.2009.5")


class ListBenchmarksTests(CalibrationStudioBenchmarkStorageTestCase):

    def test_lists_every_saved_benchmark(self):

        storage.save_benchmark(_building(title="A"), self.storage_root)
        storage.save_benchmark(_julich(title="B"), self.storage_root)

        titles = {b.title for b in storage.list_benchmarks(self.storage_root)}
        self.assertEqual(titles, {"A", "B"})

    def test_empty_storage_root_returns_no_benchmarks(self):

        self.assertEqual(storage.list_benchmarks(self.storage_root), ())


class CorruptionHandlingTests(CalibrationStudioBenchmarkStorageTestCase):

    def test_garbled_benchmark_file_raises_corrupted_record_file_error(self):

        benchmark = _building()
        json_path = storage.save_benchmark(benchmark, self.storage_root)

        json_path.write_text("{ not valid json ]]", encoding="utf-8")

        with self.assertRaises(storage.CorruptedRecordFileError):
            storage.load_benchmark(benchmark.benchmark_id, self.storage_root)


class VersionMismatchTests(CalibrationStudioBenchmarkStorageTestCase):

    def test_incompatible_schema_version_raises(self):

        benchmark = _building()
        json_path = storage.save_benchmark(benchmark, self.storage_root)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        data["schema_version"] = "calibration_studio_benchmark/99"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(storage.IncompatibleSchemaVersionError):
            storage.load_benchmark(benchmark.benchmark_id, self.storage_root)

    def test_missing_schema_version_key_raises(self):

        benchmark = _building()
        json_path = storage.save_benchmark(benchmark, self.storage_root)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        del data["schema_version"]
        json_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(storage.IncompatibleSchemaVersionError):
            storage.load_benchmark(benchmark.benchmark_id, self.storage_root)


class MissingAndUnknownFieldsOnDiskTests(CalibrationStudioBenchmarkStorageTestCase):

    def test_missing_non_essential_field_still_loads(self):

        benchmark = _building()
        json_path = storage.save_benchmark(benchmark, self.storage_root)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        del data["venue"]
        del data["notes"]
        json_path.write_text(json.dumps(data), encoding="utf-8")

        reloaded = storage.load_benchmark(benchmark.benchmark_id, self.storage_root)

        self.assertEqual(reloaded.venue, "")
        self.assertEqual(reloaded.notes, "")
        self.assertEqual(reloaded.title, benchmark.title)

    def test_unknown_future_field_survives_load_and_resave(self):

        benchmark = _building()
        json_path = storage.save_benchmark(benchmark, self.storage_root)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        data["a_field_from_a_future_calibration_studio_version"] = 42
        json_path.write_text(json.dumps(data), encoding="utf-8")

        reloaded = storage.load_benchmark(benchmark.benchmark_id, self.storage_root)
        self.assertEqual(reloaded.extra["a_field_from_a_future_calibration_studio_version"], 42)

        storage.save_benchmark(reloaded, self.storage_root)
        reloaded_again = storage.load_benchmark(benchmark.benchmark_id, self.storage_root)
        self.assertEqual(reloaded_again.extra["a_field_from_a_future_calibration_studio_version"], 42)


class FullVerifyScenarioTests(CalibrationStudioBenchmarkStorageTestCase):

    def test_register_persist_reload_search_metadata_integrity(self):

        # This milestone's own VERIFY section, end to end: registration,
        # persistence, reload, search, metadata integrity -- across
        # both benchmark kinds at once, through the real
        # PublishedBenchmarkLibrary facade.
        library = PublishedBenchmarkLibrary(storage_root=self.storage_root)

        nist = _building(title="NIST 10-Story", dataset="NIST", tags=("stair", "nist"))
        julich = _julich(title="Julich Walking Speed", dataset="Julich", tags=("walking-speed",))

        library.register(nist)
        library.register(julich)

        library.save_benchmark(nist)
        library.save_benchmark(julich)

        # reload in a fresh library instance (new process, conceptually)
        library2 = PublishedBenchmarkLibrary(storage_root=self.storage_root)
        persisted = library2.list_persisted_benchmarks()

        self.assertEqual(len(persisted), 2)

        # search after reload
        self.assertEqual(len(library2.find_by_dataset("NIST")), 1)
        self.assertEqual(len(library2.find_by_dataset("Julich")), 1)
        self.assertEqual(len(library2.find_by_tag("stair")), 1)
        self.assertEqual(
            len(library2.find_by_benchmark_type(BenchmarkType.BUILDING_RECREATION)), 1,
        )
        self.assertEqual(
            len(library2.find_by_benchmark_type(BenchmarkType.DATASET_VALIDATION)), 1,
        )

        # metadata integrity
        reloaded_nist = library2.get(nist.benchmark_id)
        self.assertEqual(reloaded_nist.title, "NIST 10-Story")
        self.assertEqual(reloaded_nist.geometry_reference, nist.geometry_reference)
        self.assertEqual(
            reloaded_nist.published_values["evacuation_time_s"].to_dict(),
            nist.published_values["evacuation_time_s"].to_dict(),
        )

        reloaded_julich = library2.get(julich.benchmark_id)
        self.assertIsNone(reloaded_julich.geometry_reference)
        self.assertEqual(reloaded_julich.doi, "10.34735/ped.2009.5")


if __name__ == "__main__":
    unittest.main()
