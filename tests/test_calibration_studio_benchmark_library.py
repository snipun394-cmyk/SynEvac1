import unittest

from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, ValidationStatus
from calibration_studio.benchmark_library import BenchmarkNotFoundError, DuplicateBenchmarkError, PublishedBenchmarkLibrary


def _building(title="NIST 10-Story", dataset="NIST", tags=()):

    return PublishedBenchmark(
        title=title, source_citation="citation", dataset=dataset,
        benchmark_type=BenchmarkType.BUILDING_RECREATION,
        geometry_reference=GeometryVersion(version="v1", ref="ref"),
        tags=tags,
    )


def _dataset_only(title="Julich Walking Speed", dataset="Julich", tags=()):

    return PublishedBenchmark(
        title=title, source_citation="citation", dataset=dataset,
        benchmark_type=BenchmarkType.DATASET_VALIDATION, tags=tags,
    )


class RegistrationTests(unittest.TestCase):

    def test_register_makes_it_visible_to_get(self):

        library = PublishedBenchmarkLibrary()
        benchmark = _building()

        library.register(benchmark)

        self.assertIs(library.get(benchmark.benchmark_id), benchmark)

    def test_register_makes_it_visible_to_list_benchmarks(self):

        library = PublishedBenchmarkLibrary()
        benchmark = _building()

        library.register(benchmark)

        self.assertIn(benchmark, library.list_benchmarks())

    def test_fresh_library_lists_nothing(self):

        self.assertEqual(PublishedBenchmarkLibrary().list_benchmarks(), ())

    def test_get_unknown_id_returns_none(self):

        self.assertIsNone(PublishedBenchmarkLibrary().get("does-not-exist"))


class DuplicateDetectionTests(unittest.TestCase):

    def test_registering_the_same_benchmark_id_twice_raises(self):

        library = PublishedBenchmarkLibrary()
        benchmark = _building()

        library.register(benchmark)

        with self.assertRaises(DuplicateBenchmarkError):
            library.register(benchmark)

    def test_two_distinct_benchmarks_register_fine(self):

        library = PublishedBenchmarkLibrary()

        library.register(_building(title="A"))
        library.register(_building(title="B"))

        self.assertEqual(len(library.list_benchmarks()), 2)

    def test_unregister_then_reregister_the_same_id_succeeds(self):

        library = PublishedBenchmarkLibrary()
        benchmark = _building()

        library.register(benchmark)
        library.unregister(benchmark.benchmark_id)
        library.register(benchmark)

        self.assertIs(library.get(benchmark.benchmark_id), benchmark)


class UnregisterTests(unittest.TestCase):

    def test_unregister_removes_it(self):

        library = PublishedBenchmarkLibrary()
        benchmark = _building()
        library.register(benchmark)

        library.unregister(benchmark.benchmark_id)

        self.assertIsNone(library.get(benchmark.benchmark_id))
        self.assertEqual(library.list_benchmarks(), ())

    def test_unregister_unknown_id_raises(self):

        with self.assertRaises(BenchmarkNotFoundError):
            PublishedBenchmarkLibrary().unregister("does-not-exist")


class SearchingTests(unittest.TestCase):

    def setUp(self):

        self.library = PublishedBenchmarkLibrary()

        self.nist_10 = _building(title="NIST 10-Story", dataset="NIST", tags=("stair", "nist"))
        self.nist_18 = _building(title="NIST 18-Story", dataset="NIST", tags=("stair", "nist", "merge"))
        self.julich = _dataset_only(title="Julich Walking Speed", dataset="Julich", tags=("walking-speed",))

        for benchmark in (self.nist_10, self.nist_18, self.julich):
            self.library.register(benchmark)

        self.nist_10.set_validation_status(ValidationStatus.KNOWN_BROKEN)

    def test_find_by_tag(self):

        self.assertEqual(set(self.library.find_by_tag("stair")), {self.nist_10, self.nist_18})
        self.assertEqual(self.library.find_by_tag("walking-speed"), (self.julich,))
        self.assertEqual(self.library.find_by_tag("merge"), (self.nist_18,))

    def test_find_by_tag_with_no_matches_returns_empty(self):

        self.assertEqual(self.library.find_by_tag("no-such-tag"), ())

    def test_find_by_dataset(self):

        self.assertEqual(set(self.library.find_by_dataset("NIST")), {self.nist_10, self.nist_18})
        self.assertEqual(self.library.find_by_dataset("Julich"), (self.julich,))

    def test_find_by_validation_status(self):

        self.assertEqual(self.library.find_by_validation_status(ValidationStatus.KNOWN_BROKEN), (self.nist_10,))
        self.assertEqual(
            set(self.library.find_by_validation_status(ValidationStatus.NOT_RUN)),
            {self.nist_18, self.julich},
        )

    def test_find_by_benchmark_type(self):

        self.assertEqual(
            set(self.library.find_by_benchmark_type(BenchmarkType.BUILDING_RECREATION)),
            {self.nist_10, self.nist_18},
        )
        self.assertEqual(
            self.library.find_by_benchmark_type(BenchmarkType.DATASET_VALIDATION), (self.julich,),
        )


class LibraryWithoutStorageRootTests(unittest.TestCase):

    def test_save_benchmark_raises_without_storage_root(self):

        library = PublishedBenchmarkLibrary()
        benchmark = _building()
        library.register(benchmark)

        with self.assertRaises(ValueError):
            library.save_benchmark(benchmark)

    def test_load_benchmark_raises_without_storage_root(self):

        with self.assertRaises(ValueError):
            PublishedBenchmarkLibrary().load_benchmark("some-id")


if __name__ == "__main__":
    unittest.main()
