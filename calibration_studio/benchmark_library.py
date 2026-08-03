from pathlib import Path
from typing import Dict, Optional, Tuple

import calibration_studio.storage as storage
from calibration_studio.benchmark import BenchmarkType, PublishedBenchmark, ValidationStatus


# =====================================================
# Calibration Studio Phase 3 -- Published Benchmark Library.
#
# PublishedBenchmarkLibrary is the registry -- the same split
# CalibrationStudio already established between in-process bookkeeping
# (register()/unregister()/list_benchmarks()/lookups, all in-memory,
# never touching disk) and durable storage (save_benchmark()/
# load_benchmark()/list_persisted_benchmarks(), delegating to
# calibration_studio/storage.py). register() is not "save" and
# unregister() is not "delete from disk" -- exactly mirroring
# CalibrationStudio.create_project() never writing to disk on its own.
# Deleting a persisted benchmark file is explicitly out of scope here
# (not named anywhere in this milestone's own SUPPORTED OPERATIONS
# list) -- a destructive disk operation deserves its own deliberate
# design, not one bolted on as a side effect of an in-memory
# unregister() call.
# =====================================================


class DuplicateBenchmarkError(Exception):
    pass


class BenchmarkNotFoundError(Exception):
    pass


class PublishedBenchmarkLibrary:

    def __init__(self, *, storage_root=None):

        self._benchmarks: Dict[str, PublishedBenchmark] = {}
        self._storage_root = Path(storage_root) if storage_root is not None else None

    def _require_storage_root(self) -> Path:

        if self._storage_root is None:
            raise ValueError(
                "This PublishedBenchmarkLibrary has no storage_root configured -- construct it "
                "as PublishedBenchmarkLibrary(storage_root=...) to use persistence methods.",
            )

        return self._storage_root

    # =====================================================
    # Registration (in-memory)
    # =====================================================

    def register(self, benchmark: PublishedBenchmark) -> None:

        if benchmark.benchmark_id in self._benchmarks:
            raise DuplicateBenchmarkError(
                f"A benchmark is already registered at benchmark_id {benchmark.benchmark_id!r} "
                f"({self._benchmarks[benchmark.benchmark_id].title!r}) -- unregister it first if "
                f"you intend to replace it.",
            )

        self._benchmarks[benchmark.benchmark_id] = benchmark

    def unregister(self, benchmark_id: str) -> None:

        if benchmark_id not in self._benchmarks:
            raise BenchmarkNotFoundError(f"No benchmark registered at benchmark_id {benchmark_id!r}")

        del self._benchmarks[benchmark_id]

    def get(self, benchmark_id: str) -> Optional[PublishedBenchmark]:

        return self._benchmarks.get(benchmark_id)

    def list_benchmarks(self) -> Tuple[PublishedBenchmark, ...]:

        return tuple(self._benchmarks.values())

    # =====================================================
    # Lookups -- every one of this milestone's own SUPPORTED OPERATIONS,
    # each a plain filter over the in-memory registry.
    # =====================================================

    def find_by_tag(self, tag: str) -> Tuple[PublishedBenchmark, ...]:

        return tuple(benchmark for benchmark in self._benchmarks.values() if tag in benchmark.tags)

    def find_by_dataset(self, dataset: str) -> Tuple[PublishedBenchmark, ...]:

        return tuple(benchmark for benchmark in self._benchmarks.values() if benchmark.dataset == dataset)

    def find_by_validation_status(self, status: ValidationStatus) -> Tuple[PublishedBenchmark, ...]:

        return tuple(
            benchmark for benchmark in self._benchmarks.values() if benchmark.validation_status is status
        )

    def find_by_benchmark_type(self, benchmark_type: BenchmarkType) -> Tuple[PublishedBenchmark, ...]:

        return tuple(
            benchmark for benchmark in self._benchmarks.values() if benchmark.benchmark_type is benchmark_type
        )

    # =====================================================
    # Persistence -- thin delegation to calibration_studio/storage.py,
    # identical shape to CalibrationStudio's own save_project()/
    # load_project()/list_persisted_projects().
    # =====================================================

    def save_benchmark(self, benchmark: PublishedBenchmark) -> Path:

        return storage.save_benchmark(benchmark, self._require_storage_root())

    def load_benchmark(self, benchmark_id: str) -> PublishedBenchmark:

        benchmark = storage.load_benchmark(benchmark_id, self._require_storage_root())
        self._benchmarks[benchmark.benchmark_id] = benchmark

        return benchmark

    def list_persisted_benchmarks(self) -> Tuple[PublishedBenchmark, ...]:

        benchmarks = storage.list_benchmarks(self._require_storage_root())

        for benchmark in benchmarks:
            self._benchmarks[benchmark.benchmark_id] = benchmark

        return benchmarks

    def __repr__(self) -> str:

        return f"PublishedBenchmarkLibrary(benchmarks={len(self._benchmarks)})"
