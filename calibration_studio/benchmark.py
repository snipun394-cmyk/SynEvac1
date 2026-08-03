import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple


# =====================================================
# Calibration Studio Phase 3 -- Published Benchmark Library.
#
# PublishedBenchmark is the ONE object model for every kind of
# real-world reference this milestone's own brief names: a full
# recreated-building benchmark (a NIST evacuation drill, with real
# geometry and a real simulation to run against it) and a
# dataset-based scalar benchmark (a Jülich walking-speed value, with
# no geometry at all -- exactly how calibration_benchmark's own
# WalkingSpeedCandidate.dataset_source already cites it today). No
# separate Dataset/Benchmark class hierarchy exists -- `benchmark_type`
# is a plain discriminator field on this one class, `geometry_reference`
# is simply populated for one kind and None for the other. This mirrors
# the approved persistent-data-model design's own resolution of exactly
# this question.
#
# Deliberately a plain, mutable object with an explicit `version`
# counter -- not a frozen dataclass + dataclasses.replace() -- for the
# identical reason CalibrationProject/CalibrationSession already are
# (see session.py's own docstring): a benchmark has identity and
# accumulates evolving state (validation_status, calibration_history,
# notes) over a lifetime spanning many separate calibration runs, not
# a single-shot value.
#
# Citation facts (title, source_citation, doi, authors, publication_year,
# venue, published_values, assumptions, dataset, benchmark_type) are
# immutable once constructed -- a published paper's own numbers do not
# change. Everything else (tags, geometry_reference, dataset_artifacts,
# validation_status, calibration_history, current_error, notes) evolves
# via explicit mutator methods, each bumping `version`/`updated_at`,
# the same discipline CalibrationProject already established.
#
# current_error is a plain, directly-settable field in this phase, not
# an auto-computed one -- correctly computing it means reading the
# latest session in calibration_history's own result, which requires
# calibration execution (the explicitly out-of-scope Calibration
# Runner) and storage access this domain object deliberately has none
# of. A future phase can compute it and call set_current_error(); this
# phase only provides the honest slot to put that answer in.
# =====================================================


SCHEMA_VERSION = "calibration_studio_benchmark/1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})


class BenchmarkType(Enum):

    BUILDING_RECREATION = "BUILDING_RECREATION"
    DATASET_VALIDATION = "DATASET_VALIDATION"


class ValidationStatus(Enum):

    # Explicitly declared, never inferred from evidence quantity alone
    # -- the same discipline ai_registry.metadata.Deployability already
    # enforces for ML models in this codebase.

    NOT_RUN = "NOT_RUN"
    RUN_WITH_DEFAULTS = "RUN_WITH_DEFAULTS"
    RUN_WITH_CANDIDATES = "RUN_WITH_CANDIDATES"
    KNOWN_BROKEN = "KNOWN_BROKEN"


class CorruptedBenchmarkRecordError(Exception):
    pass


class InvalidBenchmarkDefinitionError(Exception):
    pass


def _utc_now_iso() -> str:

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PublishedValue:

    value: float
    unit: str
    uncertainty: Optional[float] = None

    def to_dict(self) -> dict:

        return {"value": self.value, "unit": self.unit, "uncertainty": self.uncertainty}

    @classmethod
    def from_dict(cls, data: dict) -> "PublishedValue":

        return cls(value=data["value"], unit=data.get("unit", ""), uncertainty=data.get("uncertainty"))


@dataclass(frozen=True)
class GeometryVersion:

    # today: `ref` is an opaque string (e.g. a dotted path to the
    # bespoke Python construction function a NIST recreation script
    # already uses -- "scripts.run_nist_10story_validation.
    # build_nist_10story_building"); a future phase migrating to
    # serialized Building/Project geometry would populate `ref` with a
    # file path instead, with no change to this shape. Resolving/
    # importing/loading whatever `ref` names is explicitly the future
    # Calibration Runner's job, not this phase's -- this is a plain
    # data record, nothing here executes it.

    version: str
    ref: str
    superseded_by: Optional[str] = None

    def to_dict(self) -> dict:

        return {"version": self.version, "ref": self.ref, "superseded_by": self.superseded_by}

    @classmethod
    def from_dict(cls, data: dict) -> "GeometryVersion":

        return cls(version=data.get("version", ""), ref=data.get("ref", ""), superseded_by=data.get("superseded_by"))


_KNOWN_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "benchmark_id", "title", "source_citation", "dataset", "benchmark_type",
    "doi", "authors", "publication_year", "venue", "published_values", "assumptions", "tags",
    "created_at", "updated_at", "version", "geometry_reference", "dataset_artifacts",
    "validation_status", "calibration_history", "current_error", "notes", "extra",
})


class PublishedBenchmark:

    def __init__(
        self,
        *,
        title: str,
        source_citation: str,
        dataset: str,
        benchmark_type: BenchmarkType,
        doi: Optional[str] = None,
        authors: Tuple[str, ...] = (),
        publication_year: Optional[int] = None,
        venue: str = "",
        published_values: Optional[Mapping[str, PublishedValue]] = None,
        assumptions: Tuple[str, ...] = (),
        tags: Tuple[str, ...] = (),
        extra: Optional[Dict[str, Any]] = None,
        geometry_reference: Optional[GeometryVersion] = None,
        # Restoration-only parameters -- see CalibrationSession.__init__'s
        # identical convention and reasoning.
        benchmark_id: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        version: int = 1,
        dataset_artifacts: Optional[str] = None,
        validation_status: ValidationStatus = ValidationStatus.NOT_RUN,
        calibration_history: Tuple[str, ...] = (),
        current_error: Optional[dict] = None,
        notes: str = "",
    ):

        if benchmark_type is BenchmarkType.BUILDING_RECREATION and geometry_reference is None:
            raise InvalidBenchmarkDefinitionError(
                "A BUILDING_RECREATION benchmark requires a geometry_reference -- a whole-building "
                "recreation with no geometry at all is not a meaningful record. Use "
                "BenchmarkType.DATASET_VALIDATION for a scalar, geometry-free reference value.",
            )

        # Citation facts -- immutable once constructed (see this
        # module's own docstring).
        self._benchmark_id = benchmark_id if benchmark_id is not None else str(uuid.uuid4())
        self._title = title
        self._source_citation = source_citation
        self._dataset = dataset
        self._benchmark_type = benchmark_type
        self._doi = doi
        self._authors = tuple(authors)
        self._publication_year = publication_year
        self._venue = venue
        self._published_values: Dict[str, PublishedValue] = dict(published_values or {})
        self._assumptions = tuple(assumptions)
        self._created_at = created_at if created_at is not None else _utc_now_iso()

        # Evolving state.
        self._tags: List[str] = list(tags)
        self._geometry_reference = geometry_reference
        self._dataset_artifacts = dataset_artifacts
        self._validation_status = validation_status
        self._calibration_history: List[str] = list(calibration_history)
        self._current_error = current_error
        self._notes = notes
        self._updated_at = updated_at if updated_at is not None else self._created_at
        self._version = version

        self.extra: Dict[str, Any] = dict(extra) if extra else {}

    # =====================================================
    # Identity + citation facts (read-only)
    # =====================================================

    @property
    def benchmark_id(self) -> str:
        return self._benchmark_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def source_citation(self) -> str:
        return self._source_citation

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def benchmark_type(self) -> BenchmarkType:
        return self._benchmark_type

    @property
    def doi(self) -> Optional[str]:
        return self._doi

    @property
    def authors(self) -> Tuple[str, ...]:
        return self._authors

    @property
    def publication_year(self) -> Optional[int]:
        return self._publication_year

    @property
    def venue(self) -> str:
        return self._venue

    @property
    def published_values(self) -> Mapping[str, PublishedValue]:
        return dict(self._published_values)

    @property
    def assumptions(self) -> Tuple[str, ...]:
        return self._assumptions

    @property
    def created_at(self) -> str:
        return self._created_at

    # =====================================================
    # Evolving state
    # =====================================================

    @property
    def tags(self) -> Tuple[str, ...]:
        return tuple(self._tags)

    @property
    def geometry_reference(self) -> Optional[GeometryVersion]:
        return self._geometry_reference

    @property
    def dataset_artifacts(self) -> Optional[str]:
        return self._dataset_artifacts

    @property
    def validation_status(self) -> ValidationStatus:
        return self._validation_status

    @property
    def calibration_history(self) -> Tuple[str, ...]:
        return tuple(self._calibration_history)

    @property
    def current_error(self) -> Optional[dict]:
        return self._current_error

    @property
    def notes(self) -> str:
        return self._notes

    @property
    def updated_at(self) -> str:
        return self._updated_at

    @property
    def version(self) -> int:
        return self._version

    def _touch(self) -> None:

        self._updated_at = _utc_now_iso()
        self._version += 1

    def add_tag(self, tag: str) -> None:

        if tag not in self._tags:
            self._tags.append(tag)
            self._touch()

    def remove_tag(self, tag: str) -> None:

        if tag in self._tags:
            self._tags.remove(tag)
            self._touch()

    def set_geometry_reference(self, geometry_reference: Optional[GeometryVersion]) -> None:

        self._geometry_reference = geometry_reference
        self._touch()

    def set_dataset_artifacts(self, dataset_artifacts: Optional[str]) -> None:

        self._dataset_artifacts = dataset_artifacts
        self._touch()

    def set_validation_status(self, status: ValidationStatus) -> None:

        self._validation_status = status
        self._touch()

    def add_calibration_session(self, session_id: str) -> None:

        # Append-only, idempotent -- matches CalibrationProject.
        # add_benchmark_id()'s own identical convention.
        if session_id not in self._calibration_history:
            self._calibration_history.append(session_id)
            self._touch()

    def set_current_error(self, current_error: Optional[dict]) -> None:

        self._current_error = current_error
        self._touch()

    def set_notes(self, notes: str) -> None:

        self._notes = notes
        self._touch()

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": self._benchmark_id,
            "title": self._title,
            "source_citation": self._source_citation,
            "dataset": self._dataset,
            "benchmark_type": self._benchmark_type.value,
            "doi": self._doi,
            "authors": list(self._authors),
            "publication_year": self._publication_year,
            "venue": self._venue,
            "published_values": {name: value.to_dict() for name, value in self._published_values.items()},
            "assumptions": list(self._assumptions),
            "tags": list(self._tags),
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "version": self._version,
            "geometry_reference": self._geometry_reference.to_dict() if self._geometry_reference else None,
            "dataset_artifacts": self._dataset_artifacts,
            "validation_status": self._validation_status.value,
            "calibration_history": list(self._calibration_history),
            "current_error": self._current_error,
            "notes": self._notes,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PublishedBenchmark":

        # No schema_version check here -- calibration_studio/storage.py's
        # own job, exactly like CalibrationProject/CalibrationSession.
        # from_dict()'s identical division of responsibility.

        raw_type = data.get("benchmark_type")
        try:
            benchmark_type = BenchmarkType(raw_type)
        except ValueError:
            raise CorruptedBenchmarkRecordError(
                f"Unrecognised benchmark_type {raw_type!r} in stored benchmark "
                f"{data.get('benchmark_id')!r}",
            )

        raw_status = data.get("validation_status", ValidationStatus.NOT_RUN.value)
        try:
            validation_status = ValidationStatus(raw_status)
        except ValueError:
            raise CorruptedBenchmarkRecordError(
                f"Unrecognised validation_status {raw_status!r} in stored benchmark "
                f"{data.get('benchmark_id')!r}",
            )

        published_values = {
            name: PublishedValue.from_dict(raw)
            for name, raw in (data.get("published_values") or {}).items()
        }

        raw_geometry = data.get("geometry_reference")
        geometry_reference = GeometryVersion.from_dict(raw_geometry) if raw_geometry else None

        extra = dict(data.get("extra") or {})

        # Forward compatibility -- identical reasoning to
        # CalibrationSession/CalibrationProject.from_dict()'s own
        # handling: anything this schema version doesn't recognise is
        # preserved in `extra`, never silently discarded.
        for key, value in data.items():
            if key not in _KNOWN_TOP_LEVEL_KEYS:
                extra[key] = value

        return cls(
            title=data.get("title", ""),
            source_citation=data.get("source_citation", ""),
            dataset=data.get("dataset", ""),
            benchmark_type=benchmark_type,
            doi=data.get("doi"),
            authors=tuple(data.get("authors") or ()),
            publication_year=data.get("publication_year"),
            venue=data.get("venue", ""),
            published_values=published_values,
            assumptions=tuple(data.get("assumptions") or ()),
            tags=tuple(data.get("tags") or ()),
            extra=extra,
            geometry_reference=geometry_reference,
            benchmark_id=data.get("benchmark_id"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            version=data.get("version", 1),
            dataset_artifacts=data.get("dataset_artifacts"),
            validation_status=validation_status,
            calibration_history=tuple(data.get("calibration_history") or ()),
            current_error=data.get("current_error"),
            notes=data.get("notes", ""),
        )

    def __repr__(self) -> str:

        return (
            f"PublishedBenchmark(benchmark_id={self._benchmark_id!r}, title={self._title!r}, "
            f"type={self._benchmark_type.value}, status={self._validation_status.value})"
        )
