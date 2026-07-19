import csv
import hashlib
import json
import os

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from training_dataset.loader import (
    SimulationSample,
    discover_scenario_ids,
    load_campaign,
)


# =====================================================
# Dataset Validation (Phase 2) -- run before any ML training. Never
# raises on a bad campaign (that would defeat the point of a report);
# every problem loader.load_sample() can already detect (missing
# files, broken JSON, mismatched scenario_ids, ...) is instead
# collected here as a ValidationIssue, alongside the campaign-wide
# checks (duplicates, row-count consistency, invalid values, corrupted
# timelines) that only make sense once every sample is loaded.
# =====================================================

REQUIRED_SCENARIO_FEATURE_COLUMNS = ("scenario_id", "definition_id", "seed", "total_occupants")

REQUIRED_OUTCOME_COLUMNS = (
    "scenario_id",
    "total_evacuation_time",
    "people_evacuated",
    "people_trapped",
    "reachable_occupants",
    "unreachable_occupants",
    "building_cleared",
    "simulation_finished",
)

REQUIRED_ZONE_RESULT_COLUMNS = ("scenario_id", "zone_id", "initial_occupants", "evacuated", "trapped")

REQUIRED_TIMELINE_COLUMNS = ("scenario_id", "simulation_time")

NON_NEGATIVE_OUTCOME_FIELDS = (
    "people_evacuated", "people_trapped", "reachable_occupants", "unreachable_occupants",
)

CATALOG_FILENAME = "catalog.csv"


@dataclass(frozen=True)
class ValidationIssue:

    severity: str  # "error" | "warning"
    code: str
    message: str
    scenario_id: Optional[str] = None

    def to_dict(self) -> dict:

        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "scenario_id": self.scenario_id,
        }


@dataclass
class ValidationReport:

    campaign_dir: str
    scenarios_discovered: int
    samples_loaded: int
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:

        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:

        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def is_valid(self) -> bool:

        # A campaign with only warnings (e.g. a duplicate content hash,
        # or one scenario with an empty zone_results.csv) is usable for
        # training, just worth a second look -- only "error" severity
        # issues make the campaign unusable outright.

        return len(self.errors) == 0

    def to_dict(self) -> dict:

        return {
            "campaign_dir": self.campaign_dir,
            "scenarios_discovered": self.scenarios_discovered,
            "samples_loaded": self.samples_loaded,
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def sample_content_hash(sample: SimulationSample) -> str:

    # A stable fingerprint of everything about a sample except its own
    # identity (scenario_id/seed) -- two scenario_ids sharing this hash
    # means the underlying scenario_features/simulation_outcome content
    # is indistinguishable, a data-leakage risk Phase 4's splitter
    # cares about (train and test should not secretly contain the same
    # scenario twice under different ids).

    payload = {
        "scenario_features": {
            key: value
            for key, value in sample.scenario_features.items()
            if key not in ("scenario_id", "seed")
        },
        "simulation_outcome": {
            key: value for key, value in sample.simulation_outcome.items() if key != "scenario_id"
        },
    }

    encoded = json.dumps(payload, sort_keys=True, default=str)

    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# =====================================================


def validate_campaign(campaign_dir) -> ValidationReport:

    scenario_ids = discover_scenario_ids(campaign_dir)

    dataset = load_campaign(
        campaign_dir,
        strict=False,
        require_timeline=True,
        require_ground_truth=True,
        require_decision_policy=True,
    )

    issues: List[ValidationIssue] = []

    for load_error in dataset.errors:
        issues.append(
            ValidationIssue(
                severity="error",
                code=_classify_load_error(load_error.reason),
                message=load_error.reason,
                scenario_id=load_error.scenario_id,
            )
        )

    issues.extend(_check_catalog_duplicate_scenario_ids(campaign_dir))
    issues.extend(_check_duplicate_content_hashes(dataset.samples))
    issues.extend(_check_row_count_consistency(dataset.samples))

    for sample in dataset.samples:
        issues.extend(_check_empty_secondary_csvs(sample))
        issues.extend(_check_missing_columns(sample))
        issues.extend(_check_invalid_values(sample))
        issues.extend(_check_corrupted_timeline(sample))

    return ValidationReport(
        campaign_dir=str(campaign_dir),
        scenarios_discovered=len(scenario_ids),
        samples_loaded=len(dataset.samples),
        issues=issues,
    )


# =====================================================


def _classify_load_error(reason: str) -> str:

    if "missing required file" in reason:
        return "missing_file"

    if "broken JSON" in reason:
        return "broken_json"

    if "is not a JSON object" in reason:
        return "invalid_json_shape"

    if "declares scenario_id" in reason:
        return "scenario_id_mismatch"

    if "must contain exactly" in reason:
        return "invalid_row_count"

    if "corrupted" in reason:
        return "corrupted_csv"

    return "load_error"


def _check_catalog_duplicate_scenario_ids(campaign_dir) -> List[ValidationIssue]:

    # scenario_storage's own append-only catalog (scenario_storage/
    # catalog.py) -- read-only here, never written to. A scenario_id
    # appearing more than once would mean save_scenario()'s own
    # FileExistsError guard was somehow bypassed; this package can only
    # ever detect that after the fact, from the exported artifact.

    path = os.path.join(str(campaign_dir), CATALOG_FILENAME)

    if not os.path.isfile(path):
        return []

    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    counts = Counter(row.get("scenario_id") for row in rows if row.get("scenario_id"))

    return [
        ValidationIssue(
            severity="error",
            code="duplicate_scenario_id",
            message=f"scenario_id {scenario_id!r} appears {count} times in {CATALOG_FILENAME}",
            scenario_id=scenario_id,
        )
        for scenario_id, count in counts.items()
        if count > 1
    ]


def _check_duplicate_content_hashes(samples: List[SimulationSample]) -> List[ValidationIssue]:

    groups: Dict[str, List[str]] = {}

    for sample in samples:
        groups.setdefault(sample_content_hash(sample), []).append(sample.scenario_id)

    issues = []

    for digest, scenario_ids in groups.items():

        if len(scenario_ids) > 1:

            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="duplicate_content_hash",
                    message=(
                        f"scenarios {sorted(scenario_ids)} share identical "
                        f"scenario_features/simulation_outcome content (hash "
                        f"{digest[:12]}) -- possible duplicate samples"
                    ),
                    scenario_id=None,
                )
            )

    return issues


def _check_row_count_consistency(samples: List[SimulationSample]) -> List[ValidationIssue]:

    # Every scenario in one campaign shares the same Building (§
    # designer/campaign/campaign_worker.py), so zone_results.csv should
    # have the same row count (one row per zone) for every scenario --
    # a differing count usually means a corrupted/truncated export.

    counts = Counter(len(sample.zone_results) for sample in samples)

    if len(counts) <= 1:
        return []

    mode_count, _ = counts.most_common(1)[0]

    return [
        ValidationIssue(
            severity="warning",
            code="inconsistent_zone_result_count",
            message=(
                f"zone_results.csv has {len(sample.zone_results)} rows, expected "
                f"{mode_count} (the campaign-wide mode)"
            ),
            scenario_id=sample.scenario_id,
        )
        for sample in samples
        if len(sample.zone_results) != mode_count
    ]


def _check_empty_secondary_csvs(sample: SimulationSample) -> List[ValidationIssue]:

    issues = []

    if not sample.zone_results:
        issues.append(
            ValidationIssue(
                "warning", "empty_zone_results", "zone_results.csv has no rows", sample.scenario_id,
            )
        )

    if not sample.timeline:
        issues.append(
            ValidationIssue(
                "warning", "empty_timeline", "timeline.csv has no rows", sample.scenario_id,
            )
        )

    return issues


def _check_missing_columns(sample: SimulationSample) -> List[ValidationIssue]:

    issues = []

    missing_features = [
        column for column in REQUIRED_SCENARIO_FEATURE_COLUMNS
        if column not in sample.scenario_features
    ]
    if missing_features:
        issues.append(
            ValidationIssue(
                "error", "missing_required_columns",
                f"scenario_features.csv is missing columns: {missing_features}",
                sample.scenario_id,
            )
        )

    missing_outcome = [
        column for column in REQUIRED_OUTCOME_COLUMNS if column not in sample.simulation_outcome
    ]
    if missing_outcome:
        issues.append(
            ValidationIssue(
                "error", "missing_required_columns",
                f"simulation_outcomes.csv is missing columns: {missing_outcome}",
                sample.scenario_id,
            )
        )

    if sample.zone_results:

        missing_zone = [
            column for column in REQUIRED_ZONE_RESULT_COLUMNS
            if column not in sample.zone_results[0]
        ]
        if missing_zone:
            issues.append(
                ValidationIssue(
                    "error", "missing_required_columns",
                    f"zone_results.csv is missing columns: {missing_zone}",
                    sample.scenario_id,
                )
            )

    if sample.timeline:

        missing_timeline = [
            column for column in REQUIRED_TIMELINE_COLUMNS if column not in sample.timeline[0]
        ]
        if missing_timeline:
            issues.append(
                ValidationIssue(
                    "error", "missing_required_columns",
                    f"timeline.csv is missing columns: {missing_timeline}",
                    sample.scenario_id,
                )
            )

    return issues


def _check_invalid_values(sample: SimulationSample) -> List[ValidationIssue]:

    issues = []
    outcome = sample.simulation_outcome

    for field_name in NON_NEGATIVE_OUTCOME_FIELDS:

        value = outcome.get(field_name)

        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            issues.append(
                ValidationIssue(
                    "error", "invalid_value",
                    f"simulation_outcomes.{field_name} = {value!r} (expected a non-negative number)",
                    sample.scenario_id,
                )
            )

    evacuation_time = outcome.get("total_evacuation_time")

    if evacuation_time is not None and (
        not isinstance(evacuation_time, (int, float)) or evacuation_time < 0
    ):
        issues.append(
            ValidationIssue(
                "error", "invalid_value",
                f"simulation_outcomes.total_evacuation_time = {evacuation_time!r} "
                f"(expected a non-negative number or empty)",
                sample.scenario_id,
            )
        )

    total_occupants = sample.scenario_features.get("total_occupants")
    reachable = outcome.get("reachable_occupants")
    unreachable = outcome.get("unreachable_occupants")

    if (
        isinstance(total_occupants, (int, float))
        and isinstance(reachable, (int, float))
        and isinstance(unreachable, (int, float))
        and (reachable + unreachable) != total_occupants
    ):
        issues.append(
            ValidationIssue(
                "warning", "inconsistent_occupant_count",
                f"reachable_occupants ({reachable}) + unreachable_occupants ({unreachable}) "
                f"!= scenario_features.total_occupants ({total_occupants})",
                sample.scenario_id,
            )
        )

    return issues


def _check_corrupted_timeline(sample: SimulationSample) -> List[ValidationIssue]:

    issues = []
    previous_time = None

    for row in sample.timeline:

        current_time = row.get("simulation_time")

        if not isinstance(current_time, (int, float)):
            continue

        if previous_time is not None and current_time < previous_time:
            issues.append(
                ValidationIssue(
                    "error", "corrupted_timeline",
                    f"timeline.csv simulation_time decreases ({previous_time} -> {current_time})",
                    sample.scenario_id,
                )
            )

        previous_time = current_time

    return issues
