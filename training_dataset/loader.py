import csv
import json
import os

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =====================================================
# Dataset Loader (Phase 1) -- reads an already-completed Campaign
# Studio output directory and exposes one SimulationSample per
# scenario. This package is a consumer only: nothing here generates,
# validates, or (re)simulates anything -- every value is read verbatim
# from artifacts designer.campaign.campaign_worker._export_scenario_
# artifacts() already wrote (dataset_builder's datasets/<scenario_id>/,
# dataset_builder's timelines/<scenario_id>/, ground_truth's
# ground_truth/<scenario_id>/, decision_policy's decision_policy/
# <scenario_id>/).
# =====================================================

DATASETS_DIRNAME = "datasets"
TIMELINES_DIRNAME = "timelines"
GROUND_TRUTH_DIRNAME = "ground_truth"
DECISION_POLICY_DIRNAME = "decision_policy"

SCENARIO_FEATURES_FILENAME = "scenario_features.csv"
SIMULATION_OUTCOMES_FILENAME = "simulation_outcomes.csv"
ZONE_RESULTS_FILENAME = "zone_results.csv"
TIMELINE_FILENAME = "timeline.csv"
GROUND_TRUTH_FILENAME = "ground_truth.json"
DECISION_POLICY_FILENAME = "decision_policy.json"


class SampleLoadError(Exception):

    # Raised for exactly one scenario_id's worth of artifacts -- every
    # structural problem load_sample() can detect (missing file, empty
    # required CSV, broken JSON, an artifact whose own embedded
    # scenario_id disagrees with the directory it was found under)
    # raises this, never a bare Exception/KeyError, so callers can
    # catch it specifically (see load_campaign()'s strict=False path).

    def __init__(self, scenario_id: Optional[str], reason: str):

        self.scenario_id = scenario_id
        self.reason = reason

        super().__init__(f"{scenario_id}: {reason}")


@dataclass(frozen=True)
class SimulationSample:

    # One complete simulation sample -- every artifact a single
    # accepted scenario produced, already verified (by load_sample())
    # to share one scenario_id. Plain dicts/lists throughout, the same
    # convention ground_truth.GroundTruth/decision_policy.DecisionPolicy
    # already use for their own to_dict() shapes, so no further
    # unwrapping is needed before handing a sample to pandas/statistics/
    # export code.

    scenario_id: str

    scenario_features: Dict[str, Any]
    simulation_outcome: Dict[str, Any]

    zone_results: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    ground_truth: Optional[Dict[str, Any]] = None
    decision_policy: Optional[Dict[str, Any]] = None


@dataclass
class CampaignDataset:

    # A collection of successfully loaded SimulationSamples for one
    # Campaign output directory, plus any per-scenario errors
    # encountered along the way (only ever populated when load_campaign()
    # was called with strict=False -- see its docstring).

    campaign_dir: str
    samples: List[SimulationSample] = field(default_factory=list)
    errors: List[SampleLoadError] = field(default_factory=list)

    @property
    def scenario_ids(self) -> List[str]:

        return [sample.scenario_id for sample in self.samples]

    def __len__(self) -> int:

        return len(self.samples)

    def __iter__(self):

        return iter(self.samples)

    def __getitem__(self, index):

        return self.samples[index]


# =====================================================
# Discovery -- datasets/<scenario_id>/ is the one directory every
# accepted scenario is guaranteed to have (Dataset Builder V1 runs for
# every accepted scenario; timeline/ground_truth/decision_policy are
# each written immediately after in the same _export_scenario_
# artifacts() call, but datasets/ is the first and therefore the most
# conservative directory to enumerate scenario_ids from).
# =====================================================


def discover_scenario_ids(campaign_dir) -> List[str]:

    datasets_dir = os.path.join(str(campaign_dir), DATASETS_DIRNAME)

    if not os.path.isdir(datasets_dir):
        return []

    return sorted(
        name
        for name in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, name))
    )


def scenario_artifact_paths(campaign_dir, scenario_id: str) -> Dict[str, str]:

    campaign_dir = str(campaign_dir)

    return {
        "scenario_features": os.path.join(
            campaign_dir, DATASETS_DIRNAME, scenario_id, SCENARIO_FEATURES_FILENAME,
        ),
        "simulation_outcomes": os.path.join(
            campaign_dir, DATASETS_DIRNAME, scenario_id, SIMULATION_OUTCOMES_FILENAME,
        ),
        "zone_results": os.path.join(
            campaign_dir, DATASETS_DIRNAME, scenario_id, ZONE_RESULTS_FILENAME,
        ),
        "timeline": os.path.join(
            campaign_dir, TIMELINES_DIRNAME, scenario_id, TIMELINE_FILENAME,
        ),
        "ground_truth": os.path.join(
            campaign_dir, GROUND_TRUTH_DIRNAME, scenario_id, GROUND_TRUTH_FILENAME,
        ),
        "decision_policy": os.path.join(
            campaign_dir, DECISION_POLICY_DIRNAME, scenario_id, DECISION_POLICY_FILENAME,
        ),
    }


# =====================================================
# Value coercion -- every source file here is either a CSV (dataset_
# builder.exporters.export_csv() writes every value as plain text,
# with "" for None -- dataset_builder/exporters.py:_sanitized_row()) or
# JSON (already typed). CSV values are coerced back to int/float/bool
# where they round-trip cleanly, so Phase 3 statistics never has to
# re-parse strings; anything that doesn't round-trip (ids, enum names
# like "OPEN"/"FAILED") is left as the original string.
# =====================================================


def _coerce_csv_value(value: str) -> Any:

    if value is None or value == "":
        return None

    if value in ("True", "False"):
        return value == "True"

    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    return value


def _read_csv_rows(
    path: str, scenario_id: Optional[str], label: str, *, required: bool = True,
) -> Optional[List[Dict[str, Any]]]:

    if not os.path.isfile(path):

        if required:
            raise SampleLoadError(scenario_id, f"missing required file {label} ({path})")

        return None

    try:

        with open(path, "r", newline="", encoding="utf-8") as handle:

            reader = csv.DictReader(handle)

            return [
                {key: _coerce_csv_value(value) for key, value in row.items()}
                for row in reader
            ]

    except csv.Error as exc:

        raise SampleLoadError(scenario_id, f"corrupted {label} CSV ({path}): {exc}") from exc


def _read_json_object(
    path: str, scenario_id: Optional[str], label: str, *, required: bool = True,
) -> Optional[Dict[str, Any]]:

    if not os.path.isfile(path):

        if required:
            raise SampleLoadError(scenario_id, f"missing required file {label} ({path})")

        return None

    try:

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

    except json.JSONDecodeError as exc:

        raise SampleLoadError(scenario_id, f"broken JSON in {label} ({path}): {exc}") from exc

    if not isinstance(data, dict):
        raise SampleLoadError(scenario_id, f"{label} is not a JSON object ({path})")

    return data


def _check_embedded_scenario_id(value: Any, scenario_id: str, label: str) -> None:

    if value != scenario_id:
        raise SampleLoadError(
            scenario_id,
            f"{label} declares scenario_id {value!r}, expected {scenario_id!r}",
        )


# =====================================================
# Per-scenario / whole-campaign loading.
# =====================================================


def load_sample(
    campaign_dir,
    scenario_id: str,
    *,
    require_timeline: bool = True,
    require_ground_truth: bool = True,
    require_decision_policy: bool = True,
) -> SimulationSample:

    paths = scenario_artifact_paths(campaign_dir, scenario_id)

    feature_rows = _read_csv_rows(paths["scenario_features"], scenario_id, "scenario_features.csv")

    if len(feature_rows) != 1:
        raise SampleLoadError(
            scenario_id,
            f"scenario_features.csv must contain exactly 1 row, found {len(feature_rows)}",
        )

    scenario_features = feature_rows[0]
    _check_embedded_scenario_id(
        scenario_features.get("scenario_id"), scenario_id, "scenario_features.csv",
    )

    outcome_rows = _read_csv_rows(
        paths["simulation_outcomes"], scenario_id, "simulation_outcomes.csv",
    )

    if len(outcome_rows) != 1:
        raise SampleLoadError(
            scenario_id,
            f"simulation_outcomes.csv must contain exactly 1 row, found {len(outcome_rows)}",
        )

    simulation_outcome = outcome_rows[0]
    _check_embedded_scenario_id(
        simulation_outcome.get("scenario_id"), scenario_id, "simulation_outcomes.csv",
    )

    zone_results = _read_csv_rows(paths["zone_results"], scenario_id, "zone_results.csv") or []

    for row in zone_results:
        _check_embedded_scenario_id(row.get("scenario_id"), scenario_id, "zone_results.csv")

    timeline = _read_csv_rows(
        paths["timeline"], scenario_id, "timeline.csv", required=require_timeline,
    ) or []

    for row in timeline:
        _check_embedded_scenario_id(row.get("scenario_id"), scenario_id, "timeline.csv")

    ground_truth = _read_json_object(
        paths["ground_truth"], scenario_id, "ground_truth.json", required=require_ground_truth,
    )

    if ground_truth is not None:
        _check_embedded_scenario_id(
            ground_truth.get("scenario_id"), scenario_id, "ground_truth.json",
        )

    decision_policy = _read_json_object(
        paths["decision_policy"], scenario_id, "decision_policy.json",
        required=require_decision_policy,
    )

    if decision_policy is not None:
        _check_embedded_scenario_id(
            decision_policy.get("scenario_id"), scenario_id, "decision_policy.json",
        )

    return SimulationSample(
        scenario_id=scenario_id,
        scenario_features=scenario_features,
        simulation_outcome=simulation_outcome,
        zone_results=zone_results,
        timeline=timeline,
        ground_truth=ground_truth,
        decision_policy=decision_policy,
    )


def load_campaign(
    campaign_dir,
    *,
    strict: bool = True,
    require_timeline: bool = True,
    require_ground_truth: bool = True,
    require_decision_policy: bool = True,
) -> CampaignDataset:

    # strict=True (the default) raises on the first scenario that fails
    # to load -- the right default for "about to train on this," where
    # a silently-partial dataset is worse than a loud failure.
    #
    # strict=False instead collects every SampleLoadError into
    # CampaignDataset.errors and keeps going -- this is what makes
    # "loading partial campaigns" (a campaign directory where some
    # scenarios are missing artifacts, e.g. a cancelled or still-running
    # campaign) and Phase 2 validation (validator.validate_campaign()
    # reuses this exact path) possible without reimplementing the
    # per-scenario read logic a second time.

    scenario_ids = discover_scenario_ids(campaign_dir)

    samples: List[SimulationSample] = []
    errors: List[SampleLoadError] = []

    for scenario_id in scenario_ids:

        try:

            samples.append(
                load_sample(
                    campaign_dir, scenario_id,
                    require_timeline=require_timeline,
                    require_ground_truth=require_ground_truth,
                    require_decision_policy=require_decision_policy,
                )
            )

        except SampleLoadError as exc:

            if strict:
                raise

            errors.append(exc)

    return CampaignDataset(campaign_dir=str(campaign_dir), samples=samples, errors=errors)
