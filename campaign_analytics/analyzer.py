import math
import re

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import training_dataset

from campaign_analytics import distributions as distributions_module


# =====================================================
# Campaign Analytics -- a read-only audit layer over an already-
# completed Campaign Studio output directory, built to answer one
# question before any AI training starts: is this dataset realistic
# and unbiased enough to train on?
#
# The only dependency on another package in this repository is
# training_dataset.load_campaign() -- itself a consumer-only package
# reading the exact same exported artifacts (datasets/<scenario_id>/,
# timelines/<scenario_id>/, ground_truth/<scenario_id>/,
# decision_policy/<scenario_id>/) this package would otherwise have to
# parse a second time. Nothing here imports scenario_generator,
# scenario_pipeline, simulation_runtime, or any other package that
# actually produces a campaign -- see the module docstrings in
# engineering_checks.py/anomaly_detection.py/distributions.py, which
# take plain already-loaded samples and import nothing from either
# training_dataset or the simulation stack at all.
# =====================================================

DOOR_STATE_COLUMN_PATTERN = re.compile(r"^Door_\d+_State$")
EXIT_STATE_COLUMN_PATTERN = re.compile(r"^Exit_\d+_State$")
DETECTOR_STATE_COLUMN_PATTERN = re.compile(r"^Detector_\d+_State$")
CAMERA_STATE_COLUMN_PATTERN = re.compile(r"^Camera_\d+_State$")

EVACUATION_TIME_PERCENTILES = (50, 90, 95, 99)

AVAILABLE_DEVICE_STATE = "AVAILABLE"
FAILED_DEVICE_STATE = "FAILED"


@dataclass(frozen=True)
class Finding:

    # Phase 2's own rule: "Produce warnings, not failures." This
    # package audits a campaign, it never gates one -- severity is
    # therefore always "warning" or "info", never "error".

    severity: str
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


@dataclass(frozen=True)
class CampaignOverview:

    total_scenarios: int

    # Neither total_requested nor rejected is ever written to disk by
    # designer.campaign.campaign_worker (CampaignSummary is an
    # in-memory-only result, shown once in CampaignWindow and never
    # exported) -- so both stay None (never fabricated) unless the
    # caller of analyze_campaign() explicitly supplies them (e.g. from
    # a CampaignSummary captured at the end of a live run).
    total_requested: Optional[int]
    rejected: Optional[int]
    acceptance_ratio: Optional[float]

    average_evacuation_time: Optional[float]
    minimum_evacuation_time: Optional[float]
    maximum_evacuation_time: Optional[float]
    evacuation_time_percentiles: Dict[str, Optional[float]]

    average_occupants: Optional[float]

    fire_profile_frequencies: Dict[str, int]
    ignition_zone_frequencies: Dict[str, int]
    detector_failure_frequencies: Dict[str, int]
    camera_failure_frequencies: Dict[str, int]
    door_state_frequencies: Dict[str, int]
    exit_state_frequencies: Dict[str, int]

    def to_dict(self) -> dict:

        return {
            "total_scenarios": self.total_scenarios,
            "total_requested": self.total_requested,
            "rejected": self.rejected,
            "acceptance_ratio": self.acceptance_ratio,
            "average_evacuation_time": self.average_evacuation_time,
            "minimum_evacuation_time": self.minimum_evacuation_time,
            "maximum_evacuation_time": self.maximum_evacuation_time,
            "evacuation_time_percentiles": dict(self.evacuation_time_percentiles),
            "average_occupants": self.average_occupants,
            "fire_profile_frequencies": dict(self.fire_profile_frequencies),
            "ignition_zone_frequencies": dict(self.ignition_zone_frequencies),
            "detector_failure_frequencies": dict(self.detector_failure_frequencies),
            "camera_failure_frequencies": dict(self.camera_failure_frequencies),
            "door_state_frequencies": dict(self.door_state_frequencies),
            "exit_state_frequencies": dict(self.exit_state_frequencies),
        }


@dataclass(frozen=True)
class CampaignKPIs:

    # average_rset: this codebase has no separate ASET/RSET model --
    # total_evacuation_time (Simulation Runtime's own completed-run
    # measurement) is the only "required safe egress time" quantity
    # actually computed anywhere, so it is used here as-is, under its
    # fire-engineering name.
    average_rset: Optional[float]
    peak_evacuation_time: Optional[float]

    average_queue_time: Optional[float]
    average_congestion_duration: Optional[float]

    # Coverage-based, not capacity-based: no exit/stair models a
    # numeric capacity anywhere in this codebase's exported artifacts
    # (see ground_truth/bottleneck.py's own note on this), so
    # "utilization" here means "the fraction of this building's
    # exits/stairs that were ever actually used across the whole
    # campaign," not a percentage of any modeled throughput capacity.
    exit_utilization_percentage: Optional[float]
    stair_utilization_percentage: Optional[float]

    # "Coverage" here means the fraction of positional detector/camera
    # slots that were AVAILABLE (not FAILED) on average across the
    # campaign -- i.e. 1 - failure rate, expressed as a percentage.
    detector_activation_coverage: Optional[float]
    camera_coverage: Optional[float]

    # Mean of (peak current_hazard_score / simulation_time at that
    # peak) across scenarios -- a hazard-score-units-per-second rate,
    # derived only from timeline.csv's own already-computed column.
    average_hazard_growth_rate: Optional[float]

    evacuation_success_rate: Optional[float]
    unreachable_occupant_rate: Optional[float]

    def to_dict(self) -> dict:

        return {
            "average_rset": self.average_rset,
            "peak_evacuation_time": self.peak_evacuation_time,
            "average_queue_time": self.average_queue_time,
            "average_congestion_duration": self.average_congestion_duration,
            "exit_utilization_percentage": self.exit_utilization_percentage,
            "stair_utilization_percentage": self.stair_utilization_percentage,
            "detector_activation_coverage": self.detector_activation_coverage,
            "camera_coverage": self.camera_coverage,
            "average_hazard_growth_rate": self.average_hazard_growth_rate,
            "evacuation_success_rate": self.evacuation_success_rate,
            "unreachable_occupant_rate": self.unreachable_occupant_rate,
        }


@dataclass(frozen=True)
class CampaignAnalysis:

    campaign_dir: str
    overview: CampaignOverview
    kpis: CampaignKPIs
    distributions: Dict[str, Any]
    engineering_findings: Tuple[Finding, ...] = field(default_factory=tuple)
    anomaly_findings: Tuple[Finding, ...] = field(default_factory=tuple)

    # Kept for callers who want direct access to the loaded samples
    # (report.py/visualizations.py both take samples via this field
    # rather than re-loading); intentionally excluded from to_dict()
    # since a CampaignDataset isn't itself JSON-serializable.
    dataset: Any = None

    @property
    def findings(self) -> Tuple[Finding, ...]:

        return self.engineering_findings + self.anomaly_findings

    def to_dict(self) -> dict:

        return {
            "campaign_dir": self.campaign_dir,
            "overview": self.overview.to_dict(),
            "kpis": self.kpis.to_dict(),
            "distributions": self.distributions,
            "engineering_findings": [finding.to_dict() for finding in self.engineering_findings],
            "anomaly_findings": [finding.to_dict() for finding in self.anomaly_findings],
        }


# =====================================================
# Phase 1 -- Campaign Overview.
# =====================================================


def compute_overview(
    samples, *, total_requested: Optional[int] = None, rejected: Optional[int] = None,
) -> CampaignOverview:

    samples = list(samples)
    total_scenarios = len(samples)

    evacuation_times = sorted(_numeric_values(samples, _get_evacuation_time))
    occupant_counts = _numeric_values(samples, _get_total_occupants)

    acceptance_ratio = (
        (total_scenarios / total_requested) if total_requested else None
    )

    return CampaignOverview(
        total_scenarios=total_scenarios,
        total_requested=total_requested,
        rejected=rejected,
        acceptance_ratio=acceptance_ratio,
        average_evacuation_time=_mean(evacuation_times),
        minimum_evacuation_time=evacuation_times[0] if evacuation_times else None,
        maximum_evacuation_time=evacuation_times[-1] if evacuation_times else None,
        evacuation_time_percentiles={
            f"p{p}": _percentile(evacuation_times, p) for p in EVACUATION_TIME_PERCENTILES
        },
        average_occupants=_mean(occupant_counts),
        fire_profile_frequencies=_distribution(samples, lambda s: s.scenario_features.get("fire_profile")),
        ignition_zone_frequencies=_distribution(samples, lambda s: s.scenario_features.get("ignition_zone")),
        detector_failure_frequencies=_column_value_frequencies(samples, DETECTOR_STATE_COLUMN_PATTERN),
        camera_failure_frequencies=_column_value_frequencies(samples, CAMERA_STATE_COLUMN_PATTERN),
        door_state_frequencies=_column_value_frequencies(samples, DOOR_STATE_COLUMN_PATTERN),
        exit_state_frequencies=_column_value_frequencies(samples, EXIT_STATE_COLUMN_PATTERN),
    )


# =====================================================
# Phase 4 -- Engineering KPIs.
# =====================================================


def compute_kpis(samples) -> CampaignKPIs:

    samples = list(samples)

    evacuation_times = _numeric_values(samples, _get_evacuation_time)
    queue_times = [
        row.get("average_waiting_time")
        for sample in samples
        for row in sample.zone_results
        if isinstance(row.get("average_waiting_time"), (int, float))
    ]
    congestion_durations = [
        sample.ground_truth.get("congestion_duration")
        for sample in samples
        if sample.ground_truth
        and isinstance(sample.ground_truth.get("congestion_duration"), (int, float))
    ]

    exit_universe, exits_used = _exit_universe_and_usage(samples)
    stair_universe, stairs_used = _stair_universe_and_usage(samples)

    total_occupants = sum(_numeric_values(samples, _get_total_occupants))
    total_evacuated = sum(
        value for value in _numeric_values(samples, lambda s: s.simulation_outcome.get("people_evacuated"))
    )
    total_unreachable = sum(
        value
        for value in _numeric_values(samples, lambda s: s.simulation_outcome.get("unreachable_occupants"))
    )
    building_cleared_count = sum(
        1 for sample in samples if sample.simulation_outcome.get("building_cleared") is True
    )

    return CampaignKPIs(
        average_rset=_mean(evacuation_times),
        peak_evacuation_time=max(evacuation_times) if evacuation_times else None,
        average_queue_time=_mean(queue_times),
        average_congestion_duration=_mean(congestion_durations),
        exit_utilization_percentage=_percentage(len(exits_used), len(exit_universe)),
        stair_utilization_percentage=_percentage(len(stairs_used), len(stair_universe)),
        detector_activation_coverage=_device_coverage(samples, DETECTOR_STATE_COLUMN_PATTERN),
        camera_coverage=_device_coverage(samples, CAMERA_STATE_COLUMN_PATTERN),
        average_hazard_growth_rate=_average_hazard_growth_rate(samples),
        evacuation_success_rate=_percentage(building_cleared_count, len(samples)),
        unreachable_occupant_rate=_percentage(total_unreachable, total_occupants),
    )


# =====================================================
# Orchestration -- ties Phases 1-4 together over one already-loaded
# campaign. Phase 2 (engineering_checks/anomaly_detection) and Phase 3
# (distributions) modules are imported lazily inside this function
# rather than at module load time: engineering_checks.py/anomaly_
# detection.py both import Finding from this module, so importing them
# eagerly at the top of analyzer.py (before Finding is defined) would
# be a circular import; deferring the import to call time -- by which
# point this module has already finished loading -- avoids it cleanly.
# =====================================================


def analyze_campaign(
    campaign_dir,
    *,
    strict: bool = False,
    total_requested: Optional[int] = None,
    rejected: Optional[int] = None,
) -> CampaignAnalysis:

    from campaign_analytics import anomaly_detection, engineering_checks

    dataset = training_dataset.load_campaign(campaign_dir, strict=strict)
    samples = list(dataset)

    return CampaignAnalysis(
        campaign_dir=str(campaign_dir),
        overview=compute_overview(samples, total_requested=total_requested, rejected=rejected),
        kpis=compute_kpis(samples),
        distributions=distributions_module.compute_distributions(samples),
        engineering_findings=tuple(engineering_checks.run_checks(samples)),
        anomaly_findings=tuple(anomaly_detection.run_checks(samples)),
        dataset=dataset,
    )


# =====================================================
# Generic helpers.
# =====================================================


def _get_evacuation_time(sample):

    return sample.simulation_outcome.get("total_evacuation_time")


def _get_total_occupants(sample):

    return sample.scenario_features.get("total_occupants")


def _numeric_values(samples, getter) -> List[float]:

    values = []

    for sample in samples:

        value = getter(sample)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(value)

    return values


def _mean(values) -> Optional[float]:

    return (sum(values) / len(values)) if values else None


def _percentage(numerator, denominator) -> Optional[float]:

    if not denominator:
        return None

    return (numerator / denominator) * 100.0


def _percentile(sorted_values: List[float], percentile: float) -> Optional[float]:

    if not sorted_values:
        return None

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)

    if lower_index == upper_index:
        return sorted_values[int(rank)]

    lower_weight = sorted_values[lower_index] * (upper_index - rank)
    upper_weight = sorted_values[upper_index] * (rank - lower_index)

    return lower_weight + upper_weight


def _distribution(samples, getter) -> Dict[str, int]:

    counter: Counter = Counter()

    for sample in samples:

        value = getter(sample)

        if value is None or value == "":
            continue

        counter[str(value)] += 1

    return dict(counter.most_common())


def _column_value_frequencies(samples, pattern: "re.Pattern") -> Dict[str, int]:

    counter: Counter = Counter()

    for sample in samples:
        for key, value in sample.scenario_features.items():
            if pattern.match(key) and value is not None:
                counter[str(value)] += 1

    return dict(counter.most_common())


def _device_coverage(samples, pattern: "re.Pattern") -> Optional[float]:

    total = 0
    available = 0

    for sample in samples:
        for key, value in sample.scenario_features.items():
            if pattern.match(key):
                total += 1
                if value == AVAILABLE_DEVICE_STATE:
                    available += 1

    return _percentage(available, total)


def _exit_universe_and_usage(samples):

    universe = set()
    used = set()

    for sample in samples:

        if sample.ground_truth:
            for entry in sample.ground_truth.get("exit_risk_scores") or []:
                if entry.get("exit_id"):
                    universe.add(entry["exit_id"])

        for row in sample.zone_results:
            if row.get("exit_used"):
                used.add(row["exit_used"])

    return universe, used & universe


def _stair_universe_and_usage(samples):

    universe = set()
    used = set()

    for sample in samples:

        if not sample.ground_truth:
            continue

        for entry in sample.ground_truth.get("stair_risk_scores") or []:
            if entry.get("stair_id"):
                universe.add(entry["stair_id"])

        for entry in sample.ground_truth.get("zone_route_stats") or []:
            if entry.get("preferred_stair"):
                used.add(entry["preferred_stair"])

    return universe, used & universe


def _average_hazard_growth_rate(samples) -> Optional[float]:

    rates = []

    for sample in samples:

        peak_score = None
        peak_time = None

        for row in sample.timeline:

            score = row.get("current_hazard_score")
            time = row.get("simulation_time")

            if not isinstance(score, (int, float)) or not isinstance(time, (int, float)):
                continue

            if peak_score is None or score > peak_score:
                peak_score = score
                peak_time = time

        if peak_score is not None and peak_time:
            rates.append(peak_score / peak_time)

    return _mean(rates)
