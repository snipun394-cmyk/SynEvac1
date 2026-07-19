import re

from typing import List

from campaign_analytics.analyzer import Finding


# =====================================================
# Phase 2 -- Engineering Validation (domain-plausibility half).
#
# Every check here takes only an already-loaded, duck-typed iterable
# of samples -- no import of training_dataset or any simulation-stack
# package. "Produce warnings, not failures" (the objective's own
# words): every Finding below carries severity="warning", never
# "error" -- this module audits a campaign, it never gates one.
# =====================================================

CAMERA_STATE_COLUMN_PATTERN = re.compile(r"^Camera_\d+_State$")
DETECTOR_STATE_COLUMN_PATTERN = re.compile(r"^Detector_\d+_State$")

FAILED_DEVICE_STATE = "FAILED"

MAX_IDS_LISTED_IN_MESSAGE = 10


def run_checks(samples) -> List[Finding]:

    samples = list(samples)

    findings: List[Finding] = []

    findings.extend(_check_zones_never_on_fire(samples))
    findings.extend(_check_exits_never_used(samples))
    findings.extend(_check_stairs_never_used(samples))
    findings.extend(_check_device_never_fails(samples, CAMERA_STATE_COLUMN_PATTERN, "camera"))
    findings.extend(_check_device_never_fails(samples, DETECTOR_STATE_COLUMN_PATTERN, "detector"))
    findings.extend(_check_zero_congestion_scenarios(samples))
    findings.extend(_check_impossible_congestion_scenarios(samples))

    return findings


# =====================================================


def _format_id_list(ids) -> str:

    ids = sorted(ids)

    if len(ids) <= MAX_IDS_LISTED_IN_MESSAGE:
        return str(ids)

    shown = ids[:MAX_IDS_LISTED_IN_MESSAGE]

    return f"{shown} (+{len(ids) - MAX_IDS_LISTED_IN_MESSAGE} more)"


def _check_zones_never_on_fire(samples) -> List[Finding]:

    universe = set()
    ever_hazardous = set()

    for sample in samples:

        ground_truth = sample.ground_truth

        if not ground_truth:
            continue

        for entry in ground_truth.get("zone_risk_scores") or []:
            if entry.get("zone_id"):
                universe.add(entry["zone_id"])

        ever_hazardous.update(ground_truth.get("hazard_spread_order") or [])

        if ground_truth.get("first_hazardous_zone"):
            ever_hazardous.add(ground_truth["first_hazardous_zone"])

        if ground_truth.get("maximum_hazard_zone"):
            ever_hazardous.add(ground_truth["maximum_hazard_zone"])

    never_on_fire = universe - ever_hazardous

    if not universe or not never_on_fire:
        return []

    return [
        Finding(
            "warning", "zones_never_on_fire",
            f"{len(never_on_fire)}/{len(universe)} zone(s) never became hazardous in any "
            f"scenario: {_format_id_list(never_on_fire)} -- fire ignition/spread may be biased "
            f"toward a subset of the building.",
        )
    ]


def _check_exits_never_used(samples) -> List[Finding]:

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

    never_used = universe - used

    if not universe or not never_used:
        return []

    return [
        Finding(
            "warning", "exits_never_used",
            f"{len(never_used)}/{len(universe)} exit(s) were never used by any occupant across "
            f"the campaign: {_format_id_list(never_used)}.",
        )
    ]


def _check_stairs_never_used(samples) -> List[Finding]:

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

    never_used = universe - used

    if not universe or not never_used:
        return []

    return [
        Finding(
            "warning", "stairs_never_used",
            f"{len(never_used)}/{len(universe)} stair(s) were never selected as a preferred "
            f"route in any scenario: {_format_id_list(never_used)}.",
        )
    ]


def _check_device_never_fails(samples, pattern: "re.Pattern", label: str) -> List[Finding]:

    columns = set()

    for sample in samples:
        for key in sample.scenario_features:
            if pattern.match(key):
                columns.add(key)

    findings = []

    for column in sorted(columns):

        values = {sample.scenario_features.get(column) for sample in samples}

        if values and FAILED_DEVICE_STATE not in values:
            findings.append(
                Finding(
                    "warning", f"{label}_always_active",
                    f"{column} is AVAILABLE in every scenario -- this campaign never exercises "
                    f"{label} failure, a coverage gap for ML training.",
                )
            )

    return findings


def _check_zero_congestion_scenarios(samples) -> List[Finding]:

    zero_congestion_ids = [
        sample.scenario_id
        for sample in samples
        if (sample.simulation_outcome.get("maximum_congestion") or 0) == 0
    ]

    if not zero_congestion_ids:
        return []

    return [
        Finding(
            "warning", "zero_congestion_scenarios",
            f"{len(zero_congestion_ids)}/{len(samples)} scenario(s) recorded zero congestion "
            f"anywhere: {_format_id_list(zero_congestion_ids)}.",
        )
    ]


def _check_impossible_congestion_scenarios(samples) -> List[Finding]:

    findings = []

    for sample in samples:

        congestion = sample.simulation_outcome.get("maximum_congestion")
        total_occupants = sample.scenario_features.get("total_occupants")

        if (
            isinstance(congestion, (int, float))
            and isinstance(total_occupants, (int, float))
            and congestion > total_occupants
        ):
            findings.append(
                Finding(
                    "warning", "impossible_congestion",
                    f"maximum_congestion ({congestion}) exceeds total_occupants "
                    f"({total_occupants}) -- physically impossible, worth investigating.",
                    scenario_id=sample.scenario_id,
                )
            )

    return findings
