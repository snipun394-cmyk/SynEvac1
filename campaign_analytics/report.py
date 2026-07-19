from datetime import datetime, timezone
from typing import Dict, List, Optional

from campaign_analytics.analyzer import CampaignAnalysis, Finding


# =====================================================
# Phase 5 -- Automatic Report.
#
# generate_report() takes only an already-computed CampaignAnalysis
# (analyzer.analyze_campaign()'s own return value) and formats it as
# Markdown -- no artifact is read here a second time, no number here
# is recomputed from anything other than what CampaignAnalysis already
# carries.
# =====================================================

# Which Finding codes read as a *bias* worth flagging before training
# (a coverage/variance gap in the campaign's own inputs) versus a
# *rare event* worth a fire engineer's individual attention (a specific
# scenario or value that stands out). Every code either engineering_
# checks.py or anomaly_detection.py can produce is classified here;
# anything unrecognized still shows up under "Engineering Observations"
# (which lists every finding regardless of category) so nothing is
# ever silently dropped from the report.

BIAS_FINDING_CODES = frozenset(
    {
        "zones_never_on_fire",
        "exits_never_used",
        "stairs_never_used",
        "camera_always_active",
        "detector_always_active",
        "constant_occupant_count",
        "repeated_evacuation_time",
    }
)

RARE_EVENT_FINDING_CODES = frozenset(
    {
        "evacuation_time_outlier",
        "zero_congestion_scenarios",
        "impossible_congestion",
    }
)


def generate_report(analysis: CampaignAnalysis, *, generated_at: Optional[str] = None) -> str:

    lines: List[str] = []

    lines.append("# Campaign Analytics Report")
    lines.append("")
    lines.append(f"**Campaign directory:** `{analysis.campaign_dir}`")
    lines.append(f"**Generated:** {generated_at or datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    _append_campaign_summary(lines, analysis)
    _append_dataset_quality(lines, analysis)
    _append_engineering_observations(lines, analysis)
    _append_potential_biases(lines, analysis)
    _append_rare_events(lines, analysis)
    _append_recommendations(lines, analysis)

    return "\n".join(lines) + "\n"


def write_report(analysis: CampaignAnalysis, file_path: str, **kwargs) -> str:

    text = generate_report(analysis, **kwargs)

    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(text)

    return file_path


# =====================================================


def _format_value(value) -> str:

    if value is None:
        return "n/a"

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


def _format_percentage(value) -> str:

    return "n/a" if value is None else f"{value:.1f}%"


def _format_dict_bullets(data: Dict[str, int], *, limit: int = 10) -> List[str]:

    if not data:
        return ["- (no data)"]

    items = list(data.items())[:limit]
    bullets = [f"- `{key}`: {value}" for key, value in items]

    if len(data) > limit:
        bullets.append(f"- ... ({len(data) - limit} more)")

    return bullets


def _append_campaign_summary(lines: List[str], analysis: CampaignAnalysis) -> None:

    overview = analysis.overview

    lines.append("## Campaign Summary")
    lines.append("")
    lines.append(f"- Total scenarios (accepted): {overview.total_scenarios}")

    if overview.total_requested is not None:
        lines.append(f"- Total requested: {overview.total_requested}")
        lines.append(f"- Rejected: {_format_value(overview.rejected)}")
        lines.append(f"- Acceptance ratio: {_format_percentage(overview.acceptance_ratio and overview.acceptance_ratio * 100)}")
    else:
        lines.append(
            "- Total requested / rejected: n/a (CampaignSummary is not persisted by "
            "Campaign Studio; pass total_requested/rejected into analyze_campaign() to "
            "include them)"
        )

    lines.append(f"- Average evacuation time: {_format_value(overview.average_evacuation_time)} s")
    lines.append(f"- Minimum evacuation time: {_format_value(overview.minimum_evacuation_time)} s")
    lines.append(f"- Maximum evacuation time: {_format_value(overview.maximum_evacuation_time)} s")

    for label, value in overview.evacuation_time_percentiles.items():
        lines.append(f"- Evacuation time {label}: {_format_value(value)} s")

    lines.append(f"- Average occupants per scenario: {_format_value(overview.average_occupants)}")
    lines.append("")

    lines.append("**Fire profile frequencies:**")
    lines.extend(_format_dict_bullets(overview.fire_profile_frequencies))
    lines.append("")

    lines.append("**Ignition zone frequencies:**")
    lines.extend(_format_dict_bullets(overview.ignition_zone_frequencies))
    lines.append("")


def _append_dataset_quality(lines: List[str], analysis: CampaignAnalysis) -> None:

    kpis = analysis.kpis

    lines.append("## Dataset Quality")
    lines.append("")
    lines.append(f"- Average RSET (mean total evacuation time): {_format_value(kpis.average_rset)} s")
    lines.append(f"- Peak evacuation time: {_format_value(kpis.peak_evacuation_time)} s")
    lines.append(f"- Average queue time: {_format_value(kpis.average_queue_time)} s")
    lines.append(f"- Average congestion duration: {_format_value(kpis.average_congestion_duration)} s")
    lines.append(f"- Exit utilization: {_format_percentage(kpis.exit_utilization_percentage)}")
    lines.append(f"- Stair utilization: {_format_percentage(kpis.stair_utilization_percentage)}")
    lines.append(f"- Detector activation coverage: {_format_percentage(kpis.detector_activation_coverage)}")
    lines.append(f"- Camera coverage: {_format_percentage(kpis.camera_coverage)}")
    lines.append(f"- Average hazard growth rate: {_format_value(kpis.average_hazard_growth_rate)} score/s")
    lines.append(f"- Evacuation success rate: {_format_percentage(kpis.evacuation_success_rate)}")
    lines.append(f"- Unreachable occupant rate: {_format_percentage(kpis.unreachable_occupant_rate)}")
    lines.append("")


def _append_engineering_observations(lines: List[str], analysis: CampaignAnalysis) -> None:

    lines.append("## Engineering Observations")
    lines.append("")

    if not analysis.engineering_findings:
        lines.append("No engineering plausibility issues were detected.")
    else:
        for finding in analysis.engineering_findings:
            lines.append(f"- **[{finding.code}]** {finding.message}")

    lines.append("")


def _findings_with_codes(findings, codes) -> List[Finding]:

    return [finding for finding in findings if finding.code in codes]


def _append_potential_biases(lines: List[str], analysis: CampaignAnalysis) -> None:

    biases = _findings_with_codes(analysis.findings, BIAS_FINDING_CODES)

    lines.append("## Potential Biases")
    lines.append("")

    if not biases:
        lines.append("No coverage or distribution biases were detected.")
    else:
        for finding in biases:
            lines.append(f"- **[{finding.code}]** {finding.message}")

    lines.append("")


def _append_rare_events(lines: List[str], analysis: CampaignAnalysis) -> None:

    rare_events = _findings_with_codes(analysis.findings, RARE_EVENT_FINDING_CODES)

    lines.append("## Rare Events")
    lines.append("")

    if not rare_events:
        lines.append("No rare or outlier scenarios were detected.")
    else:
        for finding in rare_events:
            scenario_note = f" (scenario `{finding.scenario_id}`)" if finding.scenario_id else ""
            lines.append(f"- **[{finding.code}]** {finding.message}{scenario_note}")

    lines.append("")


def _append_recommendations(lines: List[str], analysis: CampaignAnalysis) -> None:

    lines.append("## Recommendations Before AI Training")
    lines.append("")

    recommendations = []

    if analysis.overview.total_scenarios == 0:
        recommendations.append(
            "This campaign has no loaded scenarios -- generate and export a campaign before "
            "attempting to train on it."
        )

    codes_present = {finding.code for finding in analysis.findings}

    if "constant_occupant_count" in codes_present:
        recommendations.append(
            "Vary the occupant count distribution in the Scenario Definition -- a model trained "
            "on a fixed occupant count will not generalize to other occupancy levels."
        )

    if "repeated_evacuation_time" in codes_present:
        recommendations.append(
            "Investigate why so many scenarios share the same evacuation time -- confirm fire "
            "growth and occupant placement are actually varying seed-to-seed."
        )

    if {"zones_never_on_fire", "exits_never_used", "stairs_never_used"} & codes_present:
        recommendations.append(
            "Broaden fire ignition/exit-and-stair-routing coverage -- some building elements "
            "were never exercised by this campaign and a model trained on it will have no "
            "signal for them."
        )

    if {"camera_always_active", "detector_always_active"} & codes_present:
        recommendations.append(
            "Enable sensor-failure scenarios in the Scenario Definition -- this campaign never "
            "trains a model on detector/camera failure conditions."
        )

    if "impossible_congestion" in codes_present:
        recommendations.append(
            "Investigate the impossible-congestion scenario(s) flagged above before including "
            "them in a training set -- they likely indicate an export or measurement bug."
        )

    if analysis.kpis.unreachable_occupant_rate and analysis.kpis.unreachable_occupant_rate > 0:
        recommendations.append(
            f"{analysis.kpis.unreachable_occupant_rate:.1f}% of occupants were unreachable "
            f"across this campaign -- confirm this reflects intended building/route "
            f"connectivity constraints, not a navigation defect."
        )

    if not recommendations:
        recommendations.append(
            "No significant issues were detected -- this campaign appears suitable for ML "
            "training as exported."
        )

    for recommendation in recommendations:
        lines.append(f"- {recommendation}")

    lines.append("")
