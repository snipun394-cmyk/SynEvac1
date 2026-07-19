import os

from typing import Any, Dict, List, Optional


# =====================================================
# Phase 7 -- Final Validation Report. A pure Markdown formatter over an
# already-computed ValidationBenchmarkResult -- same "one section
# function per concern, over already-computed dataclasses" pattern as
# research_framework.report.generate_research_report. No new
# computation happens here; every number below is read straight off
# the result object. Known Limitations is generated from what is
# actually None/skipped on the passed result (plus the framework's own
# disclosed, permanent limitations), never boilerplate.
# =====================================================


def generate_validation_report(result: Any) -> str:

    sections = [
        _title_section(result),
        _simulation_performance_section(result),
        _ai_performance_section(result),
        _rl_performance_section(result),
        _recommendation_performance_section(result),
        _command_center_performance_section(result),
        _dataset_statistics_section(result),
        _known_limitations_section(result),
        _future_improvements_section(result),
    ]

    return "\n\n".join(section for section in sections if section)


def write_validation_report(result: Any, file_path: str) -> str:

    text = generate_validation_report(result)

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(text)

    return file_path


# =====================================================


def _title_section(result: Any) -> str:

    lines = [
        "# SynEvac Validation & Benchmark Report",
        "",
        f"Generated: {result.generated_at}",
        f"Campaign directory: {result.campaign_dir or 'N/A (no campaign_dir supplied)'}",
    ]

    return "\n".join(lines)


# =====================================================


def _simulation_performance_section(result: Any) -> str:

    lines = ["## Simulation Performance"]

    if result.campaign_analysis is None:
        lines.append("Not evaluated -- no campaign_dir was supplied.")
        return "\n".join(lines)

    kpis = result.campaign_analysis.kpis
    overview = result.campaign_analysis.overview

    lines.append(f"- Scenarios analyzed: {overview.total_scenarios}")
    lines.append(f"- Average RSET (total_evacuation_time): {_fmt(kpis.average_rset)}")
    lines.append(f"- Peak evacuation time: {_fmt(kpis.peak_evacuation_time)}")
    lines.append(f"- Evacuation success rate: {_fmt(kpis.evacuation_success_rate)}")
    lines.append(f"- Unreachable occupant rate: {_fmt(kpis.unreachable_occupant_rate)}")
    lines.append(f"- Average queue time: {_fmt(kpis.average_queue_time)}")
    lines.append(f"- Exit utilization: {_fmt(kpis.exit_utilization_percentage)}")
    lines.append(f"- Stair utilization: {_fmt(kpis.stair_utilization_percentage)}")

    return "\n".join(lines)


# =====================================================


def _ai_performance_section(result: Any) -> str:

    lines = ["## AI Performance"]

    if not result.prediction_results:
        lines.append("Not evaluated -- no campaign_dir was supplied.")
        return "\n".join(lines)

    for name, prediction_result in result.prediction_results.items():

        lines.append(f"### {name}")

        for metric_name, value in prediction_result.metrics.items():
            lines.append(f"- {metric_name}: {_fmt(value)}")

        for ci_name, ci in prediction_result.bootstrap_ci.items():
            lines.append(
                f"- {ci_name} bootstrap {int(ci.confidence * 100)}% CI: "
                f"[{_fmt(ci.lower)}, {_fmt(ci.upper)}] (n={ci.n})"
            )

    if result.recommendation_accuracy is not None:

        lines.append("### Recommendation Accuracy (AI bottleneck-location vs. Decision Policy)")

        for key, value in result.recommendation_accuracy.items():
            lines.append(f"- {key}: {_fmt(value)}")

    if result.firefighter_monotonicity is not None:

        lines.append("### Firefighter Intelligence Consistency (monotonicity)")

        for key, value in result.firefighter_monotonicity.items():
            lines.append(f"- {key}: {_fmt(value)}")

    return "\n".join(lines)


# =====================================================


def _rl_performance_section(result: Any) -> str:

    lines = ["## RL Performance"]

    if result.policy_comparison is None:
        lines.append("Not evaluated -- building/definition/definition_id were not supplied.")
        return "\n".join(lines)

    lines.append("### Metrics by arm")

    for arm, metrics in result.policy_comparison.metrics_by_arm.items():

        lines.append(f"- **{arm}**")

        for key, value in metrics.items():
            lines.append(f"  - {key}: {_fmt(value)}")

    comparison = result.policy_comparison.statistical_comparison

    lines.append(
        f"### Statistical comparison ({comparison.metric_name}, baseline={comparison.baseline_arm})"
    )
    lines.append(
        f"- ANOVA: F={_fmt(comparison.anova.f_statistic)}, p={_fmt(comparison.anova.p_value)}"
    )

    for arm, effect in comparison.effect_sizes_vs_baseline.items():
        lines.append(f"- Cohen's d ({arm} vs. {comparison.baseline_arm}): {_fmt(effect.cohens_d)}")

    if "ai_rl" not in result.policy_comparison.reports:
        lines.append("- AI+RL arm not included in this comparison (no campaign_dir/AI prediction bundle available).")

    return "\n".join(lines)


# =====================================================


def _recommendation_performance_section(result: Any) -> str:

    lines = ["## Recommendation Performance"]

    if result.recommendation_validation is None:
        lines.append("Not evaluated -- building/definition/definition_id were not supplied.")
        return "\n".join(lines)

    validation = result.recommendation_validation

    lines.append(f"- Scenarios evaluated: {validation.scenarios_evaluated}")
    lines.append(f"- Total ticks: {validation.total_ticks}")
    lines.append(f"- Total recommendation changes: {validation.total_changes}")
    lines.append(f"- Stability score: {_fmt(validation.stability_score)}")
    lines.append(f"- Contradictory recommendations found: {len(validation.consistency_violations)}")
    lines.append(f"- Late redirects found: {len(validation.late_redirects)}")
    lines.append(
        "- False redirects: N/A -- requires a decision policy recomputed at a later tick than the "
        "one that issued the redirect; this framework computes one decision policy per scenario "
        "(matching decision_policy's own established post-hoc, single-computation convention)."
    )
    lines.append(f"- Firefighter priority deterministic: {_fmt(validation.firefighter_priority_deterministic)}")

    return "\n".join(lines)


# =====================================================


def _command_center_performance_section(result: Any) -> str:

    return (
        "## Command Center Performance\n\n"
        "Not separately tracked anywhere in the platform -- the Command Center is a "
        "visualization layer over Ground Truth/Decision Policy/Advisory System outputs and "
        "carries no timing/accuracy instrumentation of its own to validate. This section is "
        "left honestly empty rather than fabricated; see Future Improvements."
    )


# =====================================================


def _dataset_statistics_section(result: Any) -> str:

    lines = ["## Dataset Statistics"]

    if result.dataset_statistics is None:
        lines.append("Not evaluated -- no campaign_dir was supplied.")
        return "\n".join(lines)

    stats = result.dataset_statistics

    lines.append(f"- Number of scenarios: {stats.get('num_scenarios')}")
    lines.append(f"- Fire profile distribution: {stats.get('fire_profile_distribution')}")
    lines.append(f"- Ignition zone distribution: {stats.get('ignition_zone_distribution')}")
    lines.append(f"- Bottleneck frequency: {stats.get('bottleneck_frequency')}")
    lines.append(f"- Detector failure rate: {_fmt(stats.get('detector_failure_rate'))}")
    lines.append(f"- Camera failure rate: {_fmt(stats.get('camera_failure_rate'))}")

    return "\n".join(lines)


# =====================================================


def _known_limitations_section(result: Any) -> str:

    lines = ["## Known Limitations"]

    disclosed: List[str] = list(result.limitations)

    disclosed.append(
        "AI+RL comparison is an observation-augmented RL policy (AI predictions appended as "
        "extra observation features), not a rule-based hybrid -- see the module docstring in "
        "validation_framework/ai_augmented_policy.py."
    )
    disclosed.append(
        "Smoke Exposure and Travel Distance are not exposed by rl_training.evaluator."
        "EpisodeMetrics, so those two RL metrics are always reported as unavailable rather than "
        "estimated."
    )
    disclosed.append(
        "False redirect detection requires a decision policy recomputed at a later simulation "
        "tick than the redirect itself; this framework always computes exactly one decision "
        "policy per scenario, so false redirects are always reported as N/A."
    )

    if not disclosed:
        lines.append("None recorded for this run.")
    else:
        for item in disclosed:
            lines.append(f"- {item}")

    return "\n".join(lines)


# =====================================================


def _future_improvements_section(result: Any) -> str:

    items = [
        "Per-tick decision policy recomputation to enable genuine false-redirect detection.",
        "Smoke exposure and travel distance metrics for RL policy comparison, once exposed by "
        "rl_training.evaluator.EpisodeMetrics.",
        "Bootstrap confidence intervals on additional AI metrics beyond the primary scalar "
        "(currently MAE for regression, accuracy for classification).",
        "Command Center performance instrumentation (render time, refresh latency) if that "
        "becomes a tracked concern.",
        "Additional AI+RL architectures (e.g. richer AI feature sets, learned feature "
        "compression) beyond the fixed evacuation-time/bottleneck/smoke/exit-usage vector used here.",
    ]

    return "## Future Improvements\n\n" + "\n".join(f"- {item}" for item in items)


# =====================================================


def _fmt(value: Optional[Any]) -> str:

    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)
