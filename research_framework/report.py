import datetime

from typing import Any, Dict, Mapping, Optional, Sequence

from campaign_analytics import analyze_campaign

from research_framework.ablation import AblationStudyResult, NO_PERCEPTION_NOTE, NoRLAblationResult
from research_framework.runner import ExperimentRunResult
from research_framework.sensitivity import SensitivityAnalysisResult
from research_framework.statistics import ArmComparisonReport


# =====================================================
# Phase 7 -- Research Report. Pure Markdown formatting over
# already-computed results (Phase 2's ExperimentRunResult, Phase 3's
# ArmComparisonReport, Phase 4's AblationStudyResult/NoRLAblationResult,
# Phase 5's SensitivityAnalysisResult, Phase 6's figure file paths) --
# no new computation happens here, the same "formatter only" role
# campaign_analytics.report.generate_report() already plays for one
# campaign; this is that same role, one level up, across a whole
# experiment. Per-arm KPIs are read via campaign_analytics.
# analyze_campaign() directly (reused, not reimplemented) against each
# arm's own campaign_dir.
# =====================================================


def generate_research_report(
    experiment_result: ExperimentRunResult,
    arm_comparisons: Sequence[ArmComparisonReport] = (),
    ablation_result: Optional[AblationStudyResult] = None,
    no_rl_result: Optional[NoRLAblationResult] = None,
    sensitivity_result: Optional[SensitivityAnalysisResult] = None,
    figure_paths: Optional[Mapping[str, str]] = None,
    generated_at: Optional[str] = None,
) -> str:

    generated_at = generated_at or datetime.datetime.now(datetime.timezone.utc).isoformat()

    lines = []

    lines.append(f"# Research Report — {experiment_result.spec.name}")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")

    lines.extend(_experiment_summary_section(experiment_result))
    lines.extend(_per_arm_kpi_section(experiment_result))

    if arm_comparisons:
        lines.extend(_statistical_analysis_section(arm_comparisons))

    if ablation_result is not None or no_rl_result is not None:
        lines.extend(_ablation_section(ablation_result, no_rl_result))

    if sensitivity_result is not None:
        lines.extend(_sensitivity_section(sensitivity_result))

    if figure_paths:
        lines.extend(_figures_section(figure_paths))

    lines.extend(_conclusions_section(experiment_result, arm_comparisons, ablation_result))

    return "\n".join(lines) + "\n"


# =====================================================


def _experiment_summary_section(experiment_result: ExperimentRunResult):

    spec = experiment_result.spec

    return [
        "## Experiment Summary",
        "",
        f"- Samples per arm: {spec.samples_per_arm}",
        f"- Master seed: {spec.master_seed}",
        f"- Arms compared: {', '.join(arm.name for arm in spec.arms)}",
        f"- Output root: {spec.output_root}",
        "",
    ]


def _per_arm_kpi_section(experiment_result: ExperimentRunResult):

    lines = ["## Per-Arm KPIs", ""]

    for arm_result in experiment_result.arm_results:

        lines.append(f"### {arm_result.arm.name}")
        lines.append("")
        lines.append(
            f"- Accepted scenarios: {arm_result.accepted_count} / {experiment_result.spec.samples_per_arm}",
        )

        if arm_result.accepted_count > 0:

            try:

                analysis = analyze_campaign(arm_result.campaign_dir)
                overview = analysis.overview

                lines.append(f"- Average evacuation time: {overview.average_evacuation_time}")
                lines.append(f"- Evacuation time p95: {overview.evacuation_time_percentiles.get('p95')}")
                lines.append(f"- Average occupants: {overview.average_occupants}")

                for finding in analysis.findings:
                    lines.append(f"  - [{finding.severity}] {finding.code}: {finding.message}")

            except Exception as exc:

                lines.append(f"- KPI computation failed (this arm's campaign could not be analyzed): {exc}")

        if arm_result.ai_results:

            lines.append("- AI models trained:")

            for name, result in arm_result.ai_results.items():
                lines.append(f"  - `{name}`: {result.metrics}")

        if arm_result.ai_errors:

            lines.append("- AI training failures:")

            for name, error in arm_result.ai_errors.items():
                lines.append(f"  - `{name}`: {error}")

        lines.append("")

    return lines


def _statistical_analysis_section(arm_comparisons: Sequence[ArmComparisonReport]):

    lines = ["## Statistical Analysis", ""]

    for comparison in arm_comparisons:

        lines.append(f"### {comparison.metric_name}")
        lines.append("")
        lines.append(
            f"- One-way ANOVA across arms: F = {comparison.anova.f_statistic}, "
            f"p = {comparison.anova.p_value}",
        )
        lines.append(f"- Baseline arm: `{comparison.baseline_arm}`")
        lines.append("")
        lines.append("| Arm | Mean | 95% CI | n |")
        lines.append("|---|---|---|---|")

        for arm_name, ci in comparison.confidence_intervals.items():
            lines.append(f"| {arm_name} | {ci.mean} | [{ci.lower}, {ci.upper}] | {ci.n} |")

        lines.append("")

        if comparison.effect_sizes_vs_baseline:

            lines.append("| Arm vs. baseline | Cohen's d |")
            lines.append("|---|---|")

            for arm_name, effect in comparison.effect_sizes_vs_baseline.items():
                lines.append(f"| {arm_name} | {effect.cohens_d} |")

            lines.append("")

    return lines


def _ablation_section(ablation_result: Optional[AblationStudyResult], no_rl_result: Optional[NoRLAblationResult]):

    lines = ["## Ablation Studies", ""]

    if ablation_result is not None:

        lines.append("| Capability disabled | Metric | Cohen's d | Paired p-value |")
        lines.append("|---|---|---|---|")

        for result in ablation_result.results:

            paired_p = result.paired.p_value if result.paired is not None else None
            lines.append(
                f"| {result.capability} | {result.metric_name} | {result.effect_size.cohens_d} | {paired_p} |",
            )

        lines.append("")

    if no_rl_result is not None:

        lines.append("### No RL")
        lines.append("")
        lines.append(f"- With RL policy — average reward: {no_rl_result.with_rl.average_reward}")
        lines.append(f"- Without RL (no intervention) — average reward: {no_rl_result.without_rl.average_reward}")
        lines.append(f"- Reward effect size (Cohen's d): {no_rl_result.reward_effect_size.cohens_d}")
        lines.append(f"- Evacuation time effect size (Cohen's d): {no_rl_result.evacuation_time_effect_size.cohens_d}")
        lines.append("")

    lines.append(f"### No Perception")
    lines.append("")
    lines.append(NO_PERCEPTION_NOTE)
    lines.append("")

    return lines


def _sensitivity_section(sensitivity_result: SensitivityAnalysisResult):

    lines = ["## Sensitivity Analysis", ""]
    lines.append("Parameters ranked by |Pearson r| against each metric:")
    lines.append("")

    metric_names = sorted({sweep.metric_name for sweep in sensitivity_result.sweeps})

    for metric_name in metric_names:

        lines.append(f"### {metric_name}")
        lines.append("")
        lines.append("| Parameter | Pearson r | p-value | Slope | n |")
        lines.append("|---|---|---|---|---|")

        for sweep in sensitivity_result.ranked(metric_name):

            sensitivity = sweep.sensitivity
            lines.append(
                f"| {sweep.parameter_name} | {sensitivity.pearson_r} | {sensitivity.p_value} | "
                f"{sensitivity.slope} | {sensitivity.n} |",
            )

        lines.append("")

    return lines


def _figures_section(figure_paths: Mapping[str, str]):

    lines = ["## Figures", ""]

    for name, path in figure_paths.items():
        lines.append(f"- **{name}**: `{path}`")

    lines.append("")

    return lines


def _conclusions_section(
    experiment_result: ExperimentRunResult,
    arm_comparisons: Sequence[ArmComparisonReport],
    ablation_result: Optional[AblationStudyResult],
):

    lines = ["## Conclusions", ""]

    if not arm_comparisons and ablation_result is None:

        lines.append(
            "No statistical comparisons or ablation results were supplied to this report -- "
            "see Per-Arm KPIs above for the raw, per-arm outcomes this experiment produced.",
        )
        lines.append("")
        return lines

    for comparison in arm_comparisons:

        p_value = comparison.anova.p_value

        if p_value is None:

            lines.append(
                f"- {comparison.metric_name}: not enough data across arms to run ANOVA.",
            )

        elif p_value < 0.05:

            lines.append(
                f"- {comparison.metric_name}: arms differ significantly (ANOVA p = {p_value:.4f}).",
            )

        else:

            lines.append(
                f"- {comparison.metric_name}: no significant difference detected across arms "
                f"(ANOVA p = {p_value:.4f}).",
            )

    if ablation_result is not None:

        for result in ablation_result.results:

            if result.effect_size.cohens_d is None:
                continue

            magnitude = abs(result.effect_size.cohens_d)
            qualifier = "large" if magnitude >= 0.8 else "medium" if magnitude >= 0.5 else "small"

            lines.append(
                f"- Disabling `{result.capability}` has a {qualifier} effect on "
                f"{result.metric_name} (Cohen's d = {result.effect_size.cohens_d:.3f}).",
            )

    lines.append("")

    return lines


# =====================================================


def write_research_report(*args, file_path: str, **kwargs) -> str:

    report_text = generate_research_report(*args, **kwargs)

    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    return file_path
