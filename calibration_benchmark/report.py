import datetime
import math
from typing import Optional

from calibration_benchmark.harness import CalibrationBenchmarkResult, MetricComparison
from calibration_benchmark.recommendation import AdoptionRecommendation, recommend


# =====================================================
# Calibration Benchmark Framework V1 -- renders a
# CalibrationBenchmarkResult into the "publication-quality benchmarking
# report" this milestone's brief asks for: parameter values, experiment
# settings, before/after comparison, statistical significance, and an
# adopt/reject/inconclusive recommendation, as one self-contained
# Markdown document. Never writes to any SynEvac default -- this module
# only ever reads a CalibrationBenchmarkResult and an
# AdoptionRecommendation, both already pure, already-computed reports.
# =====================================================


def _fmt(value, digits: int = 3) -> str:

    if value is None:
        return "n/a"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def _p_value_cell(p_value: Optional[float]) -> str:

    if p_value is None:
        return "n/a"

    if isinstance(p_value, float) and math.isnan(p_value):
        return "identical values (no variance)"

    significance = "**significant**" if p_value < 0.05 else "not significant"
    return f"{p_value:.4f} ({significance})"


def _comparison_table_row(comparison: MetricComparison) -> str:

    return (
        f"| {comparison.metric_label} | {_fmt(comparison.baseline_mean)} "
        f"| {_fmt(comparison.candidate_mean)} "
        f"| {_fmt(comparison.paired.mean_difference)} "
        f"| {_fmt(comparison.effect_size.cohens_d)} "
        f"| {_p_value_cell(comparison.paired.p_value)} "
        f"| {comparison.n_pairs} |"
    )


def render_markdown_report(
    result: CalibrationBenchmarkResult, *, recommendation: Optional[AdoptionRecommendation] = None,
    generated_at: Optional[datetime.datetime] = None,
) -> str:

    recommendation = recommendation or recommend(result)
    generated_at = generated_at or datetime.datetime.now(datetime.timezone.utc)
    candidate = result.candidate

    lines = []

    lines.append(f"# Calibration Benchmark Report -- {candidate.name}")
    lines.append("")
    lines.append(f"*Generated {generated_at.isoformat()} by Calibration Benchmark Framework V1.*")
    lines.append("")
    lines.append(
        "This report compares one SynEvac default parameter against one dataset-derived "
        "candidate value, across an identical, paired scenario campaign. **No SynEvac default "
        "was changed to produce this report** -- both arms were run from freshly-constructed, "
        "non-mutating objects (see `calibration_benchmark/candidates.py`)."
    )
    lines.append("")

    lines.append("## 1. Parameter Under Test")
    lines.append("")
    lines.append(f"- **Subsystem:** {candidate.subsystem}")
    lines.append(f"- **Parameter:** `{candidate.name}`")
    lines.append(f"- **Calibration Mapping Report tier:** {candidate.calibration_tier}")
    lines.append(f"- **Dataset source:** {candidate.dataset_source}")
    lines.append(f"- **Current SynEvac default:** {candidate.current_value} {candidate.unit}")
    lines.append(f"- **Candidate (dataset-derived) value:** {candidate.candidate_value} {candidate.unit}")
    lines.append(f"- **Rationale:** {candidate.rationale}")
    lines.append("")

    lines.append("## 2. Experiment Settings")
    lines.append("")
    lines.append(f"- **Scenarios requested:** {result.n_scenarios_requested}")
    lines.append(f"- **Scenarios completed (paired, both arms):** {result.n_completed_pairs}")
    lines.append(
        "- **Design:** paired -- the same seeded scenario batch is run once under the current "
        "default and once under the candidate value; every comparison below is a paired "
        "(same-scenario) statistical test, not an independent-samples comparison."
    )
    lines.append(
        "- **Statistical test:** paired t-test (`research_framework.statistics.paired_comparison`), "
        "Cohen's d effect size, 95% confidence intervals -- all reused from this codebase's own "
        "existing statistics module, not reimplemented."
    )
    lines.append("")

    lines.append("## 3. Before/After Comparison")
    lines.append("")
    lines.append("| Metric | Baseline mean | Candidate mean | Mean difference (baseline − candidate) | Cohen's d | p-value | n |")
    lines.append("|---|---|---|---|---|---|---|")

    for comparison in result.comparisons.values():
        lines.append(_comparison_table_row(comparison))

    if result.additional_comparisons:

        lines.append("")
        lines.append(
            "**Additional metrics** (Prediction Accuracy / Recommendation Effectiveness -- "
            "self-contained proxies computed from this same run; see "
            "`calibration_benchmark/optional_metrics.py` for exactly what each measures and why "
            "it is scoped the way it is):"
        )
        lines.append("")
        lines.append("| Metric | Baseline mean | Candidate mean | Mean difference | Cohen's d | p-value | n |")
        lines.append("|---|---|---|---|---|---|---|")

        for comparison in result.additional_comparisons.values():
            lines.append(_comparison_table_row(comparison))

    lines.append("")

    lines.append("## 4. Statistical Significance, Per Metric")
    lines.append("")

    for verdict in recommendation.metric_verdicts:
        lines.append(f"- **{verdict.metric_name}** -> {verdict.verdict.name}: {verdict.reason}")

    lines.append("")

    lines.append("## 5. Recommendation")
    lines.append("")
    lines.append(f"**Overall verdict: {recommendation.overall_verdict.name}**")
    lines.append("")
    lines.append(recommendation.summary)
    lines.append("")
    lines.append(
        "This is a recommendation only. Adopting it requires a separate, explicit, human-made "
        "change to the relevant SynEvac source file -- this framework has no code path that "
        "writes one."
    )
    lines.append("")

    return "\n".join(lines)


def save_report(result: CalibrationBenchmarkResult, path: str, **kwargs) -> str:

    markdown = render_markdown_report(result, **kwargs)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(markdown)

    return markdown
