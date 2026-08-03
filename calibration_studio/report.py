from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from calibration_benchmark import CalibrationBenchmarkResult
from calibration_benchmark.report import render_markdown_report

from calibration_studio.benchmark import PublishedBenchmark
from calibration_studio.dashboard import ValidationDashboard
from calibration_studio.paths import report_markdown_path
from calibration_studio.project import CalibrationProject
from calibration_studio.session import CalibrationSession


# =====================================================
# Calibration Studio Phase 7 -- Report Generation.
#
# Two layers, deliberately separate, for the exact reason this
# milestone's own brief names: "remain extensible for future PDF export
# without redesigning the report model."
#
# SessionReportContent is the MODEL -- structured evidence (the live
# CalibrationBenchmarkResult when one exists, its persisted snapshot
# otherwise, project/benchmark/dashboard context, git/reproducibility/
# replay facts, and a plain list of disclosed limitations), never a
# pre-rendered string. build_session_report_content() assembles it by
# reading already-computed fields off CalibrationSession/
# CalibrationProject/PublishedBenchmark/ValidationDashboard -- it
# performs no statistics, no recommendation logic, no new aggregation
# of its own; every number in the model was already computed by
# something else, earlier.
#
# render_markdown() is the ONE renderer this phase builds. When a live
# result exists, it embeds calibration_benchmark.report.
# render_markdown_report()'s own output VERBATIM for the statistical-
# results/recommendation portion -- the existing, already-tested,
# already-"publication-quality" report this milestone's own brief says
# not to duplicate. A future PDF renderer would consume the identical
# SessionReportContent model and format it differently, without this
# model ever needing to change.
#
# THE ONE HONEST DEGRADE THIS PHASE INTRODUCES: a session reloaded from
# disk has result=None (calibration_benchmark has no MetricComparison.
# from_dict() -- confirmed in Phase 2, reused as fact again in Phase 6)
# -- render_markdown_report() cannot be called on it at all. For that
# case only, render_markdown() shows a small, clearly-labeled table
# built directly from the already-serialized result_snapshot dict
# (reading pre-computed numbers, not recomputing anything) -- never an
# attempt to recreate calibration_benchmark's own five-section report
# structure from a plain dict.
# =====================================================


def _utc_now_iso() -> str:

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SessionReportContent:

    generated_at: str

    session_id: str
    session_status: str
    failure_reason: Optional[str]

    project_id: Optional[str]
    project_name: Optional[str]
    project_status: Optional[str]

    benchmark_id: Optional[str]
    benchmark_title: Optional[str]
    benchmark_source_citation: Optional[str]
    benchmark_dataset: Optional[str]
    benchmark_validation_status: Optional[str]

    candidate_snapshot: Optional[dict]
    master_seed: Optional[int]

    # Structured evidence -- see this module's own docstring for why
    # neither of these is pre-rendered text.
    result: Optional[CalibrationBenchmarkResult]
    result_snapshot: Optional[dict]

    dashboard_parameter_confidence: Optional[dict]

    replay_output_dir: Optional[str]
    replay_scenario_id: Optional[str]

    git_commit_hash: Optional[str]
    git_dirty: Optional[bool]

    reproducible: Optional[bool]

    limitations: Tuple[str, ...]

    @property
    def replay_available(self) -> bool:

        return self.replay_output_dir is not None and self.replay_scenario_id is not None

    def to_dict(self) -> dict:

        return {
            "generated_at": self.generated_at,
            "session_id": self.session_id,
            "session_status": self.session_status,
            "failure_reason": self.failure_reason,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_status": self.project_status,
            "benchmark_id": self.benchmark_id,
            "benchmark_title": self.benchmark_title,
            "benchmark_source_citation": self.benchmark_source_citation,
            "benchmark_dataset": self.benchmark_dataset,
            "benchmark_validation_status": self.benchmark_validation_status,
            "candidate_snapshot": self.candidate_snapshot,
            "master_seed": self.master_seed,
            "result": self.result.to_dict() if self.result is not None else self.result_snapshot,
            "dashboard_parameter_confidence": self.dashboard_parameter_confidence,
            "replay_output_dir": self.replay_output_dir,
            "replay_scenario_id": self.replay_scenario_id,
            "replay_available": self.replay_available,
            "git_commit_hash": self.git_commit_hash,
            "git_dirty": self.git_dirty,
            "reproducible": self.reproducible,
            "limitations": list(self.limitations),
        }


def build_session_report_content(
    *,
    session: CalibrationSession,
    project: Optional[CalibrationProject] = None,
    benchmark: Optional[PublishedBenchmark] = None,
    dashboard: Optional[ValidationDashboard] = None,
) -> SessionReportContent:

    limitations: List[str] = []

    if session.status.value == "FAILED":
        limitations.append(f"This session failed: {session.failure_reason}")

    if session.git_dirty is True:
        limitations.append(
            "Recorded against an uncommitted working tree (git_dirty=True) -- not fully "
            "reproducible from git history alone.",
        )

    if session.reproducible is False:
        limitations.append(
            "This session's own randomness-control audit found at least one uncontrolled "
            "randomness source -- statistical results may not be exactly reproducible.",
        )
    elif session.reproducible is None:
        limitations.append("Reproducibility could not be determined for this session.")

    if session.result is None and session.result_snapshot is not None:
        limitations.append(
            "Full statistical detail and recommendation could not be regenerated for this "
            "reloaded session -- calibration_benchmark has no deserialization path for its own "
            "result/comparison types, so only the persisted summary is shown below.",
        )

    if session.candidate_snapshot is None:
        limitations.append("No parameter candidate was tested in this session (production defaults only).")

    if not (session.replay_output_dir is not None and session.replay_scenario_id is not None):
        limitations.append(
            "No replay artifacts have been recorded for this session -- visual inspection in "
            "Replay Studio is not currently available.",
        )

    if session.benchmark_id is None:
        limitations.append("This session is not associated with any Published Benchmark.")
    elif benchmark is None:
        limitations.append(
            f"Session references benchmark_id {session.benchmark_id!r}, but no PublishedBenchmark "
            f"object was supplied to this report -- benchmark information below is unavailable.",
        )

    dashboard_parameter_confidence = None

    if dashboard is None:
        limitations.append("No Validation Dashboard snapshot was supplied to this report.")
    elif session.candidate_snapshot is not None:

        parameter_name = session.candidate_snapshot.get("name")
        match = next(
            (summary for summary in dashboard.parameter_confidence if summary.parameter_name == parameter_name),
            None,
        )

        if match is not None:
            dashboard_parameter_confidence = match.to_dict()
        else:
            limitations.append(
                f"No matching Validation Dashboard parameter-confidence summary was found for "
                f"{parameter_name!r}.",
            )

    return SessionReportContent(
        generated_at=_utc_now_iso(),
        session_id=session.session_id,
        session_status=session.status.value,
        failure_reason=session.failure_reason,
        project_id=project.project_id if project is not None else None,
        project_name=project.name if project is not None else None,
        project_status=project.status.value if project is not None else None,
        benchmark_id=session.benchmark_id,
        benchmark_title=benchmark.title if benchmark is not None else None,
        benchmark_source_citation=benchmark.source_citation if benchmark is not None else None,
        benchmark_dataset=benchmark.dataset if benchmark is not None else None,
        benchmark_validation_status=benchmark.validation_status.value if benchmark is not None else None,
        candidate_snapshot=session.candidate_snapshot,
        master_seed=session.master_seed,
        result=session.result,
        result_snapshot=session.result_snapshot,
        dashboard_parameter_confidence=dashboard_parameter_confidence,
        replay_output_dir=session.replay_output_dir,
        replay_scenario_id=session.replay_scenario_id,
        git_commit_hash=session.git_commit_hash,
        git_dirty=session.git_dirty,
        reproducible=session.reproducible,
        limitations=tuple(limitations),
    )


def _snapshot_comparisons_table(result_snapshot: Dict[str, Any]) -> List[str]:

    lines = [
        "| Metric | Baseline mean | Candidate mean | p-value | Cohen's d | n |",
        "|---|---|---|---|---|---|",
    ]

    for name, comparison in (result_snapshot.get("comparisons") or {}).items():

        paired = comparison.get("paired") or {}
        effect = comparison.get("effect_size") or {}

        lines.append(
            f"| {comparison.get('metric_label', name)} | {comparison.get('baseline_mean')} "
            f"| {comparison.get('candidate_mean')} | {paired.get('p_value')} "
            f"| {effect.get('cohens_d')} | {comparison.get('n_pairs')} |",
        )

    return lines


def render_markdown(content: SessionReportContent) -> str:

    lines: List[str] = []

    lines.append(f"# Calibration Studio Session Report -- {content.session_id}")
    lines.append("")
    lines.append(f"*Generated {content.generated_at} by Calibration Studio.*")
    lines.append("")

    lines.append("## Project Metadata")
    lines.append("")
    if content.project_id is not None:
        lines.append(f"- **Project:** {content.project_name} (`{content.project_id}`)")
        lines.append(f"- **Project status:** {content.project_status}")
    else:
        lines.append("- No project context was supplied for this report.")
    lines.append("")

    lines.append("## Benchmark Information")
    lines.append("")
    if content.benchmark_title is not None:
        lines.append(f"- **Benchmark:** {content.benchmark_title} (`{content.benchmark_id}`)")
        lines.append(f"- **Source citation:** {content.benchmark_source_citation}")
        lines.append(f"- **Dataset:** {content.benchmark_dataset}")
        lines.append(f"- **Validation status:** {content.benchmark_validation_status}")
    elif content.benchmark_id is not None:
        lines.append(f"- Benchmark id `{content.benchmark_id}` is referenced, but no benchmark object was supplied.")
    else:
        lines.append("- This session is not associated with any Published Benchmark.")
    lines.append("")

    lines.append("## Parameter Under Investigation")
    lines.append("")
    if content.candidate_snapshot is not None:
        snapshot = content.candidate_snapshot
        lines.append(f"- **Parameter:** `{snapshot.get('name')}`")
        lines.append(f"- **Subsystem:** {snapshot.get('subsystem')}")
        lines.append(f"- **Current value:** {snapshot.get('current_value')} {snapshot.get('unit')}")
        lines.append(f"- **Candidate value:** {snapshot.get('candidate_value')} {snapshot.get('unit')}")
        lines.append(f"- **Dataset source:** {snapshot.get('dataset_source')}")
        lines.append(f"- **Rationale:** {snapshot.get('rationale')}")
    else:
        lines.append("- No parameter candidate was tested (production defaults only).")
    lines.append("")

    lines.append("## Calibration Settings")
    lines.append("")
    lines.append(f"- **Master seed:** {content.master_seed}")
    if content.result_snapshot is not None:
        lines.append(f"- **Scenarios requested:** {content.result_snapshot.get('n_scenarios_requested')}")
        lines.append(f"- **Scenarios completed (paired):** {content.result_snapshot.get('n_completed_pairs')}")
    lines.append("")

    lines.append("## Statistical Results & Recommendation")
    lines.append("")
    if content.result is not None:
        # Reused verbatim -- calibration_benchmark's own report,
        # never reimplemented here.
        lines.append(render_markdown_report(content.result))
    elif content.result_snapshot is not None:
        lines.append(
            "*Full statistical detail is unavailable for this reloaded session -- showing the "
            "persisted summary only (see Scientific Limitations).*",
        )
        lines.append("")
        lines.extend(_snapshot_comparisons_table(content.result_snapshot))
    else:
        lines.append("- This session has not completed -- no statistical results are available yet.")
    lines.append("")

    lines.append("## Validation Status")
    lines.append("")
    if content.dashboard_parameter_confidence is not None:
        confidence = content.dashboard_parameter_confidence
        lines.append(f"- **Sessions tested (this parameter):** {confidence['n_sessions']}")
        lines.append(
            f"- **Adopt / Reject / Inconclusive / Unknown:** "
            f"{confidence['n_adopt']} / {confidence['n_reject']} / {confidence['n_inconclusive']} / "
            f"{confidence['n_unknown']}",
        )
        lines.append(f"- **Reproducible runs:** {confidence['n_reproducible']}")
    else:
        lines.append("- No Validation Dashboard summary is available for this parameter.")
    lines.append("")

    lines.append("## Replay Availability")
    lines.append("")
    if content.replay_available:
        lines.append(f"- Recorded at `{content.replay_output_dir}`, scenario `{content.replay_scenario_id}`.")
        lines.append("- Open via `CalibrationStudio.open_in_replay_studio(session_id)`.")
    else:
        lines.append("- No replay artifacts have been recorded for this session.")
    lines.append("")

    lines.append("## Git Provenance")
    lines.append("")
    lines.append(f"- **Commit:** `{content.git_commit_hash}`")
    lines.append(f"- **Working tree dirty:** {content.git_dirty}")
    lines.append("")

    lines.append("## Reproducibility Status")
    lines.append("")
    lines.append(f"- **Reproducible:** {content.reproducible}")
    lines.append("")

    lines.append("## Scientific Limitations")
    lines.append("")
    if content.limitations:
        for limitation in content.limitations:
            lines.append(f"- {limitation}")
    else:
        lines.append("- None disclosed.")
    lines.append("")

    return "\n".join(lines)


def save_session_report_markdown(markdown: str, session_id: str, storage_root) -> Path:

    path = report_markdown_path(storage_root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")

    return path


class CalibrationReportGenerator:

    # The one public entry point this milestone's own brief names --
    # both methods are thin wrappers over the two free functions above,
    # kept as a class purely so CalibrationStudio can hold one
    # (`self.reports`) the same way it already holds one
    # PublishedBenchmarkLibrary (`self.benchmarks`).

    def generate_session_report(
        self,
        *,
        session: CalibrationSession,
        project: Optional[CalibrationProject] = None,
        benchmark: Optional[PublishedBenchmark] = None,
        dashboard: Optional[ValidationDashboard] = None,
    ) -> str:

        content = build_session_report_content(
            session=session, project=project, benchmark=benchmark, dashboard=dashboard,
        )

        return render_markdown(content)

    def save_session_report(
        self,
        *,
        session: CalibrationSession,
        storage_root,
        project: Optional[CalibrationProject] = None,
        benchmark: Optional[PublishedBenchmark] = None,
        dashboard: Optional[ValidationDashboard] = None,
    ) -> Path:

        markdown = self.generate_session_report(
            session=session, project=project, benchmark=benchmark, dashboard=dashboard,
        )

        return save_session_report_markdown(markdown, session.session_id, storage_root)
