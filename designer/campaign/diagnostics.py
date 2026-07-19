from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


# Diagnostics Mode -- a normalized, aggregate view over three
# differently-shaped issue types this codebase already produces
# (designer.validation.ValidationIssue for Building authoring,
# scenario_definition.validation.DefinitionValidationIssue for a
# ScenarioDefinition's own structural well-formedness,
# scenario_validator.issue.ScenarioValidationIssue for a sampled
# candidate) into one common display row, so CampaignWindow's
# Validation Report panel needs to know only one shape, never three.
# This module performs no validation of its own -- it only records and
# aggregates reports every one of those three, already-frozen
# validators already produces.


@dataclass(frozen=True)
class ValidationRow:

    # `source` is "BUILDING"/"DEFINITION" for a pre-flight issue, or a
    # scenario_validator.issue.FailureCategory value ("NAVIGATION",
    # "FIRE", ...) for a per-candidate one -- one field doing the job
    # of what would otherwise be two differently-typed issue classes,
    # since a display row never needs to distinguish "which validator"
    # beyond "what category of problem is this."

    source: str
    code: str
    message: str
    count: int = 1


@dataclass(frozen=True)
class PreflightResult:

    building_issues: Tuple[ValidationRow, ...] = ()
    definition_issues: Tuple[ValidationRow, ...] = ()

    # =====================================================

    @property
    def has_errors(self) -> bool:

        return bool(self.building_issues) or bool(self.definition_issues)


@dataclass(frozen=True)
class DiagnosticsSummary:

    total_candidates: int
    rejected_candidates: int

    category_counts: Mapping[str, int]
    rows: Tuple[ValidationRow, ...]

    first_rejection: Optional[ValidationRow]


def explain_total_rejection(summary: DiagnosticsSummary) -> str:

    # Requirement 7 -- a clear, specific explanation for the "0
    # accepted" case, built entirely from data this module already
    # aggregated (never re-runs anything).

    if summary.rejected_candidates == 0 or not summary.rows:
        return (
            "Every scenario was rejected, but no validation issues were "
            "recorded -- this itself is unexpected; check Diagnostics Mode "
            "is enabled."
        )

    dominant_category, dominant_count = max(
        summary.category_counts.items(), key=lambda pair: pair[1],
    )
    top_row = max(summary.rows, key=lambda row: row.count)

    return (
        f"Every generated scenario was rejected "
        f"({summary.rejected_candidates} rejected generation attempt(s) "
        f"across {summary.total_candidates} total attempt(s)). "
        f"The dominant failure category was {dominant_category} "
        f"({dominant_count} occurrence(s)). "
        f"Most common issue: [{top_row.source}/{top_row.code}] {top_row.message} "
        f"(seen {top_row.count} time(s)). "
        f"Adjust Scenario Definition to reduce this category of rejection, "
        f"or review the Validation Report panel for the full breakdown."
    )


class DiagnosticsCollector:

    # Owns exactly the aggregate state Requirement 3 asks for
    # ("rejection counts grouped by FailureCategory") plus the bounded
    # per-(category, code) row list Requirement 4's panel displays --
    # bounded because there are only FailureCategory.ALL's seven
    # categories times a small, fixed number of codes each validation
    # module defines, so this stays cheap regardless of how many
    # thousands of candidates a campaign generates (only the *counts*
    # grow, never the number of distinct rows).

    def __init__(self):

        self.total_candidates = 0
        self.rejected_candidates = 0

        self._category_counts = Counter()
        self._row_counts = Counter()
        self._row_messages = {}
        self._row_order = []

        self.first_rejection: Optional[ValidationRow] = None

    # =====================================================

    def record_candidate_report(self, report) -> Optional[ValidationRow]:

        # Returns the first-rejection row exactly once -- the first
        # call across this collector's whole lifetime that records a
        # rejected report -- and None on every other call, so a caller
        # never needs to track its own "have I already shown this"
        # flag to satisfy Requirement 2 ("display the first rejection
        # immediately").

        self.total_candidates += 1

        if report.accepted:
            return None

        self.rejected_candidates += 1

        first_row_this_report = None

        for issue in report.errors:

            key = (issue.category, issue.code)

            self._category_counts[issue.category] += 1
            self._row_counts[key] += 1

            if key not in self._row_messages:

                self._row_messages[key] = issue.message
                self._row_order.append(key)

            row = ValidationRow(
                source=issue.category, code=issue.code,
                message=self._row_messages[key], count=self._row_counts[key],
            )

            if first_row_this_report is None:
                first_row_this_report = row

        if self.first_rejection is None and first_row_this_report is not None:

            self.first_rejection = first_row_this_report
            return first_row_this_report

        return None

    # =====================================================

    def summary(self) -> DiagnosticsSummary:

        rows = tuple(
            ValidationRow(
                source=category, code=code,
                message=self._row_messages[(category, code)],
                count=self._row_counts[(category, code)],
            )
            for category, code in self._row_order
        )

        return DiagnosticsSummary(
            total_candidates=self.total_candidates,
            rejected_candidates=self.rejected_candidates,
            category_counts=dict(self._category_counts),
            rows=rows,
            first_rejection=self.first_rejection,
        )
