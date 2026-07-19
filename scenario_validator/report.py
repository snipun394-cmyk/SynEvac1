from scenario_validator.issue import ScenarioValidationIssue


class ScenarioValidationReport:

    # §5.4's frozen shape: issues / accepted (computed, never stored) /
    # metadata. `accepted` and `is_valid` are the same property under
    # two names -- `accepted` matches this pass's own vocabulary
    # ("Accepted"), `is_valid` matches the
    # navigation.validation.ValidationReport /
    # scenario_definition.validation.DefinitionValidationReport
    # convention this package's shape otherwise mirrors, so callers
    # already familiar with either sibling report type need to learn
    # nothing new here.
    #
    # Stateless with respect to anything beyond the issues handed to it
    # -- this class itself never reasons about attempts, retries, or
    # sequences of reports (§5.2/§5.8: that is exclusively
    # orchestration's job, built by collecting many of these reports
    # over time). is_single_category_attributable() and
    # error_categories()/error_codes() below expose exactly the
    # per-report signal §5.8 resolved as sufficient for that future
    # orchestration work -- they compute nothing this report doesn't
    # already carry.

    ERROR = "error"
    WARNING = "warning"

    def __init__(self, issues=None, metadata=None):

        self.issues = list(issues) if issues else []
        self.metadata = dict(metadata) if metadata else {}

    # =====================================================

    def add(self, category, severity, code, message, object_id=""):

        self.issues.append(
            ScenarioValidationIssue(
                category=category,
                severity=severity,
                code=code,
                message=message,
                object_id=object_id,
            )
        )

    # =====================================================

    def extend(self, issues):

        self.issues.extend(issues)

    # =====================================================

    @property
    def accepted(self):

        return not any(issue.severity == self.ERROR for issue in self.issues)

    # Alias -- see class docstring.
    is_valid = accepted

    # =====================================================

    @property
    def errors(self):

        return [issue for issue in self.issues if issue.severity == self.ERROR]

    # =====================================================

    @property
    def warnings(self):

        return [issue for issue in self.issues if issue.severity == self.WARNING]

    # =====================================================

    def by_code(self, code):

        return [issue for issue in self.issues if issue.code == code]

    # =====================================================

    def by_category(self, category):

        return [issue for issue in self.issues if issue.category == category]

    # =====================================================

    def error_categories(self):

        # The set of categories any ERROR-severity issue belongs to --
        # §5.8's category-attribution signal.
        return {issue.category for issue in self.errors}

    # =====================================================

    def is_single_category_attributable(self):

        # §5.8, resolved this pass: a rejection is single-category-
        # attributable exactly when every ERROR-severity issue shares
        # one category. True (vacuously) for an accepted report, since
        # there is nothing to attribute.
        return len(self.error_categories()) <= 1

    # =====================================================

    def summary(self):

        # Summary statistics (this implementation phase's own Output
        # requirement) -- a plain dict of counts, computed fresh from
        # `issues` every call rather than tracked incrementally, so it
        # can never drift out of sync with the issues actually present.
        return {
            "accepted": self.accepted,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "categories": sorted({issue.category for issue in self.issues}),
            "codes": sorted({issue.code for issue in self.issues}),
        }

    # =====================================================

    def __len__(self):

        return len(self.issues)

    def __iter__(self):

        return iter(self.issues)

    # =====================================================

    def __repr__(self):

        return (
            f"ScenarioValidationReport(accepted={self.accepted}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )
