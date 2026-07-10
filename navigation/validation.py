from dataclasses import dataclass


@dataclass
class ValidationIssue:

    code: str
    severity: str
    message: str

    object_id: str = ""
    floor_id: str = ""


class ValidationReport:

    # Data-integrity problems -- the graph itself is unreliable
    # (an id collision, or an edge pointing at a node that doesn't
    # exist). These should never happen from normal Designer use;
    # they indicate a corrupted or hand-edited project file.
    ERROR = "error"

    # Topological/completeness problems -- the graph is internally
    # consistent, but the Designer model isn't finished yet (a Door
    # nobody assigned a Zone to, a Zone nobody can escape from). Not
    # fatal to building the graph itself, only to using it fully.
    WARNING = "warning"

    def __init__(self):

        self.issues = []

    # =====================================================

    def add(
        self,
        code,
        message,
        severity="error",
        object_id="",
        floor_id="",
    ):

        self.issues.append(
            ValidationIssue(
                code=code,
                severity=severity,
                message=message,
                object_id=object_id,
                floor_id=floor_id,
            )
        )

    # =====================================================

    @property
    def is_valid(self):

        return not any(
            issue.severity == self.ERROR
            for issue in self.issues
        )

    # =====================================================

    @property
    def errors(self):

        return [
            issue
            for issue in self.issues
            if issue.severity == self.ERROR
        ]

    # =====================================================

    @property
    def warnings(self):

        return [
            issue
            for issue in self.issues
            if issue.severity == self.WARNING
        ]

    # =====================================================

    def by_code(self, code):

        return [
            issue
            for issue in self.issues
            if issue.code == code
        ]

    # =====================================================

    def __len__(self):

        return len(self.issues)

    def __iter__(self):

        return iter(self.issues)

    # =====================================================

    def __repr__(self):

        return (
            f"ValidationReport("
            f"errors={len(self.errors)}, "
            f"warnings={len(self.warnings)})"
        )
