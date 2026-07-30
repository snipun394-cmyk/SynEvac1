from PyQt6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from designer.validation import validate_building_authoring
from navigation.graph_builder import NavigationGraphGenerator
from navigation.validation import ValidationReport

from builder.validation_extras import INFO, validate_builder_extras


class ValidationPanel(QWidget):

    # Dedicated, always-visible validation surface -- per the
    # feasibility investigation's own UX recommendation (Phase 7),
    # continuous/incremental validation reads better for a first-time
    # user than a single late "Validate" gate. BuilderMainWindow calls
    # refresh() after every authoring edit (the same
    # item_changed_callback/floors_changed_callback signals already
    # driving ProjectSummaryPanel/NavigationPreviewPanel), not only
    # from a toolbar button -- the toolbar's Validate action (see
    # builder_toolbar.py) still exists for an explicit re-check, but
    # is no longer the only way results appear.
    #
    # Combines three ALREADY-EXISTING/REUSED report sources plus one
    # Builder-only additive module, all speaking the same
    # ValidationReport/ValidationIssue value type:
    #   1. designer.validation.validate_building_authoring() -- Door/
    #      Exit/Stair zone-wiring completeness (ERROR), zone-assignment
    #      completeness for the rest of the asset palette (WARNING).
    #   2. navigation.graph_builder.NavigationGraphGenerator().build()
    #      .validate() -- structural graph validity, isolated zones,
    #      disconnected floors.
    #   3. builder.validation_extras.validate_builder_extras() -- the
    #      three Builder-specific checks the milestone brief names that
    #      neither existing validator covers (overlapping geometry,
    #      missing names, missing scale calibration).
    # None of designer/validation.py, navigation/validation.py, or
    # navigation/graph_builder.py is modified by this milestone.

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.summary_label = QLabel("No project loaded.")
        layout.addWidget(self.summary_label)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        self._last_report_valid = True

    # =====================================================

    @property
    def is_valid(self):

        # True means "no ERROR-severity issue" -- the same bar
        # save_project()/export gating uses (see BuilderMainWindow).
        # Warnings/Info never block.
        return self._last_report_valid

    # =====================================================

    def refresh(self, building):

        self.list_widget.clear()

        if building is None:

            self.summary_label.setText("No project loaded.")
            self._last_report_valid = True

            return

        authoring_report = validate_building_authoring(building)
        graph_report = NavigationGraphGenerator().build(building).validate()
        extras_report = validate_builder_extras(building)

        errors = []
        warnings = []
        info = []

        for report in (authoring_report, graph_report, extras_report):

            for issue in report:

                if issue.severity == ValidationReport.ERROR:
                    errors.append(issue)
                elif issue.severity == INFO:
                    info.append(issue)
                else:
                    warnings.append(issue)

        self._last_report_valid = not errors

        self.summary_label.setText(
            f"{len(errors)} error(s), {len(warnings)} warning(s), "
            f"{len(info)} note(s)."
        )

        for issue in errors:
            self._add_row("ERROR", issue.message)

        for issue in warnings:
            self._add_row("WARNING", issue.message)

        for issue in info:
            self._add_row("INFO", issue.message)

        if not errors and not warnings and not info:
            self._add_row("OK", "No issues found.")

    # =====================================================

    def _add_row(self, severity, message):

        item = QListWidgetItem(f"[{severity}] {message}")

        self.list_widget.addItem(item)
