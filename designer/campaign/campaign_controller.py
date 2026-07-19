import hashlib
import json
import os

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox

from designer.campaign.campaign_worker import CampaignConfig, CampaignWorker


def derive_definition_id(definition) -> str:

    # No Definition Storage/Catalog exists anywhere in this codebase
    # (docs/architecture/dataset_intent.md §2, still an open gap) --
    # GenerationRequest/run_pipeline() nonetheless require a caller-
    # supplied definition_id (scenario_generator/request.py). Restated
    # locally (sha256 of a canonical JSON encoding), not imported from
    # scenario_generator.metadata_builder -- the same "same algorithm,
    # independently implemented per package" precedent already used
    # for this exact purpose in dataset_intent.md §6 and
    # reproducibility_review.md §7.4.

    canonical = json.dumps(definition.to_dict(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return f"def-{digest[:16]}"


class CampaignController:

    # The mediator between CampaignWindow (raw widgets, no backend
    # knowledge of its own beyond building a ScenarioDefinition) and
    # CampaignWorker (the only place backend pipeline calls actually
    # happen). Every button the window exposes is wired here to
    # exactly one already-existing capability -- this class calls no
    # scenario_generator/scenario_pipeline/scenario_storage/
    # scenario_runner/behaviour_profile_resolver/ai_decision/
    # simulation_runtime API directly; it only starts/pauses/resumes/
    # cancels a CampaignWorker, which does.

    def __init__(self, window, get_building):

        self.window = window
        self.get_building = get_building

        self.worker = None

        self.window.start_button.clicked.connect(self.on_start)
        self.window.pause_button.clicked.connect(self.on_pause)
        self.window.resume_button.clicked.connect(self.on_resume)
        self.window.cancel_button.clicked.connect(self.on_cancel)
        self.window.open_output_button.clicked.connect(self.on_open_output)

    # =====================================================

    def build_config(self) -> CampaignConfig:

        building = self.get_building()

        if building is None:
            raise ValueError("No Building is loaded -- open or create a project first.")

        # Re-synced here, immediately before translation, rather than
        # trusting whatever the window's zone/door/... id lists were
        # last populated with (set_building(), normally called once by
        # MainWindow.open_campaign_studio()). Found via this phase's
        # own diagnostics mode: if the Designer's current Building
        # changes while the Campaign Studio window is already open
        # (e.g. the user opens a different project without reopening
        # this window), build_scenario_definition() would otherwise
        # keep constructing Distribution maps keyed by the *previous*
        # Building's ids -- every one of those ids is then reported
        # UNKNOWN_ID by the Definition pre-flight check (Requirement
        # 6) against the *current* Building, correctly catching the
        # mismatch, but only after the fact. Resyncing here prevents it
        # instead. A CampaignController bug, not a Scenario Engine one
        # -- no backend package is touched by this fix.
        self.window.set_building(building)

        output_directory = self.window.output_directory()

        if not output_directory:
            raise ValueError("Choose an Output Directory before starting a campaign.")

        engineering_issues = self.window.validate_engineering_setup()

        if engineering_issues:
            raise ValueError("\n".join(engineering_issues))

        population_issues = self.window.validate_population_setup()

        if population_issues:
            raise ValueError("\n".join(population_issues))

        definition = self.window.build_scenario_definition()

        return CampaignConfig(
            name=self.window.campaign_name(),
            building=building,
            definition=definition,
            definition_id=derive_definition_id(definition),
            count=self.window.scenario_count(),
            master_seed=self.window.master_seed(),
            max_attempts=self.window.max_attempts(),
            output_directory=output_directory,
            diagnostics_mode=self.window.diagnostics_mode(),
        )

    # =====================================================

    def on_start(self):

        if self.worker is not None and self.worker.isRunning():
            return

        try:
            config = self.build_config()

        except ValueError as exc:

            self.window.show_error(str(exc))
            return

        self.worker = CampaignWorker(config)

        self.worker.progress_updated.connect(self.window.update_progress)
        self.worker.campaign_finished.connect(self.on_finished)
        self.worker.error_occurred.connect(self.on_error)

        self.worker.preflight_completed.connect(self.window.show_preflight)
        self.worker.first_rejection_detected.connect(self.window.show_first_rejection)
        self.worker.diagnostics_updated.connect(self.window.update_diagnostics)

        self.window.set_running_state(True)

        self.worker.start()

    # =====================================================

    def on_pause(self):

        if self.worker is not None:

            self.worker.pause()
            self.window.set_paused_state(True)

    # =====================================================

    def on_resume(self):

        if self.worker is not None:

            self.worker.resume()
            self.window.set_paused_state(False)

    # =====================================================

    def on_cancel(self):

        if self.worker is not None:
            self.worker.cancel()

    # =====================================================

    def on_finished(self, summary):

        self.window.set_running_state(False)
        self.window.show_summary(summary)

    # =====================================================

    def on_error(self, message):

        self.window.set_running_state(False)
        self.window.show_error(message)

    # =====================================================

    def on_open_output(self):

        output_directory = self.window.output_directory()

        if output_directory and os.path.isdir(output_directory):

            QDesktopServices.openUrl(QUrl.fromLocalFile(output_directory))

        else:

            QMessageBox.warning(
                self.window, "Open Output Directory",
                "The output directory does not exist yet -- run a campaign first.",
            )
