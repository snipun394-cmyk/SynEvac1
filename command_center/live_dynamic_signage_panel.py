from functools import partial

from PyQt6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from dynamic_signage.consistency import detect_inconsistencies
from dynamic_signage.models import SignageStatus

from command_center.live_operator_action_gateway import PROVIDER_CAPABILITY_NO_PROVIDER, SignageRequestStatus


# =====================================================
# LiveDynamicSignagePanel -- Live Dynamic Sign Operator Approval &
# Dispatch Completion milestone. Display + explicit operator Approve/
# Reject affordance for each PLANNED per-sign instruction, mirroring
# command_center.live_evacuation_guidance_panel.LiveEvacuationGuidancePanel's
# own established live-mode shape and command_center.building_controls_
# panel.BuildingControlsPanel's own "submit-then-decide, status tracked
# by the controller itself" workflow exactly.
#
# DECISION SUPPORT + EXPLICIT OPERATOR ACTION ONLY -- every Approve
# click routes through LiveOperatorActionGateway.approve_signage_
# instruction(), the ONLY seam that can ever reach DynamicSignageController/
# DynamicSignageProvider. This panel never imports dynamic_signage.
# controller/dynamic_signage.provider itself (mechanically enforced --
# see tests/test_live_command_center.py::CommandCenterLiveIntegrationGuardTests).
#
# Each sign's decision is entirely independent (Phase 10): approving or
# rejecting one row never touches any other row's own PENDING_APPROVAL/
# APPROVED/REJECTED state, because DynamicSignageController itself keys
# every decision on "{sign_id}::{signage_revision}" -- this panel adds
# no cross-sign logic of its own.
# =====================================================


def _format_validation(instruction, conflict, codes) -> str:

    if conflict is not None:
        return f"CONFLICT: {conflict.reason}"

    if codes:
        return "; ".join(codes)

    if instruction.status == SignageStatus.ACTIVE:
        return "OK"

    return instruction.reason or "-"


_DISPATCH_STATUS_LABELS = {
    SignageRequestStatus.REJECTED: "Rejected by operator",
    SignageRequestStatus.SUPERSEDED: "Superseded",
    SignageRequestStatus.APPROVED: "Approved",
    SignageRequestStatus.DISPATCHED: "Dispatching...",
    SignageRequestStatus.CONFIRMED: "Confirmed",
    SignageRequestStatus.FAILED: "Failed",
}


class LiveDynamicSignagePanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._signage_snapshot = None
        self._guidance_snapshot = None
        self._gateway = None
        self._time = 0.0

        layout = QVBoxLayout(self)

        note = QLabel(
            "Decision support + explicit operator action only -- a planned sign instruction never "
            "reaches a physical/simulated sign until approved here. Approving or rejecting one sign "
            "never affects any other sign, and never broadcasts a Voice Evacuation message."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        table_group = QGroupBox("Live Dynamic Signage")
        table_layout = QVBoxLayout(table_group)
        self.sign_table = QTableWidget(0, 10)
        self.sign_table.setHorizontalHeaderLabels([
            "Sign", "Zone", "Target Exit", "Indication", "Plan Revision", "Guidance Revision",
            "Availability", "Validation", "Dispatch Status", "Decision",
        ])
        self.sign_table.horizontalHeader().setStretchLastSection(True)
        self.sign_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.sign_table)
        layout.addWidget(table_group)

        layout.addStretch(1)

    # =====================================================

    def show_live(self, signage_snapshot, guidance_snapshot=None, gateway=None, time: float = 0.0) -> None:

        self._signage_snapshot = signage_snapshot
        self._guidance_snapshot = guidance_snapshot
        self._gateway = gateway
        self._time = time

        if gateway is not None:
            # Mirrors BuildingControlsPanel.show_live()'s own
            # ingest-then-render precedent exactly -- safe to call every
            # render cycle, DynamicSignageController.submit() itself
            # already dedups/supersedes (dynamic_signage/controller.py).
            gateway.ingest_signage_instructions(signage_snapshot)

        if signage_snapshot is None or not signage_snapshot.instructions:
            self.sign_table.setRowCount(0)
            return

        inconsistencies = (
            detect_inconsistencies(guidance_snapshot, signage_snapshot) if guidance_snapshot is not None else {}
        )

        sign_ids = sorted(signage_snapshot.instructions.keys())
        self.sign_table.setRowCount(len(sign_ids))

        for row_index, sign_id in enumerate(sign_ids):

            instruction = signage_snapshot.instructions[sign_id]
            conflict = signage_snapshot.conflict(sign_id)
            codes = inconsistencies.get(sign_id, ())

            self.sign_table.setItem(row_index, 0, QTableWidgetItem(sign_id))
            self.sign_table.setItem(row_index, 1, QTableWidgetItem(instruction.zone_id or "-"))
            self.sign_table.setItem(row_index, 2, QTableWidgetItem(instruction.recommended_exit_id or "-"))
            self.sign_table.setItem(row_index, 3, QTableWidgetItem(instruction.indication))
            self.sign_table.setItem(row_index, 4, QTableWidgetItem(str(instruction.signage_revision)))
            self.sign_table.setItem(row_index, 5, QTableWidgetItem(str(instruction.guidance_revision)))
            self.sign_table.setItem(row_index, 6, QTableWidgetItem(instruction.status))
            self.sign_table.setItem(row_index, 7, QTableWidgetItem(_format_validation(instruction, conflict, codes)))

            if instruction.status == SignageStatus.ACTIVE:
                dispatch_status = (
                    gateway.signage_instruction_status(instruction) if gateway is not None
                    else SignageRequestStatus.PENDING_APPROVAL
                )
            else:
                # Never submitted to the controller at all -- an
                # honestly empty dispatch status, never a fabricated
                # PENDING_APPROVAL for an instruction nothing was ever
                # asked to approve.
                dispatch_status = "NOT SUBMITTED"

            self.sign_table.setItem(row_index, 8, QTableWidgetItem(dispatch_status))
            self._set_decision_cell(row_index, instruction, dispatch_status, codes, gateway)

    # =====================================================

    def _set_decision_cell(self, row_index, instruction, dispatch_status, codes, gateway) -> None:

        if instruction.status != SignageStatus.ACTIVE:

            self.sign_table.setItem(row_index, 9, QTableWidgetItem("No actionable instruction -- nothing to approve"))
            return

        if codes:

            # Phase 7/11 -- an Approve button existing is never itself
            # permission to dispatch; an inconsistent instruction stays
            # explicitly unapprovable, with the reason stated plainly.
            self.sign_table.setItem(row_index, 9, QTableWidgetItem(f"Approval unavailable: {', '.join(codes)}"))
            return

        if dispatch_status != SignageRequestStatus.PENDING_APPROVAL:

            label = _DISPATCH_STATUS_LABELS.get(dispatch_status, dispatch_status)
            self.sign_table.setItem(row_index, 9, QTableWidgetItem(label))
            return

        capability = gateway.signage_capability if gateway is not None else PROVIDER_CAPABILITY_NO_PROVIDER

        widget = QWidget()
        decision_layout = QHBoxLayout(widget)
        decision_layout.setContentsMargins(0, 0, 0, 0)

        approve_button = QPushButton("Approve")
        approve_button.setEnabled(gateway is not None and capability != PROVIDER_CAPABILITY_NO_PROVIDER)
        approve_button.clicked.connect(partial(self._on_approve, instruction))
        decision_layout.addWidget(approve_button)

        reject_button = QPushButton("Reject")
        reject_button.setEnabled(gateway is not None)
        reject_button.clicked.connect(partial(self._on_reject, instruction))
        decision_layout.addWidget(reject_button)

        self.sign_table.setCellWidget(row_index, 9, widget)

    # =====================================================

    def _on_approve(self, instruction) -> None:

        if self._gateway is None:
            return

        if self._gateway.signage_instruction_status(instruction) == SignageRequestStatus.PENDING_APPROVAL:

            try:
                self._gateway.approve_signage_instruction(instruction, self._time, self._guidance_snapshot)
            except Exception:
                # Honest failure (e.g. Phase 7's consistency gate, or a
                # duplicate click racing a newer revision's own
                # supersession) -- never crash, the re-render below
                # shows whatever the real current state actually is.
                pass

        self.show_live(self._signage_snapshot, self._guidance_snapshot, self._gateway, self._time)

    # =====================================================

    def _on_reject(self, instruction) -> None:

        if self._gateway is None:
            return

        if self._gateway.signage_instruction_status(instruction) == SignageRequestStatus.PENDING_APPROVAL:

            try:
                self._gateway.reject_signage_instruction(instruction, self._time)
            except Exception:
                pass

        self.show_live(self._signage_snapshot, self._guidance_snapshot, self._gateway, self._time)
