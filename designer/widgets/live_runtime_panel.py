from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from live_runtime_launcher.modes import ApplicationMode


_STOPPED_DISPLAY = {
    "state": "STOPPED",
    "ai_status": "NOT CONFIGURED",
    "capabilities": {
        "camera": "NO_PROVIDER", "voice": "NO_PROVIDER",
        "dynamic_signage": "NO_PROVIDER", "building_control": "NO_PROVIDER",
    },
    "error": None,
}


class LiveRuntimePanel(QWidget):

    # Application Live Runtime Launcher milestone -- the Designer-facing
    # surface for entering LIVE/OFFLINE_DEMO application mode against
    # the currently loaded project. Same "dumb widget, controller pushes
    # updates in" convention as CameraManagerPanel/SpeakerManagerPanel:
    # this class owns no LiveRuntime/LiveRuntimeSession state of its
    # own, only the mode selector and Start/Stop/Open Command Center
    # buttons -- designer.live_runtime_controller.LiveRuntimeController
    # is the only class that ever touches live_runtime_launcher.session.
    # LiveRuntimeSession, mirroring designer.campaign.campaign_controller.
    # CampaignController's own mediator role for Campaign Studio.

    def __init__(self):

        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Application Mode:"))

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Live", ApplicationMode.LIVE)
        self.mode_combo.addItem("Offline Demo", ApplicationMode.OFFLINE_DEMO)
        mode_row.addWidget(self.mode_combo)

        layout.addLayout(mode_row)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        self.ai_status_label = QLabel()
        layout.addWidget(self.ai_status_label)

        capability_box = QGroupBox("Provider Capability")
        capability_layout = QVBoxLayout()
        capability_box.setLayout(capability_layout)

        self.camera_capability_label = QLabel()
        self.voice_capability_label = QLabel()
        self.signage_capability_label = QLabel()
        self.building_control_capability_label = QLabel()

        for label in (
            self.camera_capability_label, self.voice_capability_label,
            self.signage_capability_label, self.building_control_capability_label,
        ):
            capability_layout.addWidget(label)

        layout.addWidget(capability_box)

        button_row = QHBoxLayout()

        self.start_button = QPushButton("Start Live Runtime")
        self.stop_button = QPushButton("Stop Live Runtime")
        self.open_command_center_button = QPushButton("Open Command Center")

        for button in (self.start_button, self.stop_button, self.open_command_center_button):
            button_row.addWidget(button)

        layout.addLayout(button_row)

        self.error_label = QLabel()
        layout.addWidget(self.error_label)

        layout.addStretch(1)

        self.set_state(**_STOPPED_DISPLAY)

    # =====================================================

    def selected_mode(self) -> ApplicationMode:

        return self.mode_combo.currentData()

    # =====================================================

    def set_state(self, state, *, ai_status, capabilities, error) -> None:

        self.status_label.setText(f"Status: {state}")
        self.ai_status_label.setText(f"AI: {ai_status}")

        self.camera_capability_label.setText(f"Camera: {capabilities['camera']}")
        self.voice_capability_label.setText(f"Voice: {capabilities['voice']}")
        self.signage_capability_label.setText(f"Dynamic Signage: {capabilities['dynamic_signage']}")
        self.building_control_capability_label.setText(f"Building Control: {capabilities['building_control']}")

        self.error_label.setText(error or "")
