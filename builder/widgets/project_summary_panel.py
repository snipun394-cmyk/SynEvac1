from PyQt6.QtWidgets import QFormLayout, QLabel, QWidget


class ProjectSummaryPanel(QWidget):

    # Dumb display widget -- BuilderMainWindow pushes updates in via
    # refresh(), same convention every other Builder/Studio panel in
    # this codebase already follows (see e.g. designer/widgets/
    # camera_manager_panel.py's own "dumb widget, MainWindow pushes
    # updates in" comment). Never reads Building/ValidationPanel state
    # on its own initiative.

    def __init__(self):
        super().__init__()

        layout = QFormLayout()
        self.setLayout(layout)

        self.floor_count = QLabel("0")
        self.zone_count = QLabel("0")
        self.door_count = QLabel("0")
        self.exit_count = QLabel("0")
        self.stair_count = QLabel("0")
        self.camera_count = QLabel("0")
        self.smoke_detector_count = QLabel("0")
        self.heat_detector_count = QLabel("0")
        self.speaker_count = QLabel("0")
        self.obstacle_count = QLabel("0")
        self.total_area = QLabel("0.00 m²")
        self.scale_status = QLabel("-")
        self.validation_status = QLabel("-")

        layout.addRow("Floors", self.floor_count)
        layout.addRow("Zones", self.zone_count)
        layout.addRow("Doors", self.door_count)
        layout.addRow("Exits", self.exit_count)
        layout.addRow("Stairs", self.stair_count)
        layout.addRow("Cameras", self.camera_count)
        layout.addRow("Smoke Detectors", self.smoke_detector_count)
        layout.addRow("Heat Detectors", self.heat_detector_count)
        layout.addRow("Speakers", self.speaker_count)
        layout.addRow("Obstacles", self.obstacle_count)
        layout.addRow("Total Zone Area", self.total_area)
        layout.addRow("Scale Calibration", self.scale_status)
        layout.addRow("Validation Status", self.validation_status)

    # =====================================================

    def refresh(self, building, validation_is_valid=True, validation_summary_text=""):

        if building is None:

            for label in (
                self.floor_count, self.zone_count, self.door_count, self.exit_count,
                self.stair_count, self.camera_count, self.smoke_detector_count,
                self.heat_detector_count, self.speaker_count, self.obstacle_count,
            ):
                label.setText("0")

            self.total_area.setText("0.00 m²")
            self.scale_status.setText("-")
            self.validation_status.setText("-")

            return

        floors = building.floors

        zone_total = sum(len(floor.zones) for floor in floors)
        door_total = sum(len(floor.doors) for floor in floors)
        exit_total = sum(len(floor.exits) for floor in floors)
        stair_total = sum(len(floor.stairs) for floor in floors)
        camera_total = sum(len(floor.cameras) for floor in floors)
        smoke_total = sum(len(floor.smoke_detectors) for floor in floors)
        heat_total = sum(len(floor.heat_detectors) for floor in floors)
        speaker_total = sum(len(floor.speakers) for floor in floors)
        obstacle_total = sum(len(floor.obstacles) for floor in floors)

        area_total = sum(
            zone.area
            for floor in floors
            for zone in floor.zones
        )

        self.floor_count.setText(str(len(floors)))
        self.zone_count.setText(str(zone_total))
        self.door_count.setText(str(door_total))
        self.exit_count.setText(str(exit_total))
        self.stair_count.setText(str(stair_total))
        self.camera_count.setText(str(camera_total))
        self.smoke_detector_count.setText(str(smoke_total))
        self.heat_detector_count.setText(str(heat_total))
        self.speaker_count.setText(str(speaker_total))
        self.obstacle_count.setText(str(obstacle_total))
        self.total_area.setText(f"{area_total:.2f} m²")

        calibrated_count = sum(1 for floor in floors if floor.is_scale_calibrated)

        self.scale_status.setText(
            f"{calibrated_count} / {len(floors)} floor(s) calibrated"
        )

        self.validation_status.setText(
            ("VALID -- " if validation_is_valid else "INVALID -- ") + validation_summary_text
        )
