from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from camera_validation.validator import validate_building


class CameraValidationPanel(QWidget):

    # Camera Calibration & Placement Validation -- an on-demand
    # engineering report over camera_validation/validator.py, not a
    # continuously-live view. Unlike CameraManagerPanel/
    # PerceptionDebugPanel (refreshed on every simulation tick), this
    # panel deliberately only recomputes when the engineer explicitly
    # asks it to (the Run Validation button, or once when a project
    # first loads) -- running the Visibility Engine's ray-sweep for
    # every camera on every tick would be wasted work for a report
    # whose whole point is "does this placement look right", not "what
    # is happening on screen right now". Same "dumb widget, MainWindow
    # pushes building in" convention as every other Designer panel.

    def __init__(self):

        super().__init__()

        self._last_building = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        header_row = QHBoxLayout()

        run_button = QPushButton("Run Validation")
        run_button.clicked.connect(self._run_validation)
        header_row.addWidget(run_button)

        header_row.addWidget(QLabel("Overall Network Score:"))
        self.network_score_label = QLabel("-")
        header_row.addWidget(self.network_score_label)

        layout.addLayout(header_row)

        layout.addWidget(QLabel("Cameras"))

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(7)
        self.camera_table.setHorizontalHeaderLabels(
            ["Camera", "Floor", "Placement Score", "Coverage %", "Doors", "Exits", "Stairs"],
        )
        self.camera_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.camera_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.camera_table.setRowCount(0)
        layout.addWidget(self.camera_table)

        layout.addWidget(QLabel("Recommendations"))

        self.recommendation_table = QTableWidget()
        self.recommendation_table.setColumnCount(3)
        self.recommendation_table.setHorizontalHeaderLabels(["Subject", "Category", "Recommendation"])
        self.recommendation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.recommendation_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.recommendation_table.setRowCount(0)
        layout.addWidget(self.recommendation_table)

    # =====================================================
    # Public entry point
    # =====================================================

    def refresh(self, building):

        self._last_building = building

        self._run_validation()

    # =====================================================

    def _run_validation(self):

        if self._last_building is None:

            self.network_score_label.setText("-")
            self.camera_table.setRowCount(0)
            self.recommendation_table.setRowCount(0)

            return

        floors = self._last_building.ordered_floors()

        cameras = [camera for floor in floors for camera in floor.cameras]

        floor_names = {floor.id: floor.name for floor in floors}

        report = validate_building(cameras, self._last_building)

        self.network_score_label.setText(f"{report.overall_network_score:.1f} / 100")

        self._refresh_camera_table(report, floors, floor_names)
        self._refresh_recommendation_table(report, floors)

    # =====================================================

    def _refresh_camera_table(self, report, floors, floor_names):

        rows = []

        for floor in floors:

            floor_report = report.floor_report(floor.id)

            if floor_report is None:
                continue

            for camera in floor.cameras:

                metrics = floor_report.camera_metrics.get(camera.id)

                if metrics is None:
                    continue

                rows.append((camera, floor, metrics))

        self.camera_table.setRowCount(len(rows))

        for row, (camera, floor, metrics) in enumerate(rows):

            self.camera_table.setItem(row, 0, QTableWidgetItem(camera.name or camera.id))
            self.camera_table.setItem(row, 1, QTableWidgetItem(floor_names.get(floor.id, floor.id)))
            self.camera_table.setItem(row, 2, QTableWidgetItem(f"{metrics.placement_score:.1f}"))
            self.camera_table.setItem(row, 3, QTableWidgetItem(f"{metrics.zone_coverage_percentage:.1f}"))
            self.camera_table.setItem(row, 4, QTableWidgetItem(str(len(metrics.visible_door_ids))))
            self.camera_table.setItem(row, 5, QTableWidgetItem(str(len(metrics.visible_exit_ids))))
            self.camera_table.setItem(row, 6, QTableWidgetItem(str(len(metrics.visible_stair_ids))))

    # =====================================================

    def _refresh_recommendation_table(self, report, floors):

        recommendations = []

        for floor in floors:

            floor_report = report.floor_report(floor.id)

            if floor_report is None:
                continue

            recommendations.extend(floor_report.recommendations)

        self.recommendation_table.setRowCount(len(recommendations))

        for row, recommendation in enumerate(recommendations):

            self.recommendation_table.setItem(row, 0, QTableWidgetItem(recommendation.subject_type))
            self.recommendation_table.setItem(row, 1, QTableWidgetItem(recommendation.category))
            self.recommendation_table.setItem(row, 2, QTableWidgetItem(recommendation.message))
