from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QLabel,
    QLineEdit,
)


class PropertyPanel(QWidget):

    GRID_SIZE = 50

    def __init__(self):
        super().__init__()

        self.current_zone = None

        layout = QFormLayout()

        # =====================================================
        # General
        # =====================================================

        self.object_type = QLabel("No Selection")
        self.object_id = QLabel("-")
        self.object_name = QLineEdit()

        layout.addRow("Object", self.object_type)
        layout.addRow("ID", self.object_id)
        layout.addRow("Name", self.object_name)

        # =====================================================
        # Geometry
        # =====================================================

        self.origin_x = QLineEdit()
        self.origin_y = QLineEdit()

        self.width = QLineEdit()
        self.height = QLineEdit()

        layout.addRow("Origin X (m)", self.origin_x)
        layout.addRow("Origin Y (m)", self.origin_y)

        layout.addRow("Width (m)", self.width)
        layout.addRow("Height (m)", self.height)

        # =====================================================
        # Derived Coordinates
        # =====================================================

        self.top_left = QLabel("-")
        self.top_right = QLabel("-")
        self.bottom_right = QLabel("-")
        self.bottom_left = QLabel("-")

        self.area = QLabel("-")

        layout.addRow("Top Left", self.top_left)
        layout.addRow("Top Right", self.top_right)
        layout.addRow("Bottom Right", self.bottom_right)
        layout.addRow("Bottom Left", self.bottom_left)

        layout.addRow("Area", self.area)

        self.setLayout(layout)

        # =====================================================

        self.object_name.editingFinished.connect(
            self.rename_object
        )

        self.origin_x.editingFinished.connect(
            self.update_geometry
        )

        self.origin_y.editingFinished.connect(
            self.update_geometry
        )

        self.width.editingFinished.connect(
            self.update_geometry
        )

        self.height.editingFinished.connect(
            self.update_geometry
        )

    # =====================================================

    def show_rectangle(self, zone):

        self.current_zone = zone

        self.object_type.setText("Zone")
        self.object_id.setText(zone.zone_id)

        self.object_name.blockSignals(True)
        self.origin_x.blockSignals(True)
        self.origin_y.blockSignals(True)
        self.width.blockSignals(True)
        self.height.blockSignals(True)

        self.object_name.setText(zone.zone_name)

        tlx, tly = zone.top_left
        trx, try_ = zone.top_right
        brx, bry = zone.bottom_right
        blx, bly = zone.bottom_left

        self.origin_x.setText(f"{tlx:.2f}")
        self.origin_y.setText(f"{tly:.2f}")

        self.width.setText(f"{zone.width_m:.2f}")
        self.height.setText(f"{zone.height_m:.2f}")

        self.top_left.setText(
            f"({tlx:.2f}, {tly:.2f})"
        )

        self.top_right.setText(
            f"({trx:.2f}, {try_:.2f})"
        )

        self.bottom_right.setText(
            f"({brx:.2f}, {bry:.2f})"
        )

        self.bottom_left.setText(
            f"({blx:.2f}, {bly:.2f})"
        )

        self.area.setText(
            f"{zone.area_m2:.2f} m²"
        )

        self.object_name.blockSignals(False)
        self.origin_x.blockSignals(False)
        self.origin_y.blockSignals(False)
        self.width.blockSignals(False)
        self.height.blockSignals(False)

    # =====================================================

    def refresh(self):

        if self.current_zone:
            self.show_rectangle(
                self.current_zone
            )

    # =====================================================

    def clear(self):

        self.current_zone = None

        self.object_type.setText(
            "No Selection"
        )

        self.object_id.setText("-")

        self.object_name.clear()

        self.origin_x.clear()
        self.origin_y.clear()

        self.width.clear()
        self.height.clear()

        self.top_left.setText("-")
        self.top_right.setText("-")
        self.bottom_right.setText("-")
        self.bottom_left.setText("-")

        self.area.setText("-")

    # =====================================================

    def rename_object(self):

        if self.current_zone is None:
            return

        name = self.object_name.text().strip()

        if not name:
            return

        self.current_zone.rename(name)

    # =====================================================

    def update_geometry(self):

        if self.current_zone is None:
            return

        try:

            x = float(
                self.origin_x.text()
            )

            y = float(
                self.origin_y.text()
            )

            w = float(
                self.width.text()
            )

            h = float(
                self.height.text()
            )

        except ValueError:

            self.refresh()

            return

        self.current_zone.setPos(
            x * self.GRID_SIZE,
            y * self.GRID_SIZE,
        )

        self.current_zone.setRect(
            0,
            0,
            w * self.GRID_SIZE,
            h * self.GRID_SIZE,
        )

        self.refresh()