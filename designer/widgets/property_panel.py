from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QCheckBox,
    QLabel,
    QLineEdit,
)


class PropertyPanel(QWidget):

    GRID_SIZE = 50

    def __init__(self):
        super().__init__()

        self.current_item = None
        self._refresh_handler = None

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
        # Zone Geometry
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
        # Zone Derived Coordinates
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

        self.zone_fields = [
            self.origin_x,
            self.origin_y,
            self.width,
            self.height,
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
            self.area,
        ]

        # =====================================================
        # Exit Geometry
        # =====================================================

        self.start_x = QLineEdit()
        self.start_y = QLineEdit()

        self.end_x = QLineEdit()
        self.end_y = QLineEdit()

        self.exit_width = QLineEdit()
        self.capacity = QLineEdit()

        self.length = QLabel("-")

        self.blocked = QCheckBox()

        layout.addRow("Start X (m)", self.start_x)
        layout.addRow("Start Y (m)", self.start_y)
        layout.addRow("End X (m)", self.end_x)
        layout.addRow("End Y (m)", self.end_y)

        layout.addRow("Length", self.length)

        layout.addRow("Exit Width (m)", self.exit_width)
        layout.addRow("Capacity", self.capacity)

        layout.addRow("Blocked", self.blocked)

        self.exit_fields = [
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
            self.length,
            self.exit_width,
            self.capacity,
            self.blocked,
        ]

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

        self.start_x.editingFinished.connect(
            self.update_exit_geometry
        )

        self.start_y.editingFinished.connect(
            self.update_exit_geometry
        )

        self.end_x.editingFinished.connect(
            self.update_exit_geometry
        )

        self.end_y.editingFinished.connect(
            self.update_exit_geometry
        )

        self.exit_width.editingFinished.connect(
            self.update_exit_geometry
        )

        self.capacity.editingFinished.connect(
            self.update_exit_geometry
        )

        self.blocked.toggled.connect(
            self.update_exit_blocked
        )

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)

    # =====================================================

    def _set_fields_visible(self, fields, visible):

        for field in fields:

            field.setVisible(visible)

            label = self.layout().labelForField(field)

            if label:
                label.setVisible(visible)

    # =====================================================
    # Zone
    # =====================================================

    def show_rectangle(self, zone):

        self.current_item = zone
        self._refresh_handler = self.show_rectangle

        self._set_fields_visible(self.zone_fields, True)
        self._set_fields_visible(self.exit_fields, False)

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
    # Exit
    # =====================================================

    def show_line(self, exit_item):

        self.current_item = exit_item
        self._refresh_handler = self.show_line

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, True)

        model = exit_item.model

        self.object_type.setText("Exit")
        self.object_id.setText(exit_item.object_id)

        self.object_name.blockSignals(True)
        self.start_x.blockSignals(True)
        self.start_y.blockSignals(True)
        self.end_x.blockSignals(True)
        self.end_y.blockSignals(True)
        self.exit_width.blockSignals(True)
        self.capacity.blockSignals(True)
        self.blocked.blockSignals(True)

        self.object_name.setText(exit_item.object_name)

        if model is not None:

            sx, sy = model.start_point
            ex, ey = model.end_point

            self.start_x.setText(f"{sx:.2f}")
            self.start_y.setText(f"{sy:.2f}")
            self.end_x.setText(f"{ex:.2f}")
            self.end_y.setText(f"{ey:.2f}")

            self.length.setText(
                f"{model.length:.2f} m"
            )

            self.exit_width.setText(
                f"{model.width:.2f}"
            )

            self.capacity.setText(
                str(model.capacity)
            )

            self.blocked.setChecked(
                model.is_blocked
            )

        self.object_name.blockSignals(False)
        self.start_x.blockSignals(False)
        self.start_y.blockSignals(False)
        self.end_x.blockSignals(False)
        self.end_y.blockSignals(False)
        self.exit_width.blockSignals(False)
        self.capacity.blockSignals(False)
        self.blocked.blockSignals(False)

    # =====================================================

    def refresh(self):

        if self.current_item and self._refresh_handler:
            self._refresh_handler(self.current_item)

    # =====================================================

    def clear(self):

        self.current_item = None
        self._refresh_handler = None

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)

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

        self.start_x.clear()
        self.start_y.clear()
        self.end_x.clear()
        self.end_y.clear()

        self.exit_width.clear()
        self.capacity.clear()

        self.length.setText("-")

        self.blocked.blockSignals(True)
        self.blocked.setChecked(False)
        self.blocked.blockSignals(False)

    # =====================================================

    def rename_object(self):

        if self.current_item is None:
            return

        name = self.object_name.text().strip()

        if not name:
            return

        self.current_item.rename(name)

    # =====================================================

    def update_geometry(self):

        if self.current_item is None:
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

        self.current_item.setPos(
            x * self.GRID_SIZE,
            y * self.GRID_SIZE,
        )

        self.current_item.setRect(
            0,
            0,
            w * self.GRID_SIZE,
            h * self.GRID_SIZE,
        )

        self.refresh()

    # =====================================================

    def update_exit_geometry(self):

        if self.current_item is None:
            return

        try:

            x1 = float(self.start_x.text())
            y1 = float(self.start_y.text())

            x2 = float(self.end_x.text())
            y2 = float(self.end_y.text())

            w = float(self.exit_width.text())
            cap = int(self.capacity.text())

        except ValueError:

            self.refresh()

            return

        self.current_item.setPos(
            x1 * self.GRID_SIZE,
            y1 * self.GRID_SIZE,
        )

        self.current_item.setLine(
            0,
            0,
            (x2 - x1) * self.GRID_SIZE,
            (y2 - y1) * self.GRID_SIZE,
        )

        if self.current_item.model is not None:

            self.current_item.model.width = w
            self.current_item.model.capacity = cap

        self.refresh()

    # =====================================================

    def update_exit_blocked(self):

        if self.current_item is None:
            return

        if self.current_item.model is not None:

            self.current_item.model.is_blocked = (
                self.blocked.isChecked()
            )
