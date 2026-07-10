from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QComboBox,
    QFormLayout,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)

from models.floor import Floor


class PropertyPanel(QWidget):

    GRID_SIZE = 50

    def __init__(self):
        super().__init__()

        self.current_item = None
        self._refresh_handler = None

        # Needed to resolve Stair from_floor_id/to_floor_id into
        # actual Floor elevations for the derived traversal fields.
        self.building = None

        # Fired after a Floor's Name/Elevation/Height is edited here,
        # so MainWindow can refresh FloorList/ProjectTree (neither of
        # which is reachable directly from this widget).
        self.floor_updated_callback = None

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

        self.zone_length = QLineEdit()
        self.zone_width = QLineEdit()

        layout.addRow("Origin X (m)", self.origin_x)
        layout.addRow("Origin Y (m)", self.origin_y)

        layout.addRow("Length (m)", self.zone_length)
        layout.addRow("Width (m)", self.zone_width)

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
            self.zone_length,
            self.zone_width,
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

        # =====================================================
        # Stair Geometry
        # =====================================================

        self.stair_start_x = QLineEdit()
        self.stair_start_y = QLineEdit()

        self.stair_end_x = QLineEdit()
        self.stair_end_y = QLineEdit()

        self.stair_width = QLineEdit()

        self.stair_length = QLabel("-")

        # Destination floor -- names are shown, but the combo
        # stores each floor's UUID as itemData so the model
        # keeps storing an id, never a display name.
        self.to_floor_combo = QComboBox()

        # Read-only/debug only. The combo above is authoritative;
        # this just exposes the raw UUID for copy/paste.
        self.to_floor_uuid = QLineEdit()
        self.to_floor_uuid.setReadOnly(True)

        self.copy_to_floor_button = QPushButton("Copy")

        to_floor_id_row = QWidget()
        to_floor_id_layout = QHBoxLayout()
        to_floor_id_layout.setContentsMargins(0, 0, 0, 0)
        to_floor_id_layout.addWidget(self.to_floor_uuid)
        to_floor_id_layout.addWidget(self.copy_to_floor_button)
        to_floor_id_row.setLayout(to_floor_id_layout)

        # Derived (Building/Floor elevations) -- never editable.
        self.vertical_height = QLabel("-")
        self.travel_distance = QLabel("-")

        layout.addRow("Start X (m)", self.stair_start_x)
        layout.addRow("Start Y (m)", self.stair_start_y)
        layout.addRow("End X (m)", self.stair_end_x)
        layout.addRow("End Y (m)", self.stair_end_y)

        layout.addRow("Length", self.stair_length)

        layout.addRow("Stair Width (m)", self.stair_width)

        layout.addRow("To Floor", self.to_floor_combo)
        layout.addRow("To Floor ID", to_floor_id_row)

        layout.addRow(
            "Vertical Height (m)",
            self.vertical_height,
        )

        layout.addRow(
            "Travel Distance (m)",
            self.travel_distance,
        )

        self.stair_fields = [
            self.stair_start_x,
            self.stair_start_y,
            self.stair_end_x,
            self.stair_end_y,
            self.stair_length,
            self.stair_width,
            self.to_floor_combo,
            to_floor_id_row,
            self.vertical_height,
            self.travel_distance,
        ]

        # =====================================================
        # Floor Properties
        #
        # Name reuses self.object_name (same as Zone/Exit/Stair)
        # rather than a dedicated field, so there is a single
        # "Name" row instead of two.
        # =====================================================

        self.floor_elevation = QLineEdit()
        self.floor_height = QLineEdit()

        layout.addRow("Elevation (m)", self.floor_elevation)
        layout.addRow("Floor Height (m)", self.floor_height)

        self.floor_fields = [
            self.floor_elevation,
            self.floor_height,
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

        self.zone_length.editingFinished.connect(
            self.update_geometry
        )

        self.zone_width.editingFinished.connect(
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

        self.stair_start_x.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_start_y.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_end_x.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_end_y.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_width.editingFinished.connect(
            self.update_stair_geometry
        )

        self.to_floor_combo.currentIndexChanged.connect(
            self.update_stair_to_floor
        )

        self.copy_to_floor_button.clicked.connect(
            self.copy_to_floor_id
        )

        self.floor_elevation.editingFinished.connect(
            self.update_floor_properties
        )

        self.floor_height.editingFinished.connect(
            self.update_floor_properties
        )

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.floor_fields, False)

    # =====================================================

    def set_building(self, building):

        self.building = building

        self.refresh()

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
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        self.object_type.setText("Zone")
        self.object_id.setText(zone.zone_id)

        self.object_name.blockSignals(True)
        self.origin_x.blockSignals(True)
        self.origin_y.blockSignals(True)
        self.zone_length.blockSignals(True)
        self.zone_width.blockSignals(True)

        self.object_name.setText(zone.zone_name)

        tlx, tly = zone.top_left
        trx, try_ = zone.top_right
        brx, bry = zone.bottom_right
        blx, bly = zone.bottom_left

        self.origin_x.setText(f"{tlx:.2f}")
        self.origin_y.setText(f"{tly:.2f}")

        self.zone_length.setText(f"{zone.width_m:.2f}")
        self.zone_width.setText(f"{zone.height_m:.2f}")

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
        self.zone_length.blockSignals(False)
        self.zone_width.blockSignals(False)

    # =====================================================
    # Exit
    # =====================================================

    def show_line(self, exit_item):

        self.current_item = exit_item
        self._refresh_handler = self.show_line

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, True)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.floor_fields, False)

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
    # Stair
    # =====================================================

    def show_stair(self, stair_item):

        self.current_item = stair_item
        self._refresh_handler = self.show_stair

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, True)
        self._set_fields_visible(self.floor_fields, False)

        model = stair_item.model

        self.object_type.setText("Stair")
        self.object_id.setText(stair_item.object_id)

        self.object_name.blockSignals(True)
        self.stair_start_x.blockSignals(True)
        self.stair_start_y.blockSignals(True)
        self.stair_end_x.blockSignals(True)
        self.stair_end_y.blockSignals(True)
        self.stair_width.blockSignals(True)
        self.to_floor_combo.blockSignals(True)

        self.object_name.setText(stair_item.object_name)

        if model is not None:

            sx, sy = model.start_point
            ex, ey = model.end_point

            self.stair_start_x.setText(f"{sx:.2f}")
            self.stair_start_y.setText(f"{sy:.2f}")
            self.stair_end_x.setText(f"{ex:.2f}")
            self.stair_end_y.setText(f"{ey:.2f}")

            self.stair_length.setText(
                f"{model.length:.2f} m"
            )

            self.stair_width.setText(
                f"{model.width:.2f}"
            )

            self._populate_to_floor_combo(model)

            if self.building is not None:

                height = model.vertical_height(
                    self.building
                )

                distance = model.travel_distance(
                    self.building
                )

                self.vertical_height.setText(
                    f"{height:.2f} m"
                )

                self.travel_distance.setText(
                    f"{distance:.2f} m"
                )

            else:

                self.vertical_height.setText("-")
                self.travel_distance.setText("-")

        self.object_name.blockSignals(False)
        self.stair_start_x.blockSignals(False)
        self.stair_start_y.blockSignals(False)
        self.stair_end_x.blockSignals(False)
        self.stair_end_y.blockSignals(False)
        self.stair_width.blockSignals(False)
        self.to_floor_combo.blockSignals(False)

    # =====================================================
    # Floor
    # =====================================================

    def show_floor(self, floor):

        self.current_item = floor
        self._refresh_handler = self.show_floor

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.floor_fields, True)

        self.object_type.setText("Floor")
        self.object_id.setText(floor.id)

        self.object_name.blockSignals(True)
        self.floor_elevation.blockSignals(True)
        self.floor_height.blockSignals(True)

        self.object_name.setText(floor.name)
        self.floor_elevation.setText(f"{floor.elevation:.2f}")
        self.floor_height.setText(f"{floor.height:.2f}")

        self.object_name.blockSignals(False)
        self.floor_elevation.blockSignals(False)
        self.floor_height.blockSignals(False)

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
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        self.object_type.setText(
            "No Selection"
        )

        self.object_id.setText("-")

        self.object_name.clear()

        self.origin_x.clear()
        self.origin_y.clear()

        self.zone_length.clear()
        self.zone_width.clear()

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

        self.stair_start_x.clear()
        self.stair_start_y.clear()
        self.stair_end_x.clear()
        self.stair_end_y.clear()

        self.stair_width.clear()
        self.stair_length.setText("-")

        self.to_floor_combo.blockSignals(True)
        self.to_floor_combo.clear()
        self.to_floor_combo.blockSignals(False)

        self.to_floor_uuid.clear()

        self.vertical_height.setText("-")
        self.travel_distance.setText("-")

        self.floor_elevation.clear()
        self.floor_height.clear()

    # =====================================================

    def rename_object(self):

        if self.current_item is None:
            return

        name = self.object_name.text().strip()

        if not name:
            return

        self.current_item.rename(name)

        if isinstance(self.current_item, Floor):

            if self.floor_updated_callback:
                self.floor_updated_callback(self.current_item)

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
                self.zone_length.text()
            )

            h = float(
                self.zone_width.text()
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

    # =====================================================

    def update_stair_geometry(self):

        if self.current_item is None:
            return

        try:

            x1 = float(self.stair_start_x.text())
            y1 = float(self.stair_start_y.text())

            x2 = float(self.stair_end_x.text())
            y2 = float(self.stair_end_y.text())

            w = float(self.stair_width.text())

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

        self.refresh()

    # =====================================================

    def _populate_to_floor_combo(self, model):

        self.to_floor_combo.blockSignals(True)

        self.to_floor_combo.clear()

        if self.building is not None:

            for floor in self.building.ordered_floors():

                if floor.id == model.from_floor_id:
                    continue

                self.to_floor_combo.addItem(
                    floor.name,
                    floor.id,
                )

        index = self.to_floor_combo.findData(
            model.to_floor_id
        )

        if index == -1 and self.to_floor_combo.count() > 0:

            # No destination floor chosen yet (or it was
            # deleted) -- default to the first available
            # floor rather than leaving Vertical Height/Travel
            # Distance uncomputable.
            index = 0

            model.to_floor_id = (
                self.to_floor_combo.itemData(index)
            )

        self.to_floor_combo.setCurrentIndex(index)

        self.to_floor_combo.blockSignals(False)

        self.to_floor_uuid.setText(model.to_floor_id)

    # =====================================================

    def update_stair_to_floor(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        floor_id = self.to_floor_combo.itemData(index)

        if floor_id is None:
            return

        self.current_item.model.to_floor_id = floor_id

        self.refresh()

    # =====================================================

    def copy_to_floor_id(self):

        QApplication.clipboard().setText(
            self.to_floor_uuid.text()
        )

    # =====================================================

    def update_floor_properties(self):

        if self.current_item is None:
            return

        try:

            elevation = float(
                self.floor_elevation.text()
            )

            height = float(
                self.floor_height.text()
            )

        except ValueError:

            self.refresh()

            return

        self.current_item.elevation = elevation
        self.current_item.height = height

        self.refresh()

        if self.floor_updated_callback:
            self.floor_updated_callback(self.current_item)
