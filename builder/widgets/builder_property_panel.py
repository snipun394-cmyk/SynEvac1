from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QWidget,
)

from models import connectable_space
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.obstacle import Obstacle
from models.sensor_asset import HealthStatus
from models.speaker import Speaker
from models.staircase import Staircase
from models.zone import Zone


class BuilderPropertyPanel(QWidget):

    # A new, Builder-only property editor -- deliberately NOT a reuse
    # of designer.widgets.property_panel.PropertyPanel (7800+ lines,
    # and imports camera_calibration at module level -- a live-camera/
    # Perception dependency the milestone brief explicitly forbids
    # Builder from carrying, even dormant). This panel covers exactly
    # the asset palette the brief lists under "Building Authoring"
    # (Zones, Doors, Exits, Stairs, Cameras, Smoke Detectors, Heat
    # Detectors, Speakers, Obstacles) plus Floor's own Name/Height/
    # Scale, following the SAME field-grouping/blockSignals/
    # editingFinished conventions Studio's PropertyPanel already
    # established (see that file), so behaviour reads identically to
    # an engineer already familiar with Studio -- only the dependency
    # footprint and the set of supported types differ.

    GRID_SIZE = 50

    def __init__(self):
        super().__init__()

        self.current_item = None
        self._refresh_handler = None

        self.building = None

        # Fired after a Floor's Name/Height is edited here, so
        # BuilderMainWindow can refresh FloorList/ProjectTree (neither
        # of which is reachable directly from this widget) -- same
        # convention as Studio's PropertyPanel.floor_updated_callback.
        self.floor_updated_callback = None

        # Fired after any edit that could change validation/summary/
        # navigation-preview state -- BuilderMainWindow re-runs those
        # panels from here rather than this widget reaching for them
        # directly.
        self.item_changed_callback = None

        layout = QFormLayout()
        self.setLayout(layout)

        self.object_type = QLabel("-")
        self.object_id = QLabel("-")
        self.object_name = QLineEdit()
        self.object_name.editingFinished.connect(self.rename_object)

        layout.addRow("Type", self.object_type)
        layout.addRow("ID", self.object_id)
        layout.addRow("Name", self.object_name)

        self._build_zone_fields(layout)
        self._build_door_fields(layout)
        self._build_exit_fields(layout)
        self._build_stair_fields(layout)
        self._build_camera_fields(layout)
        self._build_smoke_detector_fields(layout)
        self._build_heat_detector_fields(layout)
        self._build_speaker_fields(layout)
        self._build_obstacle_fields(layout)
        self._build_floor_fields(layout)

        self.clear()

    # =====================================================
    # Binding
    # =====================================================

    def set_building(self, building):

        self.building = building

    # =====================================================

    def refresh(self):

        if self._refresh_handler and self.current_item is not None:
            self._refresh_handler(self.current_item)

    # =====================================================

    def clear(self):

        self.current_item = None
        self._refresh_handler = None

        self.object_type.setText("-")
        self.object_id.setText("-")
        self.object_name.clear()

        for fields in self._all_field_groups():
            self._set_fields_visible(fields, False)

    # =====================================================

    def _all_field_groups(self):

        return (
            self.zone_fields,
            self.door_fields,
            self.exit_fields,
            self.stair_fields,
            self.camera_fields,
            self.smoke_detector_fields,
            self.heat_detector_fields,
            self.speaker_fields,
            self.obstacle_fields,
            self.floor_fields,
        )

    # =====================================================

    def _set_fields_visible(self, fields, visible):

        for field in fields:

            field.setVisible(visible)

            label = self.layout().labelForField(field)

            if label:
                label.setVisible(visible)

    # =====================================================

    def _show_only(self, fields):

        for group in self._all_field_groups():
            self._set_fields_visible(group, group is fields)

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

        self._notify_changed()

    # =====================================================

    def _notify_changed(self):

        if self.item_changed_callback:
            self.item_changed_callback()

    # =====================================================
    # Shared zone-combo/checklist helpers -- reused across every
    # asset type that assigns itself to a floor's connectable spaces
    # (Zone, or an Assembly Point carried over from a Studio-authored
    # project). models.connectable_space is confirmed dependency-clean
    # (Layer 0 -- see the feasibility investigation), so it is reused
    # directly rather than re-implemented.
    # =====================================================

    def _populate_zone_combo(self, combo, floor_id, current_id, exclude_id=""):

        combo.blockSignals(True)

        combo.clear()
        combo.addItem("None", "")

        floor = self.building.get_floor(floor_id) if self.building is not None else None

        if floor is not None:

            for space_type, space in connectable_space.all_connectable_spaces(floor):

                if space.id == exclude_id:
                    continue

                combo.addItem(
                    connectable_space.label_for(space_type, space.name),
                    space.id,
                )

        index = combo.findData(current_id)

        combo.setCurrentIndex(index if index != -1 else 0)

        combo.blockSignals(False)

    # =====================================================

    def _populate_zone_checklist(self, list_widget, floor_id, current_ids):

        list_widget.blockSignals(True)

        list_widget.clear()

        floor = self.building.get_floor(floor_id) if self.building is not None else None

        if floor is not None:

            for space_type, space in connectable_space.all_connectable_spaces(floor):

                item = QListWidgetItem(
                    connectable_space.label_for(space_type, space.name)
                )

                item.setData(Qt.ItemDataRole.UserRole, space.id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

                item.setCheckState(
                    Qt.CheckState.Checked
                    if space.id in current_ids
                    else Qt.CheckState.Unchecked
                )

                list_widget.addItem(item)

        list_widget.blockSignals(False)

    # =====================================================

    def _checked_ids(self, list_widget):

        return tuple(
            list_widget.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(list_widget.count())
            if list_widget.item(row).checkState() == Qt.CheckState.Checked
        )

    # =====================================================
    # Zone
    # =====================================================

    def _build_zone_fields(self, layout):

        self.origin_x = QLineEdit()
        self.origin_y = QLineEdit()
        self.zone_length = QLineEdit()
        self.zone_width = QLineEdit()
        self.zone_type = QComboBox()
        self.zone_type.addItems(Zone.ZONE_TYPES)
        self.zone_max_occupancy = QLineEdit()
        self.zone_area = QLabel("-")

        for field in (self.origin_x, self.origin_y, self.zone_length, self.zone_width):
            field.editingFinished.connect(self._update_zone_geometry)

        self.zone_type.currentIndexChanged.connect(self._update_zone_type)
        self.zone_max_occupancy.editingFinished.connect(self._update_zone_max_occupancy)

        self.zone_fields = (
            self.origin_x, self.origin_y, self.zone_length, self.zone_width,
            self.zone_type, self.zone_max_occupancy, self.zone_area,
        )

        layout.addRow("Origin X (m)", self.origin_x)
        layout.addRow("Origin Y (m)", self.origin_y)
        layout.addRow("Length (m)", self.zone_length)
        layout.addRow("Width (m)", self.zone_width)
        layout.addRow("Zone Type", self.zone_type)
        layout.addRow("Max Occupancy", self.zone_max_occupancy)
        layout.addRow("Area", self.zone_area)

    # =====================================================

    def show_zone(self, zone_item):

        self.current_item = zone_item
        self._refresh_handler = self.show_zone

        self._show_only(self.zone_fields)

        self.object_type.setText("Zone")
        self.object_id.setText(zone_item.zone_id)

        self.object_name.blockSignals(True)
        self.origin_x.blockSignals(True)
        self.origin_y.blockSignals(True)
        self.zone_length.blockSignals(True)
        self.zone_width.blockSignals(True)
        self.zone_type.blockSignals(True)
        self.zone_max_occupancy.blockSignals(True)

        self.object_name.setText(zone_item.zone_name)

        tlx, tly = zone_item.top_left

        self.origin_x.setText(f"{tlx:.2f}")
        self.origin_y.setText(f"{tly:.2f}")
        self.zone_length.setText(f"{zone_item.width_m:.2f}")
        self.zone_width.setText(f"{zone_item.height_m:.2f}")
        self.zone_area.setText(f"{zone_item.area_m2:.2f} m²")

        if zone_item.model is not None:

            type_index = self.zone_type.findText(zone_item.model.zone_type)

            if type_index != -1:
                self.zone_type.setCurrentIndex(type_index)

            self.zone_max_occupancy.setText(str(zone_item.model.max_occupancy))

        self.object_name.blockSignals(False)
        self.origin_x.blockSignals(False)
        self.origin_y.blockSignals(False)
        self.zone_length.blockSignals(False)
        self.zone_width.blockSignals(False)
        self.zone_type.blockSignals(False)
        self.zone_max_occupancy.blockSignals(False)

    # =====================================================

    def _update_zone_geometry(self):

        if self.current_item is None:
            return

        try:

            x = float(self.origin_x.text())
            y = float(self.origin_y.text())
            w = float(self.zone_length.text())
            h = float(self.zone_width.text())

        except ValueError:

            self.refresh()

            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.setRect(0, 0, w * self.GRID_SIZE, h * self.GRID_SIZE)

        self.refresh()
        self._notify_changed()

    # =====================================================

    def _update_zone_type(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.zone_type = self.zone_type.itemText(index)

        self._notify_changed()

    # =====================================================

    def _update_zone_max_occupancy(self):

        if self.current_item is None or self.current_item.model is None:
            return

        try:
            value = int(self.zone_max_occupancy.text())
        except ValueError:
            self.refresh()
            return

        self.current_item.model.max_occupancy = value

        self._notify_changed()

    # =====================================================
    # Door
    # =====================================================

    def _build_door_fields(self, layout):

        self.door_start_x = QLineEdit()
        self.door_start_y = QLineEdit()
        self.door_end_x = QLineEdit()
        self.door_end_y = QLineEdit()
        self.door_width = QLineEdit()
        self.door_type = QComboBox()
        self.door_type.addItems(Door.DOOR_TYPES)
        self.door_normally_open = QCheckBox()
        self.door_locked = QCheckBox()
        self.door_active = QCheckBox()
        self.door_zone_a = QComboBox()
        self.door_zone_b = QComboBox()

        for field in (self.door_start_x, self.door_start_y, self.door_end_x, self.door_end_y, self.door_width):
            field.editingFinished.connect(self._update_door_geometry)

        self.door_type.currentIndexChanged.connect(self._update_door_type)
        self.door_normally_open.toggled.connect(self._update_door_flags)
        self.door_locked.toggled.connect(self._update_door_flags)
        self.door_active.toggled.connect(self._update_door_flags)
        self.door_zone_a.currentIndexChanged.connect(self._update_door_zones)
        self.door_zone_b.currentIndexChanged.connect(self._update_door_zones)

        self.door_fields = (
            self.door_start_x, self.door_start_y, self.door_end_x, self.door_end_y,
            self.door_width, self.door_type, self.door_normally_open, self.door_locked,
            self.door_active, self.door_zone_a, self.door_zone_b,
        )

        layout.addRow("Start X (m)", self.door_start_x)
        layout.addRow("Start Y (m)", self.door_start_y)
        layout.addRow("End X (m)", self.door_end_x)
        layout.addRow("End Y (m)", self.door_end_y)
        layout.addRow("Width (m)", self.door_width)
        layout.addRow("Door Type", self.door_type)
        layout.addRow("Normally Open", self.door_normally_open)
        layout.addRow("Locked", self.door_locked)
        layout.addRow("Active", self.door_active)
        layout.addRow("Zone A", self.door_zone_a)
        layout.addRow("Zone B", self.door_zone_b)

    # =====================================================

    def show_door(self, door_item):

        self.current_item = door_item
        self._refresh_handler = self.show_door

        self._show_only(self.door_fields)

        model = door_item.model

        self.object_type.setText("Door")
        self.object_id.setText(door_item.object_id)

        for field in (
            self.object_name, self.door_start_x, self.door_start_y, self.door_end_x,
            self.door_end_y, self.door_width, self.door_type, self.door_normally_open,
            self.door_locked, self.door_active, self.door_zone_a, self.door_zone_b,
        ):
            field.blockSignals(True)

        self.object_name.setText(door_item.object_name)

        if model is not None:

            sx, sy = model.start_point
            ex, ey = model.end_point

            self.door_start_x.setText(f"{sx:.2f}")
            self.door_start_y.setText(f"{sy:.2f}")
            self.door_end_x.setText(f"{ex:.2f}")
            self.door_end_y.setText(f"{ey:.2f}")
            self.door_width.setText(f"{model.width:.2f}")

            type_index = self.door_type.findText(model.door_type)

            if type_index != -1:
                self.door_type.setCurrentIndex(type_index)

            self.door_normally_open.setChecked(model.normally_open)
            self.door_locked.setChecked(model.locked)
            self.door_active.setChecked(model.active)

            self._populate_zone_combo(self.door_zone_a, model.floor_id, model.zone_a_id, model.zone_b_id)
            self._populate_zone_combo(self.door_zone_b, model.floor_id, model.zone_b_id, model.zone_a_id)

        for field in (
            self.object_name, self.door_start_x, self.door_start_y, self.door_end_x,
            self.door_end_y, self.door_width, self.door_type, self.door_normally_open,
            self.door_locked, self.door_active, self.door_zone_a, self.door_zone_b,
        ):
            field.blockSignals(False)

    # =====================================================

    def _update_door_geometry(self):

        if self.current_item is None:
            return

        try:

            x1 = float(self.door_start_x.text())
            y1 = float(self.door_start_y.text())
            x2 = float(self.door_end_x.text())
            y2 = float(self.door_end_y.text())
            width = float(self.door_width.text())

        except ValueError:

            self.refresh()

            return

        self.current_item.setPos(x1 * self.GRID_SIZE, y1 * self.GRID_SIZE)

        self.current_item.setLine(
            0, 0, (x2 - x1) * self.GRID_SIZE, (y2 - y1) * self.GRID_SIZE,
        )

        if self.current_item.model is not None:
            self.current_item.model.width = width

        self.refresh()
        self._notify_changed()

    # =====================================================

    def _update_door_type(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.door_type = self.door_type.itemText(index)

        self._notify_changed()

    # =====================================================

    def _update_door_flags(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.normally_open = self.door_normally_open.isChecked()
        self.current_item.model.locked = self.door_locked.isChecked()
        self.current_item.model.active = self.door_active.isChecked()

        self.current_item.refresh_geometry()

        self._notify_changed()

    # =====================================================

    def _update_door_zones(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.zone_a_id = self.door_zone_a.currentData()
        self.current_item.model.zone_b_id = self.door_zone_b.currentData()

        self._notify_changed()

    # =====================================================
    # Exit
    # =====================================================

    def _build_exit_fields(self, layout):

        self.exit_start_x = QLineEdit()
        self.exit_start_y = QLineEdit()
        self.exit_end_x = QLineEdit()
        self.exit_end_y = QLineEdit()
        self.exit_width = QLineEdit()
        self.exit_capacity = QLineEdit()
        self.exit_blocked = QCheckBox()
        self.exit_zone = QComboBox()

        for field in (self.exit_start_x, self.exit_start_y, self.exit_end_x, self.exit_end_y, self.exit_width):
            field.editingFinished.connect(self._update_exit_geometry)

        self.exit_capacity.editingFinished.connect(self._update_exit_capacity)
        self.exit_blocked.toggled.connect(self._update_exit_flags)
        self.exit_zone.currentIndexChanged.connect(self._update_exit_zone)

        self.exit_fields = (
            self.exit_start_x, self.exit_start_y, self.exit_end_x, self.exit_end_y,
            self.exit_width, self.exit_capacity, self.exit_blocked, self.exit_zone,
        )

        layout.addRow("Start X (m)", self.exit_start_x)
        layout.addRow("Start Y (m)", self.exit_start_y)
        layout.addRow("End X (m)", self.exit_end_x)
        layout.addRow("End Y (m)", self.exit_end_y)
        layout.addRow("Width (m)", self.exit_width)
        layout.addRow("Capacity", self.exit_capacity)
        layout.addRow("Blocked", self.exit_blocked)
        layout.addRow("Zone", self.exit_zone)

    # =====================================================

    def show_exit(self, exit_item):

        self.current_item = exit_item
        self._refresh_handler = self.show_exit

        self._show_only(self.exit_fields)

        model = exit_item.model

        self.object_type.setText("Exit")
        self.object_id.setText(exit_item.object_id)

        for field in (
            self.object_name, self.exit_start_x, self.exit_start_y, self.exit_end_x,
            self.exit_end_y, self.exit_width, self.exit_capacity, self.exit_blocked, self.exit_zone,
        ):
            field.blockSignals(True)

        self.object_name.setText(exit_item.object_name)

        if model is not None:

            sx, sy = model.start_point
            ex, ey = model.end_point

            self.exit_start_x.setText(f"{sx:.2f}")
            self.exit_start_y.setText(f"{sy:.2f}")
            self.exit_end_x.setText(f"{ex:.2f}")
            self.exit_end_y.setText(f"{ey:.2f}")
            self.exit_width.setText(f"{model.width:.2f}")
            self.exit_capacity.setText(str(model.capacity))
            self.exit_blocked.setChecked(model.is_blocked)

            self._populate_zone_combo(self.exit_zone, model.floor_id, model.zone_id)

        for field in (
            self.object_name, self.exit_start_x, self.exit_start_y, self.exit_end_x,
            self.exit_end_y, self.exit_width, self.exit_capacity, self.exit_blocked, self.exit_zone,
        ):
            field.blockSignals(False)

    # =====================================================

    def _update_exit_geometry(self):

        if self.current_item is None:
            return

        try:

            x1 = float(self.exit_start_x.text())
            y1 = float(self.exit_start_y.text())
            x2 = float(self.exit_end_x.text())
            y2 = float(self.exit_end_y.text())
            width = float(self.exit_width.text())

        except ValueError:

            self.refresh()

            return

        self.current_item.setPos(x1 * self.GRID_SIZE, y1 * self.GRID_SIZE)

        self.current_item.setLine(
            0, 0, (x2 - x1) * self.GRID_SIZE, (y2 - y1) * self.GRID_SIZE,
        )

        if self.current_item.model is not None:
            self.current_item.model.width = width

        self.refresh()
        self._notify_changed()

    # =====================================================

    def _update_exit_capacity(self):

        if self.current_item is None or self.current_item.model is None:
            return

        try:
            value = int(self.exit_capacity.text())
        except ValueError:
            self.refresh()
            return

        self.current_item.model.capacity = value

        self._notify_changed()

    # =====================================================

    def _update_exit_flags(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.is_blocked = self.exit_blocked.isChecked()

        self.current_item.refresh_geometry()

        self._notify_changed()

    # =====================================================

    def _update_exit_zone(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.zone_id = self.exit_zone.currentData()

        self._notify_changed()

    # =====================================================
    # Stair
    #
    # Position (from_position/to_position) is deliberately read-only
    # here -- a Staircase is one shared object rendered as two markers
    # on two different floors (see designer/items/stair_item.py's own
    # docstring); repositioning either marker already works via drag
    # (ItemIsMovable), and the Stair Tool's own guided placement flow
    # (entrance click -> destination floor -> landing click, mediated
    # by BuilderMainWindow exactly as Studio's MainWindow already does)
    # is how a new Stair's geometry and destination floor are set.
    # =====================================================

    def _build_stair_fields(self, layout):

        self.stair_from_floor = QLabel("-")
        self.stair_from_position = QLabel("-")
        self.stair_to_floor = QComboBox()
        self.stair_to_position = QLabel("-")
        self.stair_width = QLineEdit()
        self.stair_from_zone = QComboBox()
        self.stair_to_zone = QComboBox()
        self.stair_vertical_height = QLabel("-")
        self.stair_travel_distance = QLabel("-")

        self.stair_width.editingFinished.connect(self._update_stair_width)
        self.stair_from_zone.currentIndexChanged.connect(self._update_stair_zones)
        self.stair_to_zone.currentIndexChanged.connect(self._update_stair_zones)
        self.stair_to_floor.currentIndexChanged.connect(self._update_stair_destination_floor)

        self.stair_fields = (
            self.stair_from_floor, self.stair_from_position, self.stair_to_floor,
            self.stair_to_position, self.stair_width, self.stair_from_zone,
            self.stair_to_zone, self.stair_vertical_height, self.stair_travel_distance,
        )

        layout.addRow("From Floor", self.stair_from_floor)
        layout.addRow("From Position (m)", self.stair_from_position)
        layout.addRow("To Floor", self.stair_to_floor)
        layout.addRow("To Position (m)", self.stair_to_position)
        layout.addRow("Width (m)", self.stair_width)
        layout.addRow("From Zone", self.stair_from_zone)
        layout.addRow("To Zone", self.stair_to_zone)
        layout.addRow("Vertical Height", self.stair_vertical_height)
        layout.addRow("Travel Distance", self.stair_travel_distance)

    # =====================================================

    def show_stair(self, stair_item):

        self.current_item = stair_item
        self._refresh_handler = self.show_stair

        self._show_only(self.stair_fields)

        model = stair_item.model

        self.object_type.setText("Stair")
        self.object_id.setText(stair_item.object_id)

        for field in (
            self.object_name, self.stair_width, self.stair_to_floor,
            self.stair_from_zone, self.stair_to_zone,
        ):
            field.blockSignals(True)

        self.object_name.setText(stair_item.object_name)

        if model is not None:

            fx, fy = model.from_position
            tx, ty = model.to_position

            self.stair_from_position.setText(f"({fx:.2f}, {fy:.2f})")
            self.stair_to_position.setText(f"({tx:.2f}, {ty:.2f})")
            self.stair_width.setText(f"{model.width:.2f}")

            if self.building is not None:

                from_floor = self.building.get_floor(model.from_floor_id)

                self.stair_from_floor.setText(from_floor.name if from_floor is not None else "-")

                self._populate_to_floor_combo(model)

            self._populate_zone_combo(self.stair_from_zone, model.from_floor_id, model.from_zone_id)
            self._populate_zone_combo(self.stair_to_zone, model.to_floor_id, model.to_zone_id)

            if self.building is not None:

                self.stair_vertical_height.setText(f"{model.vertical_height(self.building):.2f} m")
                self.stair_travel_distance.setText(f"{model.travel_distance(self.building):.2f} m")

            else:

                self.stair_vertical_height.setText("-")
                self.stair_travel_distance.setText("-")

        for field in (
            self.object_name, self.stair_width, self.stair_to_floor,
            self.stair_from_zone, self.stair_to_zone,
        ):
            field.blockSignals(False)

    # =====================================================

    def _populate_to_floor_combo(self, model):

        self.stair_to_floor.blockSignals(True)

        self.stair_to_floor.clear()

        if self.building is not None:

            for floor in self.building.ordered_floors():

                if floor.id == model.from_floor_id:
                    continue

                self.stair_to_floor.addItem(floor.name, floor.id)

        index = self.stair_to_floor.findData(model.to_floor_id)

        self.stair_to_floor.setCurrentIndex(index if index != -1 else -1)

        self.stair_to_floor.blockSignals(False)

    # =====================================================

    def _update_stair_destination_floor(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        model = self.current_item.model

        new_floor_id = self.stair_to_floor.itemData(index)

        if not new_floor_id or new_floor_id == model.to_floor_id:
            return

        model.to_floor_id = new_floor_id
        model.to_zone_id = ""

        self.refresh()
        self._notify_changed()

    # =====================================================

    def _update_stair_width(self):

        if self.current_item is None or self.current_item.model is None:
            return

        try:
            value = float(self.stair_width.text())
        except ValueError:
            self.refresh()
            return

        self.current_item.model.width = value

        self._notify_changed()

    # =====================================================

    def _update_stair_zones(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.from_zone_id = self.stair_from_zone.currentData()
        self.current_item.model.to_zone_id = self.stair_to_zone.currentData()

        self._notify_changed()

    # =====================================================
    # Camera
    # =====================================================

    def _build_camera_fields(self, layout):

        self.camera_x = QLineEdit()
        self.camera_y = QLineEdit()
        self.camera_rotation = QLineEdit()
        self.camera_fov = QLineEdit()
        self.camera_range = QLineEdit()
        self.camera_mount_height = QLineEdit()
        self.camera_active = QCheckBox()
        self.camera_zone = QComboBox()

        for field in (self.camera_x, self.camera_y):
            field.editingFinished.connect(self._update_camera_position)

        for field in (self.camera_rotation, self.camera_fov, self.camera_range, self.camera_mount_height):
            field.editingFinished.connect(self._update_camera_geometry)

        self.camera_active.toggled.connect(self._update_camera_flags)
        self.camera_zone.currentIndexChanged.connect(self._update_camera_zone)

        self.camera_fields = (
            self.camera_x, self.camera_y, self.camera_rotation, self.camera_fov,
            self.camera_range, self.camera_mount_height, self.camera_active, self.camera_zone,
        )

        layout.addRow("X (m)", self.camera_x)
        layout.addRow("Y (m)", self.camera_y)
        layout.addRow("Rotation (deg)", self.camera_rotation)
        layout.addRow("Horizontal FOV (deg)", self.camera_fov)
        layout.addRow("Max Range (m)", self.camera_range)
        layout.addRow("Mount Height (m)", self.camera_mount_height)
        layout.addRow("Active", self.camera_active)
        layout.addRow("Zone", self.camera_zone)

    # =====================================================

    def show_camera(self, camera_item):

        self.current_item = camera_item
        self._refresh_handler = self.show_camera

        self._show_only(self.camera_fields)

        model = camera_item.model

        self.object_type.setText("Camera")
        self.object_id.setText(camera_item.object_id)

        for field in (
            self.object_name, self.camera_x, self.camera_y, self.camera_rotation,
            self.camera_fov, self.camera_range, self.camera_mount_height,
            self.camera_active, self.camera_zone,
        ):
            field.blockSignals(True)

        self.object_name.setText(camera_item.object_name)

        if model is not None:

            px, py = model.position

            self.camera_x.setText(f"{px:.2f}")
            self.camera_y.setText(f"{py:.2f}")
            self.camera_rotation.setText(f"{model.rotation:.1f}")
            self.camera_fov.setText(f"{model.horizontal_fov:.1f}")
            self.camera_range.setText(f"{model.max_range:.2f}")
            self.camera_mount_height.setText(f"{model.mount_height:.2f}")
            self.camera_active.setChecked(model.active)

            self._populate_zone_combo(
                self.camera_zone, model.floor_id, model.zone_ids[0] if model.zone_ids else "",
            )

        for field in (
            self.object_name, self.camera_x, self.camera_y, self.camera_rotation,
            self.camera_fov, self.camera_range, self.camera_mount_height,
            self.camera_active, self.camera_zone,
        ):
            field.blockSignals(False)

    # =====================================================

    def _update_camera_position(self):

        if self.current_item is None:
            return

        try:
            x = float(self.camera_x.text())
            y = float(self.camera_y.text())
        except ValueError:
            self.refresh()
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)

        self.refresh()
        self._notify_changed()

    # =====================================================

    def _update_camera_geometry(self):

        if self.current_item is None or self.current_item.model is None:
            return

        try:

            rotation = float(self.camera_rotation.text())
            fov = float(self.camera_fov.text())
            max_range = float(self.camera_range.text())
            mount_height = float(self.camera_mount_height.text())

        except ValueError:

            self.refresh()

            return

        self.current_item.set_rotation_degrees(rotation)

        self.current_item.model.horizontal_fov = fov
        self.current_item.model.max_range = max_range
        self.current_item.model.mount_height = mount_height

        self.current_item.refresh_geometry()

        self._notify_changed()

    # =====================================================

    def _update_camera_flags(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.camera_active.isChecked()

        self.current_item.refresh_geometry()

        self._notify_changed()

    # =====================================================

    def _update_camera_zone(self):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.camera_zone.currentData()

        self.current_item.model.zone_ids = (zone_id,) if zone_id else ()

        self._notify_changed()

    # =====================================================
    # Smoke Detector / Heat Detector -- SensorAsset-based, identical
    # shape (see models/sensor_asset.py). Two near-identical sections
    # rather than one parametrized one, matching Studio's own
    # PropertyPanel convention of a dedicated show_*/field set per
    # concrete type.
    # =====================================================

    def _build_smoke_detector_fields(self, layout):

        self.smoke_x = QLineEdit()
        self.smoke_y = QLineEdit()
        self.smoke_active = QCheckBox()
        self.smoke_health = QComboBox()
        self.smoke_health.addItems(HealthStatus.ALL)
        self.smoke_threshold = QLineEdit()
        self.smoke_installation_date = QLineEdit()
        self.smoke_zone = QComboBox()

        for field in (self.smoke_x, self.smoke_y):
            field.editingFinished.connect(self._update_smoke_detector_position)

        self.smoke_active.toggled.connect(self._update_smoke_detector_fields)
        self.smoke_health.currentIndexChanged.connect(self._update_smoke_detector_fields)
        self.smoke_threshold.editingFinished.connect(self._update_smoke_detector_fields)
        self.smoke_installation_date.editingFinished.connect(self._update_smoke_detector_fields)
        self.smoke_zone.currentIndexChanged.connect(self._update_smoke_detector_zone)

        self.smoke_detector_fields = (
            self.smoke_x, self.smoke_y, self.smoke_active, self.smoke_health,
            self.smoke_threshold, self.smoke_installation_date, self.smoke_zone,
        )

        layout.addRow("X (m)", self.smoke_x)
        layout.addRow("Y (m)", self.smoke_y)
        layout.addRow("Active", self.smoke_active)
        layout.addRow("Health Status", self.smoke_health)
        layout.addRow("Activation Threshold", self.smoke_threshold)
        layout.addRow("Installation Date", self.smoke_installation_date)
        layout.addRow("Zone", self.smoke_zone)

    # =====================================================

    def show_smoke_detector(self, item):

        self.current_item = item
        self._refresh_handler = self.show_smoke_detector

        self._show_only(self.smoke_detector_fields)

        self._show_sensor(item, "Smoke Detector", self.smoke_x, self.smoke_y, self.smoke_active,
                           self.smoke_health, self.smoke_threshold, self.smoke_installation_date,
                           self.smoke_zone)

    # =====================================================

    def _update_smoke_detector_position(self):

        self._update_sensor_position(self.smoke_x, self.smoke_y)

    def _update_smoke_detector_fields(self):

        self._update_sensor_fields(self.smoke_active, self.smoke_health, self.smoke_threshold,
                                    self.smoke_installation_date)

    def _update_smoke_detector_zone(self):

        self._update_sensor_zone(self.smoke_zone)

    # =====================================================

    def _build_heat_detector_fields(self, layout):

        self.heat_x = QLineEdit()
        self.heat_y = QLineEdit()
        self.heat_active = QCheckBox()
        self.heat_health = QComboBox()
        self.heat_health.addItems(HealthStatus.ALL)
        self.heat_threshold = QLineEdit()
        self.heat_installation_date = QLineEdit()
        self.heat_zone = QComboBox()

        for field in (self.heat_x, self.heat_y):
            field.editingFinished.connect(self._update_heat_detector_position)

        self.heat_active.toggled.connect(self._update_heat_detector_fields)
        self.heat_health.currentIndexChanged.connect(self._update_heat_detector_fields)
        self.heat_threshold.editingFinished.connect(self._update_heat_detector_fields)
        self.heat_installation_date.editingFinished.connect(self._update_heat_detector_fields)
        self.heat_zone.currentIndexChanged.connect(self._update_heat_detector_zone)

        self.heat_detector_fields = (
            self.heat_x, self.heat_y, self.heat_active, self.heat_health,
            self.heat_threshold, self.heat_installation_date, self.heat_zone,
        )

        layout.addRow("X (m)", self.heat_x)
        layout.addRow("Y (m)", self.heat_y)
        layout.addRow("Active", self.heat_active)
        layout.addRow("Health Status", self.heat_health)
        layout.addRow("Activation Threshold", self.heat_threshold)
        layout.addRow("Installation Date", self.heat_installation_date)
        layout.addRow("Zone", self.heat_zone)

    # =====================================================

    def show_heat_detector(self, item):

        self.current_item = item
        self._refresh_handler = self.show_heat_detector

        self._show_only(self.heat_detector_fields)

        self._show_sensor(item, "Heat Detector", self.heat_x, self.heat_y, self.heat_active,
                           self.heat_health, self.heat_threshold, self.heat_installation_date,
                           self.heat_zone)

    # =====================================================

    def _update_heat_detector_position(self):

        self._update_sensor_position(self.heat_x, self.heat_y)

    def _update_heat_detector_fields(self):

        self._update_sensor_fields(self.heat_active, self.heat_health, self.heat_threshold,
                                    self.heat_installation_date)

    def _update_heat_detector_zone(self):

        self._update_sensor_zone(self.heat_zone)

    # =====================================================
    # Shared SensorAsset (Smoke/Heat Detector) plumbing
    # =====================================================

    def _show_sensor(self, item, label, x_field, y_field, active_field, health_field,
                      threshold_field, installation_field, zone_field):

        model = item.model

        self.object_type.setText(label)
        self.object_id.setText(item.object_id)

        fields = (self.object_name, x_field, y_field, active_field, health_field,
                  threshold_field, installation_field, zone_field)

        for field in fields:
            field.blockSignals(True)

        self.object_name.setText(item.object_name)

        if model is not None:

            px, py = model.position

            x_field.setText(f"{px:.2f}")
            y_field.setText(f"{py:.2f}")
            active_field.setChecked(model.active)

            health_index = health_field.findText(model.health_status)

            if health_index != -1:
                health_field.setCurrentIndex(health_index)

            threshold_field.setText(f"{model.activation_threshold:.2f}")
            installation_field.setText(model.installation_date)

            self._populate_zone_combo(
                zone_field, model.floor_id, model.zone_ids[0] if model.zone_ids else "",
            )

        for field in fields:
            field.blockSignals(False)

    # =====================================================

    def _update_sensor_position(self, x_field, y_field):

        if self.current_item is None:
            return

        try:
            x = float(x_field.text())
            y = float(y_field.text())
        except ValueError:
            self.refresh()
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)

        self.refresh()
        self._notify_changed()

    # =====================================================

    def _update_sensor_fields(self, active_field, health_field, threshold_field, installation_field):

        if self.current_item is None or self.current_item.model is None:
            return

        model = self.current_item.model

        try:
            threshold = float(threshold_field.text())
        except ValueError:
            self.refresh()
            return

        model.active = active_field.isChecked()
        model.health_status = health_field.currentText()
        model.activation_threshold = threshold
        model.installation_date = installation_field.text()

        self.current_item.refresh_geometry()

        self._notify_changed()

    # =====================================================

    def _update_sensor_zone(self, zone_field):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = zone_field.currentData()

        self.current_item.model.zone_ids = (zone_id,) if zone_id else ()

        self._notify_changed()

    # =====================================================
    # Speaker -- SensorAsset-based, but zone_ids is a MULTI-select
    # checklist rather than a single combo (a real voice-evacuation
    # speaker routinely serves several zones at once; a Smoke/Heat
    # Detector's own single-zone assignment is a different, narrower
    # need). Same convention Studio's PropertyPanel already uses.
    # =====================================================

    def _build_speaker_fields(self, layout):

        self.speaker_x = QLineEdit()
        self.speaker_y = QLineEdit()
        self.speaker_active = QCheckBox()
        self.speaker_type = QComboBox()
        self.speaker_type.addItems(Speaker.SPEAKER_TYPES)
        self.speaker_volume = QLineEdit()
        self.speaker_installation_date = QLineEdit()
        self.speaker_zones = QListWidget()
        self.speaker_zones.setMaximumHeight(120)

        for field in (self.speaker_x, self.speaker_y):
            field.editingFinished.connect(self._update_speaker_position)

        self.speaker_active.toggled.connect(self._update_speaker_fields)
        self.speaker_type.currentIndexChanged.connect(self._update_speaker_fields)
        self.speaker_volume.editingFinished.connect(self._update_speaker_fields)
        self.speaker_installation_date.editingFinished.connect(self._update_speaker_fields)
        self.speaker_zones.itemChanged.connect(self._update_speaker_zones)

        self.speaker_fields = (
            self.speaker_x, self.speaker_y, self.speaker_active, self.speaker_type,
            self.speaker_volume, self.speaker_installation_date, self.speaker_zones,
        )

        layout.addRow("X (m)", self.speaker_x)
        layout.addRow("Y (m)", self.speaker_y)
        layout.addRow("Active", self.speaker_active)
        layout.addRow("Speaker Type", self.speaker_type)
        layout.addRow("Volume Level (dB)", self.speaker_volume)
        layout.addRow("Installation Date", self.speaker_installation_date)
        layout.addRow("Zones", self.speaker_zones)

    # =====================================================

    def show_speaker(self, speaker_item):

        self.current_item = speaker_item
        self._refresh_handler = self.show_speaker

        self._show_only(self.speaker_fields)

        model = speaker_item.model

        self.object_type.setText("Speaker")
        self.object_id.setText(speaker_item.object_id)

        fields = (
            self.object_name, self.speaker_x, self.speaker_y, self.speaker_active,
            self.speaker_type, self.speaker_volume, self.speaker_installation_date,
            self.speaker_zones,
        )

        for field in fields:
            field.blockSignals(True)

        self.object_name.setText(speaker_item.object_name)

        if model is not None:

            px, py = model.position

            self.speaker_x.setText(f"{px:.2f}")
            self.speaker_y.setText(f"{py:.2f}")
            self.speaker_active.setChecked(model.active)

            type_index = self.speaker_type.findText(model.speaker_type)

            if type_index != -1:
                self.speaker_type.setCurrentIndex(type_index)

            self.speaker_volume.setText(f"{model.volume_level:.1f}")
            self.speaker_installation_date.setText(model.installation_date)

            self._populate_zone_checklist(self.speaker_zones, model.floor_id, model.zone_ids)

        for field in fields:
            field.blockSignals(False)

    # =====================================================

    def _update_speaker_position(self):

        if self.current_item is None:
            return

        try:
            x = float(self.speaker_x.text())
            y = float(self.speaker_y.text())
        except ValueError:
            self.refresh()
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)

        self.refresh()
        self._notify_changed()

    # =====================================================

    def _update_speaker_fields(self):

        if self.current_item is None or self.current_item.model is None:
            return

        model = self.current_item.model

        try:
            volume = float(self.speaker_volume.text())
        except ValueError:
            self.refresh()
            return

        model.active = self.speaker_active.isChecked()
        model.speaker_type = self.speaker_type.currentText()
        model.volume_level = volume
        model.installation_date = self.speaker_installation_date.text()

        self.current_item.refresh_geometry()

        self._notify_changed()

    # =====================================================

    def _update_speaker_zones(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.zone_ids = self._checked_ids(self.speaker_zones)

        self._notify_changed()

    # =====================================================
    # Obstacle -- position is deliberately NOT an editable field here
    # (see models/obstacle.py's own comment: "placement/repositioning
    # happens via the Tool and Move", matching Studio's own scope for
    # this asset), only Length/Width/Type/Traversability/Traversal
    # Cost/Active.
    # =====================================================

    def _build_obstacle_fields(self, layout):

        self.obstacle_length = QLineEdit()
        self.obstacle_width = QLineEdit()
        self.obstacle_type = QComboBox()
        self.obstacle_type.addItems(Obstacle.OBSTACLE_TYPES)
        self.obstacle_traversability = QComboBox()
        self.obstacle_traversability.addItems(Obstacle.TRAVERSABILITY_OPTIONS)
        self.obstacle_traversal_cost = QLineEdit()
        self.obstacle_active = QCheckBox()

        for field in (self.obstacle_length, self.obstacle_width):
            field.editingFinished.connect(self._update_obstacle_geometry)

        self.obstacle_type.currentIndexChanged.connect(self._update_obstacle_fields)
        self.obstacle_traversability.currentIndexChanged.connect(self._update_obstacle_fields)
        self.obstacle_traversal_cost.editingFinished.connect(self._update_obstacle_fields)
        self.obstacle_active.toggled.connect(self._update_obstacle_fields)

        self.obstacle_fields = (
            self.obstacle_length, self.obstacle_width, self.obstacle_type,
            self.obstacle_traversability, self.obstacle_traversal_cost, self.obstacle_active,
        )

        layout.addRow("Length (m)", self.obstacle_length)
        layout.addRow("Width (m)", self.obstacle_width)
        layout.addRow("Obstacle Type", self.obstacle_type)
        layout.addRow("Traversability", self.obstacle_traversability)
        layout.addRow("Traversal Cost", self.obstacle_traversal_cost)
        layout.addRow("Active", self.obstacle_active)

    # =====================================================

    def show_obstacle(self, obstacle_item):

        self.current_item = obstacle_item
        self._refresh_handler = self.show_obstacle

        self._show_only(self.obstacle_fields)

        model = obstacle_item.model

        self.object_type.setText("Obstacle")
        self.object_id.setText(obstacle_item.object_id)

        fields = (
            self.object_name, self.obstacle_length, self.obstacle_width, self.obstacle_type,
            self.obstacle_traversability, self.obstacle_traversal_cost, self.obstacle_active,
        )

        for field in fields:
            field.blockSignals(True)

        self.object_name.setText(obstacle_item.object_name)

        if model is not None:

            self.obstacle_length.setText(f"{model.length:.2f}")
            self.obstacle_width.setText(f"{model.width:.2f}")

            type_index = self.obstacle_type.findText(model.obstacle_type)

            if type_index != -1:
                self.obstacle_type.setCurrentIndex(type_index)

            traversability_index = self.obstacle_traversability.findText(model.traversability)

            if traversability_index != -1:
                self.obstacle_traversability.setCurrentIndex(traversability_index)

            self.obstacle_traversal_cost.setText(f"{model.traversal_cost:.2f}")
            self.obstacle_active.setChecked(model.active)

        for field in fields:
            field.blockSignals(False)

    # =====================================================

    def _update_obstacle_geometry(self):

        if self.current_item is None:
            return

        try:
            length = float(self.obstacle_length.text())
            width = float(self.obstacle_width.text())
        except ValueError:
            self.refresh()
            return

        self.current_item.setRect(0, 0, length * self.GRID_SIZE, width * self.GRID_SIZE)

        self.refresh()
        self._notify_changed()

    # =====================================================

    def _update_obstacle_fields(self):

        if self.current_item is None or self.current_item.model is None:
            return

        model = self.current_item.model

        try:
            traversal_cost = float(self.obstacle_traversal_cost.text())
        except ValueError:
            self.refresh()
            return

        model.obstacle_type = self.obstacle_type.currentText()
        model.traversability = self.obstacle_traversability.currentText()
        model.traversal_cost = traversal_cost
        model.active = self.obstacle_active.isChecked()

        self.current_item.refresh_geometry()

        self._notify_changed()

    # =====================================================
    # Floor
    # =====================================================

    def _build_floor_fields(self, layout):

        self.floor_height = QLineEdit()
        self.floor_scale = QLabel("Not Calibrated")

        self.floor_height.editingFinished.connect(self._update_floor_height)

        self.floor_fields = (
            self.floor_height, self.floor_scale,
        )

        layout.addRow("Height (m)", self.floor_height)
        layout.addRow("Scale", self.floor_scale)

    # =====================================================

    def show_floor(self, floor):

        self.current_item = floor
        self._refresh_handler = self.show_floor

        self._show_only(self.floor_fields)

        self.object_type.setText("Floor")
        self.object_id.setText(floor.id)

        self.object_name.blockSignals(True)
        self.floor_height.blockSignals(True)

        self.object_name.setText(floor.name)
        self.floor_height.setText(f"{floor.height:.2f}")

        if floor.is_scale_calibrated:
            self.floor_scale.setText(f"{floor.floor_plan_scale:.2f} px = 1 m")
        else:
            self.floor_scale.setText("Not Calibrated")

        self.object_name.blockSignals(False)
        self.floor_height.blockSignals(False)

    # =====================================================

    def _update_floor_height(self):

        if self.current_item is None or not isinstance(self.current_item, Floor):
            return

        try:
            value = float(self.floor_height.text())
        except ValueError:
            self.refresh()
            return

        self.current_item.height = value

        if self.floor_updated_callback:
            self.floor_updated_callback(self.current_item)

        self._notify_changed()
