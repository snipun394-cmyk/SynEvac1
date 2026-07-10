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

from models.detector import Detector
from models.door import Door
from models.floor import Floor
from models.obstacle import Obstacle
from models.zone import Zone


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

        # Read-only, but copyable -- every object type (Stair
        # included) shows its id here via this one shared field,
        # so making it copyable here covers all of them at once
        # rather than duplicating the id+Copy-button row per type.
        self.object_id = QLineEdit("-")
        self.object_id.setReadOnly(True)

        self.copy_object_id_button = QPushButton("Copy")

        object_id_row = QWidget()
        object_id_layout = QHBoxLayout()
        object_id_layout.setContentsMargins(0, 0, 0, 0)
        object_id_layout.addWidget(self.object_id)
        object_id_layout.addWidget(self.copy_object_id_button)
        object_id_row.setLayout(object_id_layout)

        self.object_name = QLineEdit()

        layout.addRow("Object", self.object_type)
        layout.addRow("ID", object_id_row)
        layout.addRow("Name", self.object_name)

        # =====================================================
        # Zone Geometry
        # =====================================================

        self.origin_x = QLineEdit()
        self.origin_y = QLineEdit()

        self.zone_length = QLineEdit()
        self.zone_width = QLineEdit()

        self.zone_type = QComboBox()

        for zone_type in Zone.ZONE_TYPES:
            self.zone_type.addItem(zone_type)

        layout.addRow("Origin X (m)", self.origin_x)
        layout.addRow("Origin Y (m)", self.origin_y)

        layout.addRow("Length (m)", self.zone_length)
        layout.addRow("Width (m)", self.zone_width)

        layout.addRow("Zone Type", self.zone_type)

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
            self.zone_type,
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
        # Camera Geometry
        # =====================================================

        self.camera_x = QLineEdit()
        self.camera_y = QLineEdit()

        self.camera_rotation = QLineEdit()

        self.camera_fov = QLineEdit()
        self.camera_range = QLineEdit()
        self.camera_mount_height = QLineEdit()

        self.camera_active = QCheckBox()

        layout.addRow("Position X (m)", self.camera_x)
        layout.addRow("Position Y (m)", self.camera_y)

        layout.addRow("Rotation (deg)", self.camera_rotation)

        layout.addRow("Horizontal FOV (deg)", self.camera_fov)
        layout.addRow("Maximum Range (m)", self.camera_range)
        layout.addRow("Mount Height (m)", self.camera_mount_height)

        layout.addRow("Active", self.camera_active)

        self.camera_fields = [
            self.camera_x,
            self.camera_y,
            self.camera_rotation,
            self.camera_fov,
            self.camera_range,
            self.camera_mount_height,
            self.camera_active,
        ]

        # =====================================================
        # Detector Geometry
        # =====================================================

        self.detector_x = QLineEdit()
        self.detector_y = QLineEdit()

        self.detector_coverage_radius = QLineEdit()

        self.detector_mount_height = QLineEdit()

        self.detector_type = QComboBox()

        for detector_type in Detector.DETECTOR_TYPES:
            self.detector_type.addItem(detector_type)

        self.detector_active = QCheckBox()

        layout.addRow("Position X (m)", self.detector_x)
        layout.addRow("Position Y (m)", self.detector_y)

        layout.addRow("Coverage Radius (m)", self.detector_coverage_radius)

        layout.addRow("Mount Height (m)", self.detector_mount_height)

        layout.addRow("Detector Type", self.detector_type)

        layout.addRow("Active", self.detector_active)

        self.detector_fields = [
            self.detector_x,
            self.detector_y,
            self.detector_coverage_radius,
            self.detector_mount_height,
            self.detector_type,
            self.detector_active,
        ]

        # =====================================================
        # Assembly Point Geometry
        # =====================================================

        self.assembly_x = QLineEdit()
        self.assembly_y = QLineEdit()

        self.assembly_length = QLineEdit()
        self.assembly_width = QLineEdit()

        # Optional -- left blank means "unspecified", not zero
        # capacity; see update_assembly_point_geometry().
        self.assembly_capacity = QLineEdit()

        self.assembly_description = QLineEdit()

        self.assembly_active = QCheckBox()

        layout.addRow("Position X (m)", self.assembly_x)
        layout.addRow("Position Y (m)", self.assembly_y)

        layout.addRow("Length (m)", self.assembly_length)
        layout.addRow("Width (m)", self.assembly_width)

        layout.addRow("Capacity", self.assembly_capacity)

        layout.addRow("Description", self.assembly_description)

        layout.addRow("Active", self.assembly_active)

        self.assembly_fields = [
            self.assembly_x,
            self.assembly_y,
            self.assembly_length,
            self.assembly_width,
            self.assembly_capacity,
            self.assembly_description,
            self.assembly_active,
        ]

        # =====================================================
        # Obstacle Geometry
        # =====================================================

        self.obstacle_length = QLineEdit()
        self.obstacle_width = QLineEdit()

        self.obstacle_type = QComboBox()

        for obstacle_type in Obstacle.OBSTACLE_TYPES:
            self.obstacle_type.addItem(obstacle_type)

        self.obstacle_traversability = QComboBox()

        for traversability in Obstacle.TRAVERSABILITY_OPTIONS:
            self.obstacle_traversability.addItem(traversability)

        self.obstacle_traversal_cost = QLineEdit()

        self.obstacle_active = QCheckBox()

        layout.addRow("Length (m)", self.obstacle_length)
        layout.addRow("Width (m)", self.obstacle_width)

        layout.addRow("Type", self.obstacle_type)

        layout.addRow("Traversability", self.obstacle_traversability)

        layout.addRow("Traversal Cost", self.obstacle_traversal_cost)

        layout.addRow("Active", self.obstacle_active)

        self.obstacle_fields = [
            self.obstacle_length,
            self.obstacle_width,
            self.obstacle_type,
            self.obstacle_traversability,
            self.obstacle_traversal_cost,
            self.obstacle_active,
        ]

        # =====================================================
        # Door Geometry
        # =====================================================

        self.door_start_x = QLineEdit()
        self.door_start_y = QLineEdit()

        self.door_end_x = QLineEdit()
        self.door_end_y = QLineEdit()

        self.door_length = QLabel("-")

        self.door_width = QLineEdit()

        self.door_type = QComboBox()

        for door_type in Door.DOOR_TYPES:
            self.door_type.addItem(door_type)

        self.door_normally_open = QCheckBox()
        self.door_locked = QCheckBox()
        self.door_active = QCheckBox()

        # Connectivity -- the two Zones this Door joins. Zone
        # touching geometrically never implies connectivity;
        # only this explicit link does. Populated from whichever
        # floor the Door itself belongs to; see
        # _populate_zone_combo().
        self.door_zone_a = QComboBox()
        self.door_zone_b = QComboBox()

        layout.addRow("Start X (m)", self.door_start_x)
        layout.addRow("Start Y (m)", self.door_start_y)
        layout.addRow("End X (m)", self.door_end_x)
        layout.addRow("End Y (m)", self.door_end_y)

        layout.addRow("Length", self.door_length)

        layout.addRow("Door Width (m)", self.door_width)

        layout.addRow("Door Type", self.door_type)

        layout.addRow("Normally Open", self.door_normally_open)
        layout.addRow("Locked", self.door_locked)
        layout.addRow("Active", self.door_active)

        layout.addRow("Zone A", self.door_zone_a)
        layout.addRow("Zone B", self.door_zone_b)

        self.door_fields = [
            self.door_start_x,
            self.door_start_y,
            self.door_end_x,
            self.door_end_y,
            self.door_length,
            self.door_width,
            self.door_type,
            self.door_normally_open,
            self.door_locked,
            self.door_active,
            self.door_zone_a,
            self.door_zone_b,
        ]

        # =====================================================
        # Floor Properties
        #
        # Name reuses self.object_name (same as Zone/Exit/Stair)
        # rather than a dedicated field, so there is a single
        # "Name" row instead of two.
        # =====================================================

        # Derived -- see Building.floor_elevation(). Never
        # editable; always the cumulative height of every floor
        # below this one, so it can never drift out of sync with
        # height/order the way a freely-typed value could.
        self.floor_elevation = QLabel("-")

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

        self.zone_type.currentIndexChanged.connect(
            self.update_zone_type
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

        self.copy_object_id_button.clicked.connect(
            self.copy_object_id
        )

        self.camera_x.editingFinished.connect(
            self.update_camera_geometry
        )

        self.camera_y.editingFinished.connect(
            self.update_camera_geometry
        )

        self.camera_rotation.editingFinished.connect(
            self.update_camera_geometry
        )

        self.camera_fov.editingFinished.connect(
            self.update_camera_geometry
        )

        self.camera_range.editingFinished.connect(
            self.update_camera_geometry
        )

        self.camera_mount_height.editingFinished.connect(
            self.update_camera_geometry
        )

        self.camera_active.toggled.connect(
            self.update_camera_active
        )

        self.detector_x.editingFinished.connect(
            self.update_detector_geometry
        )

        self.detector_y.editingFinished.connect(
            self.update_detector_geometry
        )

        self.detector_coverage_radius.editingFinished.connect(
            self.update_detector_geometry
        )

        self.detector_mount_height.editingFinished.connect(
            self.update_detector_geometry
        )

        self.detector_type.currentIndexChanged.connect(
            self.update_detector_type
        )

        self.detector_active.toggled.connect(
            self.update_detector_active
        )

        self.assembly_x.editingFinished.connect(
            self.update_assembly_point_geometry
        )

        self.assembly_y.editingFinished.connect(
            self.update_assembly_point_geometry
        )

        self.assembly_length.editingFinished.connect(
            self.update_assembly_point_geometry
        )

        self.assembly_width.editingFinished.connect(
            self.update_assembly_point_geometry
        )

        self.assembly_capacity.editingFinished.connect(
            self.update_assembly_point_geometry
        )

        self.assembly_description.editingFinished.connect(
            self.update_assembly_point_geometry
        )

        self.assembly_active.toggled.connect(
            self.update_assembly_point_active
        )

        self.obstacle_length.editingFinished.connect(
            self.update_obstacle_geometry
        )

        self.obstacle_width.editingFinished.connect(
            self.update_obstacle_geometry
        )

        self.obstacle_traversal_cost.editingFinished.connect(
            self.update_obstacle_geometry
        )

        self.obstacle_type.currentIndexChanged.connect(
            self.update_obstacle_type
        )

        self.obstacle_traversability.currentIndexChanged.connect(
            self.update_obstacle_traversability
        )

        self.obstacle_active.toggled.connect(
            self.update_obstacle_active
        )

        self.door_start_x.editingFinished.connect(
            self.update_door_geometry
        )

        self.door_start_y.editingFinished.connect(
            self.update_door_geometry
        )

        self.door_end_x.editingFinished.connect(
            self.update_door_geometry
        )

        self.door_end_y.editingFinished.connect(
            self.update_door_geometry
        )

        self.door_width.editingFinished.connect(
            self.update_door_geometry
        )

        self.door_type.currentIndexChanged.connect(
            self.update_door_type
        )

        self.door_normally_open.toggled.connect(
            self.update_door_normally_open
        )

        self.door_locked.toggled.connect(
            self.update_door_locked
        )

        self.door_active.toggled.connect(
            self.update_door_active
        )

        self.door_zone_a.currentIndexChanged.connect(
            self.update_door_zone_a
        )

        self.door_zone_b.currentIndexChanged.connect(
            self.update_door_zone_b
        )

        self.floor_height.editingFinished.connect(
            self.update_floor_properties
        )

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
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
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        self.object_type.setText("Zone")
        self.object_id.setText(zone.zone_id)

        self.object_name.blockSignals(True)
        self.origin_x.blockSignals(True)
        self.origin_y.blockSignals(True)
        self.zone_length.blockSignals(True)
        self.zone_width.blockSignals(True)
        self.zone_type.blockSignals(True)

        self.object_name.setText(zone.zone_name)

        tlx, tly = zone.top_left
        trx, try_ = zone.top_right
        brx, bry = zone.bottom_right
        blx, bly = zone.bottom_left

        self.origin_x.setText(f"{tlx:.2f}")
        self.origin_y.setText(f"{tly:.2f}")

        self.zone_length.setText(f"{zone.width_m:.2f}")
        self.zone_width.setText(f"{zone.height_m:.2f}")

        if zone.model is not None:

            type_index = self.zone_type.findText(
                zone.model.zone_type
            )

            if type_index != -1:
                self.zone_type.setCurrentIndex(type_index)

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
        self.zone_type.blockSignals(False)

    # =====================================================
    # Exit
    # =====================================================

    def show_line(self, exit_item):

        self.current_item = exit_item
        self._refresh_handler = self.show_line

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, True)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
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
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
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
    # Camera
    # =====================================================

    def show_camera(self, camera_item):

        self.current_item = camera_item
        self._refresh_handler = self.show_camera

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, True)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = camera_item.model

        self.object_type.setText("Camera")
        self.object_id.setText(camera_item.object_id)

        self.object_name.blockSignals(True)
        self.camera_x.blockSignals(True)
        self.camera_y.blockSignals(True)
        self.camera_rotation.blockSignals(True)
        self.camera_fov.blockSignals(True)
        self.camera_range.blockSignals(True)
        self.camera_mount_height.blockSignals(True)
        self.camera_active.blockSignals(True)

        self.object_name.setText(camera_item.object_name)

        if model is not None:

            px, py = model.position

            self.camera_x.setText(f"{px:.2f}")
            self.camera_y.setText(f"{py:.2f}")

            self.camera_rotation.setText(
                f"{model.rotation:.1f}"
            )

            self.camera_fov.setText(
                f"{model.horizontal_fov:.1f}"
            )

            self.camera_range.setText(
                f"{model.max_range:.2f}"
            )

            self.camera_mount_height.setText(
                f"{model.mount_height:.2f}"
            )

            self.camera_active.setChecked(
                model.active
            )

        self.object_name.blockSignals(False)
        self.camera_x.blockSignals(False)
        self.camera_y.blockSignals(False)
        self.camera_rotation.blockSignals(False)
        self.camera_fov.blockSignals(False)
        self.camera_range.blockSignals(False)
        self.camera_mount_height.blockSignals(False)
        self.camera_active.blockSignals(False)

    # =====================================================
    # Detector
    # =====================================================

    def show_detector(self, detector_item):

        self.current_item = detector_item
        self._refresh_handler = self.show_detector

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, True)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = detector_item.model

        self.object_type.setText("Detector")
        self.object_id.setText(detector_item.object_id)

        self.object_name.blockSignals(True)
        self.detector_x.blockSignals(True)
        self.detector_y.blockSignals(True)
        self.detector_coverage_radius.blockSignals(True)
        self.detector_mount_height.blockSignals(True)
        self.detector_type.blockSignals(True)
        self.detector_active.blockSignals(True)

        self.object_name.setText(detector_item.object_name)

        if model is not None:

            px, py = model.position

            self.detector_x.setText(f"{px:.2f}")
            self.detector_y.setText(f"{py:.2f}")

            self.detector_coverage_radius.setText(
                f"{model.coverage_radius:.2f}"
            )

            self.detector_mount_height.setText(
                f"{model.mount_height:.2f}"
            )

            index = self.detector_type.findText(
                model.detector_type
            )

            if index != -1:
                self.detector_type.setCurrentIndex(index)

            self.detector_active.setChecked(
                model.active
            )

        self.object_name.blockSignals(False)
        self.detector_x.blockSignals(False)
        self.detector_y.blockSignals(False)
        self.detector_coverage_radius.blockSignals(False)
        self.detector_mount_height.blockSignals(False)
        self.detector_type.blockSignals(False)
        self.detector_active.blockSignals(False)

    # =====================================================
    # Assembly Point
    # =====================================================

    def show_assembly_point(self, assembly_point_item):

        self.current_item = assembly_point_item
        self._refresh_handler = self.show_assembly_point

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.assembly_fields, True)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = assembly_point_item.model

        self.object_type.setText("Assembly Point")
        self.object_id.setText(assembly_point_item.object_id)

        self.object_name.blockSignals(True)
        self.assembly_x.blockSignals(True)
        self.assembly_y.blockSignals(True)
        self.assembly_length.blockSignals(True)
        self.assembly_width.blockSignals(True)
        self.assembly_capacity.blockSignals(True)
        self.assembly_description.blockSignals(True)
        self.assembly_active.blockSignals(True)

        self.object_name.setText(assembly_point_item.object_name)

        if model is not None:

            px, py = model.position

            self.assembly_x.setText(f"{px:.2f}")
            self.assembly_y.setText(f"{py:.2f}")

            self.assembly_length.setText(
                f"{model.length:.2f}"
            )

            self.assembly_width.setText(
                f"{model.width:.2f}"
            )

            self.assembly_capacity.setText(
                str(model.capacity) if model.capacity else ""
            )

            self.assembly_description.setText(
                model.description
            )

            self.assembly_active.setChecked(
                model.active
            )

        self.object_name.blockSignals(False)
        self.assembly_x.blockSignals(False)
        self.assembly_y.blockSignals(False)
        self.assembly_length.blockSignals(False)
        self.assembly_width.blockSignals(False)
        self.assembly_capacity.blockSignals(False)
        self.assembly_description.blockSignals(False)
        self.assembly_active.blockSignals(False)

    # =====================================================
    # Obstacle
    # =====================================================

    def show_obstacle(self, obstacle_item):

        self.current_item = obstacle_item
        self._refresh_handler = self.show_obstacle

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, True)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = obstacle_item.model

        self.object_type.setText("Obstacle")
        self.object_id.setText(obstacle_item.object_id)

        self.object_name.blockSignals(True)
        self.obstacle_length.blockSignals(True)
        self.obstacle_width.blockSignals(True)
        self.obstacle_type.blockSignals(True)
        self.obstacle_traversability.blockSignals(True)
        self.obstacle_traversal_cost.blockSignals(True)
        self.obstacle_active.blockSignals(True)

        self.object_name.setText(obstacle_item.object_name)

        if model is not None:

            self.obstacle_length.setText(
                f"{model.length:.2f}"
            )

            self.obstacle_width.setText(
                f"{model.width:.2f}"
            )

            type_index = self.obstacle_type.findText(
                model.obstacle_type
            )

            if type_index != -1:
                self.obstacle_type.setCurrentIndex(type_index)

            traversability_index = self.obstacle_traversability.findText(
                model.traversability
            )

            if traversability_index != -1:
                self.obstacle_traversability.setCurrentIndex(
                    traversability_index
                )

            self.obstacle_traversal_cost.setText(
                f"{model.traversal_cost:.2f}"
            )

            self.obstacle_active.setChecked(
                model.active
            )

        self.object_name.blockSignals(False)
        self.obstacle_length.blockSignals(False)
        self.obstacle_width.blockSignals(False)
        self.obstacle_type.blockSignals(False)
        self.obstacle_traversability.blockSignals(False)
        self.obstacle_traversal_cost.blockSignals(False)
        self.obstacle_active.blockSignals(False)

    # =====================================================
    # Door
    # =====================================================

    def show_door(self, door_item):

        self.current_item = door_item
        self._refresh_handler = self.show_door

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, True)
        self._set_fields_visible(self.floor_fields, False)

        model = door_item.model

        self.object_type.setText("Door")
        self.object_id.setText(door_item.object_id)

        self.object_name.blockSignals(True)
        self.door_start_x.blockSignals(True)
        self.door_start_y.blockSignals(True)
        self.door_end_x.blockSignals(True)
        self.door_end_y.blockSignals(True)
        self.door_width.blockSignals(True)
        self.door_type.blockSignals(True)
        self.door_normally_open.blockSignals(True)
        self.door_locked.blockSignals(True)
        self.door_active.blockSignals(True)

        self.object_name.setText(door_item.object_name)

        if model is not None:

            sx, sy = model.start_point
            ex, ey = model.end_point

            self.door_start_x.setText(f"{sx:.2f}")
            self.door_start_y.setText(f"{sy:.2f}")
            self.door_end_x.setText(f"{ex:.2f}")
            self.door_end_y.setText(f"{ey:.2f}")

            self.door_length.setText(
                f"{model.length:.2f} m"
            )

            self.door_width.setText(
                f"{model.width:.2f}"
            )

            type_index = self.door_type.findText(
                model.door_type
            )

            if type_index != -1:
                self.door_type.setCurrentIndex(type_index)

            self.door_normally_open.setChecked(
                model.normally_open
            )

            self.door_locked.setChecked(
                model.locked
            )

            self.door_active.setChecked(
                model.active
            )

            self._populate_zone_combo(
                self.door_zone_a,
                model,
                model.zone_a_id,
                model.zone_b_id,
            )

            self._populate_zone_combo(
                self.door_zone_b,
                model,
                model.zone_b_id,
                model.zone_a_id,
            )

        self.object_name.blockSignals(False)
        self.door_start_x.blockSignals(False)
        self.door_start_y.blockSignals(False)
        self.door_end_x.blockSignals(False)
        self.door_end_y.blockSignals(False)
        self.door_width.blockSignals(False)
        self.door_type.blockSignals(False)
        self.door_normally_open.blockSignals(False)
        self.door_locked.blockSignals(False)
        self.door_active.blockSignals(False)

    # =====================================================
    # Floor
    # =====================================================

    def show_floor(self, floor):

        self.current_item = floor
        self._refresh_handler = self.show_floor

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, True)

        self.object_type.setText("Floor")
        self.object_id.setText(floor.id)

        self.object_name.blockSignals(True)
        self.floor_height.blockSignals(True)

        self.object_name.setText(floor.name)

        if self.building is not None:

            elevation = self.building.floor_elevation(floor)

            self.floor_elevation.setText(f"{elevation:.2f}")

        else:

            self.floor_elevation.setText("-")

        self.floor_height.setText(f"{floor.height:.2f}")

        self.object_name.blockSignals(False)
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
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
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

        self.zone_type.blockSignals(True)
        self.zone_type.setCurrentIndex(0)
        self.zone_type.blockSignals(False)

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

        self.camera_x.clear()
        self.camera_y.clear()

        self.camera_rotation.clear()

        self.camera_fov.clear()
        self.camera_range.clear()
        self.camera_mount_height.clear()

        self.camera_active.blockSignals(True)
        self.camera_active.setChecked(False)
        self.camera_active.blockSignals(False)

        self.detector_x.clear()
        self.detector_y.clear()

        self.detector_coverage_radius.clear()

        self.detector_mount_height.clear()

        self.detector_type.blockSignals(True)
        self.detector_type.setCurrentIndex(0)
        self.detector_type.blockSignals(False)

        self.detector_active.blockSignals(True)
        self.detector_active.setChecked(False)
        self.detector_active.blockSignals(False)

        self.assembly_x.clear()
        self.assembly_y.clear()

        self.assembly_length.clear()
        self.assembly_width.clear()
        self.assembly_capacity.clear()
        self.assembly_description.clear()

        self.assembly_active.blockSignals(True)
        self.assembly_active.setChecked(False)
        self.assembly_active.blockSignals(False)

        self.obstacle_length.clear()
        self.obstacle_width.clear()

        self.obstacle_type.blockSignals(True)
        self.obstacle_type.setCurrentIndex(0)
        self.obstacle_type.blockSignals(False)

        self.obstacle_traversability.blockSignals(True)
        self.obstacle_traversability.setCurrentIndex(0)
        self.obstacle_traversability.blockSignals(False)

        self.obstacle_traversal_cost.clear()

        self.obstacle_active.blockSignals(True)
        self.obstacle_active.setChecked(False)
        self.obstacle_active.blockSignals(False)

        self.door_start_x.clear()
        self.door_start_y.clear()
        self.door_end_x.clear()
        self.door_end_y.clear()

        self.door_length.setText("-")

        self.door_width.clear()

        self.door_type.blockSignals(True)
        self.door_type.setCurrentIndex(0)
        self.door_type.blockSignals(False)

        self.door_normally_open.blockSignals(True)
        self.door_normally_open.setChecked(False)
        self.door_normally_open.blockSignals(False)

        self.door_locked.blockSignals(True)
        self.door_locked.setChecked(False)
        self.door_locked.blockSignals(False)

        self.door_active.blockSignals(True)
        self.door_active.setChecked(False)
        self.door_active.blockSignals(False)

        self.door_zone_a.blockSignals(True)
        self.door_zone_a.clear()
        self.door_zone_a.blockSignals(False)

        self.door_zone_b.blockSignals(True)
        self.door_zone_b.clear()
        self.door_zone_b.blockSignals(False)

        self.floor_elevation.setText("-")
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

    def update_zone_type(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.zone_type = (
            self.zone_type.itemText(index)
        )

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

        # "None" is a valid, persistent selection -- a newly
        # placed Stair must never silently claim a destination
        # floor the user didn't choose. That auto-pick was exactly
        # what made Vertical Height look "connected" when it
        # wasn't, and is why users ended up creating a second,
        # unrelated Stair instead of intentionally linking this
        # one. Vertical Height/Travel Distance correctly read 0.0
        # until a destination is chosen deliberately.
        self.to_floor_combo.addItem("None", "")

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

        if index == -1:
            index = 0

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

    def copy_object_id(self):

        QApplication.clipboard().setText(
            self.object_id.text()
        )

    # =====================================================

    def update_camera_geometry(self):

        if self.current_item is None:
            return

        try:

            x = float(self.camera_x.text())
            y = float(self.camera_y.text())

            rotation = float(
                self.camera_rotation.text()
            )

            fov = float(self.camera_fov.text())
            max_range = float(self.camera_range.text())

            mount_height = float(
                self.camera_mount_height.text()
            )

        except ValueError:

            self.refresh()

            return

        self.current_item.setPos(
            x * self.GRID_SIZE,
            y * self.GRID_SIZE,
        )

        self.current_item.set_rotation_degrees(
            rotation
        )

        if self.current_item.model is not None:

            self.current_item.model.horizontal_fov = fov
            self.current_item.model.max_range = max_range
            self.current_item.model.mount_height = mount_height

        self.current_item.refresh_geometry()

        self.refresh()

    # =====================================================

    def update_camera_active(self):

        if self.current_item is None:
            return

        if self.current_item.model is not None:

            self.current_item.model.active = (
                self.camera_active.isChecked()
            )

        self.current_item.refresh_geometry()

    # =====================================================

    def update_detector_geometry(self):

        if self.current_item is None:
            return

        try:

            x = float(self.detector_x.text())
            y = float(self.detector_y.text())

            coverage_radius = float(
                self.detector_coverage_radius.text()
            )

            mount_height = float(
                self.detector_mount_height.text()
            )

        except ValueError:

            self.refresh()

            return

        self.current_item.setPos(
            x * self.GRID_SIZE,
            y * self.GRID_SIZE,
        )

        if self.current_item.model is not None:

            self.current_item.model.coverage_radius = coverage_radius
            self.current_item.model.mount_height = mount_height

        self.current_item.refresh_geometry()

        self.refresh()

    # =====================================================

    def update_detector_type(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.detector_type = (
            self.detector_type.itemText(index)
        )

        self.current_item.refresh_geometry()

    # =====================================================

    def update_detector_active(self):

        if self.current_item is None:
            return

        if self.current_item.model is not None:

            self.current_item.model.active = (
                self.detector_active.isChecked()
            )

        self.current_item.refresh_geometry()

    # =====================================================

    def update_assembly_point_geometry(self):

        if self.current_item is None:
            return

        try:

            x = float(self.assembly_x.text())
            y = float(self.assembly_y.text())

            length = float(self.assembly_length.text())
            width = float(self.assembly_width.text())

            capacity_text = self.assembly_capacity.text().strip()
            capacity = int(capacity_text) if capacity_text else 0

        except ValueError:

            self.refresh()

            return

        self.current_item.setPos(
            x * self.GRID_SIZE,
            y * self.GRID_SIZE,
        )

        self.current_item.set_size(length, width)

        if self.current_item.model is not None:

            self.current_item.model.capacity = capacity
            self.current_item.model.description = (
                self.assembly_description.text()
            )

        self.refresh()

    # =====================================================

    def update_assembly_point_active(self):

        if self.current_item is None:
            return

        if self.current_item.model is not None:

            self.current_item.model.active = (
                self.assembly_active.isChecked()
            )

        self.current_item.refresh_geometry()

    # =====================================================

    def update_obstacle_geometry(self):

        if self.current_item is None:
            return

        try:

            length = float(self.obstacle_length.text())
            width = float(self.obstacle_width.text())

            traversal_cost = float(
                self.obstacle_traversal_cost.text()
            )

        except ValueError:

            self.refresh()

            return

        self.current_item.setRect(
            0,
            0,
            length * self.GRID_SIZE,
            width * self.GRID_SIZE,
        )

        if self.current_item.model is not None:

            self.current_item.model.traversal_cost = traversal_cost

        self.refresh()

    # =====================================================

    def update_obstacle_type(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.obstacle_type = (
            self.obstacle_type.itemText(index)
        )

        self.current_item.refresh_geometry()

    # =====================================================

    def update_obstacle_traversability(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.traversability = (
            self.obstacle_traversability.itemText(index)
        )

        self.current_item.refresh_geometry()

    # =====================================================

    def update_obstacle_active(self):

        if self.current_item is None:
            return

        if self.current_item.model is not None:

            self.current_item.model.active = (
                self.obstacle_active.isChecked()
            )

        self.current_item.refresh_geometry()

    # =====================================================

    def update_door_geometry(self):

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

            self.current_item.model.width = width

        self.refresh()

    # =====================================================

    def update_door_type(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.door_type = (
            self.door_type.itemText(index)
        )

        self.current_item.refresh_geometry()

    # =====================================================

    def update_door_normally_open(self):

        if self.current_item is None:
            return

        if self.current_item.model is not None:

            self.current_item.model.normally_open = (
                self.door_normally_open.isChecked()
            )

    # =====================================================

    def update_door_locked(self):

        if self.current_item is None:
            return

        if self.current_item.model is not None:

            self.current_item.model.locked = (
                self.door_locked.isChecked()
            )

        self.current_item.refresh_geometry()

    # =====================================================

    def update_door_active(self):

        if self.current_item is None:
            return

        if self.current_item.model is not None:

            self.current_item.model.active = (
                self.door_active.isChecked()
            )

        self.current_item.refresh_geometry()

    # =====================================================
    # Populates a Zone-selection combo from whichever floor the
    # Door itself belongs to (never the whole Building, unlike
    # Stair's To Floor combo -- a Door only ever connects two
    # spaces on its own floor). "None" is a valid, persistent
    # selection here (unlike Stair, which always forces a default)
    # since a Door may be placed before its connections are made.
    # =====================================================

    def _populate_zone_combo(self, combo, model, current_zone_id, exclude_zone_id):

        combo.blockSignals(True)

        combo.clear()

        combo.addItem("None", "")

        if self.building is not None:

            floor = self.building.get_floor(model.floor_id)

            if floor is not None:

                for zone in floor.zones:

                    if zone.id == exclude_zone_id:
                        continue

                    combo.addItem(
                        zone.name,
                        zone.id,
                    )

        index = combo.findData(current_zone_id)

        if index == -1:
            index = 0

        combo.setCurrentIndex(index)

        combo.blockSignals(False)

    # =====================================================

    def update_door_zone_a(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        zone_id = self.door_zone_a.itemData(index)

        self.current_item.model.zone_a_id = zone_id or ""

        self.refresh()

    # =====================================================

    def update_door_zone_b(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        zone_id = self.door_zone_b.itemData(index)

        self.current_item.model.zone_b_id = zone_id or ""

        self.refresh()

    # =====================================================

    def update_floor_properties(self):

        if self.current_item is None:
            return

        try:

            height = float(
                self.floor_height.text()
            )

        except ValueError:

            self.refresh()

            return

        self.current_item.height = height

        self.refresh()

        if self.floor_updated_callback:
            self.floor_updated_callback(self.current_item)
