from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from camera_calibration.calibration import CalibrationRegistry
from camera_calibration.calibration_loader import (
    CalibrationLoadError, calibration_from_camera, load_calibration_json, save_calibration_json,
)
from camera_calibration.camera_model import calibration_status_text

from models import connectable_space
from models.detector import Detector
from models.door import Door
from models.emergency_light import EmergencyLight, EmergencyLightAvailability
from models.engineering_asset import DeviceMode
from models.fire_extinguisher import FireExtinguisher
from models.fire_hydrant import FireHydrant
from models.fire_pump import FirePump
from models.fire_service_inlet import FireServiceInlet
from models.fire_water_system import MEMBERSHIP_FIELDS, assign_asset_to_system, system_containing_asset
from models.fire_water_tank import FireWaterTank
from models.floor import Floor
from models.hose_reel import HoseReel
from models.jockey_pump import JockeyPump
from models.obstacle import Obstacle
from models.pump_asset import PumpControlMode
from models.sensor_asset import HealthStatus
from models.speaker import Speaker
from models.staircase import StairObservableRegion
from models.zone import Zone

from navigation.node import Node

from sandbox.occupant import SandboxDestinationType


class PropertyPanel(QWidget):

    GRID_SIZE = 50

    def __init__(self):
        super().__init__()

        self.current_item = None
        self._refresh_handler = None

        # Set once by MainWindow -- notified whenever an edit made
        # through this panel could change what's rendered beyond the
        # panel/item itself (today: Camera geometry/active edits,
        # which the Camera Coverage overlay -- designer/scene/
        # graphics_scene.py::refresh_camera_coverage() -- needs to
        # react to). None is a valid, common state (no such listener
        # registered) and every call site guards for it.
        self.on_visual_change = None

        # Needed to resolve Stair from_floor_id/to_floor_id into
        # actual Floor elevations for the derived traversal fields,
        # and (for Occupant) to resolve node ids into readable Zone/
        # Assembly Point names and to recompute routes.
        self.building = None

        # Manual Simulation Sandbox -- needed to recompute an
        # Occupant's route when its destination type changes here.
        # Never used for anything else this panel does.
        self.sandbox_manager = None

        # Real Camera Calibration & World-Coordinate Validation
        # milestone, Phase 13 -- the minimum useful calibration
        # visibility this milestone asks for: a per-camera_id
        # CalibrationRegistry, exactly the same production type
        # live_runtime/camera_calibration already establish, owned here
        # in-memory for THIS Designer session only. Deliberately NOT
        # yet persisted with the project file (no calibration field
        # exists in models.camera.Camera or the project serialization
        # format -- adding one is a genuine future extension, not done
        # here to avoid touching project file format/versioning as part
        # of this milestone). A calibration loaded or saved here via
        # the "Calibrate Camera..." dialog is lost when the Designer
        # session ends unless the operator separately saved it to a
        # calibration JSON file on disk (scripts/calibrate_camera_scene.py
        # produces exactly that file shape).
        self.calibration_registry = CalibrationRegistry()

        # Fired after a Floor's Name/Elevation/Height is edited here,
        # so MainWindow can refresh FloorList/ProjectTree (neither of
        # which is reachable directly from this widget).
        self.floor_updated_callback = None

        # Fired after an Occupant's destination (and therefore route)
        # is recomputed here, so MainWindow can refresh the on-canvas
        # route highlight (not reachable directly from this widget).
        self.occupant_route_changed_callback = None

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

        # Connectivity -- the one Zone this Exit leads out of. Same
        # convention as Door's Zone A/Zone B combos below: never
        # inferred from geometry, only this explicit link. Previously
        # missing entirely from this panel -- an Exit's zone_id had no
        # way to be set through the Designer UI at all, which is what
        # silently left NavigationGraphGenerator with no Zone to
        # attach an Exit edge to even after zone_id looked "assigned"
        # in the user's intent.
        self.exit_zone = QComboBox()

        layout.addRow("Start X (m)", self.start_x)
        layout.addRow("Start Y (m)", self.start_y)
        layout.addRow("End X (m)", self.end_x)
        layout.addRow("End Y (m)", self.end_y)

        layout.addRow("Length", self.length)

        layout.addRow("Exit Width (m)", self.exit_width)
        layout.addRow("Capacity", self.capacity)

        layout.addRow("Blocked", self.blocked)

        layout.addRow("Zone", self.exit_zone)

        self.exit_fields = [
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
            self.length,
            self.exit_width,
            self.capacity,
            self.blocked,
            self.exit_zone,
        ]

        # =====================================================
        # Stair Geometry
        #
        # A Staircase is one shared object with a position on
        # EACH of the two floors it connects -- both are always
        # shown and editable here regardless of which marker
        # (entrance or landing) is actually selected, since
        # selecting either one edits the same engineering object.
        # Editing the position that belongs to the floor NOT
        # currently displayed still writes straight onto the
        # model; only the marker actually on screen gets its
        # on-canvas position moved (see update_stair_geometry()).
        # =====================================================

        # Read-only -- From Floor is fixed at placement time by
        # the guided Stair Tool workflow, not editable after the
        # fact from here.
        self.stair_from_floor = QLabel("-")

        self.stair_from_x = QLineEdit()
        self.stair_from_y = QLineEdit()

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

        self.stair_to_x = QLineEdit()
        self.stair_to_y = QLineEdit()

        self.stair_width = QLineEdit()

        # Derived (Building/Floor elevations) -- never editable.
        self.vertical_height = QLabel("-")
        self.travel_distance = QLabel("-")

        # Connectivity -- the Zone at each end this Staircase actually
        # opens into. Same explicit-reference convention as Door's own
        # Zone A/Zone B combos below (never inferred from geometry,
        # only this explicit link) -- previously missing entirely from
        # this panel, which is exactly what let a Stair look "placed"
        # on both floors while still producing no Navigation Graph
        # edge at all.
        self.stair_from_zone = QComboBox()
        self.stair_to_zone = QComboBox()

        # =====================================================
        # Observable Stair Perception milestone -- the smallest Designer
        # authoring support for the OPTIONAL, per-floor-side observable
        # region a calibrated stair camera needs (models.staircase.
        # StairObservableRegion). Deliberately reuses the existing Stair
        # workflow's own From/To split rather than a new tool: an axis-
        # aligned rectangle CENTERED on that side's already-placed
        # from_position/to_position anchor, with an explicit width/depth
        # the operator types in. Left blank (either field empty or <= 0),
        # the region stays None -- STAIR LOCALIZATION UNAVAILABLE for
        # that side, never a fabricated default (the audit's own Phase 2
        # requirement). Existing Stair placement/geometry fields above
        # are completely unaffected -- purely additive.
        # =====================================================

        self.stair_from_region_width = QLineEdit()
        self.stair_from_region_depth = QLineEdit()

        self.stair_to_region_width = QLineEdit()
        self.stair_to_region_depth = QLineEdit()

        layout.addRow("From Floor", self.stair_from_floor)
        layout.addRow("From Position X (m)", self.stair_from_x)
        layout.addRow("From Position Y (m)", self.stair_from_y)
        layout.addRow("From Zone", self.stair_from_zone)
        layout.addRow("From Observable Region Width (m)", self.stair_from_region_width)
        layout.addRow("From Observable Region Depth (m)", self.stair_from_region_depth)

        layout.addRow("To Floor", self.to_floor_combo)
        layout.addRow("To Floor ID", to_floor_id_row)
        layout.addRow("To Position X (m)", self.stair_to_x)
        layout.addRow("To Position Y (m)", self.stair_to_y)
        layout.addRow("To Zone", self.stair_to_zone)
        layout.addRow("To Observable Region Width (m)", self.stair_to_region_width)
        layout.addRow("To Observable Region Depth (m)", self.stair_to_region_depth)

        layout.addRow("Stair Width (m)", self.stair_width)

        layout.addRow(
            "Vertical Height (m)",
            self.vertical_height,
        )

        layout.addRow(
            "Travel Distance (m)",
            self.travel_distance,
        )

        self.stair_fields = [
            self.stair_from_floor,
            self.stair_from_x,
            self.stair_from_y,
            self.stair_from_zone,
            self.stair_from_region_width,
            self.stair_from_region_depth,
            self.to_floor_combo,
            to_floor_id_row,
            self.stair_to_x,
            self.stair_to_y,
            self.stair_to_zone,
            self.stair_to_region_width,
            self.stair_to_region_depth,
            self.stair_width,
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

        # Assigned Zone -- reuses the same _populate_zone_combo() Door's
        # Zone A/Zone B combos already establish, resolved from the
        # camera's own floor_id. V1 assigns a single zone through this
        # combo; the model underneath (Camera.zone_ids) already holds a
        # tuple so a future multi-select UI can widen this without any
        # model change.
        self.camera_zone = QComboBox()

        self.camera_resolution = QLineEdit()
        self.camera_fps = QLineEdit()

        # Simulation / Replay / Live -- see models.engineering_asset.
        # DeviceMode. Only Simulation is meaningful today; this combo
        # just records the user's choice for a future Replay/Live
        # integration to read.
        self.camera_mode = QComboBox()

        for mode in DeviceMode.ALL:
            self.camera_mode.addItem(mode)

        # Connection Info -- placeholders only, never read/connected to
        # by anything today (see models.engineering_asset.ConnectionInfo).
        self.camera_rtsp = QLineEdit()
        self.camera_ip = QLineEdit()
        self.camera_username = QLineEdit()

        self.camera_password = QLineEdit()
        self.camera_password.setEchoMode(
            QLineEdit.EchoMode.Password
        )

        layout.addRow("Position X (m)", self.camera_x)
        layout.addRow("Position Y (m)", self.camera_y)

        layout.addRow("Rotation (deg)", self.camera_rotation)

        layout.addRow("Horizontal FOV (deg)", self.camera_fov)
        layout.addRow("Maximum Range (m)", self.camera_range)
        layout.addRow("Mount Height (m)", self.camera_mount_height)

        layout.addRow("Active", self.camera_active)

        layout.addRow("Assigned Zone", self.camera_zone)

        layout.addRow("Resolution", self.camera_resolution)
        layout.addRow("FPS", self.camera_fps)

        layout.addRow("Mode", self.camera_mode)

        layout.addRow("RTSP Address", self.camera_rtsp)
        layout.addRow("IP Address", self.camera_ip)
        layout.addRow("Username", self.camera_username)
        layout.addRow("Password", self.camera_password)

        # Camera Coverage & Visibility Engine -- derived, never
        # editable (same convention as Stair's Vertical Height/Travel
        # Distance above), recomputed by VisibilityEngine every time
        # show_camera() runs. See visibility/engine.py.
        self.camera_visible_zones = QLabel("-")
        self.camera_partial_zones = QLabel("-")
        self.camera_hidden_zones = QLabel("-")
        self.camera_max_visible_distance = QLabel("-")

        layout.addRow("Visible Zones", self.camera_visible_zones)
        layout.addRow("Partially Visible Zones", self.camera_partial_zones)
        layout.addRow("Hidden Zones", self.camera_hidden_zones)
        layout.addRow("Max Visible Distance (m)", self.camera_max_visible_distance)

        # Real Camera Calibration & World-Coordinate Validation
        # milestone, Phase 13 -- the one status line + one action this
        # milestone asks Designer to expose: "NOT CONFIGURED" /
        # "CONFIGURED -- UNVALIDATED" / "VALIDATED -- RMSE: X m", never
        # raw matrices in the main panel (a dedicated dialog, opened by
        # the button below, is where Load/Save actually happens).
        self.camera_calibration_status = QLabel("CALIBRATION: NOT CONFIGURED")
        self.camera_calibrate_button = QPushButton("Calibrate Camera...")
        self.camera_calibrate_button.clicked.connect(self._open_calibration_dialog)

        layout.addRow("Calibration", self.camera_calibration_status)
        layout.addRow("", self.camera_calibrate_button)

        self.camera_fields = [
            self.camera_x,
            self.camera_y,
            self.camera_rotation,
            self.camera_fov,
            self.camera_range,
            self.camera_mount_height,
            self.camera_active,
            self.camera_zone,
            self.camera_resolution,
            self.camera_fps,
            self.camera_mode,
            self.camera_rtsp,
            self.camera_ip,
            self.camera_username,
            self.camera_password,
            self.camera_visible_zones,
            self.camera_partial_zones,
            self.camera_hidden_zones,
            self.camera_max_visible_distance,
            self.camera_calibration_status,
            self.camera_calibrate_button,
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
        # Smoke Detector (Building Sensor Network Framework) --
        # additive, alongside (never replacing) the generic Detector
        # section above.
        # =====================================================

        self.smoke_detector_x = QLineEdit()
        self.smoke_detector_y = QLineEdit()

        self.smoke_detector_active = QCheckBox()

        self.smoke_detector_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.smoke_detector_health.addItem(health_status)

        self.smoke_detector_mode = QComboBox()

        for mode in DeviceMode.ALL:
            self.smoke_detector_mode.addItem(mode)

        self.smoke_detector_threshold = QLineEdit()

        self.smoke_detector_installation_date = QLineEdit()

        # A manually-entered smoke_level reading -- this framework has
        # no live hazard simulation wired into the Designer (see
        # designer/perception_debug_runner.py's own identical note for
        # Perception), so this lets an engineer directly test a
        # detector's activation_threshold against a hand-entered
        # reading, the same "hand-set, no time evolution" role
        # PerceptionDebugPanel's own Hazard Injector plays -- an
        # independent, self-contained control, not wired to that panel.
        self.smoke_detector_test_level = QLineEdit()

        self.smoke_detector_state = QLabel("-")

        # Digital Twin Asset -> Zone Assignment & Live FACP Runtime
        # milestone -- Zone.contains() auto-assigns this on placement
        # (see GraphicsScene._find_unambiguous_zone_at()); this combo is
        # both the visible confirmation of that and the manual override/
        # reassignment path, same "Assigned Zone" convention Camera/
        # DynamicSign already established (single zone -- a point
        # detector's zone_ids represents the one physical zone containing
        # it, not a multi-zone service-coverage concept; see this
        # milestone's own architecture doc for the full reasoning).
        self.smoke_detector_zone = QComboBox()

        self.smoke_detector_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.smoke_detector_zone_warning.setWordWrap(True)
        self.smoke_detector_zone_warning.setStyleSheet("color: #b45309;")

        layout.addRow("Position X (m)", self.smoke_detector_x)
        layout.addRow("Position Y (m)", self.smoke_detector_y)

        layout.addRow("Active", self.smoke_detector_active)

        layout.addRow("Health Status", self.smoke_detector_health)
        layout.addRow("Mode", self.smoke_detector_mode)

        layout.addRow("Activation Threshold (smoke level 0-1)", self.smoke_detector_threshold)
        layout.addRow("Installation Date", self.smoke_detector_installation_date)

        layout.addRow("Test Smoke Level (0-1)", self.smoke_detector_test_level)
        layout.addRow("Current State", self.smoke_detector_state)

        layout.addRow("Assigned Zone", self.smoke_detector_zone)
        layout.addRow("", self.smoke_detector_zone_warning)

        self.smoke_detector_fields = [
            self.smoke_detector_x,
            self.smoke_detector_y,
            self.smoke_detector_active,
            self.smoke_detector_health,
            self.smoke_detector_mode,
            self.smoke_detector_threshold,
            self.smoke_detector_installation_date,
            self.smoke_detector_test_level,
            self.smoke_detector_state,
            self.smoke_detector_zone,
            self.smoke_detector_zone_warning,
        ]

        # =====================================================
        # Heat Detector (Building Sensor Network Framework) -- reuses
        # the exact same framework/state model as Smoke Detector above.
        # =====================================================

        self.heat_detector_x = QLineEdit()
        self.heat_detector_y = QLineEdit()

        self.heat_detector_active = QCheckBox()

        self.heat_detector_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.heat_detector_health.addItem(health_status)

        self.heat_detector_mode = QComboBox()

        for mode in DeviceMode.ALL:
            self.heat_detector_mode.addItem(mode)

        self.heat_detector_threshold = QLineEdit()

        self.heat_detector_installation_date = QLineEdit()

        self.heat_detector_test_temperature = QLineEdit()

        self.heat_detector_state = QLabel("-")

        # Same auto-assignment/manual-override convention as Smoke
        # Detector immediately above.
        self.heat_detector_zone = QComboBox()

        self.heat_detector_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.heat_detector_zone_warning.setWordWrap(True)
        self.heat_detector_zone_warning.setStyleSheet("color: #b45309;")

        layout.addRow("Position X (m)", self.heat_detector_x)
        layout.addRow("Position Y (m)", self.heat_detector_y)

        layout.addRow("Active", self.heat_detector_active)

        layout.addRow("Health Status", self.heat_detector_health)
        layout.addRow("Mode", self.heat_detector_mode)

        layout.addRow("Activation Threshold (°C)", self.heat_detector_threshold)
        layout.addRow("Installation Date", self.heat_detector_installation_date)

        layout.addRow("Test Temperature (°C)", self.heat_detector_test_temperature)
        layout.addRow("Current State", self.heat_detector_state)

        layout.addRow("Assigned Zone", self.heat_detector_zone)
        layout.addRow("", self.heat_detector_zone_warning)

        self.heat_detector_fields = [
            self.heat_detector_x,
            self.heat_detector_y,
            self.heat_detector_active,
            self.heat_detector_health,
            self.heat_detector_mode,
            self.heat_detector_threshold,
            self.heat_detector_installation_date,
            self.heat_detector_test_temperature,
            self.heat_detector_state,
            self.heat_detector_zone,
            self.heat_detector_zone_warning,
        ]

        # =====================================================
        # Speaker (Zoned Voice Evacuation & Speaker Network Framework)
        # -- reuses the same SensorAsset foundation as Smoke/Heat
        # Detector above, minus the detector-only test-reading/current-
        # state controls (a Speaker is an output device -- it has no
        # compute_state()/alarm concept of its own to test against).
        # =====================================================

        self.speaker_x = QLineEdit()
        self.speaker_y = QLineEdit()

        self.speaker_active = QCheckBox()

        self.speaker_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.speaker_health.addItem(health_status)

        self.speaker_mode = QComboBox()

        for mode in DeviceMode.ALL:
            self.speaker_mode.addItem(mode)

        self.speaker_type = QComboBox()

        for speaker_type in Speaker.SPEAKER_TYPES:
            self.speaker_type.addItem(speaker_type)

        self.speaker_volume = QLineEdit()

        self.speaker_installation_date = QLineEdit()

        # Digital Twin Asset -> Zone Assignment & Live FACP Runtime
        # milestone -- Speaker.zone_ids is SERVICE/BROADCAST COVERAGE,
        # not physical location (a speaker mounted in one zone may
        # legitimately serve others -- Phase 2's own explicit semantics).
        # Genuinely multi-select (see _populate_zone_checklist()'s own
        # docstring for why this is a checklist, not a QComboBox), and
        # deliberately NEVER auto-assigned from position (Phase 4's own
        # explicit "do not automatically assign Speaker coverage based
        # solely on position" instruction) -- always starts empty on
        # placement, manual assignment only.
        self.speaker_zones = QListWidget()
        self.speaker_zones.setMaximumHeight(90)

        self.speaker_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.speaker_zone_warning.setWordWrap(True)
        self.speaker_zone_warning.setStyleSheet("color: #b45309;")

        layout.addRow("Position X (m)", self.speaker_x)
        layout.addRow("Position Y (m)", self.speaker_y)

        layout.addRow("Active", self.speaker_active)

        layout.addRow("Health Status", self.speaker_health)
        layout.addRow("Mode", self.speaker_mode)

        layout.addRow("Speaker Type", self.speaker_type)
        layout.addRow("Volume Level (dB)", self.speaker_volume)
        layout.addRow("Installation Date", self.speaker_installation_date)

        layout.addRow("Covered Zone(s)", self.speaker_zones)
        layout.addRow("", self.speaker_zone_warning)

        self.speaker_fields = [
            self.speaker_x,
            self.speaker_y,
            self.speaker_active,
            self.speaker_health,
            self.speaker_mode,
            self.speaker_type,
            self.speaker_volume,
            self.speaker_installation_date,
            self.speaker_zones,
            self.speaker_zone_warning,
        ]

        # =====================================================
        # Dynamic Evacuation Sign (Live Dynamic Evacuation Signage
        # milestone) -- Phase 19's own required minimum: Name (shared
        # object_name field), Active, Orientation, Covered Zone(s). No
        # graphical sign editor, no supported-indications multi-select
        # UI (the model's own DEFAULT_SUPPORTED already covers every
        # Designer-placed sign; narrowing it is a project-file/scripted
        # edit, not a Property Panel concern this milestone requires).
        # =====================================================

        self.sign_x = QLineEdit()
        self.sign_y = QLineEdit()

        self.sign_orientation = QLineEdit()

        self.sign_active = QCheckBox()

        # Single-zone assignment, same V1 simplicity convention as
        # self.camera_zone above -- Sign.zone_ids already holds a tuple,
        # so a future multi-select UI can widen this without any model
        # change.
        self.sign_zone = QComboBox()

        layout.addRow("Position X (m)", self.sign_x)
        layout.addRow("Position Y (m)", self.sign_y)
        layout.addRow("Orientation (deg)", self.sign_orientation)
        layout.addRow("Active", self.sign_active)
        layout.addRow("Covered Zone", self.sign_zone)

        self.sign_fields = [
            self.sign_x,
            self.sign_y,
            self.sign_orientation,
            self.sign_active,
            self.sign_zone,
        ]

        # =====================================================
        # Manual Call Point (Manual Call Points & Emergency Lighting
        # milestone) -- reuses the exact same SensorAsset foundation as
        # Smoke/Heat Detector, minus the continuous-reading "test level"
        # control neither needs: an MCP's own `activated` field IS the
        # ground truth (a direct human action on the device, not an
        # external hazard reading -- see models.manual_call_point.
        # ManualCallPoint's own docstring), so it gets a plain "Activated"
        # checkbox instead.
        # =====================================================

        self.mcp_x = QLineEdit()
        self.mcp_y = QLineEdit()

        self.mcp_active = QCheckBox()
        self.mcp_activated = QCheckBox()

        self.mcp_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.mcp_health.addItem(health_status)

        self.mcp_mode = QComboBox()

        for mode in DeviceMode.ALL:
            self.mcp_mode.addItem(mode)

        self.mcp_installation_date = QLineEdit()

        self.mcp_state = QLabel("-")

        # Single-zone physical-location assignment, same convention as
        # Camera/Sign/Smoke-Heat Detector.
        self.mcp_zone = QComboBox()

        self.mcp_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.mcp_zone_warning.setWordWrap(True)
        self.mcp_zone_warning.setStyleSheet("color: #b45309;")

        layout.addRow("Position X (m)", self.mcp_x)
        layout.addRow("Position Y (m)", self.mcp_y)

        layout.addRow("Active", self.mcp_active)
        layout.addRow("Activated", self.mcp_activated)

        layout.addRow("Health Status", self.mcp_health)
        layout.addRow("Mode", self.mcp_mode)
        layout.addRow("Installation Date", self.mcp_installation_date)
        layout.addRow("Current State", self.mcp_state)

        layout.addRow("Assigned Zone", self.mcp_zone)
        layout.addRow("", self.mcp_zone_warning)

        self.mcp_fields = [
            self.mcp_x,
            self.mcp_y,
            self.mcp_active,
            self.mcp_activated,
            self.mcp_health,
            self.mcp_mode,
            self.mcp_installation_date,
            self.mcp_state,
            self.mcp_zone,
            self.mcp_zone_warning,
        ]

        # =====================================================
        # Emergency Light (Manual Call Points & Emergency Lighting
        # milestone) -- a building safety OUTPUT asset, not a sensor
        # (see models.emergency_light.EmergencyLight's own docstring) --
        # no alarm/current-state concept, only availability.
        # =====================================================

        self.emergency_light_x = QLineEdit()
        self.emergency_light_y = QLineEdit()

        self.emergency_light_active = QCheckBox()

        self.emergency_light_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.emergency_light_health.addItem(health_status)

        self.emergency_light_type = QComboBox()

        for light_type in EmergencyLight.LIGHT_TYPES:
            self.emergency_light_type.addItem(light_type)

        self.emergency_light_availability = QLabel("-")

        self.emergency_light_zone = QComboBox()

        self.emergency_light_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.emergency_light_zone_warning.setWordWrap(True)
        self.emergency_light_zone_warning.setStyleSheet("color: #b45309;")

        layout.addRow("Position X (m)", self.emergency_light_x)
        layout.addRow("Position Y (m)", self.emergency_light_y)

        layout.addRow("Active", self.emergency_light_active)

        layout.addRow("Health Status", self.emergency_light_health)
        layout.addRow("Light Type", self.emergency_light_type)
        layout.addRow("Availability", self.emergency_light_availability)

        layout.addRow("Assigned Zone", self.emergency_light_zone)
        layout.addRow("", self.emergency_light_zone_warning)

        self.emergency_light_fields = [
            self.emergency_light_x,
            self.emergency_light_y,
            self.emergency_light_active,
            self.emergency_light_health,
            self.emergency_light_type,
            self.emergency_light_availability,
            self.emergency_light_zone,
            self.emergency_light_zone_warning,
        ]

        # =====================================================
        # Sprinkler (Fire Suppression & Water-Based Safety Asset
        # Digital Twin milestone) -- reuses the exact same SensorAsset
        # foundation and "Test Temperature" manual-reading convention
        # Heat Detector already establishes above (no live hazard
        # simulation is wired into the Designer -- see
        # models.sprinkler.Sprinkler's own docstring for why this
        # produces a SprinklerActivationState, never a DetectorState).
        # =====================================================

        self.sprinkler_x = QLineEdit()
        self.sprinkler_y = QLineEdit()

        self.sprinkler_active = QCheckBox()

        self.sprinkler_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.sprinkler_health.addItem(health_status)

        self.sprinkler_mode = QComboBox()

        for mode in DeviceMode.ALL:
            self.sprinkler_mode.addItem(mode)

        self.sprinkler_activation_temperature = QLineEdit()

        self.sprinkler_installation_date = QLineEdit()

        self.sprinkler_test_temperature = QLineEdit()

        self.sprinkler_state = QLabel("-")

        self.sprinkler_zone = QComboBox()

        self.sprinkler_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.sprinkler_zone_warning.setWordWrap(True)
        self.sprinkler_zone_warning.setStyleSheet("color: #b45309;")

        self.sprinkler_fire_water_system = QComboBox()

        layout.addRow("Position X (m)", self.sprinkler_x)
        layout.addRow("Position Y (m)", self.sprinkler_y)

        layout.addRow("Active", self.sprinkler_active)

        layout.addRow("Health Status", self.sprinkler_health)
        layout.addRow("Mode", self.sprinkler_mode)

        layout.addRow("Activation Temperature (°C)", self.sprinkler_activation_temperature)
        layout.addRow("Installation Date", self.sprinkler_installation_date)

        layout.addRow("Test Temperature (°C)", self.sprinkler_test_temperature)
        layout.addRow("Current State", self.sprinkler_state)

        layout.addRow("Assigned Zone", self.sprinkler_zone)
        layout.addRow("", self.sprinkler_zone_warning)
        layout.addRow("Fire Water System", self.sprinkler_fire_water_system)

        self.sprinkler_fields = [
            self.sprinkler_x,
            self.sprinkler_y,
            self.sprinkler_active,
            self.sprinkler_health,
            self.sprinkler_mode,
            self.sprinkler_activation_temperature,
            self.sprinkler_installation_date,
            self.sprinkler_test_temperature,
            self.sprinkler_state,
            self.sprinkler_zone,
            self.sprinkler_zone_warning,
            self.sprinkler_fire_water_system,
        ]

        # =====================================================
        # Fire Extinguisher (Fire Suppression & Water-Based Safety
        # Asset Digital Twin milestone) -- a passive, manually-operated
        # resource, not a sensor -- no current-state/test-reading
        # controls, only availability (see models.fire_extinguisher.
        # FireExtinguisher's own docstring).
        # =====================================================

        self.fire_extinguisher_x = QLineEdit()
        self.fire_extinguisher_y = QLineEdit()

        self.fire_extinguisher_active = QCheckBox()

        self.fire_extinguisher_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.fire_extinguisher_health.addItem(health_status)

        self.fire_extinguisher_type = QComboBox()

        for extinguisher_type in FireExtinguisher.EXTINGUISHER_TYPES:
            self.fire_extinguisher_type.addItem(extinguisher_type)

        self.fire_extinguisher_availability = QLabel("-")

        self.fire_extinguisher_zone = QComboBox()

        self.fire_extinguisher_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.fire_extinguisher_zone_warning.setWordWrap(True)
        self.fire_extinguisher_zone_warning.setStyleSheet("color: #b45309;")

        layout.addRow("Position X (m)", self.fire_extinguisher_x)
        layout.addRow("Position Y (m)", self.fire_extinguisher_y)

        layout.addRow("Active", self.fire_extinguisher_active)

        layout.addRow("Health Status", self.fire_extinguisher_health)
        layout.addRow("Extinguisher Type", self.fire_extinguisher_type)
        layout.addRow("Availability", self.fire_extinguisher_availability)

        layout.addRow("Assigned Zone", self.fire_extinguisher_zone)
        layout.addRow("", self.fire_extinguisher_zone_warning)

        self.fire_extinguisher_fields = [
            self.fire_extinguisher_x,
            self.fire_extinguisher_y,
            self.fire_extinguisher_active,
            self.fire_extinguisher_health,
            self.fire_extinguisher_type,
            self.fire_extinguisher_availability,
            self.fire_extinguisher_zone,
            self.fire_extinguisher_zone_warning,
        ]

        # =====================================================
        # Fire Hydrant / Landing Valve (Fire Suppression & Water-Based
        # Safety Asset Digital Twin milestone) -- same passive-resource
        # shape as Fire Extinguisher above.
        # =====================================================

        self.fire_hydrant_x = QLineEdit()
        self.fire_hydrant_y = QLineEdit()

        self.fire_hydrant_active = QCheckBox()

        self.fire_hydrant_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.fire_hydrant_health.addItem(health_status)

        self.fire_hydrant_type = QComboBox()

        for hydrant_type in FireHydrant.HYDRANT_TYPES:
            self.fire_hydrant_type.addItem(hydrant_type)

        self.fire_hydrant_availability = QLabel("-")

        self.fire_hydrant_zone = QComboBox()

        self.fire_hydrant_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.fire_hydrant_zone_warning.setWordWrap(True)
        self.fire_hydrant_zone_warning.setStyleSheet("color: #b45309;")

        # Fire Water Supply & Suppression Infrastructure milestone --
        # which FireWaterSystem (models/fire_water_system.py) this
        # Hydrant is traced to, same single-select combo convention as
        # Assigned Zone above (never affects zone_ids).
        self.fire_hydrant_fire_water_system = QComboBox()

        layout.addRow("Position X (m)", self.fire_hydrant_x)
        layout.addRow("Position Y (m)", self.fire_hydrant_y)

        layout.addRow("Active", self.fire_hydrant_active)

        layout.addRow("Health Status", self.fire_hydrant_health)
        layout.addRow("Hydrant Type", self.fire_hydrant_type)
        layout.addRow("Availability", self.fire_hydrant_availability)

        layout.addRow("Assigned Zone", self.fire_hydrant_zone)
        layout.addRow("", self.fire_hydrant_zone_warning)
        layout.addRow("Fire Water System", self.fire_hydrant_fire_water_system)

        self.fire_hydrant_fields = [
            self.fire_hydrant_x,
            self.fire_hydrant_y,
            self.fire_hydrant_active,
            self.fire_hydrant_health,
            self.fire_hydrant_type,
            self.fire_hydrant_availability,
            self.fire_hydrant_zone,
            self.fire_hydrant_zone_warning,
            self.fire_hydrant_fire_water_system,
        ]

        # =====================================================
        # Hose Reel (Fire Suppression & Water-Based Safety Asset
        # Digital Twin milestone) -- same passive-resource shape as
        # Fire Extinguisher/Fire Hydrant above, minus a type combo
        # (see models.hose_reel.HoseReel's own docstring for why).
        # =====================================================

        self.hose_reel_x = QLineEdit()
        self.hose_reel_y = QLineEdit()

        self.hose_reel_active = QCheckBox()

        self.hose_reel_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.hose_reel_health.addItem(health_status)

        self.hose_reel_availability = QLabel("-")

        self.hose_reel_zone = QComboBox()

        self.hose_reel_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.hose_reel_zone_warning.setWordWrap(True)
        self.hose_reel_zone_warning.setStyleSheet("color: #b45309;")

        self.hose_reel_fire_water_system = QComboBox()

        layout.addRow("Position X (m)", self.hose_reel_x)
        layout.addRow("Position Y (m)", self.hose_reel_y)

        layout.addRow("Active", self.hose_reel_active)

        layout.addRow("Health Status", self.hose_reel_health)
        layout.addRow("Availability", self.hose_reel_availability)

        layout.addRow("Assigned Zone", self.hose_reel_zone)
        layout.addRow("", self.hose_reel_zone_warning)
        layout.addRow("Fire Water System", self.hose_reel_fire_water_system)

        self.hose_reel_fields = [
            self.hose_reel_x,
            self.hose_reel_y,
            self.hose_reel_active,
            self.hose_reel_health,
            self.hose_reel_availability,
            self.hose_reel_zone,
            self.hose_reel_zone_warning,
            self.hose_reel_fire_water_system,
        ]

        # =====================================================
        # Fire Water Tank (Fire Water Supply & Suppression
        # Infrastructure milestone) -- an EngineeringAsset, not a
        # sensor/pump: no continuous reading, only capacity/level and
        # availability (see models.fire_water_tank.FireWaterTank's own
        # docstring).
        # =====================================================

        self.fire_water_tank_x = QLineEdit()
        self.fire_water_tank_y = QLineEdit()

        self.fire_water_tank_active = QCheckBox()

        self.fire_water_tank_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.fire_water_tank_health.addItem(health_status)

        self.fire_water_tank_capacity = QLineEdit()
        self.fire_water_tank_level = QLineEdit()

        self.fire_water_tank_state = QLabel("-")

        self.fire_water_tank_zone = QComboBox()

        self.fire_water_tank_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.fire_water_tank_zone_warning.setWordWrap(True)
        self.fire_water_tank_zone_warning.setStyleSheet("color: #b45309;")

        self.fire_water_tank_fire_water_system = QComboBox()

        layout.addRow("Position X (m)", self.fire_water_tank_x)
        layout.addRow("Position Y (m)", self.fire_water_tank_y)

        layout.addRow("Active", self.fire_water_tank_active)

        layout.addRow("Health Status", self.fire_water_tank_health)
        layout.addRow("Capacity (L)", self.fire_water_tank_capacity)
        layout.addRow("Current Level (L, blank = unmeasured)", self.fire_water_tank_level)
        layout.addRow("Operational State", self.fire_water_tank_state)

        layout.addRow("Assigned Zone", self.fire_water_tank_zone)
        layout.addRow("", self.fire_water_tank_zone_warning)
        layout.addRow("Fire Water System", self.fire_water_tank_fire_water_system)

        self.fire_water_tank_fields = [
            self.fire_water_tank_x,
            self.fire_water_tank_y,
            self.fire_water_tank_active,
            self.fire_water_tank_health,
            self.fire_water_tank_capacity,
            self.fire_water_tank_level,
            self.fire_water_tank_state,
            self.fire_water_tank_zone,
            self.fire_water_tank_zone_warning,
            self.fire_water_tank_fire_water_system,
        ]

        # =====================================================
        # Fire Pump (Fire Water Supply & Suppression Infrastructure
        # milestone) -- see models.pump_asset.PumpAsset's own docstring
        # for the run-state/control-mode semantics shared with Jockey
        # Pump.
        # =====================================================

        self.fire_pump_x = QLineEdit()
        self.fire_pump_y = QLineEdit()

        self.fire_pump_active = QCheckBox()

        self.fire_pump_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.fire_pump_health.addItem(health_status)

        self.fire_pump_control_mode = QComboBox()

        for control_mode in PumpControlMode.ALL:
            self.fire_pump_control_mode.addItem(control_mode)

        self.fire_pump_running = QCheckBox()

        self.fire_pump_state = QLabel("-")

        self.fire_pump_zone = QComboBox()

        self.fire_pump_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.fire_pump_zone_warning.setWordWrap(True)
        self.fire_pump_zone_warning.setStyleSheet("color: #b45309;")

        self.fire_pump_fire_water_system = QComboBox()

        layout.addRow("Position X (m)", self.fire_pump_x)
        layout.addRow("Position Y (m)", self.fire_pump_y)

        layout.addRow("Active", self.fire_pump_active)

        layout.addRow("Health Status", self.fire_pump_health)
        layout.addRow("Control Mode", self.fire_pump_control_mode)
        layout.addRow("Running", self.fire_pump_running)
        layout.addRow("Operational State", self.fire_pump_state)

        layout.addRow("Assigned Zone", self.fire_pump_zone)
        layout.addRow("", self.fire_pump_zone_warning)
        layout.addRow("Fire Water System", self.fire_pump_fire_water_system)

        self.fire_pump_fields = [
            self.fire_pump_x,
            self.fire_pump_y,
            self.fire_pump_active,
            self.fire_pump_health,
            self.fire_pump_control_mode,
            self.fire_pump_running,
            self.fire_pump_state,
            self.fire_pump_zone,
            self.fire_pump_zone_warning,
            self.fire_pump_fire_water_system,
        ]

        # =====================================================
        # Jockey Pump (Fire Water Supply & Suppression Infrastructure
        # milestone) -- same PumpAsset shape as Fire Pump above.
        # =====================================================

        self.jockey_pump_x = QLineEdit()
        self.jockey_pump_y = QLineEdit()

        self.jockey_pump_active = QCheckBox()

        self.jockey_pump_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.jockey_pump_health.addItem(health_status)

        self.jockey_pump_control_mode = QComboBox()

        for control_mode in PumpControlMode.ALL:
            self.jockey_pump_control_mode.addItem(control_mode)

        self.jockey_pump_running = QCheckBox()

        self.jockey_pump_state = QLabel("-")

        self.jockey_pump_zone = QComboBox()

        self.jockey_pump_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.jockey_pump_zone_warning.setWordWrap(True)
        self.jockey_pump_zone_warning.setStyleSheet("color: #b45309;")

        self.jockey_pump_fire_water_system = QComboBox()

        layout.addRow("Position X (m)", self.jockey_pump_x)
        layout.addRow("Position Y (m)", self.jockey_pump_y)

        layout.addRow("Active", self.jockey_pump_active)

        layout.addRow("Health Status", self.jockey_pump_health)
        layout.addRow("Control Mode", self.jockey_pump_control_mode)
        layout.addRow("Running", self.jockey_pump_running)
        layout.addRow("Operational State", self.jockey_pump_state)

        layout.addRow("Assigned Zone", self.jockey_pump_zone)
        layout.addRow("", self.jockey_pump_zone_warning)
        layout.addRow("Fire Water System", self.jockey_pump_fire_water_system)

        self.jockey_pump_fields = [
            self.jockey_pump_x,
            self.jockey_pump_y,
            self.jockey_pump_active,
            self.jockey_pump_health,
            self.jockey_pump_control_mode,
            self.jockey_pump_running,
            self.jockey_pump_state,
            self.jockey_pump_zone,
            self.jockey_pump_zone_warning,
            self.jockey_pump_fire_water_system,
        ]

        # =====================================================
        # Fire Service Inlet / Breeching Inlet (Fire Water Supply &
        # Suppression Infrastructure milestone) -- a passive resource,
        # same shape as Fire Extinguisher/Fire Hydrant/Hose Reel.
        # =====================================================

        self.fire_service_inlet_x = QLineEdit()
        self.fire_service_inlet_y = QLineEdit()

        self.fire_service_inlet_active = QCheckBox()

        self.fire_service_inlet_health = QComboBox()

        for health_status in HealthStatus.ALL:
            self.fire_service_inlet_health.addItem(health_status)

        self.fire_service_inlet_type = QComboBox()

        for inlet_type in FireServiceInlet.INLET_TYPES:
            self.fire_service_inlet_type.addItem(inlet_type)

        self.fire_service_inlet_availability = QLabel("-")

        self.fire_service_inlet_zone = QComboBox()

        self.fire_service_inlet_zone_warning = QLabel(
            "Zone assignment required for live operation."
        )
        self.fire_service_inlet_zone_warning.setWordWrap(True)
        self.fire_service_inlet_zone_warning.setStyleSheet("color: #b45309;")

        self.fire_service_inlet_fire_water_system = QComboBox()

        layout.addRow("Position X (m)", self.fire_service_inlet_x)
        layout.addRow("Position Y (m)", self.fire_service_inlet_y)

        layout.addRow("Active", self.fire_service_inlet_active)

        layout.addRow("Health Status", self.fire_service_inlet_health)
        layout.addRow("Inlet Type", self.fire_service_inlet_type)
        layout.addRow("Availability", self.fire_service_inlet_availability)

        layout.addRow("Assigned Zone", self.fire_service_inlet_zone)
        layout.addRow("", self.fire_service_inlet_zone_warning)
        layout.addRow("Fire Water System", self.fire_service_inlet_fire_water_system)

        self.fire_service_inlet_fields = [
            self.fire_service_inlet_x,
            self.fire_service_inlet_y,
            self.fire_service_inlet_active,
            self.fire_service_inlet_health,
            self.fire_service_inlet_type,
            self.fire_service_inlet_availability,
            self.fire_service_inlet_zone,
            self.fire_service_inlet_zone_warning,
            self.fire_service_inlet_fire_water_system,
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

        # =====================================================
        # Occupant (Manual Simulation Sandbox)
        #
        # Read-only "Occupant Information" fields per the milestone,
        # plus the one editable field: destination type. Everything
        # else here is derived from sandbox.occupant.SandboxOccupant/
        # Route, never stored redundantly -- same convention Stair's
        # Vertical Height/Travel Distance already follow.
        # =====================================================

        self.occupant_current_floor = QLabel("-")
        self.occupant_current_zone = QLabel("-")
        self.occupant_current_node = QLabel("-")

        self.occupant_destination_type = QComboBox()

        for destination_type in SandboxDestinationType.OPTIONS:
            self.occupant_destination_type.addItem(destination_type)

        self.occupant_destination_node = QLabel("-")
        self.occupant_next_node = QLabel("-")
        self.occupant_remaining_distance = QLabel("-")
        self.occupant_current_speed = QLabel("-")
        self.occupant_state = QLabel("-")

        layout.addRow("Current Floor", self.occupant_current_floor)
        layout.addRow("Current Zone", self.occupant_current_zone)
        layout.addRow("Current Node", self.occupant_current_node)

        layout.addRow("Destination Type", self.occupant_destination_type)
        layout.addRow("Destination Node", self.occupant_destination_node)
        layout.addRow("Next Node", self.occupant_next_node)
        layout.addRow("Remaining Distance (m)", self.occupant_remaining_distance)
        layout.addRow("Current Speed (m/s)", self.occupant_current_speed)
        layout.addRow("State", self.occupant_state)

        self.occupant_fields = [
            self.occupant_current_floor,
            self.occupant_current_zone,
            self.occupant_current_node,
            self.occupant_destination_type,
            self.occupant_destination_node,
            self.occupant_next_node,
            self.occupant_remaining_distance,
            self.occupant_current_speed,
            self.occupant_state,
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

        self.exit_zone.currentIndexChanged.connect(
            self.update_exit_zone
        )

        self.stair_from_x.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_from_y.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_to_x.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_to_y.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_width.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_from_region_width.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_from_region_depth.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_to_region_width.editingFinished.connect(
            self.update_stair_geometry
        )

        self.stair_to_region_depth.editingFinished.connect(
            self.update_stair_geometry
        )

        self.to_floor_combo.currentIndexChanged.connect(
            self.update_stair_to_floor
        )

        self.stair_from_zone.currentIndexChanged.connect(
            self.update_stair_from_zone
        )

        self.stair_to_zone.currentIndexChanged.connect(
            self.update_stair_to_zone
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

        self.camera_zone.currentIndexChanged.connect(
            self.update_camera_zone
        )

        self.camera_resolution.editingFinished.connect(
            self.update_camera_metadata
        )

        self.camera_fps.editingFinished.connect(
            self.update_camera_metadata
        )

        self.camera_mode.currentIndexChanged.connect(
            self.update_camera_mode
        )

        self.camera_rtsp.editingFinished.connect(
            self.update_camera_connection
        )

        self.camera_ip.editingFinished.connect(
            self.update_camera_connection
        )

        self.camera_username.editingFinished.connect(
            self.update_camera_connection
        )

        self.camera_password.editingFinished.connect(
            self.update_camera_connection
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

        self.smoke_detector_x.editingFinished.connect(
            self.update_smoke_detector_geometry
        )

        self.smoke_detector_y.editingFinished.connect(
            self.update_smoke_detector_geometry
        )

        self.smoke_detector_active.toggled.connect(
            self.update_smoke_detector_active
        )

        self.smoke_detector_health.currentIndexChanged.connect(
            self.update_smoke_detector_health
        )

        self.smoke_detector_mode.currentIndexChanged.connect(
            self.update_smoke_detector_mode
        )

        self.smoke_detector_threshold.editingFinished.connect(
            self.update_smoke_detector_threshold
        )

        self.smoke_detector_installation_date.editingFinished.connect(
            self.update_smoke_detector_installation_date
        )

        self.smoke_detector_test_level.editingFinished.connect(
            self.update_smoke_detector_test_reading
        )

        self.smoke_detector_zone.currentIndexChanged.connect(
            self.update_smoke_detector_zone
        )

        self.heat_detector_x.editingFinished.connect(
            self.update_heat_detector_geometry
        )

        self.heat_detector_y.editingFinished.connect(
            self.update_heat_detector_geometry
        )

        self.heat_detector_active.toggled.connect(
            self.update_heat_detector_active
        )

        self.heat_detector_health.currentIndexChanged.connect(
            self.update_heat_detector_health
        )

        self.heat_detector_mode.currentIndexChanged.connect(
            self.update_heat_detector_mode
        )

        self.heat_detector_threshold.editingFinished.connect(
            self.update_heat_detector_threshold
        )

        self.heat_detector_installation_date.editingFinished.connect(
            self.update_heat_detector_installation_date
        )

        self.heat_detector_test_temperature.editingFinished.connect(
            self.update_heat_detector_test_reading
        )

        self.heat_detector_zone.currentIndexChanged.connect(
            self.update_heat_detector_zone
        )

        self.speaker_x.editingFinished.connect(
            self.update_speaker_geometry
        )
        self.speaker_y.editingFinished.connect(
            self.update_speaker_geometry
        )
        self.speaker_active.toggled.connect(
            self.update_speaker_active
        )
        self.speaker_health.currentIndexChanged.connect(
            self.update_speaker_health
        )
        self.speaker_mode.currentIndexChanged.connect(
            self.update_speaker_mode
        )
        self.speaker_type.currentIndexChanged.connect(
            self.update_speaker_type
        )
        self.speaker_volume.editingFinished.connect(
            self.update_speaker_volume
        )
        self.speaker_installation_date.editingFinished.connect(
            self.update_speaker_installation_date
        )

        self.speaker_zones.itemChanged.connect(
            self.update_speaker_zones
        )

        self.sign_x.editingFinished.connect(
            self.update_sign_geometry
        )
        self.sign_y.editingFinished.connect(
            self.update_sign_geometry
        )
        self.sign_orientation.editingFinished.connect(
            self.update_sign_orientation
        )
        self.sign_active.toggled.connect(
            self.update_sign_active
        )
        self.sign_zone.currentIndexChanged.connect(
            self.update_sign_zone
        )

        self.mcp_x.editingFinished.connect(
            self.update_mcp_geometry
        )
        self.mcp_y.editingFinished.connect(
            self.update_mcp_geometry
        )
        self.mcp_active.toggled.connect(
            self.update_mcp_active
        )
        self.mcp_activated.toggled.connect(
            self.update_mcp_activated
        )
        self.mcp_health.currentIndexChanged.connect(
            self.update_mcp_health
        )
        self.mcp_mode.currentIndexChanged.connect(
            self.update_mcp_mode
        )
        self.mcp_installation_date.editingFinished.connect(
            self.update_mcp_installation_date
        )
        self.mcp_zone.currentIndexChanged.connect(
            self.update_mcp_zone
        )

        self.emergency_light_x.editingFinished.connect(
            self.update_emergency_light_geometry
        )
        self.emergency_light_y.editingFinished.connect(
            self.update_emergency_light_geometry
        )
        self.emergency_light_active.toggled.connect(
            self.update_emergency_light_active
        )
        self.emergency_light_health.currentIndexChanged.connect(
            self.update_emergency_light_health
        )
        self.emergency_light_type.currentIndexChanged.connect(
            self.update_emergency_light_type
        )
        self.emergency_light_zone.currentIndexChanged.connect(
            self.update_emergency_light_zone
        )

        self.sprinkler_x.editingFinished.connect(
            self.update_sprinkler_geometry
        )
        self.sprinkler_y.editingFinished.connect(
            self.update_sprinkler_geometry
        )
        self.sprinkler_active.toggled.connect(
            self.update_sprinkler_active
        )
        self.sprinkler_health.currentIndexChanged.connect(
            self.update_sprinkler_health
        )
        self.sprinkler_mode.currentIndexChanged.connect(
            self.update_sprinkler_mode
        )
        self.sprinkler_activation_temperature.editingFinished.connect(
            self.update_sprinkler_activation_temperature
        )
        self.sprinkler_installation_date.editingFinished.connect(
            self.update_sprinkler_installation_date
        )
        self.sprinkler_test_temperature.editingFinished.connect(
            self.update_sprinkler_test_reading
        )
        self.sprinkler_zone.currentIndexChanged.connect(
            self.update_sprinkler_zone
        )
        self.sprinkler_fire_water_system.currentIndexChanged.connect(
            self.update_sprinkler_fire_water_system
        )

        self.fire_extinguisher_x.editingFinished.connect(
            self.update_fire_extinguisher_geometry
        )
        self.fire_extinguisher_y.editingFinished.connect(
            self.update_fire_extinguisher_geometry
        )
        self.fire_extinguisher_active.toggled.connect(
            self.update_fire_extinguisher_active
        )
        self.fire_extinguisher_health.currentIndexChanged.connect(
            self.update_fire_extinguisher_health
        )
        self.fire_extinguisher_type.currentIndexChanged.connect(
            self.update_fire_extinguisher_type
        )
        self.fire_extinguisher_zone.currentIndexChanged.connect(
            self.update_fire_extinguisher_zone
        )

        self.fire_hydrant_x.editingFinished.connect(
            self.update_fire_hydrant_geometry
        )
        self.fire_hydrant_y.editingFinished.connect(
            self.update_fire_hydrant_geometry
        )
        self.fire_hydrant_active.toggled.connect(
            self.update_fire_hydrant_active
        )
        self.fire_hydrant_health.currentIndexChanged.connect(
            self.update_fire_hydrant_health
        )
        self.fire_hydrant_type.currentIndexChanged.connect(
            self.update_fire_hydrant_type
        )
        self.fire_hydrant_zone.currentIndexChanged.connect(
            self.update_fire_hydrant_zone
        )
        self.fire_hydrant_fire_water_system.currentIndexChanged.connect(
            self.update_fire_hydrant_fire_water_system
        )

        self.hose_reel_x.editingFinished.connect(
            self.update_hose_reel_geometry
        )
        self.hose_reel_y.editingFinished.connect(
            self.update_hose_reel_geometry
        )
        self.hose_reel_active.toggled.connect(
            self.update_hose_reel_active
        )
        self.hose_reel_health.currentIndexChanged.connect(
            self.update_hose_reel_health
        )
        self.hose_reel_zone.currentIndexChanged.connect(
            self.update_hose_reel_zone
        )
        self.hose_reel_fire_water_system.currentIndexChanged.connect(
            self.update_hose_reel_fire_water_system
        )

        self.fire_water_tank_x.editingFinished.connect(
            self.update_fire_water_tank_geometry
        )
        self.fire_water_tank_y.editingFinished.connect(
            self.update_fire_water_tank_geometry
        )
        self.fire_water_tank_active.toggled.connect(
            self.update_fire_water_tank_active
        )
        self.fire_water_tank_health.currentIndexChanged.connect(
            self.update_fire_water_tank_health
        )
        self.fire_water_tank_capacity.editingFinished.connect(
            self.update_fire_water_tank_capacity
        )
        self.fire_water_tank_level.editingFinished.connect(
            self.update_fire_water_tank_level
        )
        self.fire_water_tank_zone.currentIndexChanged.connect(
            self.update_fire_water_tank_zone
        )
        self.fire_water_tank_fire_water_system.currentIndexChanged.connect(
            self.update_fire_water_tank_fire_water_system
        )

        self.fire_pump_x.editingFinished.connect(
            self.update_fire_pump_geometry
        )
        self.fire_pump_y.editingFinished.connect(
            self.update_fire_pump_geometry
        )
        self.fire_pump_active.toggled.connect(
            self.update_fire_pump_active
        )
        self.fire_pump_health.currentIndexChanged.connect(
            self.update_fire_pump_health
        )
        self.fire_pump_control_mode.currentIndexChanged.connect(
            self.update_fire_pump_control_mode
        )
        self.fire_pump_running.toggled.connect(
            self.update_fire_pump_running
        )
        self.fire_pump_zone.currentIndexChanged.connect(
            self.update_fire_pump_zone
        )
        self.fire_pump_fire_water_system.currentIndexChanged.connect(
            self.update_fire_pump_fire_water_system
        )

        self.jockey_pump_x.editingFinished.connect(
            self.update_jockey_pump_geometry
        )
        self.jockey_pump_y.editingFinished.connect(
            self.update_jockey_pump_geometry
        )
        self.jockey_pump_active.toggled.connect(
            self.update_jockey_pump_active
        )
        self.jockey_pump_health.currentIndexChanged.connect(
            self.update_jockey_pump_health
        )
        self.jockey_pump_control_mode.currentIndexChanged.connect(
            self.update_jockey_pump_control_mode
        )
        self.jockey_pump_running.toggled.connect(
            self.update_jockey_pump_running
        )
        self.jockey_pump_zone.currentIndexChanged.connect(
            self.update_jockey_pump_zone
        )
        self.jockey_pump_fire_water_system.currentIndexChanged.connect(
            self.update_jockey_pump_fire_water_system
        )

        self.fire_service_inlet_x.editingFinished.connect(
            self.update_fire_service_inlet_geometry
        )
        self.fire_service_inlet_y.editingFinished.connect(
            self.update_fire_service_inlet_geometry
        )
        self.fire_service_inlet_active.toggled.connect(
            self.update_fire_service_inlet_active
        )
        self.fire_service_inlet_health.currentIndexChanged.connect(
            self.update_fire_service_inlet_health
        )
        self.fire_service_inlet_type.currentIndexChanged.connect(
            self.update_fire_service_inlet_type
        )
        self.fire_service_inlet_zone.currentIndexChanged.connect(
            self.update_fire_service_inlet_zone
        )
        self.fire_service_inlet_fire_water_system.currentIndexChanged.connect(
            self.update_fire_service_inlet_fire_water_system
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

        self.occupant_destination_type.currentIndexChanged.connect(
            self.update_occupant_destination
        )

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)
        self._set_fields_visible(self.occupant_fields, False)

    # =====================================================

    def set_building(self, building):

        self.building = building

        self.refresh()

    # =====================================================

    def set_sandbox_manager(self, sandbox_manager):

        self.sandbox_manager = sandbox_manager

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
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
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
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
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

            self._populate_exit_zone_combo(model)

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
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = stair_item.model

        self.object_type.setText("Stair")
        self.object_id.setText(stair_item.object_id)

        self.object_name.blockSignals(True)
        self.stair_from_x.blockSignals(True)
        self.stair_from_y.blockSignals(True)
        self.stair_to_x.blockSignals(True)
        self.stair_to_y.blockSignals(True)
        self.stair_width.blockSignals(True)
        self.to_floor_combo.blockSignals(True)
        self.stair_from_region_width.blockSignals(True)
        self.stair_from_region_depth.blockSignals(True)
        self.stair_to_region_width.blockSignals(True)
        self.stair_to_region_depth.blockSignals(True)

        # stair_from_zone/stair_to_zone are populated further below via
        # _populate_stair_zone_combo(), which already blocks/unblocks
        # their own signals internally (same convention
        # _populate_to_floor_combo() already uses for to_floor_combo).

        self.object_name.setText(stair_item.object_name)

        if model is not None:

            fx, fy = model.from_position
            tx, ty = model.to_position

            self.stair_from_x.setText(f"{fx:.2f}")
            self.stair_from_y.setText(f"{fy:.2f}")
            self.stair_to_x.setText(f"{tx:.2f}")
            self.stair_to_y.setText(f"{ty:.2f}")

            self.stair_width.setText(
                f"{model.width:.2f}"
            )

            # Observable Stair Perception milestone -- blank means
            # "no observable region authored for this side" (None),
            # never a fabricated 0.00 (which would look like a
            # deliberately zero-sized region rather than "not set").
            if model.from_observable_region is not None:
                self.stair_from_region_width.setText(f"{model.from_observable_region.width:.2f}")
                self.stair_from_region_depth.setText(f"{model.from_observable_region.depth:.2f}")
            else:
                self.stair_from_region_width.setText("")
                self.stair_from_region_depth.setText("")

            if model.to_observable_region is not None:
                self.stair_to_region_width.setText(f"{model.to_observable_region.width:.2f}")
                self.stair_to_region_depth.setText(f"{model.to_observable_region.depth:.2f}")
            else:
                self.stair_to_region_width.setText("")
                self.stair_to_region_depth.setText("")

            if self.building is not None:

                from_floor = self.building.get_floor(
                    model.from_floor_id
                )

                self.stair_from_floor.setText(
                    from_floor.name
                    if from_floor is not None
                    else "-"
                )

            else:

                self.stair_from_floor.setText("-")

            self._populate_to_floor_combo(model)

            self._populate_stair_zone_combo(
                self.stair_from_zone, model.from_floor_id, model.from_zone_id,
            )

            self._populate_stair_zone_combo(
                self.stair_to_zone, model.to_floor_id, model.to_zone_id,
            )

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
        self.stair_from_x.blockSignals(False)
        self.stair_from_y.blockSignals(False)
        self.stair_to_x.blockSignals(False)
        self.stair_to_y.blockSignals(False)
        self.stair_width.blockSignals(False)
        self.to_floor_combo.blockSignals(False)
        self.stair_from_region_width.blockSignals(False)
        self.stair_from_region_depth.blockSignals(False)
        self.stair_to_region_width.blockSignals(False)
        self.stair_to_region_depth.blockSignals(False)

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
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
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
        self.camera_zone.blockSignals(True)
        self.camera_resolution.blockSignals(True)
        self.camera_fps.blockSignals(True)
        self.camera_mode.blockSignals(True)
        self.camera_rtsp.blockSignals(True)
        self.camera_ip.blockSignals(True)
        self.camera_username.blockSignals(True)
        self.camera_password.blockSignals(True)

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

            self._populate_zone_combo(
                self.camera_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )

            self.camera_resolution.setText(model.resolution)
            self.camera_fps.setText(str(model.fps))

            self._refresh_calibration_status(model)

            mode_index = self.camera_mode.findText(model.mode)

            if mode_index != -1:
                self.camera_mode.setCurrentIndex(mode_index)

            self.camera_rtsp.setText(model.connection.rtsp_address)
            self.camera_ip.setText(model.connection.ip_address)
            self.camera_username.setText(model.connection.username)
            self.camera_password.setText(model.connection.password)

            self._update_camera_visibility_stats(model)

        self.object_name.blockSignals(False)
        self.camera_x.blockSignals(False)
        self.camera_y.blockSignals(False)
        self.camera_rotation.blockSignals(False)
        self.camera_fov.blockSignals(False)
        self.camera_range.blockSignals(False)
        self.camera_mount_height.blockSignals(False)
        self.camera_active.blockSignals(False)
        self.camera_zone.blockSignals(False)
        self.camera_resolution.blockSignals(False)
        self.camera_fps.blockSignals(False)
        self.camera_mode.blockSignals(False)
        self.camera_rtsp.blockSignals(False)
        self.camera_ip.blockSignals(False)
        self.camera_username.blockSignals(False)
        self.camera_password.blockSignals(False)

    # =====================================================

    def _refresh_calibration_status(self, model) -> None:

        # Real Camera Calibration & World-Coordinate Validation
        # milestone, Phase 13 -- exactly the three states Phase 13
        # names, never a fourth invented one. profile.quality is None
        # unless a genuine camera_calibration.validation.validate_
        # calibration() run was recorded onto it (see camera_calibration/
        # camera_model.py's own CalibrationQuality docstring) -- this
        # label can only ever say VALIDATED when that genuinely happened.

        profile = self.calibration_registry.get(model.id)
        self.camera_calibration_status.setText(f"CALIBRATION: {calibration_status_text(profile)}")

    # =====================================================

    def _open_calibration_dialog(self) -> None:

        # Real Camera Calibration & World-Coordinate Validation
        # milestone, Phase 13 -- "a dedicated calibration action/dialog
        # is preferable if the UI requires more detail" than the one
        # status line above. Deliberately minimal: load an already-
        # produced calibration JSON (the normal output of
        # scripts/calibrate_camera_scene.py, run separately -- Designer
        # is NOT where correspondence-based fitting happens, see that
        # script's own docstring), or save a quick manual-entry
        # calibration built directly from this Camera Asset's own
        # existing position/mount_height/rotation/horizontal_fov (the
        # same calibration_from_camera() bridge camera_calibration.
        # calibration_loader already establishes) -- never a second,
        # competing calibration UI.

        if self.current_item is None or self.current_item.model is None:
            return

        model = self.current_item.model

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Calibrate Camera -- {model.id}")

        layout = QVBoxLayout(dialog)

        status_label = QLabel()
        layout.addWidget(status_label)

        def _refresh_dialog_status():
            profile = self.calibration_registry.get(model.id)
            if profile is None:
                status_label.setText("Status: NOT CONFIGURED")
            elif profile.quality is None:
                status_label.setText("Status: CONFIGURED -- UNVALIDATED")
            elif profile.quality.rmse_m is None:
                status_label.setText("Status: VALIDATION ATTEMPTED, NO POINTS PROJECTED")
            else:
                status_label.setText(
                    f"Status: VALIDATED -- RMSE {profile.quality.rmse_m:.3f} m "
                    f"({profile.quality.validated_point_count}/{profile.quality.reference_point_count} points)"
                )

        _refresh_dialog_status()

        load_button = QPushButton("Load Calibration JSON...")
        save_manual_button = QPushButton("Save Manual (angle-based) Calibration As...")

        def _load():
            path, _ = QFileDialog.getOpenFileName(dialog, "Load Calibration JSON", "", "JSON Files (*.json)")
            if not path:
                return
            try:
                profile = load_calibration_json(path)
            except CalibrationLoadError as exc:
                QMessageBox.critical(dialog, "Calibration Load Failed", str(exc))
                return
            if profile.camera_id != model.id:
                QMessageBox.warning(
                    dialog, "Camera ID Mismatch",
                    f"This calibration file is for camera_id={profile.camera_id!r}, "
                    f"not this camera ({model.id!r}). Loaded anyway under this camera's own id -- "
                    f"verify this is the file you intended.",
                )
            self.calibration_registry.set(profile)
            _refresh_dialog_status()
            self._refresh_calibration_status(model)

        def _save_manual():
            try:
                width, height = (int(part) for part in model.resolution.lower().split("x"))
            except (ValueError, AttributeError):
                width, height = 1920, 1080
            profile = calibration_from_camera(model, image_width=width, image_height=height)
            path, _ = QFileDialog.getSaveFileName(dialog, "Save Manual Calibration As", "", "JSON Files (*.json)")
            if not path:
                return
            save_calibration_json(profile, path)
            self.calibration_registry.set(profile)
            _refresh_dialog_status()
            self._refresh_calibration_status(model)

        load_button.clicked.connect(_load)
        save_manual_button.clicked.connect(_save_manual)

        layout.addWidget(load_button)
        layout.addWidget(save_manual_button)

        note = QLabel(
            "To fit calibration from measured floor reference points instead of a manual\n"
            "angle entry, use scripts/calibrate_camera_scene.py, then load its output here."
        )
        layout.addWidget(note)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.exec()

    # =====================================================

    def _update_camera_visibility_stats(self, model):

        # Only meaningful once a Building is known (set_building()) --
        # a Camera Item can exist in isolation in a few tests/tools
        # that never call set_building(), same guard every other
        # Building-dependent Camera/Stair field already uses.
        if self.building is None:

            self.camera_visible_zones.setText("-")
            self.camera_partial_zones.setText("-")
            self.camera_hidden_zones.setText("-")
            self.camera_max_visible_distance.setText("-")

            return

        from visibility.engine import VisibilityEngine

        visibility = VisibilityEngine().compute_camera_visibility(model, self.building)

        self.camera_visible_zones.setText(str(len(visibility.visible_zone_ids)))
        self.camera_partial_zones.setText(str(len(visibility.partially_visible_zone_ids)))
        self.camera_hidden_zones.setText(str(len(visibility.hidden_zone_ids)))
        self.camera_max_visible_distance.setText(
            f"{visibility.max_visible_distance:.2f}"
        )

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
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
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
    # Smoke Detector (Building Sensor Network Framework)
    # =====================================================

    def show_smoke_detector(self, smoke_detector_item):

        self.current_item = smoke_detector_item
        self._refresh_handler = self.show_smoke_detector

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, True)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = smoke_detector_item.model

        self.object_type.setText("Smoke Detector")
        self.object_id.setText(smoke_detector_item.object_id)

        self.object_name.blockSignals(True)
        self.smoke_detector_x.blockSignals(True)
        self.smoke_detector_y.blockSignals(True)
        self.smoke_detector_active.blockSignals(True)
        self.smoke_detector_health.blockSignals(True)
        self.smoke_detector_mode.blockSignals(True)
        self.smoke_detector_threshold.blockSignals(True)
        self.smoke_detector_installation_date.blockSignals(True)
        self.smoke_detector_test_level.blockSignals(True)
        self.smoke_detector_zone.blockSignals(True)

        self.object_name.setText(smoke_detector_item.object_name)

        if model is not None:

            px, py = model.position

            self.smoke_detector_x.setText(f"{px:.2f}")
            self.smoke_detector_y.setText(f"{py:.2f}")

            self.smoke_detector_active.setChecked(model.active)

            health_index = self.smoke_detector_health.findText(model.health_status)

            if health_index != -1:
                self.smoke_detector_health.setCurrentIndex(health_index)

            mode_index = self.smoke_detector_mode.findText(model.mode)

            if mode_index != -1:
                self.smoke_detector_mode.setCurrentIndex(mode_index)

            self.smoke_detector_threshold.setText(f"{model.activation_threshold:.2f}")
            self.smoke_detector_installation_date.setText(model.installation_date)

            self._populate_zone_combo(
                self.smoke_detector_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.smoke_detector_zone_warning, model.zone_ids)

            self._refresh_smoke_detector_state(smoke_detector_item)

        self.object_name.blockSignals(False)
        self.smoke_detector_x.blockSignals(False)
        self.smoke_detector_y.blockSignals(False)
        self.smoke_detector_active.blockSignals(False)
        self.smoke_detector_health.blockSignals(False)
        self.smoke_detector_mode.blockSignals(False)
        self.smoke_detector_threshold.blockSignals(False)
        self.smoke_detector_installation_date.blockSignals(False)
        self.smoke_detector_test_level.blockSignals(False)
        self.smoke_detector_zone.blockSignals(False)

    # =====================================================

    def _refresh_smoke_detector_state(self, smoke_detector_item):

        # Recomputes and displays Current State from whatever "Test
        # Smoke Level" is currently entered -- see this class's own
        # module-level note by smoke_detector_test_level on why this
        # manual reading exists (no live hazard simulation is wired
        # into the Designer). Blank input is treated as "no reading"
        # (None), the same convention SmokeDetector.compute_state()
        # itself already documents, never a guessed 0.0.

        model = smoke_detector_item.model

        if model is None:
            return

        text = self.smoke_detector_test_level.text().strip()

        try:
            smoke_level = float(text) if text else None
        except ValueError:
            smoke_level = None

        state = model.compute_state(smoke_level)

        self.smoke_detector_state.setText(state.name)

        smoke_detector_item.current_state = state
        smoke_detector_item.refresh_geometry()

    # =====================================================
    # Heat Detector (Building Sensor Network Framework)
    # =====================================================

    def show_heat_detector(self, heat_detector_item):

        self.current_item = heat_detector_item
        self._refresh_handler = self.show_heat_detector

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, True)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = heat_detector_item.model

        self.object_type.setText("Heat Detector")
        self.object_id.setText(heat_detector_item.object_id)

        self.object_name.blockSignals(True)
        self.heat_detector_x.blockSignals(True)
        self.heat_detector_y.blockSignals(True)
        self.heat_detector_active.blockSignals(True)
        self.heat_detector_health.blockSignals(True)
        self.heat_detector_mode.blockSignals(True)
        self.heat_detector_threshold.blockSignals(True)
        self.heat_detector_installation_date.blockSignals(True)
        self.heat_detector_test_temperature.blockSignals(True)
        self.heat_detector_zone.blockSignals(True)

        self.object_name.setText(heat_detector_item.object_name)

        if model is not None:

            px, py = model.position

            self.heat_detector_x.setText(f"{px:.2f}")
            self.heat_detector_y.setText(f"{py:.2f}")

            self.heat_detector_active.setChecked(model.active)

            health_index = self.heat_detector_health.findText(model.health_status)

            if health_index != -1:
                self.heat_detector_health.setCurrentIndex(health_index)

            mode_index = self.heat_detector_mode.findText(model.mode)

            if mode_index != -1:
                self.heat_detector_mode.setCurrentIndex(mode_index)

            self.heat_detector_threshold.setText(f"{model.activation_threshold:.2f}")
            self.heat_detector_installation_date.setText(model.installation_date)

            self._populate_zone_combo(
                self.heat_detector_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.heat_detector_zone_warning, model.zone_ids)

            self._refresh_heat_detector_state(heat_detector_item)

        self.object_name.blockSignals(False)
        self.heat_detector_x.blockSignals(False)
        self.heat_detector_y.blockSignals(False)
        self.heat_detector_active.blockSignals(False)
        self.heat_detector_health.blockSignals(False)
        self.heat_detector_mode.blockSignals(False)
        self.heat_detector_threshold.blockSignals(False)
        self.heat_detector_installation_date.blockSignals(False)
        self.heat_detector_test_temperature.blockSignals(False)
        self.heat_detector_zone.blockSignals(False)

    # =====================================================

    def _refresh_heat_detector_state(self, heat_detector_item):

        model = heat_detector_item.model

        if model is None:
            return

        text = self.heat_detector_test_temperature.text().strip()

        try:
            temperature = float(text) if text else None
        except ValueError:
            temperature = None

        state = model.compute_state(temperature)

        self.heat_detector_state.setText(state.name)

        heat_detector_item.current_state = state
        heat_detector_item.refresh_geometry()

    # =====================================================
    # Speaker (Zoned Voice Evacuation & Speaker Network Framework)
    # =====================================================

    def show_speaker(self, speaker_item):

        self.current_item = speaker_item
        self._refresh_handler = self.show_speaker

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, True)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = speaker_item.model

        self.object_type.setText("Speaker")
        self.object_id.setText(speaker_item.object_id)

        self.object_name.blockSignals(True)
        self.speaker_x.blockSignals(True)
        self.speaker_y.blockSignals(True)
        self.speaker_active.blockSignals(True)
        self.speaker_health.blockSignals(True)
        self.speaker_mode.blockSignals(True)
        self.speaker_type.blockSignals(True)
        self.speaker_volume.blockSignals(True)
        self.speaker_installation_date.blockSignals(True)
        self.speaker_zones.blockSignals(True)

        self.object_name.setText(speaker_item.object_name)

        if model is not None:

            px, py = model.position

            self.speaker_x.setText(f"{px:.2f}")
            self.speaker_y.setText(f"{py:.2f}")

            self.speaker_active.setChecked(model.active)

            health_index = self.speaker_health.findText(model.health_status)

            if health_index != -1:
                self.speaker_health.setCurrentIndex(health_index)

            mode_index = self.speaker_mode.findText(model.mode)

            if mode_index != -1:
                self.speaker_mode.setCurrentIndex(mode_index)

            type_index = self.speaker_type.findText(model.speaker_type)

            if type_index != -1:
                self.speaker_type.setCurrentIndex(type_index)

            self.speaker_volume.setText(f"{model.volume_level:.1f}")
            self.speaker_installation_date.setText(model.installation_date)

            self._populate_zone_checklist(self.speaker_zones, model, model.zone_ids)
            self._update_zone_warning(self.speaker_zone_warning, model.zone_ids)

        self.object_name.blockSignals(False)
        self.speaker_x.blockSignals(False)
        self.speaker_y.blockSignals(False)
        self.speaker_active.blockSignals(False)
        self.speaker_health.blockSignals(False)
        self.speaker_mode.blockSignals(False)
        self.speaker_type.blockSignals(False)
        self.speaker_volume.blockSignals(False)
        self.speaker_installation_date.blockSignals(False)
        self.speaker_zones.blockSignals(False)

    # =====================================================
    # Dynamic Evacuation Sign (Live Dynamic Evacuation Signage milestone)
    # =====================================================

    def show_sign(self, sign_item):

        self.current_item = sign_item
        self._refresh_handler = self.show_sign

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, True)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = sign_item.model

        self.object_type.setText("Dynamic Sign")
        self.object_id.setText(sign_item.object_id)

        self.object_name.blockSignals(True)
        self.sign_x.blockSignals(True)
        self.sign_y.blockSignals(True)
        self.sign_orientation.blockSignals(True)
        self.sign_active.blockSignals(True)
        self.sign_zone.blockSignals(True)

        self.object_name.setText(sign_item.object_name)

        if model is not None:

            px, py = model.position

            self.sign_x.setText(f"{px:.2f}")
            self.sign_y.setText(f"{py:.2f}")

            self.sign_orientation.setText(f"{model.orientation:.1f}")

            self.sign_active.setChecked(model.active)

            self._populate_zone_combo(
                self.sign_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )

        self.object_name.blockSignals(False)
        self.sign_x.blockSignals(False)
        self.sign_y.blockSignals(False)
        self.sign_orientation.blockSignals(False)
        self.sign_active.blockSignals(False)
        self.sign_zone.blockSignals(False)

    # =====================================================
    # Manual Call Point (Manual Call Points & Emergency Lighting
    # milestone)
    # =====================================================

    def show_manual_call_point(self, mcp_item):

        self.current_item = mcp_item
        self._refresh_handler = self.show_manual_call_point

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, True)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = mcp_item.model

        self.object_type.setText("Manual Call Point")
        self.object_id.setText(mcp_item.object_id)

        self.object_name.blockSignals(True)
        self.mcp_x.blockSignals(True)
        self.mcp_y.blockSignals(True)
        self.mcp_active.blockSignals(True)
        self.mcp_activated.blockSignals(True)
        self.mcp_health.blockSignals(True)
        self.mcp_mode.blockSignals(True)
        self.mcp_installation_date.blockSignals(True)
        self.mcp_zone.blockSignals(True)

        self.object_name.setText(mcp_item.object_name)

        if model is not None:

            px, py = model.position

            self.mcp_x.setText(f"{px:.2f}")
            self.mcp_y.setText(f"{py:.2f}")

            self.mcp_active.setChecked(model.active)
            self.mcp_activated.setChecked(model.activated)

            health_index = self.mcp_health.findText(model.health_status)

            if health_index != -1:
                self.mcp_health.setCurrentIndex(health_index)

            mode_index = self.mcp_mode.findText(model.mode)

            if mode_index != -1:
                self.mcp_mode.setCurrentIndex(mode_index)

            self.mcp_installation_date.setText(model.installation_date)

            self._populate_zone_combo(
                self.mcp_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.mcp_zone_warning, model.zone_ids)

            self._refresh_mcp_state(mcp_item)

        self.object_name.blockSignals(False)
        self.mcp_x.blockSignals(False)
        self.mcp_y.blockSignals(False)
        self.mcp_active.blockSignals(False)
        self.mcp_activated.blockSignals(False)
        self.mcp_health.blockSignals(False)
        self.mcp_mode.blockSignals(False)
        self.mcp_installation_date.blockSignals(False)
        self.mcp_zone.blockSignals(False)

    # =====================================================

    def _refresh_mcp_state(self, mcp_item):

        model = mcp_item.model

        if model is None:
            return

        state = model.compute_state()

        self.mcp_state.setText(state.name)

        mcp_item.current_state = state
        mcp_item.refresh_geometry()

    # =====================================================

    def update_mcp_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.mcp_x.text())
            y = float(self.mcp_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_mcp_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.mcp_active.isChecked()
        self._refresh_mcp_state(self.current_item)

    # =====================================================

    def update_mcp_activated(self):

        if self.current_item is None or self.current_item.model is None:
            return

        if self.mcp_activated.isChecked():
            self.current_item.model.activate()
        else:
            self.current_item.model.restore()

        self._refresh_mcp_state(self.current_item)

    # =====================================================

    def update_mcp_health(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.health_status = self.mcp_health.itemText(index)
        self._refresh_mcp_state(self.current_item)

    # =====================================================

    def update_mcp_mode(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.mode = self.mcp_mode.itemText(index)

    # =====================================================

    def update_mcp_installation_date(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.installation_date = self.mcp_installation_date.text()

    # =====================================================

    def update_mcp_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.mcp_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.mcp_zone_warning, self.current_item.model.zone_ids)

    # =====================================================
    # Emergency Light (Manual Call Points & Emergency Lighting
    # milestone)
    # =====================================================

    def show_emergency_light(self, light_item):

        self.current_item = light_item
        self._refresh_handler = self.show_emergency_light

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, True)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = light_item.model

        self.object_type.setText("Emergency Light")
        self.object_id.setText(light_item.object_id)

        self.object_name.blockSignals(True)
        self.emergency_light_x.blockSignals(True)
        self.emergency_light_y.blockSignals(True)
        self.emergency_light_active.blockSignals(True)
        self.emergency_light_health.blockSignals(True)
        self.emergency_light_type.blockSignals(True)
        self.emergency_light_zone.blockSignals(True)

        self.object_name.setText(light_item.object_name)

        if model is not None:

            px, py = model.position

            self.emergency_light_x.setText(f"{px:.2f}")
            self.emergency_light_y.setText(f"{py:.2f}")

            self.emergency_light_active.setChecked(model.active)

            health_index = self.emergency_light_health.findText(model.health_status)

            if health_index != -1:
                self.emergency_light_health.setCurrentIndex(health_index)

            type_index = self.emergency_light_type.findText(model.light_type)

            if type_index != -1:
                self.emergency_light_type.setCurrentIndex(type_index)

            self._populate_zone_combo(
                self.emergency_light_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.emergency_light_zone_warning, model.zone_ids)

            self._refresh_emergency_light_availability(light_item)

        self.object_name.blockSignals(False)
        self.emergency_light_x.blockSignals(False)
        self.emergency_light_y.blockSignals(False)
        self.emergency_light_active.blockSignals(False)
        self.emergency_light_health.blockSignals(False)
        self.emergency_light_type.blockSignals(False)
        self.emergency_light_zone.blockSignals(False)

    # =====================================================

    def _refresh_emergency_light_availability(self, light_item):

        model = light_item.model

        if model is None:
            return

        availability = model.compute_availability()

        self.emergency_light_availability.setText(availability)

        light_item.current_availability = availability
        light_item.refresh_geometry()

    # =====================================================

    def update_emergency_light_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.emergency_light_x.text())
            y = float(self.emergency_light_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_emergency_light_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.emergency_light_active.isChecked()
        self._refresh_emergency_light_availability(self.current_item)

    # =====================================================

    def update_emergency_light_health(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.health_status = self.emergency_light_health.itemText(index)
        self._refresh_emergency_light_availability(self.current_item)

    # =====================================================

    def update_emergency_light_type(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.light_type = self.emergency_light_type.itemText(index)

    # =====================================================

    def update_emergency_light_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.emergency_light_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.emergency_light_zone_warning, self.current_item.model.zone_ids)

    # =====================================================
    # Sprinkler (Fire Suppression & Water-Based Safety Asset Digital
    # Twin milestone)
    # =====================================================

    def show_sprinkler(self, sprinkler_item):

        self.current_item = sprinkler_item
        self._refresh_handler = self.show_sprinkler

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, True)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = sprinkler_item.model

        self.object_type.setText("Sprinkler")
        self.object_id.setText(sprinkler_item.object_id)

        self.object_name.blockSignals(True)
        self.sprinkler_x.blockSignals(True)
        self.sprinkler_y.blockSignals(True)
        self.sprinkler_active.blockSignals(True)
        self.sprinkler_health.blockSignals(True)
        self.sprinkler_mode.blockSignals(True)
        self.sprinkler_activation_temperature.blockSignals(True)
        self.sprinkler_installation_date.blockSignals(True)
        self.sprinkler_test_temperature.blockSignals(True)
        self.sprinkler_zone.blockSignals(True)

        self.object_name.setText(sprinkler_item.object_name)

        if model is not None:

            px, py = model.position

            self.sprinkler_x.setText(f"{px:.2f}")
            self.sprinkler_y.setText(f"{py:.2f}")

            self.sprinkler_active.setChecked(model.active)

            health_index = self.sprinkler_health.findText(model.health_status)

            if health_index != -1:
                self.sprinkler_health.setCurrentIndex(health_index)

            mode_index = self.sprinkler_mode.findText(model.mode)

            if mode_index != -1:
                self.sprinkler_mode.setCurrentIndex(mode_index)

            self.sprinkler_activation_temperature.setText(f"{model.activation_temperature:.2f}")
            self.sprinkler_installation_date.setText(model.installation_date)

            self._populate_zone_combo(
                self.sprinkler_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.sprinkler_zone_warning, model.zone_ids)

            self._populate_fire_water_system_combo(self.sprinkler_fire_water_system, model.id, "sprinkler_ids")

            self._refresh_sprinkler_state(sprinkler_item)

        self.object_name.blockSignals(False)
        self.sprinkler_x.blockSignals(False)
        self.sprinkler_y.blockSignals(False)
        self.sprinkler_active.blockSignals(False)
        self.sprinkler_health.blockSignals(False)
        self.sprinkler_mode.blockSignals(False)
        self.sprinkler_activation_temperature.blockSignals(False)
        self.sprinkler_installation_date.blockSignals(False)
        self.sprinkler_test_temperature.blockSignals(False)
        self.sprinkler_zone.blockSignals(False)

    # =====================================================

    def _refresh_sprinkler_state(self, sprinkler_item):

        model = sprinkler_item.model

        if model is None:
            return

        text = self.sprinkler_test_temperature.text().strip()

        try:
            temperature = float(text) if text else None
        except ValueError:
            temperature = None

        state = model.compute_state(temperature)

        self.sprinkler_state.setText(state.name)

        sprinkler_item.current_state = state
        sprinkler_item.refresh_geometry()

    # =====================================================

    def update_sprinkler_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.sprinkler_x.text())
            y = float(self.sprinkler_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_sprinkler_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.sprinkler_active.isChecked()
        self._refresh_sprinkler_state(self.current_item)

    # =====================================================

    def update_sprinkler_health(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.health_status = self.sprinkler_health.itemText(index)
        self._refresh_sprinkler_state(self.current_item)

    # =====================================================

    def update_sprinkler_mode(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.mode = self.sprinkler_mode.itemText(index)

    # =====================================================

    def update_sprinkler_activation_temperature(self):

        if self.current_item is None or self.current_item.model is None:
            return

        try:
            threshold = float(self.sprinkler_activation_temperature.text())
        except ValueError:
            self.refresh()
            return

        self.current_item.model.activation_temperature = threshold
        self._refresh_sprinkler_state(self.current_item)

    # =====================================================

    def update_sprinkler_installation_date(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.installation_date = self.sprinkler_installation_date.text()

    # =====================================================

    def update_sprinkler_test_reading(self):

        if self.current_item is None:
            return

        self._refresh_sprinkler_state(self.current_item)

    # =====================================================

    def update_sprinkler_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.sprinkler_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.sprinkler_zone_warning, self.current_item.model.zone_ids)

    # =====================================================

    def update_sprinkler_fire_water_system(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self._update_asset_fire_water_system(
            self.sprinkler_fire_water_system, self.current_item.model.id, "sprinkler_ids",
        )

    # =====================================================
    # Fire Extinguisher (Fire Suppression & Water-Based Safety Asset
    # Digital Twin milestone) -- a passive resource, no current-state
    # concept, only availability (mirrors Emergency Light exactly).
    # =====================================================

    def show_fire_extinguisher(self, extinguisher_item):

        self.current_item = extinguisher_item
        self._refresh_handler = self.show_fire_extinguisher

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, True)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = extinguisher_item.model

        self.object_type.setText("Fire Extinguisher")
        self.object_id.setText(extinguisher_item.object_id)

        self.object_name.blockSignals(True)
        self.fire_extinguisher_x.blockSignals(True)
        self.fire_extinguisher_y.blockSignals(True)
        self.fire_extinguisher_active.blockSignals(True)
        self.fire_extinguisher_health.blockSignals(True)
        self.fire_extinguisher_type.blockSignals(True)
        self.fire_extinguisher_zone.blockSignals(True)

        self.object_name.setText(extinguisher_item.object_name)

        if model is not None:

            px, py = model.position

            self.fire_extinguisher_x.setText(f"{px:.2f}")
            self.fire_extinguisher_y.setText(f"{py:.2f}")

            self.fire_extinguisher_active.setChecked(model.active)

            health_index = self.fire_extinguisher_health.findText(model.health_status)

            if health_index != -1:
                self.fire_extinguisher_health.setCurrentIndex(health_index)

            type_index = self.fire_extinguisher_type.findText(model.extinguisher_type)

            if type_index != -1:
                self.fire_extinguisher_type.setCurrentIndex(type_index)

            self._populate_zone_combo(
                self.fire_extinguisher_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.fire_extinguisher_zone_warning, model.zone_ids)

            self._refresh_fire_extinguisher_availability(extinguisher_item)

        self.object_name.blockSignals(False)
        self.fire_extinguisher_x.blockSignals(False)
        self.fire_extinguisher_y.blockSignals(False)
        self.fire_extinguisher_active.blockSignals(False)
        self.fire_extinguisher_health.blockSignals(False)
        self.fire_extinguisher_type.blockSignals(False)
        self.fire_extinguisher_zone.blockSignals(False)

    # =====================================================

    def _refresh_fire_extinguisher_availability(self, extinguisher_item):

        model = extinguisher_item.model

        if model is None:
            return

        availability = model.compute_availability()

        self.fire_extinguisher_availability.setText(availability)

        extinguisher_item.current_availability = availability
        extinguisher_item.refresh_geometry()

    # =====================================================

    def update_fire_extinguisher_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.fire_extinguisher_x.text())
            y = float(self.fire_extinguisher_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_fire_extinguisher_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.fire_extinguisher_active.isChecked()
        self._refresh_fire_extinguisher_availability(self.current_item)

    # =====================================================

    def update_fire_extinguisher_health(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.health_status = self.fire_extinguisher_health.itemText(index)
        self._refresh_fire_extinguisher_availability(self.current_item)

    # =====================================================

    def update_fire_extinguisher_type(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.extinguisher_type = self.fire_extinguisher_type.itemText(index)

    # =====================================================

    def update_fire_extinguisher_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.fire_extinguisher_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.fire_extinguisher_zone_warning, self.current_item.model.zone_ids)

    # =====================================================
    # Fire Hydrant / Landing Valve (Fire Suppression & Water-Based
    # Safety Asset Digital Twin milestone)
    # =====================================================

    def show_fire_hydrant(self, hydrant_item):

        self.current_item = hydrant_item
        self._refresh_handler = self.show_fire_hydrant

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, True)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = hydrant_item.model

        self.object_type.setText("Fire Hydrant")
        self.object_id.setText(hydrant_item.object_id)

        self.object_name.blockSignals(True)
        self.fire_hydrant_x.blockSignals(True)
        self.fire_hydrant_y.blockSignals(True)
        self.fire_hydrant_active.blockSignals(True)
        self.fire_hydrant_health.blockSignals(True)
        self.fire_hydrant_type.blockSignals(True)
        self.fire_hydrant_zone.blockSignals(True)

        self.object_name.setText(hydrant_item.object_name)

        if model is not None:

            px, py = model.position

            self.fire_hydrant_x.setText(f"{px:.2f}")
            self.fire_hydrant_y.setText(f"{py:.2f}")

            self.fire_hydrant_active.setChecked(model.active)

            health_index = self.fire_hydrant_health.findText(model.health_status)

            if health_index != -1:
                self.fire_hydrant_health.setCurrentIndex(health_index)

            type_index = self.fire_hydrant_type.findText(model.hydrant_type)

            if type_index != -1:
                self.fire_hydrant_type.setCurrentIndex(type_index)

            self._populate_zone_combo(
                self.fire_hydrant_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.fire_hydrant_zone_warning, model.zone_ids)

            self._populate_fire_water_system_combo(self.fire_hydrant_fire_water_system, model.id, "hydrant_ids")

            self._refresh_fire_hydrant_availability(hydrant_item)

        self.object_name.blockSignals(False)
        self.fire_hydrant_x.blockSignals(False)
        self.fire_hydrant_y.blockSignals(False)
        self.fire_hydrant_active.blockSignals(False)
        self.fire_hydrant_health.blockSignals(False)
        self.fire_hydrant_type.blockSignals(False)
        self.fire_hydrant_zone.blockSignals(False)

    # =====================================================

    def _refresh_fire_hydrant_availability(self, hydrant_item):

        model = hydrant_item.model

        if model is None:
            return

        availability = model.compute_availability()

        self.fire_hydrant_availability.setText(availability)

        hydrant_item.current_availability = availability
        hydrant_item.refresh_geometry()

    # =====================================================

    def update_fire_hydrant_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.fire_hydrant_x.text())
            y = float(self.fire_hydrant_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_fire_hydrant_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.fire_hydrant_active.isChecked()
        self._refresh_fire_hydrant_availability(self.current_item)

    # =====================================================

    def update_fire_hydrant_health(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.health_status = self.fire_hydrant_health.itemText(index)
        self._refresh_fire_hydrant_availability(self.current_item)

    # =====================================================

    def update_fire_hydrant_type(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.hydrant_type = self.fire_hydrant_type.itemText(index)

    # =====================================================

    def update_fire_hydrant_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.fire_hydrant_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.fire_hydrant_zone_warning, self.current_item.model.zone_ids)

    # =====================================================

    def update_fire_hydrant_fire_water_system(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self._update_asset_fire_water_system(
            self.fire_hydrant_fire_water_system, self.current_item.model.id, "hydrant_ids",
        )

    # =====================================================
    # Hose Reel (Fire Suppression & Water-Based Safety Asset Digital
    # Twin milestone)
    # =====================================================

    def show_hose_reel(self, hose_reel_item):

        self.current_item = hose_reel_item
        self._refresh_handler = self.show_hose_reel

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, True)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = hose_reel_item.model

        self.object_type.setText("Hose Reel")
        self.object_id.setText(hose_reel_item.object_id)

        self.object_name.blockSignals(True)
        self.hose_reel_x.blockSignals(True)
        self.hose_reel_y.blockSignals(True)
        self.hose_reel_active.blockSignals(True)
        self.hose_reel_health.blockSignals(True)
        self.hose_reel_zone.blockSignals(True)

        self.object_name.setText(hose_reel_item.object_name)

        if model is not None:

            px, py = model.position

            self.hose_reel_x.setText(f"{px:.2f}")
            self.hose_reel_y.setText(f"{py:.2f}")

            self.hose_reel_active.setChecked(model.active)

            health_index = self.hose_reel_health.findText(model.health_status)

            if health_index != -1:
                self.hose_reel_health.setCurrentIndex(health_index)

            self._populate_zone_combo(
                self.hose_reel_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.hose_reel_zone_warning, model.zone_ids)

            self._populate_fire_water_system_combo(self.hose_reel_fire_water_system, model.id, "hose_reel_ids")

            self._refresh_hose_reel_availability(hose_reel_item)

        self.object_name.blockSignals(False)
        self.hose_reel_x.blockSignals(False)
        self.hose_reel_y.blockSignals(False)
        self.hose_reel_active.blockSignals(False)
        self.hose_reel_health.blockSignals(False)
        self.hose_reel_zone.blockSignals(False)

    # =====================================================

    def _refresh_hose_reel_availability(self, hose_reel_item):

        model = hose_reel_item.model

        if model is None:
            return

        availability = model.compute_availability()

        self.hose_reel_availability.setText(availability)

        hose_reel_item.current_availability = availability
        hose_reel_item.refresh_geometry()

    # =====================================================

    def update_hose_reel_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.hose_reel_x.text())
            y = float(self.hose_reel_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_hose_reel_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.hose_reel_active.isChecked()
        self._refresh_hose_reel_availability(self.current_item)

    # =====================================================

    def update_hose_reel_health(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.health_status = self.hose_reel_health.itemText(index)
        self._refresh_hose_reel_availability(self.current_item)

    # =====================================================

    def update_hose_reel_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.hose_reel_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.hose_reel_zone_warning, self.current_item.model.zone_ids)

    # =====================================================

    def update_hose_reel_fire_water_system(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self._update_asset_fire_water_system(
            self.hose_reel_fire_water_system, self.current_item.model.id, "hose_reel_ids",
        )

    # =====================================================
    # Fire Water Tank (Fire Water Supply & Suppression Infrastructure
    # milestone)
    # =====================================================

    def show_fire_water_tank(self, tank_item):

        self.current_item = tank_item
        self._refresh_handler = self.show_fire_water_tank

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, True)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = tank_item.model

        self.object_type.setText("Fire Water Tank")
        self.object_id.setText(tank_item.object_id)

        self.object_name.blockSignals(True)
        self.fire_water_tank_x.blockSignals(True)
        self.fire_water_tank_y.blockSignals(True)
        self.fire_water_tank_active.blockSignals(True)
        self.fire_water_tank_health.blockSignals(True)
        self.fire_water_tank_capacity.blockSignals(True)
        self.fire_water_tank_level.blockSignals(True)
        self.fire_water_tank_zone.blockSignals(True)

        self.object_name.setText(tank_item.object_name)

        if model is not None:

            px, py = model.position

            self.fire_water_tank_x.setText(f"{px:.2f}")
            self.fire_water_tank_y.setText(f"{py:.2f}")

            self.fire_water_tank_active.setChecked(model.active)

            health_index = self.fire_water_tank_health.findText(model.health_status)

            if health_index != -1:
                self.fire_water_tank_health.setCurrentIndex(health_index)

            self.fire_water_tank_capacity.setText(f"{model.capacity_liters:.2f}")
            self.fire_water_tank_level.setText(
                f"{model.current_level_liters:.2f}" if model.current_level_liters is not None else ""
            )

            self._populate_zone_combo(
                self.fire_water_tank_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.fire_water_tank_zone_warning, model.zone_ids)

            self._populate_fire_water_system_combo(self.fire_water_tank_fire_water_system, model.id, "tank_ids")

            self._refresh_fire_water_tank_state(tank_item)

        self.object_name.blockSignals(False)
        self.fire_water_tank_x.blockSignals(False)
        self.fire_water_tank_y.blockSignals(False)
        self.fire_water_tank_active.blockSignals(False)
        self.fire_water_tank_health.blockSignals(False)
        self.fire_water_tank_capacity.blockSignals(False)
        self.fire_water_tank_level.blockSignals(False)
        self.fire_water_tank_zone.blockSignals(False)

    # =====================================================

    def _refresh_fire_water_tank_state(self, tank_item):

        model = tank_item.model

        if model is None:
            return

        state = model.compute_state()

        self.fire_water_tank_state.setText(state)

        tank_item.current_state = state
        tank_item.refresh_geometry()

    # =====================================================

    def update_fire_water_tank_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.fire_water_tank_x.text())
            y = float(self.fire_water_tank_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_fire_water_tank_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.fire_water_tank_active.isChecked()
        self._refresh_fire_water_tank_state(self.current_item)

    # =====================================================

    def update_fire_water_tank_health(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.health_status = self.fire_water_tank_health.itemText(index)
        self._refresh_fire_water_tank_state(self.current_item)

    # =====================================================

    def update_fire_water_tank_capacity(self):

        if self.current_item is None or self.current_item.model is None:
            return

        try:
            capacity = float(self.fire_water_tank_capacity.text())
        except ValueError:
            self.refresh()
            return

        self.current_item.model.capacity_liters = capacity
        self._refresh_fire_water_tank_state(self.current_item)

    # =====================================================

    def update_fire_water_tank_level(self):

        if self.current_item is None or self.current_item.model is None:
            return

        text = self.fire_water_tank_level.text().strip()

        if not text:
            self.current_item.model.current_level_liters = None
            self._refresh_fire_water_tank_state(self.current_item)
            return

        try:
            level = float(text)
        except ValueError:
            self.refresh()
            return

        self.current_item.model.current_level_liters = level
        self._refresh_fire_water_tank_state(self.current_item)

    # =====================================================

    def update_fire_water_tank_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.fire_water_tank_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.fire_water_tank_zone_warning, self.current_item.model.zone_ids)

    # =====================================================

    def update_fire_water_tank_fire_water_system(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self._update_asset_fire_water_system(
            self.fire_water_tank_fire_water_system, self.current_item.model.id, "tank_ids",
        )

    # =====================================================
    # Fire Pump (Fire Water Supply & Suppression Infrastructure
    # milestone)
    # =====================================================

    def show_fire_pump(self, pump_item):

        self.current_item = pump_item
        self._refresh_handler = self.show_fire_pump

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, True)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = pump_item.model

        self.object_type.setText("Fire Pump")
        self.object_id.setText(pump_item.object_id)

        self.object_name.blockSignals(True)
        self.fire_pump_x.blockSignals(True)
        self.fire_pump_y.blockSignals(True)
        self.fire_pump_active.blockSignals(True)
        self.fire_pump_health.blockSignals(True)
        self.fire_pump_control_mode.blockSignals(True)
        self.fire_pump_running.blockSignals(True)
        self.fire_pump_zone.blockSignals(True)

        self.object_name.setText(pump_item.object_name)

        if model is not None:

            px, py = model.position

            self.fire_pump_x.setText(f"{px:.2f}")
            self.fire_pump_y.setText(f"{py:.2f}")

            self.fire_pump_active.setChecked(model.active)

            health_index = self.fire_pump_health.findText(model.health_status)

            if health_index != -1:
                self.fire_pump_health.setCurrentIndex(health_index)

            control_mode_index = self.fire_pump_control_mode.findText(model.control_mode)

            if control_mode_index != -1:
                self.fire_pump_control_mode.setCurrentIndex(control_mode_index)

            self.fire_pump_running.setChecked(model.running)

            self._populate_zone_combo(
                self.fire_pump_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.fire_pump_zone_warning, model.zone_ids)

            self._populate_fire_water_system_combo(self.fire_pump_fire_water_system, model.id, "pump_ids")

            self._refresh_fire_pump_state(pump_item)

        self.object_name.blockSignals(False)
        self.fire_pump_x.blockSignals(False)
        self.fire_pump_y.blockSignals(False)
        self.fire_pump_active.blockSignals(False)
        self.fire_pump_health.blockSignals(False)
        self.fire_pump_control_mode.blockSignals(False)
        self.fire_pump_running.blockSignals(False)
        self.fire_pump_zone.blockSignals(False)

    # =====================================================

    def _refresh_fire_pump_state(self, pump_item):

        model = pump_item.model

        if model is None:
            return

        state = model.compute_state()

        self.fire_pump_state.setText(state)

        pump_item.current_state = state
        pump_item.refresh_geometry()

    # =====================================================

    def update_fire_pump_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.fire_pump_x.text())
            y = float(self.fire_pump_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_fire_pump_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.fire_pump_active.isChecked()
        self._refresh_fire_pump_state(self.current_item)

    # =====================================================

    def update_fire_pump_health(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.health_status = self.fire_pump_health.itemText(index)
        self._refresh_fire_pump_state(self.current_item)

    # =====================================================

    def update_fire_pump_control_mode(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.control_mode = self.fire_pump_control_mode.itemText(index)

    # =====================================================

    def update_fire_pump_running(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.running = self.fire_pump_running.isChecked()
        self._refresh_fire_pump_state(self.current_item)

    # =====================================================

    def update_fire_pump_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.fire_pump_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.fire_pump_zone_warning, self.current_item.model.zone_ids)

    # =====================================================

    def update_fire_pump_fire_water_system(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self._update_asset_fire_water_system(
            self.fire_pump_fire_water_system, self.current_item.model.id, "pump_ids",
        )

    # =====================================================
    # Jockey Pump (Fire Water Supply & Suppression Infrastructure
    # milestone) -- same PumpAsset shape as Fire Pump above.
    # =====================================================

    def show_jockey_pump(self, jockey_pump_item):

        self.current_item = jockey_pump_item
        self._refresh_handler = self.show_jockey_pump

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, True)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = jockey_pump_item.model

        self.object_type.setText("Jockey Pump")
        self.object_id.setText(jockey_pump_item.object_id)

        self.object_name.blockSignals(True)
        self.jockey_pump_x.blockSignals(True)
        self.jockey_pump_y.blockSignals(True)
        self.jockey_pump_active.blockSignals(True)
        self.jockey_pump_health.blockSignals(True)
        self.jockey_pump_control_mode.blockSignals(True)
        self.jockey_pump_running.blockSignals(True)
        self.jockey_pump_zone.blockSignals(True)

        self.object_name.setText(jockey_pump_item.object_name)

        if model is not None:

            px, py = model.position

            self.jockey_pump_x.setText(f"{px:.2f}")
            self.jockey_pump_y.setText(f"{py:.2f}")

            self.jockey_pump_active.setChecked(model.active)

            health_index = self.jockey_pump_health.findText(model.health_status)

            if health_index != -1:
                self.jockey_pump_health.setCurrentIndex(health_index)

            control_mode_index = self.jockey_pump_control_mode.findText(model.control_mode)

            if control_mode_index != -1:
                self.jockey_pump_control_mode.setCurrentIndex(control_mode_index)

            self.jockey_pump_running.setChecked(model.running)

            self._populate_zone_combo(
                self.jockey_pump_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.jockey_pump_zone_warning, model.zone_ids)

            self._populate_fire_water_system_combo(self.jockey_pump_fire_water_system, model.id, "jockey_pump_ids")

            self._refresh_jockey_pump_state(jockey_pump_item)

        self.object_name.blockSignals(False)
        self.jockey_pump_x.blockSignals(False)
        self.jockey_pump_y.blockSignals(False)
        self.jockey_pump_active.blockSignals(False)
        self.jockey_pump_health.blockSignals(False)
        self.jockey_pump_control_mode.blockSignals(False)
        self.jockey_pump_running.blockSignals(False)
        self.jockey_pump_zone.blockSignals(False)

    # =====================================================

    def _refresh_jockey_pump_state(self, jockey_pump_item):

        model = jockey_pump_item.model

        if model is None:
            return

        state = model.compute_state()

        self.jockey_pump_state.setText(state)

        jockey_pump_item.current_state = state
        jockey_pump_item.refresh_geometry()

    # =====================================================

    def update_jockey_pump_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.jockey_pump_x.text())
            y = float(self.jockey_pump_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_jockey_pump_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.jockey_pump_active.isChecked()
        self._refresh_jockey_pump_state(self.current_item)

    # =====================================================

    def update_jockey_pump_health(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.health_status = self.jockey_pump_health.itemText(index)
        self._refresh_jockey_pump_state(self.current_item)

    # =====================================================

    def update_jockey_pump_control_mode(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.control_mode = self.jockey_pump_control_mode.itemText(index)

    # =====================================================

    def update_jockey_pump_running(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.running = self.jockey_pump_running.isChecked()
        self._refresh_jockey_pump_state(self.current_item)

    # =====================================================

    def update_jockey_pump_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.jockey_pump_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.jockey_pump_zone_warning, self.current_item.model.zone_ids)

    # =====================================================

    def update_jockey_pump_fire_water_system(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self._update_asset_fire_water_system(
            self.jockey_pump_fire_water_system, self.current_item.model.id, "jockey_pump_ids",
        )

    # =====================================================
    # Fire Service Inlet / Breeching Inlet (Fire Water Supply &
    # Suppression Infrastructure milestone)
    # =====================================================

    def show_fire_service_inlet(self, inlet_item):

        self.current_item = inlet_item
        self._refresh_handler = self.show_fire_service_inlet

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, True)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)

        model = inlet_item.model

        self.object_type.setText("Fire Service Inlet")
        self.object_id.setText(inlet_item.object_id)

        self.object_name.blockSignals(True)
        self.fire_service_inlet_x.blockSignals(True)
        self.fire_service_inlet_y.blockSignals(True)
        self.fire_service_inlet_active.blockSignals(True)
        self.fire_service_inlet_health.blockSignals(True)
        self.fire_service_inlet_type.blockSignals(True)
        self.fire_service_inlet_zone.blockSignals(True)

        self.object_name.setText(inlet_item.object_name)

        if model is not None:

            px, py = model.position

            self.fire_service_inlet_x.setText(f"{px:.2f}")
            self.fire_service_inlet_y.setText(f"{py:.2f}")

            self.fire_service_inlet_active.setChecked(model.active)

            health_index = self.fire_service_inlet_health.findText(model.health_status)

            if health_index != -1:
                self.fire_service_inlet_health.setCurrentIndex(health_index)

            type_index = self.fire_service_inlet_type.findText(model.inlet_type)

            if type_index != -1:
                self.fire_service_inlet_type.setCurrentIndex(type_index)

            self._populate_zone_combo(
                self.fire_service_inlet_zone,
                model,
                model.zone_ids[0] if model.zone_ids else "",
                "",
            )
            self._update_zone_warning(self.fire_service_inlet_zone_warning, model.zone_ids)

            self._populate_fire_water_system_combo(self.fire_service_inlet_fire_water_system, model.id, "inlet_ids")

            self._refresh_fire_service_inlet_availability(inlet_item)

        self.object_name.blockSignals(False)
        self.fire_service_inlet_x.blockSignals(False)
        self.fire_service_inlet_y.blockSignals(False)
        self.fire_service_inlet_active.blockSignals(False)
        self.fire_service_inlet_health.blockSignals(False)
        self.fire_service_inlet_type.blockSignals(False)
        self.fire_service_inlet_zone.blockSignals(False)

    # =====================================================

    def _refresh_fire_service_inlet_availability(self, inlet_item):

        model = inlet_item.model

        if model is None:
            return

        availability = model.compute_availability()

        self.fire_service_inlet_availability.setText(availability)

        inlet_item.current_availability = availability
        inlet_item.refresh_geometry()

    # =====================================================

    def update_fire_service_inlet_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.fire_service_inlet_x.text())
            y = float(self.fire_service_inlet_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_fire_service_inlet_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.fire_service_inlet_active.isChecked()
        self._refresh_fire_service_inlet_availability(self.current_item)

    # =====================================================

    def update_fire_service_inlet_health(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.health_status = self.fire_service_inlet_health.itemText(index)
        self._refresh_fire_service_inlet_availability(self.current_item)

    # =====================================================

    def update_fire_service_inlet_type(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.inlet_type = self.fire_service_inlet_type.itemText(index)

    # =====================================================

    def update_fire_service_inlet_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.fire_service_inlet_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.fire_service_inlet_zone_warning, self.current_item.model.zone_ids)

    # =====================================================

    def update_fire_service_inlet_fire_water_system(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self._update_asset_fire_water_system(
            self.fire_service_inlet_fire_water_system, self.current_item.model.id, "inlet_ids",
        )

    # =====================================================

    def update_speaker_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.speaker_x.text())
            y = float(self.speaker_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_speaker_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.speaker_active.isChecked()
        self.current_item.refresh_geometry()

    # =====================================================

    def update_speaker_health(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.health_status = self.speaker_health.itemText(index)

    # =====================================================

    def update_speaker_mode(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.mode = self.speaker_mode.itemText(index)

    # =====================================================

    def update_speaker_type(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.speaker_type = self.speaker_type.itemText(index)

    # =====================================================

    def update_speaker_volume(self):

        if self.current_item is None or self.current_item.model is None:
            return

        try:
            volume = float(self.speaker_volume.text())
        except ValueError:
            return

        self.current_item.model.volume_level = volume

    # =====================================================

    def update_speaker_installation_date(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.installation_date = self.speaker_installation_date.text()

    # =====================================================

    def update_speaker_zones(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.zone_ids = self._checked_zone_ids(self.speaker_zones)
        self._update_zone_warning(self.speaker_zone_warning, self.current_item.model.zone_ids)

    # =====================================================
    # Dynamic Evacuation Sign
    # =====================================================

    def update_sign_geometry(self):

        if self.current_item is None:
            return

        try:
            x = float(self.sign_x.text())
            y = float(self.sign_y.text())
        except ValueError:
            return

        self.current_item.setPos(x * self.GRID_SIZE, y * self.GRID_SIZE)
        self.current_item.sync_to_model()

    # =====================================================

    def update_sign_orientation(self):

        if self.current_item is None:
            return

        try:
            degrees = float(self.sign_orientation.text())
        except ValueError:
            return

        self.current_item.set_orientation_degrees(degrees)

    # =====================================================

    def update_sign_active(self):

        if self.current_item is None or self.current_item.model is None:
            return

        self.current_item.model.active = self.sign_active.isChecked()
        self.current_item.refresh_geometry()

    # =====================================================

    def update_sign_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.sign_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )

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
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
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
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
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
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
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
    # Occupant (Manual Simulation Sandbox)
    # =====================================================

    def show_occupant(self, occupant_item):

        self.current_item = occupant_item
        self._refresh_handler = self.show_occupant

        self._set_fields_visible(self.zone_fields, False)
        self._set_fields_visible(self.exit_fields, False)
        self._set_fields_visible(self.stair_fields, False)
        self._set_fields_visible(self.camera_fields, False)
        self._set_fields_visible(self.detector_fields, False)
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)
        self._set_fields_visible(self.occupant_fields, True)

        occupant = occupant_item.occupant

        self.object_type.setText("Occupant")
        self.object_id.setText(occupant_item.object_id)

        self.object_name.blockSignals(True)
        self.occupant_destination_type.blockSignals(True)

        self.object_name.setText(occupant_item.object_name)

        if self.building is not None:

            floor = self.building.get_floor(occupant.floor_id)

            self.occupant_current_floor.setText(
                floor.name if floor is not None else "-"
            )

        else:

            self.occupant_current_floor.setText("-")

        self.occupant_current_zone.setText(
            self._resolve_node_label(occupant.zone_id)
        )

        self.occupant_current_node.setText(
            self._resolve_node_label(occupant.current_node_id)
        )

        if occupant.destination_type is not None:

            index = self.occupant_destination_type.findText(
                occupant.destination_type
            )

            if index != -1:
                self.occupant_destination_type.setCurrentIndex(index)

        self.occupant_destination_node.setText(
            self._resolve_node_label(occupant.destination_node_id)
        )

        self.occupant_next_node.setText(
            self._resolve_node_label(occupant.next_node_id)
        )

        remaining = occupant.remaining_distance

        self.occupant_remaining_distance.setText(
            f"{remaining:.2f}" if remaining is not None else "-"
        )

        self.occupant_current_speed.setText(
            f"{occupant.current_speed:.2f}"
        )

        self.occupant_state.setText(
            occupant.state.name
        )

        self.object_name.blockSignals(False)
        self.occupant_destination_type.blockSignals(False)

    # =====================================================
    # node_id -> a readable label: the matching Zone/Assembly Point's
    # own name if one exists in the current Building, "Outside" for
    # the Navigation Graph's single shared exterior node, the raw id
    # as a last resort (e.g. building is None), "-" for no node at
    # all. Never a second source of truth for a name -- always
    # resolved live from whichever Building is currently set.
    # =====================================================

    def _resolve_node_label(self, node_id):

        if node_id is None:
            return "-"

        if node_id == Node.OUTSIDE_NODE_ID:
            return "Outside"

        if self.building is not None:

            for floor in self.building.floors:

                for zone in floor.zones:

                    if zone.id == node_id:
                        return zone.name

                for assembly_point in floor.assembly_points:

                    if assembly_point.id == node_id:
                        return assembly_point.name

        return node_id

    # =====================================================

    def update_occupant_destination(self, index):

        if self.current_item is None:
            return

        if self.sandbox_manager is None or self.building is None:
            return

        destination_type = self.occupant_destination_type.itemText(index)

        self.sandbox_manager.compute_route(
            self.current_item.occupant, self.building, destination_type,
        )

        self.current_item.sync_from_occupant()

        self.refresh()

        if self.occupant_route_changed_callback:
            self.occupant_route_changed_callback(self.current_item.occupant)

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
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
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
        self._set_fields_visible(self.smoke_detector_fields, False)
        self._set_fields_visible(self.heat_detector_fields, False)
        self._set_fields_visible(self.speaker_fields, False)
        self._set_fields_visible(self.sign_fields, False)
        self._set_fields_visible(self.mcp_fields, False)
        self._set_fields_visible(self.emergency_light_fields, False)
        self._set_fields_visible(self.sprinkler_fields, False)
        self._set_fields_visible(self.fire_extinguisher_fields, False)
        self._set_fields_visible(self.fire_hydrant_fields, False)
        self._set_fields_visible(self.hose_reel_fields, False)
        self._set_fields_visible(self.fire_water_tank_fields, False)
        self._set_fields_visible(self.fire_pump_fields, False)
        self._set_fields_visible(self.jockey_pump_fields, False)
        self._set_fields_visible(self.fire_service_inlet_fields, False)
        self._set_fields_visible(self.assembly_fields, False)
        self._set_fields_visible(self.obstacle_fields, False)
        self._set_fields_visible(self.door_fields, False)
        self._set_fields_visible(self.floor_fields, False)
        self._set_fields_visible(self.occupant_fields, False)

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

        self.exit_zone.blockSignals(True)
        self.exit_zone.clear()
        self.exit_zone.blockSignals(False)

        self.stair_from_floor.setText("-")

        self.stair_from_x.clear()
        self.stair_from_y.clear()
        self.stair_to_x.clear()
        self.stair_to_y.clear()

        self.stair_width.clear()

        self.to_floor_combo.blockSignals(True)
        self.to_floor_combo.clear()
        self.to_floor_combo.blockSignals(False)

        self.to_floor_uuid.clear()

        self.stair_from_zone.blockSignals(True)
        self.stair_from_zone.clear()
        self.stair_from_zone.blockSignals(False)

        self.stair_to_zone.blockSignals(True)
        self.stair_to_zone.clear()
        self.stair_to_zone.blockSignals(False)

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

        self.camera_zone.blockSignals(True)
        self.camera_zone.clear()
        self.camera_zone.blockSignals(False)

        self.camera_resolution.clear()
        self.camera_fps.clear()

        self.camera_mode.blockSignals(True)
        self.camera_mode.setCurrentIndex(0)
        self.camera_mode.blockSignals(False)

        self.camera_rtsp.clear()
        self.camera_ip.clear()
        self.camera_username.clear()
        self.camera_password.clear()

        self.camera_visible_zones.setText("-")
        self.camera_partial_zones.setText("-")
        self.camera_hidden_zones.setText("-")
        self.camera_max_visible_distance.setText("-")

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

        self.smoke_detector_x.clear()
        self.smoke_detector_y.clear()

        self.smoke_detector_active.blockSignals(True)
        self.smoke_detector_active.setChecked(False)
        self.smoke_detector_active.blockSignals(False)

        self.smoke_detector_health.blockSignals(True)
        self.smoke_detector_health.setCurrentIndex(0)
        self.smoke_detector_health.blockSignals(False)

        self.smoke_detector_mode.blockSignals(True)
        self.smoke_detector_mode.setCurrentIndex(0)
        self.smoke_detector_mode.blockSignals(False)

        self.smoke_detector_threshold.clear()
        self.smoke_detector_installation_date.clear()
        self.smoke_detector_test_level.clear()
        self.smoke_detector_state.setText("-")

        self.smoke_detector_zone.blockSignals(True)
        self.smoke_detector_zone.clear()
        self.smoke_detector_zone.blockSignals(False)
        self.smoke_detector_zone_warning.setVisible(False)

        self.heat_detector_x.clear()
        self.heat_detector_y.clear()

        self.heat_detector_active.blockSignals(True)
        self.heat_detector_active.setChecked(False)
        self.heat_detector_active.blockSignals(False)

        self.heat_detector_health.blockSignals(True)
        self.heat_detector_health.setCurrentIndex(0)
        self.heat_detector_health.blockSignals(False)

        self.heat_detector_mode.blockSignals(True)
        self.heat_detector_mode.setCurrentIndex(0)
        self.heat_detector_mode.blockSignals(False)

        self.heat_detector_threshold.clear()
        self.heat_detector_installation_date.clear()
        self.heat_detector_test_temperature.clear()
        self.heat_detector_state.setText("-")

        self.heat_detector_zone.blockSignals(True)
        self.heat_detector_zone.clear()
        self.heat_detector_zone.blockSignals(False)
        self.heat_detector_zone_warning.setVisible(False)

        self.speaker_x.clear()
        self.speaker_y.clear()

        self.speaker_active.blockSignals(True)
        self.speaker_active.setChecked(False)
        self.speaker_active.blockSignals(False)

        self.speaker_health.blockSignals(True)
        self.speaker_health.setCurrentIndex(0)
        self.speaker_health.blockSignals(False)

        self.speaker_mode.blockSignals(True)
        self.speaker_mode.setCurrentIndex(0)
        self.speaker_mode.blockSignals(False)

        self.speaker_type.blockSignals(True)
        self.speaker_type.setCurrentIndex(0)
        self.speaker_type.blockSignals(False)

        self.speaker_volume.clear()
        self.speaker_installation_date.clear()

        self.speaker_zones.blockSignals(True)
        self.speaker_zones.clear()
        self.speaker_zones.blockSignals(False)
        self.speaker_zone_warning.setVisible(False)

        self.sign_x.clear()
        self.sign_y.clear()
        self.sign_orientation.clear()

        self.sign_active.blockSignals(True)
        self.sign_active.setChecked(False)
        self.sign_active.blockSignals(False)

        self.sign_zone.blockSignals(True)
        self.sign_zone.clear()
        self.sign_zone.blockSignals(False)

        self.mcp_x.clear()
        self.mcp_y.clear()

        self.mcp_active.blockSignals(True)
        self.mcp_active.setChecked(False)
        self.mcp_active.blockSignals(False)

        self.mcp_activated.blockSignals(True)
        self.mcp_activated.setChecked(False)
        self.mcp_activated.blockSignals(False)

        self.mcp_health.blockSignals(True)
        self.mcp_health.setCurrentIndex(0)
        self.mcp_health.blockSignals(False)

        self.mcp_mode.blockSignals(True)
        self.mcp_mode.setCurrentIndex(0)
        self.mcp_mode.blockSignals(False)

        self.mcp_installation_date.clear()
        self.mcp_state.setText("-")

        self.mcp_zone.blockSignals(True)
        self.mcp_zone.clear()
        self.mcp_zone.blockSignals(False)
        self.mcp_zone_warning.setVisible(False)

        self.emergency_light_x.clear()
        self.emergency_light_y.clear()

        self.emergency_light_active.blockSignals(True)
        self.emergency_light_active.setChecked(False)
        self.emergency_light_active.blockSignals(False)

        self.emergency_light_health.blockSignals(True)
        self.emergency_light_health.setCurrentIndex(0)
        self.emergency_light_health.blockSignals(False)

        self.emergency_light_type.blockSignals(True)
        self.emergency_light_type.setCurrentIndex(0)
        self.emergency_light_type.blockSignals(False)

        self.emergency_light_availability.setText("-")

        self.emergency_light_zone.blockSignals(True)
        self.emergency_light_zone.clear()
        self.emergency_light_zone.blockSignals(False)
        self.emergency_light_zone_warning.setVisible(False)

        self.sprinkler_x.clear()
        self.sprinkler_y.clear()

        self.sprinkler_active.blockSignals(True)
        self.sprinkler_active.setChecked(False)
        self.sprinkler_active.blockSignals(False)

        self.sprinkler_health.blockSignals(True)
        self.sprinkler_health.setCurrentIndex(0)
        self.sprinkler_health.blockSignals(False)

        self.sprinkler_mode.blockSignals(True)
        self.sprinkler_mode.setCurrentIndex(0)
        self.sprinkler_mode.blockSignals(False)

        self.sprinkler_activation_temperature.clear()
        self.sprinkler_installation_date.clear()
        self.sprinkler_test_temperature.clear()
        self.sprinkler_state.setText("-")

        self.sprinkler_zone.blockSignals(True)
        self.sprinkler_zone.clear()
        self.sprinkler_zone.blockSignals(False)
        self.sprinkler_zone_warning.setVisible(False)

        self.sprinkler_fire_water_system.blockSignals(True)
        self.sprinkler_fire_water_system.clear()
        self.sprinkler_fire_water_system.blockSignals(False)

        self.fire_extinguisher_x.clear()
        self.fire_extinguisher_y.clear()

        self.fire_extinguisher_active.blockSignals(True)
        self.fire_extinguisher_active.setChecked(False)
        self.fire_extinguisher_active.blockSignals(False)

        self.fire_extinguisher_health.blockSignals(True)
        self.fire_extinguisher_health.setCurrentIndex(0)
        self.fire_extinguisher_health.blockSignals(False)

        self.fire_extinguisher_type.blockSignals(True)
        self.fire_extinguisher_type.setCurrentIndex(0)
        self.fire_extinguisher_type.blockSignals(False)

        self.fire_extinguisher_availability.setText("-")

        self.fire_extinguisher_zone.blockSignals(True)
        self.fire_extinguisher_zone.clear()
        self.fire_extinguisher_zone.blockSignals(False)
        self.fire_extinguisher_zone_warning.setVisible(False)

        self.fire_hydrant_x.clear()
        self.fire_hydrant_y.clear()

        self.fire_hydrant_active.blockSignals(True)
        self.fire_hydrant_active.setChecked(False)
        self.fire_hydrant_active.blockSignals(False)

        self.fire_hydrant_health.blockSignals(True)
        self.fire_hydrant_health.setCurrentIndex(0)
        self.fire_hydrant_health.blockSignals(False)

        self.fire_hydrant_type.blockSignals(True)
        self.fire_hydrant_type.setCurrentIndex(0)
        self.fire_hydrant_type.blockSignals(False)

        self.fire_hydrant_availability.setText("-")

        self.fire_hydrant_zone.blockSignals(True)
        self.fire_hydrant_zone.clear()
        self.fire_hydrant_zone.blockSignals(False)
        self.fire_hydrant_zone_warning.setVisible(False)

        self.fire_hydrant_fire_water_system.blockSignals(True)
        self.fire_hydrant_fire_water_system.clear()
        self.fire_hydrant_fire_water_system.blockSignals(False)

        self.hose_reel_x.clear()
        self.hose_reel_y.clear()

        self.hose_reel_active.blockSignals(True)
        self.hose_reel_active.setChecked(False)
        self.hose_reel_active.blockSignals(False)

        self.hose_reel_health.blockSignals(True)
        self.hose_reel_health.setCurrentIndex(0)
        self.hose_reel_health.blockSignals(False)

        self.hose_reel_availability.setText("-")

        self.hose_reel_zone.blockSignals(True)
        self.hose_reel_zone.clear()
        self.hose_reel_zone.blockSignals(False)
        self.hose_reel_zone_warning.setVisible(False)

        self.hose_reel_fire_water_system.blockSignals(True)
        self.hose_reel_fire_water_system.clear()
        self.hose_reel_fire_water_system.blockSignals(False)

        self.fire_water_tank_x.clear()
        self.fire_water_tank_y.clear()

        self.fire_water_tank_active.blockSignals(True)
        self.fire_water_tank_active.setChecked(False)
        self.fire_water_tank_active.blockSignals(False)

        self.fire_water_tank_health.blockSignals(True)
        self.fire_water_tank_health.setCurrentIndex(0)
        self.fire_water_tank_health.blockSignals(False)

        self.fire_water_tank_capacity.clear()
        self.fire_water_tank_level.clear()
        self.fire_water_tank_state.setText("-")

        self.fire_water_tank_zone.blockSignals(True)
        self.fire_water_tank_zone.clear()
        self.fire_water_tank_zone.blockSignals(False)
        self.fire_water_tank_zone_warning.setVisible(False)

        self.fire_water_tank_fire_water_system.blockSignals(True)
        self.fire_water_tank_fire_water_system.clear()
        self.fire_water_tank_fire_water_system.blockSignals(False)

        self.fire_pump_x.clear()
        self.fire_pump_y.clear()

        self.fire_pump_active.blockSignals(True)
        self.fire_pump_active.setChecked(False)
        self.fire_pump_active.blockSignals(False)

        self.fire_pump_health.blockSignals(True)
        self.fire_pump_health.setCurrentIndex(0)
        self.fire_pump_health.blockSignals(False)

        self.fire_pump_control_mode.blockSignals(True)
        self.fire_pump_control_mode.setCurrentIndex(0)
        self.fire_pump_control_mode.blockSignals(False)

        self.fire_pump_running.blockSignals(True)
        self.fire_pump_running.setChecked(False)
        self.fire_pump_running.blockSignals(False)

        self.fire_pump_state.setText("-")

        self.fire_pump_zone.blockSignals(True)
        self.fire_pump_zone.clear()
        self.fire_pump_zone.blockSignals(False)
        self.fire_pump_zone_warning.setVisible(False)

        self.fire_pump_fire_water_system.blockSignals(True)
        self.fire_pump_fire_water_system.clear()
        self.fire_pump_fire_water_system.blockSignals(False)

        self.jockey_pump_x.clear()
        self.jockey_pump_y.clear()

        self.jockey_pump_active.blockSignals(True)
        self.jockey_pump_active.setChecked(False)
        self.jockey_pump_active.blockSignals(False)

        self.jockey_pump_health.blockSignals(True)
        self.jockey_pump_health.setCurrentIndex(0)
        self.jockey_pump_health.blockSignals(False)

        self.jockey_pump_control_mode.blockSignals(True)
        self.jockey_pump_control_mode.setCurrentIndex(0)
        self.jockey_pump_control_mode.blockSignals(False)

        self.jockey_pump_running.blockSignals(True)
        self.jockey_pump_running.setChecked(False)
        self.jockey_pump_running.blockSignals(False)

        self.jockey_pump_state.setText("-")

        self.jockey_pump_zone.blockSignals(True)
        self.jockey_pump_zone.clear()
        self.jockey_pump_zone.blockSignals(False)
        self.jockey_pump_zone_warning.setVisible(False)

        self.jockey_pump_fire_water_system.blockSignals(True)
        self.jockey_pump_fire_water_system.clear()
        self.jockey_pump_fire_water_system.blockSignals(False)

        self.fire_service_inlet_x.clear()
        self.fire_service_inlet_y.clear()

        self.fire_service_inlet_active.blockSignals(True)
        self.fire_service_inlet_active.setChecked(False)
        self.fire_service_inlet_active.blockSignals(False)

        self.fire_service_inlet_health.blockSignals(True)
        self.fire_service_inlet_health.setCurrentIndex(0)
        self.fire_service_inlet_health.blockSignals(False)

        self.fire_service_inlet_type.blockSignals(True)
        self.fire_service_inlet_type.setCurrentIndex(0)
        self.fire_service_inlet_type.blockSignals(False)

        self.fire_service_inlet_availability.setText("-")

        self.fire_service_inlet_zone.blockSignals(True)
        self.fire_service_inlet_zone.clear()
        self.fire_service_inlet_zone.blockSignals(False)
        self.fire_service_inlet_zone_warning.setVisible(False)

        self.fire_service_inlet_fire_water_system.blockSignals(True)
        self.fire_service_inlet_fire_water_system.clear()
        self.fire_service_inlet_fire_water_system.blockSignals(False)

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

        self.occupant_current_floor.setText("-")
        self.occupant_current_zone.setText("-")
        self.occupant_current_node.setText("-")

        self.occupant_destination_type.blockSignals(True)
        self.occupant_destination_type.setCurrentIndex(0)
        self.occupant_destination_type.blockSignals(False)

        self.occupant_destination_node.setText("-")
        self.occupant_next_node.setText("-")
        self.occupant_remaining_distance.setText("-")
        self.occupant_current_speed.setText("-")
        self.occupant_state.setText("-")

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
    # Populates the Exit's Zone combo from whichever floor the Exit
    # itself belongs to -- Zones only, deliberately not every
    # connectable space the way Door's own combo is: NavigationGraph
    # Generator._add_exit_edges() resolves an Exit's endpoint with the
    # default allowed_types=(Node.ZONE,), never Assembly Point, so
    # offering Assembly Point here would let a user pick a selection
    # that silently produces no edge at all -- the exact "looks
    # configured but isn't" trap zone_id being unreachable through
    # this panel already was.
    # =====================================================

    def _populate_exit_zone_combo(self, model):

        self.exit_zone.blockSignals(True)

        self.exit_zone.clear()

        self.exit_zone.addItem("None", "")

        if self.building is not None:

            floor = self.building.get_floor(model.floor_id)

            if floor is not None:

                for zone in floor.zones:

                    self.exit_zone.addItem(
                        zone.name,
                        zone.id,
                    )

        index = self.exit_zone.findData(model.zone_id)

        if index == -1:
            index = 0

        self.exit_zone.setCurrentIndex(index)

        self.exit_zone.blockSignals(False)

    # =====================================================

    def update_exit_zone(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        zone_id = self.exit_zone.itemData(index)

        self.current_item.model.zone_id = zone_id or ""

    # =====================================================

    def _parse_optional_stair_region(self, width_field, depth_field, center_x, center_y):

        # Observable Stair Perception milestone -- BOTH width and depth
        # must be present and > 0 for a region to exist at all; either
        # field blank (or non-positive) means "not authored", the region
        # stays None -- never a fabricated default. Center always tracks
        # this side's CURRENT from_position/to_position (passed in
        # freshly parsed by the caller), so moving the stair marker later
        # keeps an already-authored region correctly anchored without the
        # operator having to retype width/depth.

        width_text = width_field.text().strip()
        depth_text = depth_field.text().strip()

        if not width_text and not depth_text:
            return None, True

        try:

            width = float(width_text)
            depth = float(depth_text)

        except ValueError:

            return None, False

        if width <= 0.0 or depth <= 0.0:
            return None, True

        return StairObservableRegion(center_x=center_x, center_y=center_y, width=width, depth=depth), True

    # =====================================================

    def update_stair_geometry(self):

        if self.current_item is None:
            return

        try:

            fx = float(self.stair_from_x.text())
            fy = float(self.stair_from_y.text())

            tx = float(self.stair_to_x.text())
            ty = float(self.stair_to_y.text())

            width = float(self.stair_width.text())

        except ValueError:

            self.refresh()

            return

        from_region, from_region_ok = self._parse_optional_stair_region(
            self.stair_from_region_width, self.stair_from_region_depth, fx, fy,
        )
        to_region, to_region_ok = self._parse_optional_stair_region(
            self.stair_to_region_width, self.stair_to_region_depth, tx, ty,
        )

        if not from_region_ok or not to_region_ok:

            self.refresh()

            return

        model = self.current_item.model

        if model is not None:

            # Both ends are always writable from here regardless
            # of which marker is actually selected -- the other
            # end's floor may not even be the one on screen right
            # now, so there is no graphics item to move for it;
            # writing the model directly is enough, and it will
            # render correctly next time that floor is shown.
            model.from_position = (fx, fy)
            model.to_position = (tx, ty)
            model.width = width
            model.from_observable_region = from_region
            model.to_observable_region = to_region

        # Only the marker actually being displayed gets its
        # on-canvas position moved.
        if self.current_item.role == "from":

            self.current_item.setPos(
                fx * self.GRID_SIZE,
                fy * self.GRID_SIZE,
            )

        else:

            self.current_item.setPos(
                tx * self.GRID_SIZE,
                ty * self.GRID_SIZE,
            )

        self.current_item.refresh_geometry()

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
    # Populates a Stair endpoint's Zone combo from a given floor_id --
    # deliberately not the same _populate_zone_combo() Door already
    # uses, since Door's version reads a single model.floor_id shared
    # by both ends, while a Stair's two ends belong to two different
    # floors. Zone-only (not Assembly Point, unlike Door's combo):
    # NavigationGraphGenerator._add_stair_edges() resolves both ends
    # with the default allowed_types=(Node.ZONE,), so offering
    # Assembly Point here would let a user pick a selection that
    # silently produces no edge at all -- the same reasoning
    # _populate_exit_zone_combo() already documents.
    # =====================================================

    def _populate_stair_zone_combo(self, combo, floor_id, current_zone_id):

        combo.blockSignals(True)

        combo.clear()

        combo.addItem("None", "")

        if self.building is not None:

            floor = self.building.get_floor(floor_id)

            if floor is not None:

                for zone in floor.zones:

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

    def update_stair_from_zone(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        zone_id = self.stair_from_zone.itemData(index)

        self.current_item.model.from_zone_id = zone_id or ""

    # =====================================================

    def update_stair_to_zone(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        zone_id = self.stair_to_zone.itemData(index)

        self.current_item.model.to_zone_id = zone_id or ""

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

        # Whatever To Zone was previously selected belonged to the
        # OLD destination floor -- meaningless (and likely invalid)
        # against the new one, so it must not silently carry over.
        self.current_item.model.to_zone_id = ""

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

        if self.on_visual_change:
            self.on_visual_change()

    # =====================================================

    def update_camera_active(self):

        if self.current_item is None:
            return

        if self.current_item.model is not None:

            self.current_item.model.active = (
                self.camera_active.isChecked()
            )

        self.current_item.refresh_geometry()

        if self.on_visual_change:
            self.on_visual_change()

    # =====================================================

    def update_camera_zone(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        zone_id = self.camera_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )

    # =====================================================

    def update_camera_metadata(self):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.resolution = (
            self.camera_resolution.text()
        )

        try:

            self.current_item.model.fps = int(
                self.camera_fps.text()
            )

        except ValueError:

            self.refresh()

    # =====================================================

    def update_camera_mode(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.mode = (
            self.camera_mode.itemText(index)
        )

    # =====================================================

    def update_camera_connection(self):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        connection = self.current_item.model.connection

        connection.rtsp_address = self.camera_rtsp.text()
        connection.ip_address = self.camera_ip.text()
        connection.username = self.camera_username.text()
        connection.password = self.camera_password.text()

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
    # Smoke Detector (Building Sensor Network Framework)
    # =====================================================

    def update_smoke_detector_geometry(self):

        if self.current_item is None:
            return

        try:

            x = float(self.smoke_detector_x.text())
            y = float(self.smoke_detector_y.text())

        except ValueError:

            self.refresh()

            return

        self.current_item.setPos(
            x * self.GRID_SIZE,
            y * self.GRID_SIZE,
        )

        self.current_item.refresh_geometry()

        self.refresh()

    # =====================================================

    def update_smoke_detector_active(self):

        if self.current_item is None:
            return

        if self.current_item.model is not None:

            self.current_item.model.active = (
                self.smoke_detector_active.isChecked()
            )

        self._refresh_smoke_detector_state(self.current_item)

    # =====================================================

    def update_smoke_detector_health(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.health_status = (
            self.smoke_detector_health.itemText(index)
        )

        self._refresh_smoke_detector_state(self.current_item)

    # =====================================================

    def update_smoke_detector_mode(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.mode = (
            self.smoke_detector_mode.itemText(index)
        )

    # =====================================================

    def update_smoke_detector_threshold(self):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        try:

            threshold = float(self.smoke_detector_threshold.text())

        except ValueError:

            self.refresh()

            return

        self.current_item.model.activation_threshold = threshold

        self._refresh_smoke_detector_state(self.current_item)

    # =====================================================

    def update_smoke_detector_installation_date(self):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.installation_date = (
            self.smoke_detector_installation_date.text()
        )

    # =====================================================

    def update_smoke_detector_test_reading(self):

        if self.current_item is None:
            return

        self._refresh_smoke_detector_state(self.current_item)

    # =====================================================

    def update_smoke_detector_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.smoke_detector_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.smoke_detector_zone_warning, self.current_item.model.zone_ids)

    # =====================================================
    # Heat Detector (Building Sensor Network Framework)
    # =====================================================

    def update_heat_detector_geometry(self):

        if self.current_item is None:
            return

        try:

            x = float(self.heat_detector_x.text())
            y = float(self.heat_detector_y.text())

        except ValueError:

            self.refresh()

            return

        self.current_item.setPos(
            x * self.GRID_SIZE,
            y * self.GRID_SIZE,
        )

        self.current_item.refresh_geometry()

        self.refresh()

    # =====================================================

    def update_heat_detector_active(self):

        if self.current_item is None:
            return

        if self.current_item.model is not None:

            self.current_item.model.active = (
                self.heat_detector_active.isChecked()
            )

        self._refresh_heat_detector_state(self.current_item)

    # =====================================================

    def update_heat_detector_health(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.health_status = (
            self.heat_detector_health.itemText(index)
        )

        self._refresh_heat_detector_state(self.current_item)

    # =====================================================

    def update_heat_detector_mode(self, index):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.mode = (
            self.heat_detector_mode.itemText(index)
        )

    # =====================================================

    def update_heat_detector_threshold(self):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        try:

            threshold = float(self.heat_detector_threshold.text())

        except ValueError:

            self.refresh()

            return

        self.current_item.model.activation_threshold = threshold

        self._refresh_heat_detector_state(self.current_item)

    # =====================================================

    def update_heat_detector_installation_date(self):

        if self.current_item is None:
            return

        if self.current_item.model is None:
            return

        self.current_item.model.installation_date = (
            self.heat_detector_installation_date.text()
        )

    # =====================================================

    def update_heat_detector_test_reading(self):

        if self.current_item is None:
            return

        self._refresh_heat_detector_state(self.current_item)

    # =====================================================

    def update_heat_detector_zone(self, index):

        if self.current_item is None or self.current_item.model is None:
            return

        zone_id = self.heat_detector_zone.itemData(index)

        self.current_item.model.zone_ids = (
            (zone_id,) if zone_id else ()
        )
        self._update_zone_warning(self.heat_detector_zone_warning, self.current_item.model.zone_ids)

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

        # A Door connects any navigable space registered in
        # models/connectable_space.py (Zone, Assembly Point today),
        # not just a Zone -- the field/param names here stay as
        # "zone" for the same backward-compatibility reason
        # Door.zone_a_id/zone_b_id kept their names (see door.py).
        # itemData is still just the plain object id: ids are unique
        # across every connectable type, so no separate "which type"
        # data needs to travel with it.

        combo.blockSignals(True)

        combo.clear()

        combo.addItem("None", "")

        if self.building is not None:

            floor = self.building.get_floor(model.floor_id)

            if floor is not None:

                for space_type, space in connectable_space.all_connectable_spaces(floor):

                    if space.id == exclude_zone_id:
                        continue

                    combo.addItem(
                        connectable_space.label_for(
                            space_type,
                            space.name,
                        ),
                        space.id,
                    )

        index = combo.findData(current_zone_id)

        if index == -1:
            index = 0

        combo.setCurrentIndex(index)

        combo.blockSignals(False)

    # =====================================================
    # Digital Twin Asset -> Zone Assignment & Live FACP Runtime milestone,
    # Phase 3 -- the genuine multi-select sibling of _populate_zone_combo()
    # above, for Speaker.zone_ids specifically. speaker_manager.
    # SpeakerManager.active_speakers_in_zone()/voice_evacuation.controller.
    # VoiceEvacuationController already treat zone_ids as an ordinary
    # tuple-membership test with no cardinality limit (confirmed by
    # reading both directly) -- a single QComboBox would silently reduce
    # a Speaker to at most one served zone, which Phase 3 explicitly
    # forbids doing "merely because a single QComboBox is easier". Each
    # row is a checkable QListWidgetItem (never a modal/dialog selector)
    # so "which zones are currently checked" is always visible at a
    # glance, same "show current state, not just an editor" spirit as
    # every other field in this panel.
    # =====================================================

    def _populate_zone_checklist(self, list_widget, model, current_zone_ids):

        list_widget.blockSignals(True)

        list_widget.clear()

        if self.building is not None:

            floor = self.building.get_floor(model.floor_id)

            if floor is not None:

                for space_type, space in connectable_space.all_connectable_spaces(floor):

                    item = QListWidgetItem(
                        connectable_space.label_for(space_type, space.name)
                    )
                    item.setData(Qt.ItemDataRole.UserRole, space.id)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

                    item.setCheckState(
                        Qt.CheckState.Checked if space.id in current_zone_ids
                        else Qt.CheckState.Unchecked
                    )

                    list_widget.addItem(item)

        list_widget.blockSignals(False)

    # =====================================================

    def _checked_zone_ids(self, list_widget):

        return tuple(
            list_widget.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(list_widget.count())
            if list_widget.item(row).checkState() == Qt.CheckState.Checked
        )

    # =====================================================
    # Phase 5 -- a modest, reused-everywhere inline hint (never a new
    # validation framework): visible exactly when an asset whose
    # downstream operation depends on zone_ids currently has none.
    # designer/validation.py's own validate_building_authoring() reports
    # the identical condition as a project-wide WARNING for the
    # "Validate Project" dialog -- this is the same fact, surfaced
    # inline for whichever single asset is currently selected.
    # =====================================================

    def _update_zone_warning(self, warning_label, zone_ids):

        warning_label.setVisible(not zone_ids)

    # =====================================================
    # Fire Water Supply & Suppression Infrastructure milestone -- the
    # "Fire Water System" combo every Tank/Pump/Jockey Pump/Inlet/
    # Sprinkler/Hydrant/Hose Reel section shares. Membership lives on
    # FireWaterSystem itself (models/fire_water_system.py), never on
    # the asset -- this combo's "current selection" is therefore looked
    # up by scanning building.fire_water_systems for whichever one's
    # own field_name tuple already contains this asset's id, the same
    # "asset never stores a reverse-reference" discipline that keeps
    # membership single-sourced.
    # =====================================================

    def _populate_fire_water_system_combo(self, combo, asset_id, field_name):

        combo.blockSignals(True)

        combo.clear()

        combo.addItem("None", "")

        if self.building is not None:

            for system in self.building.fire_water_systems:
                combo.addItem(system.name, system.id)

        current_system = (
            system_containing_asset(self.building, asset_id, field_name)
            if self.building is not None else None
        )
        current_system_id = current_system.id if current_system is not None else ""

        index = combo.findData(current_system_id)

        if index == -1:
            index = 0

        combo.setCurrentIndex(index)

        combo.blockSignals(False)

    # =====================================================

    def _update_asset_fire_water_system(self, combo, asset_id, field_name):

        if self.building is None:
            return

        target_system_id = combo.itemData(combo.currentIndex()) or None

        assign_asset_to_system(self.building, asset_id, field_name, target_system_id)

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
