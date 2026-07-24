from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QBrush, QKeyEvent, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
)

from designer.items.assembly_point_item import AssemblyPointItem
from designer.items.camera_item import CameraItem
from designer.items.detector_item import DetectorItem
from designer.items.door_item import DoorItem
from designer.items.exit_item import ExitItem
from designer.items.heat_detector_item import HeatDetectorItem
from designer.items.obstacle_item import ObstacleItem
from designer.items.occupant_item import OccupantItem
from designer.items.sign_item import SignItem
from designer.items.manual_call_point_item import ManualCallPointItem
from designer.items.emergency_light_item import EmergencyLightItem
from designer.items.sprinkler_item import SprinklerItem
from designer.items.fire_extinguisher_item import FireExtinguisherItem
from designer.items.fire_hydrant_item import FireHydrantItem
from designer.items.hose_reel_item import HoseReelItem
from designer.items.fire_water_tank_item import FireWaterTankItem
from designer.items.fire_pump_item import FirePumpItem
from designer.items.jockey_pump_item import JockeyPumpItem
from designer.items.fire_service_inlet_item import FireServiceInletItem
from designer.items.smoke_detector_item import SmokeDetectorItem
from designer.items.speaker_item import SpeakerItem
from designer.items.stair_item import StairItem
from designer.items.zone_rectangle import ZoneRectangle

from models.project import Project
from models.assembly_point import AssemblyPoint
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.heat_detector import HeatDetector
from models.obstacle import Obstacle
from models.smoke_detector import SmokeDetector
from models.speaker import Speaker
from models.dynamic_sign import DynamicEvacuationSign
from models.manual_call_point import ManualCallPoint
from models.emergency_light import EmergencyLight
from models.sprinkler import Sprinkler
from models.fire_extinguisher import FireExtinguisher
from models.fire_hydrant import FireHydrant
from models.hose_reel import HoseReel
from models.fire_water_tank import FireWaterTank
from models.fire_pump import FirePump
from models.jockey_pump import JockeyPump
from models.fire_service_inlet import FireServiceInlet
from models.staircase import Staircase
from models.zone import Zone

from sandbox.manager import SandboxManager
from sandbox.occupant import SandboxDestinationType


class GraphicsScene(QGraphicsScene):

    GRID_SIZE = 50
    GRID_SCALE = 1.0

    def __init__(self):
        super().__init__()

        self.setSceneRect(
            -5000,
            -5000,
            10000,
            10000,
        )

        # -------------------------------------------------
        # Project Model
        #
        # GraphicsScene never constructs Building/Floor
        # itself -- Project.new_default() is the domain-level
        # factory that owns default project shape. GraphicsScene
        # only renders/edits whichever floor it is told is
        # current.
        # -------------------------------------------------

        self.project = Project.new_default()

        self.current_floor = (
            self.project.building.ordered_floors()[0]
        )

        # -------------------------------------------------

        self.current_tool = "select"

        self.floor_plan_item = None

        self.start_point = None
        self.preview_rect = None
        self.preview_line = None
        self.dimension_text = None

        # -------------------------------------------------
        # Stair Tool -- a Staircase is a single object spanning
        # two floors, placed across a guided three-step flow
        # (entrance click -> destination floor chosen -> landing
        # click). pending_stair holds the in-progress model
        # between those steps; it is never added to any floor's
        # Floor.stairs until the landing click completes it, so
        # a cancelled placement never leaves a half-formed stair
        # behind. GraphicsScene never shows the destination-floor
        # picker itself or switches floors on its own -- both are
        # requested through callbacks MainWindow provides, the
        # same "Scene never owns dialogs/coordination" convention
        # FloorList/MainWindow already follow.
        # -------------------------------------------------

        self.pending_stair = None

        self.floor_picker_callback = None
        self.floor_switch_requested_callback = None

        # Asked after a destination floor is chosen, only when a
        # Stair already connects that floor pair -- must return True
        # to proceed, False (or anything falsy) to abandon this
        # placement. Same "Scene never shows a dialog itself"
        # contract as floor_picker_callback above.
        self.duplicate_stair_confirmation_callback = None

        self.selected_item = None

        self.selection_changed_callback = None

        # -------------------------------------------------
        # Manual Simulation Sandbox (Simulation V0) -- occupants are
        # temporary debugging objects, never part of Project/Building,
        # never touched by Serializer. sandbox_manager is the single
        # source of truth for what occupants exist; occupant_items is
        # only a floor_id-scoped rendering cache (whichever
        # OccupantItem currently represents a given occupant.id on
        # THIS floor's QGraphicsScene, same "graphics item is a view
        # over the real state" convention every other item already
        # follows for its own model).
        # -------------------------------------------------

        self.sandbox_manager = SandboxManager()
        self.occupant_items = {}
        self._highlighted_route_items = []

        # -------------------------------------------------
        # Camera Coverage & Visibility Engine -- purely a rendering
        # concern, same "graphics item is a view over the real state"
        # pattern as everything else in this class. show_camera_
        # coverage is never serialized (Serializer never touches
        # GraphicsScene at all); _coverage_overlay_items are plain
        # QGraphicsPolygonItem/QGraphicsRectItem instances added on
        # TOP of the real ZoneRectangle/CameraItem items, never
        # replacing them, and are excluded from clear_graphics_items()'s
        # isinstance() filter so they need their own bookkeeping here.
        # -------------------------------------------------

        self.show_camera_coverage = False
        self._coverage_overlay_items = []

        # Asked after the Occupant Tool's drag-rectangle is released --
        # must return (count, distribution) or None (cancelled). Same
        # "Scene never shows a dialog itself" contract as
        # floor_picker_callback above.
        self.occupant_generation_callback = None

        self.draw_grid()

    # =====================================================

    def draw_grid(self):

        pen = QPen(QColor(60, 60, 60))

        left = int(self.sceneRect().left())
        right = int(self.sceneRect().right())
        top = int(self.sceneRect().top())
        bottom = int(self.sceneRect().bottom())

        for x in range(
            left,
            right,
            self.GRID_SIZE,
        ):

            self.addLine(
                x,
                top,
                x,
                bottom,
                pen,
            )

        for y in range(
            top,
            bottom,
            self.GRID_SIZE,
        ):

            self.addLine(
                left,
                y,
                right,
                y,
                pen,
            )

    # =====================================================

    def snap(self, point):

        x = (
            round(point.x() / self.GRID_SIZE)
            * self.GRID_SIZE
        )

        y = (
            round(point.y() / self.GRID_SIZE)
            * self.GRID_SIZE
        )

        return x, y

    # =====================================================
    # Which Zone (if any) on `floor` contains this point -- the same
    # "click inside a Zone" resolution the Occupant Tool already uses
    # (see sandbox.manager.SandboxManager._find_zone), duplicated here
    # rather than imported: it is three lines over Zone's own public
    # contains(), not logic worth coupling GraphicsScene's Stair
    # authoring to the unrelated Manual Simulation Sandbox package for.
    # =====================================================

    def _find_zone_at(self, floor, x_m, y_m):

        for zone in floor.zones:

            if zone.contains(x_m, y_m):
                return zone

        return None

    # =====================================================
    # Digital Twin Asset -> Zone Assignment & Live FACP Runtime milestone,
    # Phase 4 -- a deliberately STRICTER sibling of _find_zone_at() above,
    # used only for auto-assigning a newly-placed Smoke/Heat Detector's
    # zone_ids. _find_zone_at() (Stair authoring, untouched) returns the
    # FIRST matching zone and is fine with that -- a Stair landing click
    # always has an explicit human confirming the floor/zone right there.
    # Auto-assignment has no such confirmation step, so it must never
    # guess: if a position falls inside more than one zone (overlapping
    # zones on the same floor), this returns None -- the honest "cannot
    # auto-assign unambiguously" answer -- rather than silently picking
    # whichever zone happened to be first in the list. Manual Property
    # Panel assignment always remains available regardless.
    # =====================================================

    def _find_unambiguous_zone_at(self, floor, x_m, y_m):

        matches = [zone for zone in floor.zones if zone.contains(x_m, y_m)]

        return matches[0] if len(matches) == 1 else None

    # =====================================================
    # Whether a Stair already connects this exact pair of floors, in
    # either direction -- used only to decide whether the duplicate-
    # confirmation callback needs to be asked before starting a new
    # placement. A Staircase can only ever be stored on its own
    # from_floor's Floor.stairs list (see StairItem's own docstring),
    # so checking both candidate floors' own lists is sufficient --
    # no need to scan the whole Building.
    # =====================================================

    def _stair_connects(self, floor_a, floor_b):

        floor_pair = {floor_a.id, floor_b.id}

        for floor in (floor_a, floor_b):

            for stair in floor.stairs:

                if {stair.from_floor_id, stair.to_floor_id} == floor_pair:
                    return True

        return False

    # =====================================================

    def set_tool(self, tool):

        if tool != "stair":

            # Abandon an in-progress placement rather than let a
            # stale entrance from a previous "stair" session get
            # silently completed by a later, unrelated click.
            self.pending_stair = None

        self.current_tool = tool

    # =====================================================
    # Active Floor
    #
    # GraphicsScene never decides which floor is active --
    # it only renders whichever floor it is handed. Activation
    # is a MainWindow/Building concern (Floor List -> MainWindow
    # -> Building -> here).
    # =====================================================

    def set_current_floor(self, floor):

        if floor is None or floor is self.current_floor:
            return

        self.current_floor = floor

        self.rebuild_scene()

    # =====================================================
    # Floor Plan
    #
    # The image path is owned by Floor.floor_plan. GraphicsScene
    # only displays whatever the current floor's model says.
    # =====================================================

    def load_floor_plan(self, image_path):

        self.current_floor.floor_plan = image_path

        self._display_floor_plan()

    # =====================================================

    def _display_floor_plan(self):

        if self.floor_plan_item:

            self.removeItem(self.floor_plan_item)

            self.floor_plan_item = None

        path = self.current_floor.floor_plan

        if not path:
            return

        self.floor_plan_item = QGraphicsPixmapItem(
            QPixmap(path)
        )

        self.floor_plan_item.setZValue(-100)

        self.addItem(self.floor_plan_item)

    # =====================================================

    def mousePressEvent(self, event):

        # -------------------------------------------------
        # Select Tool
        # -------------------------------------------------

        if self.current_tool == "select":

            super().mousePressEvent(event)

            item = self.itemAt(
                event.scenePos(),
                self.views()[0].transform(),
            )

            if self.selected_item:
                self.selected_item.set_selected(False)

            if isinstance(item, (ZoneRectangle, ExitItem, StairItem, CameraItem, DetectorItem, SmokeDetectorItem, HeatDetectorItem, SpeakerItem, SignItem, ManualCallPointItem, EmergencyLightItem, SprinklerItem, FireExtinguisherItem, FireHydrantItem, HoseReelItem, FireWaterTankItem, FirePumpItem, JockeyPumpItem, FireServiceInletItem, AssemblyPointItem, ObstacleItem, DoorItem, OccupantItem)):

                self.selected_item = item

                item.set_selected(True)

                if isinstance(item, OccupantItem):
                    self._highlight_route(item.occupant)
                else:
                    self._clear_route_highlight()

                if self.selection_changed_callback:
                    self.selection_changed_callback(item)

            else:

                self.selected_item = None

                self._clear_route_highlight()

                if self.selection_changed_callback:
                    self.selection_changed_callback(None)

            return

        # -------------------------------------------------
        # Zone Tool
        # -------------------------------------------------

        if self.current_tool == "zone":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            if self.start_point is None:

                self.start_point = (x, y)

                self.preview_rect = QGraphicsRectItem()

                self.preview_rect.setBrush(
                    QBrush(
                        QColor(
                            0,
                            255,
                            255,
                            40,
                        )
                    )
                )

                self.preview_rect.setPen(
                    QPen(
                        QColor(
                            0,
                            255,
                            255,
                        ),
                        2,
                    )
                )

                self.addItem(
                    self.preview_rect
                )

                self.dimension_text = (
                    QGraphicsSimpleTextItem()
                )

                self.dimension_text.setBrush(
                    QBrush(
                        QColor(
                            255,
                            255,
                            0,
                        )
                    )
                )

                self.dimension_text.setZValue(
                    1000
                )

                self.addItem(
                    self.dimension_text
                )

            else:

                x1, y1 = self.start_point

                rect = QRectF(
                    min(x1, x),
                    min(y1, y),
                    abs(x - x1),
                    abs(y - y1),
                )

                zone_model = Zone(
                    name=self.project.building.next_zone_name(),
                    x=rect.x() / self.GRID_SIZE,
                    y=rect.y() / self.GRID_SIZE,
                    width=rect.width() / self.GRID_SIZE,
                    height=rect.height() / self.GRID_SIZE,
                )

                self.current_floor.add_zone(
                    zone_model
                )

                zone = ZoneRectangle(
                    rect.x(),
                    rect.y(),
                    rect.width(),
                    rect.height(),
                    model=zone_model,
                )

                self.addItem(zone)

                if self.preview_rect:

                    self.removeItem(
                        self.preview_rect
                    )

                if self.dimension_text:

                    self.removeItem(
                        self.dimension_text
                    )

                self.preview_rect = None
                self.dimension_text = None
                self.start_point = None

            return

        # -------------------------------------------------
        # Exit Tool
        # -------------------------------------------------

        if self.current_tool == "exit":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            if self.start_point is None:

                self.start_point = (x, y)

                self.preview_line = QGraphicsLineItem(
                    0,
                    0,
                    0,
                    0,
                )

                self.preview_line.setPos(x, y)

                self.preview_line.setPen(
                    QPen(
                        QColor(
                            0,
                            255,
                            0,
                        ),
                        2,
                    )
                )

                self.addItem(
                    self.preview_line
                )

                self.dimension_text = (
                    QGraphicsSimpleTextItem()
                )

                self.dimension_text.setBrush(
                    QBrush(
                        QColor(
                            255,
                            255,
                            0,
                        )
                    )
                )

                self.dimension_text.setZValue(
                    1000
                )

                self.addItem(
                    self.dimension_text
                )

            else:

                x1, y1 = self.start_point

                exit_model = Exit(
                    name=f"Exit {self.current_floor.exit_count + 1}",
                    start_point=(
                        x1 / self.GRID_SIZE,
                        y1 / self.GRID_SIZE,
                    ),
                    end_point=(
                        x / self.GRID_SIZE,
                        y / self.GRID_SIZE,
                    ),
                    floor_id=self.current_floor.id,
                )

                self.current_floor.add_exit(
                    exit_model
                )

                exit_item = ExitItem(
                    x1,
                    y1,
                    x,
                    y,
                    model=exit_model,
                )

                self.addItem(exit_item)

                if self.preview_line:

                    self.removeItem(
                        self.preview_line
                    )

                if self.dimension_text:

                    self.removeItem(
                        self.dimension_text
                    )

                self.preview_line = None
                self.dimension_text = None
                self.start_point = None

            return

        # -------------------------------------------------
        # Stair Tool
        #
        # Guided three-step flow, spread across two clicks with a
        # floor switch in between rather than a drag-preview (the
        # two ends aren't even on the same canvas, so a live
        # preview line between them makes no sense here the way
        # it does for Exit/Door). Nothing is added to any floor's
        # model until the landing click completes it.
        #
        # Both clicks must land inside a Zone -- same "click inside a
        # Zone" contract Occupant/Camera/Detector already use -- and
        # both zone ids are captured automatically at creation, so a
        # completed Stair is *always* fully wired (from_zone_id AND
        # to_zone_id set) the moment it exists. There is no longer a
        # way to finish placing a Stair that produces no Navigation
        # Graph edge: the geometry and the engineering relationship
        # are established in the same two clicks, never a separate
        # manual step afterward.
        # -------------------------------------------------

        if self.current_tool == "stair":

            if self.pending_stair is None:

                if self.current_floor.locked:
                    return

                x, y = self.snap(
                    event.scenePos()
                )

                origin_zone = self._find_zone_at(
                    self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
                )

                if origin_zone is None:
                    return

                candidate_floors = [
                    floor
                    for floor in self.project.building.ordered_floors()
                    if floor.id != self.current_floor.id
                    and not floor.locked
                ]

                if not candidate_floors:
                    return

                if self.floor_picker_callback is None:
                    return

                destination_floor = self.floor_picker_callback(
                    candidate_floors
                )

                if destination_floor is None:
                    return

                if self._stair_connects(self.current_floor, destination_floor):

                    # A stair already connects these two floors --
                    # this is exactly the situation that used to
                    # invite a second, independent Staircase object
                    # for what should be one physical connector.
                    # MainWindow decides how to ask (Scene never
                    # shows a dialog itself, same convention
                    # floor_picker_callback already follows); no
                    # callback registered is treated conservatively
                    # as "don't proceed" rather than silently allowing
                    # a duplicate.
                    if self.duplicate_stair_confirmation_callback is None:
                        return

                    if not self.duplicate_stair_confirmation_callback(
                        self.current_floor, destination_floor,
                    ):
                        return

                self.pending_stair = Staircase(
                    name=f"Stair {self.current_floor.stair_count + 1}",
                    from_position=(
                        x / self.GRID_SIZE,
                        y / self.GRID_SIZE,
                    ),
                    from_floor_id=self.current_floor.id,
                    to_floor_id=destination_floor.id,
                    from_zone_id=origin_zone.id,
                )

                if self.floor_switch_requested_callback:

                    self.floor_switch_requested_callback(
                        destination_floor
                    )

            else:

                if self.current_floor.locked:

                    # Shouldn't happen -- locked floors are
                    # filtered out of the picker -- but never
                    # complete a placement on one regardless.
                    self.pending_stair = None

                    return

                x, y = self.snap(
                    event.scenePos()
                )

                landing_zone = self._find_zone_at(
                    self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
                )

                if landing_zone is None:

                    # Stays pending -- same "wait for a valid click"
                    # behavior a missed first click already has,
                    # rather than abandoning a placement the user is
                    # still in the middle of.
                    return

                self.pending_stair.to_position = (
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                )

                self.pending_stair.to_zone_id = landing_zone.id

                from_floor = self.project.building.get_floor(
                    self.pending_stair.from_floor_id
                )

                from_floor.add_stair(
                    self.pending_stair
                )

                stair_item = StairItem(
                    x,
                    y,
                    self.pending_stair.width,
                    "to",
                    model=self.pending_stair,
                )

                self.addItem(stair_item)

                self.pending_stair = None

            return

        # -------------------------------------------------
        # Camera Tool
        #
        # A Camera is a point object (position + rotation), not
        # a two-point line like Exit/Stair, so it is placed with
        # a single click instead of a click-drag-click sequence.
        # -------------------------------------------------

        if self.current_tool == "camera":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            camera_model = Camera(
                name=f"Camera {self.current_floor.camera_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            self.current_floor.add_camera(
                camera_model
            )

            camera_item = CameraItem(
                x,
                y,
                model=camera_model,
            )

            self.addItem(camera_item)

            return

        # -------------------------------------------------
        # Detector Tool
        #
        # A Detector is a point object (position + coverage
        # rectangle), placed with a single click just like Camera.
        # -------------------------------------------------

        if self.current_tool == "detector":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            detector_model = Detector(
                name=f"Detector {self.current_floor.detector_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            self.current_floor.add_detector(
                detector_model
            )

            detector_item = DetectorItem(
                x,
                y,
                model=detector_model,
            )

            self.addItem(detector_item)

            return

        # -------------------------------------------------
        # Smoke Detector Tool (Building Sensor Network Framework) --
        # a point object placed with a single click just like Camera/
        # Detector.
        # -------------------------------------------------

        if self.current_tool == "smoke_detector":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            smoke_detector_model = SmokeDetector(
                name=f"Smoke Detector {self.current_floor.smoke_detector_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            # Phase 4 -- a point detector auto-assigns the single zone
            # containing its position; ambiguous (overlapping zones) or
            # outside every zone both honestly leave zone_ids empty
            # rather than fabricating a nearest/first-match guess.
            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                smoke_detector_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_smoke_detector(
                smoke_detector_model
            )

            smoke_detector_item = SmokeDetectorItem(
                x,
                y,
                model=smoke_detector_model,
            )

            self.addItem(smoke_detector_item)

            return

        # -------------------------------------------------
        # Heat Detector Tool (Building Sensor Network Framework) --
        # a point object placed with a single click just like Camera/
        # Detector.
        # -------------------------------------------------

        if self.current_tool == "heat_detector":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            heat_detector_model = HeatDetector(
                name=f"Heat Detector {self.current_floor.heat_detector_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            # Phase 4 -- same auto-assignment rule as Smoke Detector
            # immediately above.
            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                heat_detector_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_heat_detector(
                heat_detector_model
            )

            heat_detector_item = HeatDetectorItem(
                x,
                y,
                model=heat_detector_model,
            )

            self.addItem(heat_detector_item)

            return

        # -------------------------------------------------
        # Speaker Tool (Zoned Voice Evacuation & Speaker Network
        # Framework) -- a point object placed with a single click just
        # like Camera/Detector/Smoke Detector/Heat Detector.
        # -------------------------------------------------

        if self.current_tool == "speaker":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            speaker_model = Speaker(
                name=f"Speaker {self.current_floor.speaker_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            self.current_floor.add_speaker(
                speaker_model
            )

            speaker_item = SpeakerItem(
                x,
                y,
                model=speaker_model,
            )

            self.addItem(speaker_item)

            return

        # -------------------------------------------------
        # Dynamic Sign Tool (Live Dynamic Evacuation Signage milestone)
        # -- a point object placed with a single click just like
        # Camera/Detector/Speaker.
        # -------------------------------------------------

        if self.current_tool == "sign":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            sign_model = DynamicEvacuationSign(
                name=f"Sign {self.current_floor.sign_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            self.current_floor.add_sign(
                sign_model
            )

            sign_item = SignItem(
                x,
                y,
                model=sign_model,
            )

            self.addItem(sign_item)

            return

        # -------------------------------------------------
        # Manual Call Point Tool (Manual Call Points & Emergency
        # Lighting milestone) -- a point object placed with a single
        # click just like Smoke/Heat Detector, with the exact same
        # unambiguous-zone auto-assignment (Phase 5).
        # -------------------------------------------------

        if self.current_tool == "manual_call_point":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            mcp_model = ManualCallPoint(
                name=f"Manual Call Point {self.current_floor.manual_call_point_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                mcp_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_manual_call_point(
                mcp_model
            )

            mcp_item = ManualCallPointItem(
                x,
                y,
                model=mcp_model,
            )

            self.addItem(mcp_item)

            return

        # -------------------------------------------------
        # Emergency Light Tool (Manual Call Points & Emergency Lighting
        # milestone) -- a point object placed with a single click,
        # same unambiguous-zone auto-assignment as Manual Call Point/
        # Smoke/Heat Detector above (Phase 7's own "the zone it
        # illuminates is, for any realistic placement, the zone
        # containing its own position" reasoning -- see docs/
        # architecture/manual_call_point_and_emergency_lighting.md).
        # -------------------------------------------------

        if self.current_tool == "emergency_light":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            light_model = EmergencyLight(
                name=f"Emergency Light {self.current_floor.emergency_light_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                light_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_emergency_light(
                light_model
            )

            light_item = EmergencyLightItem(
                x,
                y,
                model=light_model,
            )

            self.addItem(light_item)

            return

        # -------------------------------------------------
        # Sprinkler Tool (Fire Suppression & Water-Based Safety Asset
        # Digital Twin milestone) -- a point object placed with a
        # single click, same unambiguous-zone auto-assignment as
        # Manual Call Point/Smoke/Heat Detector above.
        # -------------------------------------------------

        if self.current_tool == "sprinkler":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            sprinkler_model = Sprinkler(
                name=f"Sprinkler {self.current_floor.sprinkler_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                sprinkler_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_sprinkler(
                sprinkler_model
            )

            sprinkler_item = SprinklerItem(
                x,
                y,
                model=sprinkler_model,
            )

            self.addItem(sprinkler_item)

            return

        # -------------------------------------------------
        # Fire Extinguisher Tool (Fire Suppression & Water-Based Safety
        # Asset Digital Twin milestone) -- same placement convention
        # as Sprinkler above.
        # -------------------------------------------------

        if self.current_tool == "fire_extinguisher":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            extinguisher_model = FireExtinguisher(
                name=f"Fire Extinguisher {self.current_floor.fire_extinguisher_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                extinguisher_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_fire_extinguisher(
                extinguisher_model
            )

            extinguisher_item = FireExtinguisherItem(
                x,
                y,
                model=extinguisher_model,
            )

            self.addItem(extinguisher_item)

            return

        # -------------------------------------------------
        # Fire Hydrant / Landing Valve Tool (Fire Suppression &
        # Water-Based Safety Asset Digital Twin milestone) -- same
        # placement convention as Sprinkler/Fire Extinguisher above.
        # -------------------------------------------------

        if self.current_tool == "fire_hydrant":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            hydrant_model = FireHydrant(
                name=f"Fire Hydrant {self.current_floor.fire_hydrant_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                hydrant_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_fire_hydrant(
                hydrant_model
            )

            hydrant_item = FireHydrantItem(
                x,
                y,
                model=hydrant_model,
            )

            self.addItem(hydrant_item)

            return

        # -------------------------------------------------
        # Hose Reel Tool (Fire Suppression & Water-Based Safety Asset
        # Digital Twin milestone) -- same placement convention as
        # Sprinkler/Fire Extinguisher/Fire Hydrant above.
        # -------------------------------------------------

        if self.current_tool == "hose_reel":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            hose_reel_model = HoseReel(
                name=f"Hose Reel {self.current_floor.hose_reel_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                hose_reel_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_hose_reel(
                hose_reel_model
            )

            hose_reel_item = HoseReelItem(
                x,
                y,
                model=hose_reel_model,
            )

            self.addItem(hose_reel_item)

            return

        # -------------------------------------------------
        # Fire Water Tank Tool (Fire Water Supply & Suppression
        # Infrastructure milestone) -- same placement convention as
        # every other point device above.
        # -------------------------------------------------

        if self.current_tool == "fire_water_tank":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            tank_model = FireWaterTank(
                name=f"Fire Water Tank {self.current_floor.fire_water_tank_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                tank_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_fire_water_tank(
                tank_model
            )

            tank_item = FireWaterTankItem(
                x,
                y,
                model=tank_model,
            )

            self.addItem(tank_item)

            return

        # -------------------------------------------------
        # Fire Pump Tool (Fire Water Supply & Suppression Infrastructure
        # milestone) -- same placement convention as above.
        # -------------------------------------------------

        if self.current_tool == "fire_pump":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            pump_model = FirePump(
                name=f"Fire Pump {self.current_floor.fire_pump_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                pump_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_fire_pump(
                pump_model
            )

            pump_item = FirePumpItem(
                x,
                y,
                model=pump_model,
            )

            self.addItem(pump_item)

            return

        # -------------------------------------------------
        # Jockey Pump Tool (Fire Water Supply & Suppression
        # Infrastructure milestone) -- same placement convention as
        # Fire Pump above.
        # -------------------------------------------------

        if self.current_tool == "jockey_pump":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            jockey_pump_model = JockeyPump(
                name=f"Jockey Pump {self.current_floor.jockey_pump_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                jockey_pump_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_jockey_pump(
                jockey_pump_model
            )

            jockey_pump_item = JockeyPumpItem(
                x,
                y,
                model=jockey_pump_model,
            )

            self.addItem(jockey_pump_item)

            return

        # -------------------------------------------------
        # Fire Service Inlet / Breeching Inlet Tool (Fire Water Supply &
        # Suppression Infrastructure milestone) -- same placement
        # convention as above.
        # -------------------------------------------------

        if self.current_tool == "fire_service_inlet":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            inlet_model = FireServiceInlet(
                name=f"Fire Service Inlet {self.current_floor.fire_service_inlet_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            containing_zone = self._find_unambiguous_zone_at(
                self.current_floor, x / self.GRID_SIZE, y / self.GRID_SIZE,
            )

            if containing_zone is not None:
                inlet_model.zone_ids = (containing_zone.id,)

            self.current_floor.add_fire_service_inlet(
                inlet_model
            )

            inlet_item = FireServiceInletItem(
                x,
                y,
                model=inlet_model,
            )

            self.addItem(inlet_item)

            return

        # -------------------------------------------------
        # Assembly Point Tool
        #
        # A permanent, purely geometric safe-destination marker,
        # placed with a single click just like Camera/Detector.
        # -------------------------------------------------

        if self.current_tool == "assembly_point":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            assembly_point_model = AssemblyPoint(
                name=f"Assembly Point {self.current_floor.assembly_point_count + 1}",
                position=(
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                ),
                floor_id=self.current_floor.id,
            )

            self.current_floor.add_assembly_point(
                assembly_point_model
            )

            assembly_point_item = AssemblyPointItem(
                x,
                y,
                assembly_point_model.length,
                assembly_point_model.width,
                model=assembly_point_model,
            )

            self.addItem(assembly_point_item)

            return

        # -------------------------------------------------
        # Obstacle Tool
        #
        # Click-drag-click rectangle, same interaction as the
        # Zone Tool -- Obstacle is a filled-rectangle object too.
        # -------------------------------------------------

        if self.current_tool == "obstacle":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            if self.start_point is None:

                self.start_point = (x, y)

                self.preview_rect = QGraphicsRectItem()

                self.preview_rect.setBrush(
                    QBrush(
                        QColor(
                            210,
                            60,
                            40,
                            60,
                        )
                    )
                )

                self.preview_rect.setPen(
                    QPen(
                        QColor(
                            210,
                            60,
                            40,
                        ),
                        2,
                    )
                )

                self.addItem(
                    self.preview_rect
                )

                self.dimension_text = (
                    QGraphicsSimpleTextItem()
                )

                self.dimension_text.setBrush(
                    QBrush(
                        QColor(
                            255,
                            255,
                            0,
                        )
                    )
                )

                self.dimension_text.setZValue(
                    1000
                )

                self.addItem(
                    self.dimension_text
                )

            else:

                x1, y1 = self.start_point

                rect = QRectF(
                    min(x1, x),
                    min(y1, y),
                    abs(x - x1),
                    abs(y - y1),
                )

                obstacle_model = Obstacle(
                    name=f"Obstacle {self.current_floor.obstacle_count + 1}",
                    x=rect.x() / self.GRID_SIZE,
                    y=rect.y() / self.GRID_SIZE,
                    length=rect.width() / self.GRID_SIZE,
                    width=rect.height() / self.GRID_SIZE,
                    floor_id=self.current_floor.id,
                )

                self.current_floor.add_obstacle(
                    obstacle_model
                )

                obstacle_item = ObstacleItem(
                    rect.x(),
                    rect.y(),
                    obstacle_model.length,
                    obstacle_model.width,
                    model=obstacle_model,
                )

                self.addItem(obstacle_item)

                if self.preview_rect:

                    self.removeItem(
                        self.preview_rect
                    )

                if self.dimension_text:

                    self.removeItem(
                        self.dimension_text
                    )

                self.preview_rect = None
                self.dimension_text = None
                self.start_point = None

            return

        # -------------------------------------------------
        # Door Tool
        #
        # Click-drag-click line, same interaction as Exit/Stair --
        # a Door is a traversable-connection line too. Connectivity
        # (Zone A / Zone B) is set afterwards in the Property
        # Panel, not by this placement click.
        # -------------------------------------------------

        if self.current_tool == "door":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            if self.start_point is None:

                self.start_point = (x, y)

                self.preview_line = QGraphicsLineItem(
                    0,
                    0,
                    0,
                    0,
                )

                self.preview_line.setPos(x, y)

                self.preview_line.setPen(
                    QPen(
                        QColor(
                            170,
                            100,
                            220,
                        ),
                        2,
                    )
                )

                self.addItem(
                    self.preview_line
                )

                self.dimension_text = (
                    QGraphicsSimpleTextItem()
                )

                self.dimension_text.setBrush(
                    QBrush(
                        QColor(
                            255,
                            255,
                            0,
                        )
                    )
                )

                self.dimension_text.setZValue(
                    1000
                )

                self.addItem(
                    self.dimension_text
                )

            else:

                x1, y1 = self.start_point

                door_model = Door(
                    name=f"Door {self.current_floor.door_count + 1}",
                    start_point=(
                        x1 / self.GRID_SIZE,
                        y1 / self.GRID_SIZE,
                    ),
                    end_point=(
                        x / self.GRID_SIZE,
                        y / self.GRID_SIZE,
                    ),
                    floor_id=self.current_floor.id,
                )

                self.current_floor.add_door(
                    door_model
                )

                door_item = DoorItem(
                    x1,
                    y1,
                    x,
                    y,
                    model=door_model,
                )

                self.addItem(door_item)

                if self.preview_line:

                    self.removeItem(
                        self.preview_line
                    )

                if self.dimension_text:

                    self.removeItem(
                        self.dimension_text
                    )

                self.preview_line = None
                self.dimension_text = None
                self.start_point = None

            return

        # -------------------------------------------------
        # Occupant Tool (Manual Simulation Sandbox)
        #
        # Click-drag-click rectangle, the same two-click interaction
        # Zone/Obstacle already use ("drag" in this Designer has
        # always meant "click one corner, move, click the opposite
        # corner" -- there is no mouseReleaseEvent-based dragging
        # anywhere in this scene, Zone included). The rectangle itself
        # is only a placement aid: nothing is added to Floor, and
        # nothing is added to the scene as a permanent item for it --
        # it exists only as a temporary preview between the two
        # clicks, same as Zone/Obstacle's own preview_rect.
        #
        # The actual generation (how many occupants, which
        # distribution) is decided by a dialog GraphicsScene never
        # shows itself -- occupant_generation_callback is MainWindow's
        # seam for that, the same "Scene never owns dialogs" contract
        # the Stair Tool's floor_picker_callback already established.
        # -------------------------------------------------

        if self.current_tool == "occupant":

            if self.current_floor.locked:
                return

            x, y = self.snap(
                event.scenePos()
            )

            if self.start_point is None:

                self.start_point = (x, y)

                self.preview_rect = QGraphicsRectItem()

                self.preview_rect.setBrush(
                    QBrush(
                        QColor(
                            255,
                            255,
                            255,
                            30,
                        )
                    )
                )

                self.preview_rect.setPen(
                    QPen(
                        QColor(
                            255,
                            255,
                            255,
                        ),
                        2,
                        Qt.PenStyle.DashLine,
                    )
                )

                self.addItem(
                    self.preview_rect
                )

                self.dimension_text = (
                    QGraphicsSimpleTextItem()
                )

                self.dimension_text.setBrush(
                    QBrush(
                        QColor(
                            255,
                            255,
                            0,
                        )
                    )
                )

                self.dimension_text.setZValue(
                    1000
                )

                self.addItem(
                    self.dimension_text
                )

            else:

                x1, y1 = self.start_point

                if self.preview_rect:

                    self.removeItem(
                        self.preview_rect
                    )

                if self.dimension_text:

                    self.removeItem(
                        self.dimension_text
                    )

                self.preview_rect = None
                self.dimension_text = None
                self.start_point = None

                self._generate_occupants_in_rectangle(
                    x1 / self.GRID_SIZE,
                    y1 / self.GRID_SIZE,
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                )

            return

        super().mousePressEvent(event)    # =====================================================

    def mouseMoveEvent(self, event):

        if (
            self.current_tool in ("zone", "obstacle", "occupant")
            and self.start_point
            and self.preview_rect
        ):

            x, y = self.snap(
                event.scenePos()
            )

            x1, y1 = self.start_point

            rect = QRectF(
                min(x1, x),
                min(y1, y),
                abs(x - x1),
                abs(y - y1),
            )

            self.preview_rect.setRect(rect)

            width = (
                rect.width()
                / self.GRID_SIZE
            )

            height = (
                rect.height()
                / self.GRID_SIZE
            )

            self.dimension_text.setText(
                f"{width:.1f} m × {height:.1f} m"
            )

            self.dimension_text.setPos(
                rect.center().x() - 35,
                rect.center().y() - 10,
            )

        if (
            self.current_tool in ("exit", "door")
            and self.start_point
            and self.preview_line
        ):

            x, y = self.snap(
                event.scenePos()
            )

            x1, y1 = self.start_point

            self.preview_line.setLine(
                0,
                0,
                x - x1,
                y - y1,
            )

            length = (
                (
                    (x - x1) ** 2
                    + (y - y1) ** 2
                )
                ** 0.5
            ) / self.GRID_SIZE

            self.dimension_text.setText(
                f"{length:.1f} m"
            )

            self.dimension_text.setPos(
                (x1 + x) / 2 - 20,
                (y1 + y) / 2 - 20,
            )

        super().mouseMoveEvent(event)

    # =====================================================

    def keyPressEvent(
        self,
        event: QKeyEvent,
    ):

        if event.key() == Qt.Key.Key_Delete:

            if (
                self.selected_item
                and not self.current_floor.locked
            ):

                if isinstance(
                    self.selected_item,
                    ZoneRectangle,
                ):

                    self.current_floor.remove_zone(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    ExitItem,
                ):

                    self.current_floor.remove_exit(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    StairItem,
                ):

                    # A Staircase always lives in its from_floor's
                    # Floor.stairs, regardless of which marker
                    # (entrance or landing) was actually clicked --
                    # deleting either one must remove the whole
                    # shared object, not just fail silently when
                    # viewed from the to-floor.
                    owning_floor = self.project.building.get_floor(
                        self.selected_item.model.from_floor_id
                    )

                    if owning_floor is not None:

                        owning_floor.remove_stair(
                            self.selected_item.model
                        )

                elif isinstance(
                    self.selected_item,
                    CameraItem,
                ):

                    self.current_floor.remove_camera(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    DetectorItem,
                ):

                    self.current_floor.remove_detector(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    SmokeDetectorItem,
                ):

                    self.current_floor.remove_smoke_detector(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    HeatDetectorItem,
                ):

                    self.current_floor.remove_heat_detector(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    SpeakerItem,
                ):

                    self.current_floor.remove_speaker(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    SignItem,
                ):

                    self.current_floor.remove_sign(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    ManualCallPointItem,
                ):

                    self.current_floor.remove_manual_call_point(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    EmergencyLightItem,
                ):

                    self.current_floor.remove_emergency_light(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    AssemblyPointItem,
                ):

                    self.current_floor.remove_assembly_point(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    ObstacleItem,
                ):

                    self.current_floor.remove_obstacle(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    DoorItem,
                ):

                    self.current_floor.remove_door(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    OccupantItem,
                ):

                    # Never touches Floor/Building -- an Occupant was
                    # never added to either in the first place.
                    self.sandbox_manager.remove_occupant(
                        self.selected_item.occupant
                    )

                    self.occupant_items.pop(
                        self.selected_item.occupant.occupant_id, None,
                    )

                self._clear_route_highlight()

                # A selected OccupantItem can legitimately no longer
                # be a member of this scene at all -- see
                # sync_occupants(): an occupant that walked onto a
                # different floor than the one displayed keeps its
                # selection so the Property Panel can keep tracking
                # it, but its old item was already removed. Every
                # other selectable item type is always still in this
                # scene when selected, so this changes nothing for them.
                if self.selected_item.scene() is self:

                    self.removeItem(
                        self.selected_item
                    )

                self.selected_item = None

                if (
                    self.selection_changed_callback
                ):

                    self.selection_changed_callback(
                        None
                    )

                return

        super().keyPressEvent(event)

    # =====================================================

    def get_project(self):

        return self.project

    # =====================================================

    def get_current_floor(self):

        return self.current_floor

    # =====================================================

    def get_zone_models(self):

        return self.current_floor.zones

    # =====================================================

    def clear_graphics_items(self):

        # Clears rendered items only. Must never mutate the
        # floor's model data -- Building/Floor own that. OccupantItem
        # is included for the same reason (a rendering cache over
        # sandbox_manager.occupants, not the source of truth itself)
        # even though there is no Floor list to leave untouched for it.
        for item in list(self.items()):

            if isinstance(
                item,
                (ZoneRectangle, ExitItem, StairItem, CameraItem, DetectorItem, SmokeDetectorItem, HeatDetectorItem, SpeakerItem, SignItem, ManualCallPointItem, EmergencyLightItem, SprinklerItem, FireExtinguisherItem, FireHydrantItem, HoseReelItem, FireWaterTankItem, FirePumpItem, JockeyPumpItem, FireServiceInletItem, AssemblyPointItem, ObstacleItem, DoorItem, OccupantItem),
            ):
                self.removeItem(item)

        self.selected_item = None

        self.occupant_items = {}
        self._highlighted_route_items = []

        for item in self._coverage_overlay_items:
            self.removeItem(item)

        self._coverage_overlay_items = []

    # =====================================================

    def rebuild_scene(self):

        self.clear_graphics_items()

        if self.selection_changed_callback:
            self.selection_changed_callback(None)

        self._display_floor_plan()

        movable = not self.current_floor.locked

        for zone in self.current_floor.zones:

            rect = ZoneRectangle(
                zone.x * self.GRID_SIZE,
                zone.y * self.GRID_SIZE,
                zone.width * self.GRID_SIZE,
                zone.height * self.GRID_SIZE,
                model=zone,
            )

            rect.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(rect)

        for exit_obj in self.current_floor.exits:

            x1, y1 = exit_obj.start_point
            x2, y2 = exit_obj.end_point

            exit_item = ExitItem(
                x1 * self.GRID_SIZE,
                y1 * self.GRID_SIZE,
                x2 * self.GRID_SIZE,
                y2 * self.GRID_SIZE,
                model=exit_obj,
            )

            exit_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(exit_item)

        # A Staircase is one object rendered on BOTH floors it
        # connects: the entrance marker where it's actually owned
        # (self.current_floor.stairs), and the landing marker by
        # scanning every other floor for a stair whose
        # to_floor_id names this floor. Both markers share the
        # same model -- selecting/moving/deleting either one acts
        # on the one real Staircase.

        for stair_obj in self.current_floor.stairs:

            x, y = stair_obj.from_position

            stair_item = StairItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                stair_obj.width,
                "from",
                model=stair_obj,
            )

            stair_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(stair_item)

        for other_floor in self.project.building.floors:

            if other_floor.id == self.current_floor.id:
                continue

            for stair_obj in other_floor.stairs:

                if stair_obj.to_floor_id != self.current_floor.id:
                    continue

                x, y = stair_obj.to_position

                stair_item = StairItem(
                    x * self.GRID_SIZE,
                    y * self.GRID_SIZE,
                    stair_obj.width,
                    "to",
                    model=stair_obj,
                )

                stair_item.setFlag(
                    QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                    movable,
                )

                self.addItem(stair_item)

        for camera_obj in self.current_floor.cameras:

            x, y = camera_obj.position

            camera_item = CameraItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=camera_obj,
            )

            camera_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(camera_item)

        for detector_obj in self.current_floor.detectors:

            x, y = detector_obj.position

            detector_item = DetectorItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=detector_obj,
            )

            detector_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(detector_item)

        for smoke_detector_obj in self.current_floor.smoke_detectors:

            x, y = smoke_detector_obj.position

            smoke_detector_item = SmokeDetectorItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=smoke_detector_obj,
            )

            smoke_detector_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(smoke_detector_item)

        for heat_detector_obj in self.current_floor.heat_detectors:

            x, y = heat_detector_obj.position

            heat_detector_item = HeatDetectorItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=heat_detector_obj,
            )

            heat_detector_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(heat_detector_item)

        for speaker_obj in self.current_floor.speakers:

            x, y = speaker_obj.position

            speaker_item = SpeakerItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=speaker_obj,
            )

            speaker_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(speaker_item)

        for sign_obj in self.current_floor.signs:

            x, y = sign_obj.position

            sign_item = SignItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=sign_obj,
            )

            sign_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(sign_item)

        for mcp_obj in self.current_floor.manual_call_points:

            x, y = mcp_obj.position

            mcp_item = ManualCallPointItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=mcp_obj,
            )

            mcp_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(mcp_item)

        for light_obj in self.current_floor.emergency_lights:

            x, y = light_obj.position

            light_item = EmergencyLightItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=light_obj,
            )

            light_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(light_item)

        # SynEvac Designer Simplification & Product-Boundary Cleanup
        # milestone, Phase 8 -- a genuine, pre-existing gap this
        # milestone's own backward-compatibility testing surfaced:
        # every one of these eight fire-safety/water-infrastructure
        # asset types is placed correctly (click-to-place already
        # constructs the matching *Item below, see this method's own
        # mousePressEvent branches) and serializes/deserializes
        # correctly (Floor.sprinklers etc. round-trip via to_dict/
        # from_dict, unchanged), but rebuild_scene() itself -- the one
        # method that reconstructs every graphics item from the model
        # after a project load or a floor switch -- never iterated
        # these eight lists at all. A loaded project containing a
        # Sprinkler (or any of its seven siblings below) would
        # therefore silently vanish from the canvas on open/floor-
        # switch, despite remaining fully present, editable, and
        # save-able in the underlying model. Fixed here, following the
        # exact same per-item construction pattern EmergencyLight
        # (immediately above) already establishes -- no new behavior,
        # no new model, only restoring the same rendering every other
        # asset type already had.

        for sprinkler_obj in self.current_floor.sprinklers:

            x, y = sprinkler_obj.position

            sprinkler_item = SprinklerItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=sprinkler_obj,
            )

            sprinkler_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(sprinkler_item)

        for fire_extinguisher_obj in self.current_floor.fire_extinguishers:

            x, y = fire_extinguisher_obj.position

            fire_extinguisher_item = FireExtinguisherItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=fire_extinguisher_obj,
            )

            fire_extinguisher_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(fire_extinguisher_item)

        for fire_hydrant_obj in self.current_floor.fire_hydrants:

            x, y = fire_hydrant_obj.position

            fire_hydrant_item = FireHydrantItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=fire_hydrant_obj,
            )

            fire_hydrant_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(fire_hydrant_item)

        for hose_reel_obj in self.current_floor.hose_reels:

            x, y = hose_reel_obj.position

            hose_reel_item = HoseReelItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=hose_reel_obj,
            )

            hose_reel_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(hose_reel_item)

        for fire_water_tank_obj in self.current_floor.fire_water_tanks:

            x, y = fire_water_tank_obj.position

            fire_water_tank_item = FireWaterTankItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=fire_water_tank_obj,
            )

            fire_water_tank_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(fire_water_tank_item)

        for fire_pump_obj in self.current_floor.fire_pumps:

            x, y = fire_pump_obj.position

            fire_pump_item = FirePumpItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=fire_pump_obj,
            )

            fire_pump_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(fire_pump_item)

        for jockey_pump_obj in self.current_floor.jockey_pumps:

            x, y = jockey_pump_obj.position

            jockey_pump_item = JockeyPumpItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=jockey_pump_obj,
            )

            jockey_pump_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(jockey_pump_item)

        for fire_service_inlet_obj in self.current_floor.fire_service_inlets:

            x, y = fire_service_inlet_obj.position

            fire_service_inlet_item = FireServiceInletItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                model=fire_service_inlet_obj,
            )

            fire_service_inlet_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(fire_service_inlet_item)

        for assembly_point_obj in self.current_floor.assembly_points:

            x, y = assembly_point_obj.position

            assembly_point_item = AssemblyPointItem(
                x * self.GRID_SIZE,
                y * self.GRID_SIZE,
                assembly_point_obj.length,
                assembly_point_obj.width,
                model=assembly_point_obj,
            )

            assembly_point_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(assembly_point_item)

        for obstacle_obj in self.current_floor.obstacles:

            obstacle_item = ObstacleItem(
                obstacle_obj.x * self.GRID_SIZE,
                obstacle_obj.y * self.GRID_SIZE,
                obstacle_obj.length,
                obstacle_obj.width,
                model=obstacle_obj,
            )

            obstacle_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(obstacle_item)

        for door_obj in self.current_floor.doors:

            x1, y1 = door_obj.start_point
            x2, y2 = door_obj.end_point

            door_item = DoorItem(
                x1 * self.GRID_SIZE,
                y1 * self.GRID_SIZE,
                x2 * self.GRID_SIZE,
                y2 * self.GRID_SIZE,
                model=door_obj,
            )

            door_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                movable,
            )

            self.addItem(door_item)

        # Manual Simulation Sandbox -- occupants are never read from
        # Floor (there is no Floor.occupants), only from
        # sandbox_manager, and only whichever ones currently stand on
        # THIS floor. An occupant elsewhere in the building simply has
        # no on-screen item until its floor is shown, exactly like a
        # Staircase's far marker.
        for occupant in self.sandbox_manager.occupants_on_floor(self.current_floor.id):

            occupant_item = OccupantItem(occupant)

            self.occupant_items[occupant.occupant_id] = occupant_item

            self.addItem(occupant_item)

        self.refresh_camera_coverage()

    # =====================================================
    # Camera Coverage & Visibility Engine -- Designer visualization
    #
    # Purely additive over the always-on naive FOV cone every
    # CameraItem already draws (designer/items/camera_item.py,
    # unchanged) -- this overlay shows the occlusion-aware truth
    # (visibility/engine.py, visibility/coverage.py) on top of it,
    # only while toggled on, and recomputed from scratch on every
    # call rather than incrementally patched. For a Designer-sized
    # floor plan (tens of zones/obstacles, a handful of cameras) this
    # is comfortably sub-frame; see visibility/engine.py's own
    # performance notes for the cost model if that assumption ever
    # needs revisiting for a much larger floor.
    # =====================================================

    def set_show_camera_coverage(self, enabled):

        self.show_camera_coverage = enabled

        self.refresh_camera_coverage()

    # =====================================================

    def refresh_camera_coverage(self):

        for item in self._coverage_overlay_items:
            self.removeItem(item)

        self._coverage_overlay_items = []

        if not self.show_camera_coverage:
            return

        if self.current_floor is None or self.project.building is None:
            return

        from visibility.coverage import compute_floor_coverage

        result = compute_floor_coverage(
            self.current_floor.cameras, self.project.building, self.current_floor,
        )

        self._draw_zone_coverage_overlays(result)
        self._draw_camera_visibility_polygons(result)

    # =====================================================

    def _draw_zone_coverage_overlays(self, floor_coverage):

        for zone in self.current_floor.zones:

            if zone.id in floor_coverage.uncovered_zone_ids:
                color = QColor(220, 40, 40, 90)
            elif zone.id in floor_coverage.overlapping_zone_ids:
                color = QColor(180, 40, 220, 90)
            else:
                continue

            overlay = QGraphicsRectItem(
                zone.x * self.GRID_SIZE,
                zone.y * self.GRID_SIZE,
                zone.width * self.GRID_SIZE,
                zone.height * self.GRID_SIZE,
            )

            overlay.setBrush(QBrush(color))
            overlay.setPen(QPen(Qt.PenStyle.NoPen))

            self._add_coverage_overlay(overlay, z_value=5)

    # =====================================================

    def _draw_camera_visibility_polygons(self, floor_coverage):

        for camera_id, visibility in floor_coverage.per_camera.items():

            if len(visibility.visibility_polygon) < 3:
                continue

            polygon = QPolygonF(
                [
                    QPointF(x * self.GRID_SIZE, y * self.GRID_SIZE)
                    for x, y in visibility.visibility_polygon
                ]
            )

            overlay = self.addPolygon(
                polygon,
                QPen(QColor(0, 255, 140, 200), 2),
                QBrush(QColor(0, 255, 140, 45)),
            )

            self._add_coverage_overlay(overlay, z_value=6)

    # =====================================================

    def _add_coverage_overlay(self, item, z_value):

        item.setZValue(z_value)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self._coverage_overlay_items.append(item)

        if item.scene() is None:
            self.addItem(item)

    # =====================================================
    # Manual Simulation Sandbox -- route highlighting
    #
    # Only ever touches items already on screen (self.items()), so
    # this naturally highlights just the portion of a route that lives
    # on the currently displayed floor -- the same single-floor-view
    # constraint every other cross-floor object (Staircase) already
    # has. Selecting a different item, switching floors (rebuild_scene
    # already clears selection first), or deleting the occupant all
    # clear this the same way selection itself already does.
    # =====================================================

    def _clear_route_highlight(self):

        for item in self._highlighted_route_items:
            item.set_highlighted(False)

        self._highlighted_route_items = []

    # =====================================================

    def _highlight_route(self, occupant):

        self._clear_route_highlight()

        if occupant is None or occupant.route is None:
            return

        node_ids = set(occupant.route.node_ids)
        edge_ids = set(occupant.route.edge_ids)

        for item in self.items():

            if (
                isinstance(item, ZoneRectangle)
                and item.model is not None
                and item.model.id in node_ids
            ):

                item.set_highlighted(True)
                self._highlighted_route_items.append(item)

            elif (
                isinstance(item, (DoorItem, StairItem))
                and item.model is not None
                and item.model.id in edge_ids
            ):

                item.set_highlighted(True)
                self._highlighted_route_items.append(item)

    # =====================================================
    # Called by MainWindow after the Property Panel recomputes an
    # occupant's route (destination changed) -- re-draws the
    # highlight for whichever occupant is still actually selected,
    # a no-op if the user has since selected something else.
    # =====================================================

    def refresh_occupant_route_highlight(self, occupant):

        if (
            isinstance(self.selected_item, OccupantItem)
            and self.selected_item.occupant is occupant
        ):
            self._highlight_route(occupant)

    # =====================================================
    # Manual Simulation Sandbox -- called by MainWindow after every
    # tick()/step() across every occupant (not just rebuild_scene()).
    #
    # rebuild_scene() alone is not enough here: it only re-renders
    # occupants when the DISPLAYED floor changes (a user action), but
    # an occupant can cross a Stair onto a different floor mid-
    # simulation while the user keeps watching the floor they started
    # on. Without this, that occupant's OccupantItem was never removed
    # -- it kept rendering on the wrong floor's view, repositioned to
    # its new floor's local meter coordinates as if they belonged to
    # the floor still on screen (two floors' Zones are not required to
    # share a coordinate system, so this could place the marker
    # anywhere). This reconciles occupant_items with "who is actually
    # on the displayed floor right now" every single step/tick, the
    # same way rebuild_scene() already does on a floor switch --
    # removing items for occupants who left, adding items for ones who
    # arrived (e.g. via a Stair from a floor that wasn't shown), and
    # only ever syncing position/appearance for ones who stayed.
    # =====================================================

    def sync_occupants(self):

        # An occupant merely walking onto a different floor than the
        # one on screen is not the same event as being deleted -- the
        # occupant is still alive and simulating correctly, just not
        # renderable on this floor right now. Capturing the selected
        # occupant (not the transient OccupantItem instance, which is
        # about to be thrown away) is what lets the Property Panel
        # keep tracking it across a Stair crossing instead of the
        # selection silently clearing out from under whoever was
        # watching it -- previously indistinguishable from a crash.
        selected_occupant = (
            self.selected_item.occupant
            if isinstance(self.selected_item, OccupantItem)
            else None
        )

        current_ids = {
            occupant.occupant_id
            for occupant in self.sandbox_manager.occupants_on_floor(self.current_floor.id)
        }

        for occupant_id in list(self.occupant_items.keys()):

            if occupant_id in current_ids:
                continue

            item = self.occupant_items.pop(occupant_id)

            if item in self._highlighted_route_items:
                self._highlighted_route_items.remove(item)

            # Deliberately does NOT clear self.selected_item or fire
            # selection_changed_callback here -- see selected_occupant
            # above. If this occupant reappears on the now-displayed
            # floor later in this same call (having arrived via
            # another Stair), the loop below re-attaches the
            # selection to its new item; if not, selected_item is left
            # pointing at this now-scene-less item, which still holds
            # a perfectly live `occupant` reference for the Property
            # Panel to keep reading.
            self.removeItem(item)

        occupants_by_id = {
            occupant.occupant_id: occupant
            for occupant in self.sandbox_manager.occupants
        }

        for occupant_id in current_ids:

            if occupant_id in self.occupant_items:

                self.occupant_items[occupant_id].sync_from_occupant()

            else:

                occupant = occupants_by_id[occupant_id]
                occupant_item = OccupantItem(occupant)

                self.occupant_items[occupant_id] = occupant_item

                self.addItem(occupant_item)

                if selected_occupant is occupant:

                    # The previously-selected occupant just arrived on
                    # THIS floor (e.g. via a Stair from one that
                    # wasn't displayed) -- re-attach the selection to
                    # its new item so it renders highlighted again,
                    # exactly as if it had never left.
                    self.selected_item = occupant_item

                    occupant_item.set_selected(True)

    # =====================================================
    # Manual Simulation Sandbox -- the Occupant Tool's drag-rectangle
    # workflow finishes here. Asks MainWindow (via
    # occupant_generation_callback) for how many occupants and which
    # distribution, same "Scene never shows a dialog" contract the
    # Stair Tool's floor_picker_callback already established -- a
    # cancelled dialog (or no callback registered at all) generates
    # nothing. Every resulting occupant is given the same default
    # destination (Exit) a single click used to assign immediately,
    # so there is always a route to see/animate without an extra step.
    # =====================================================

    def _generate_occupants_in_rectangle(self, x1_m, y1_m, x2_m, y2_m):

        if self.occupant_generation_callback is None:
            return

        result = self.occupant_generation_callback()

        if result is None:
            return

        count, distribution = result

        occupants = self.sandbox_manager.generate_occupants(
            self.current_floor, x1_m, y1_m, x2_m, y2_m, count, distribution,
        )

        for occupant in occupants:

            self.sandbox_manager.compute_route(
                occupant, self.project.building, SandboxDestinationType.EXIT,
            )

            occupant_item = OccupantItem(occupant)

            self.occupant_items[occupant.occupant_id] = occupant_item

            self.addItem(occupant_item)