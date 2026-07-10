from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QBrush, QKeyEvent, QPen, QPixmap
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
from designer.items.obstacle_item import ObstacleItem
from designer.items.stair_item import StairItem
from designer.items.zone_rectangle import ZoneRectangle

from models.project import Project
from models.assembly_point import AssemblyPoint
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.obstacle import Obstacle
from models.staircase import Staircase
from models.zone import Zone


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

        self.selected_item = None

        self.selection_changed_callback = None

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

            if isinstance(item, (ZoneRectangle, ExitItem, StairItem, CameraItem, DetectorItem, AssemblyPointItem, ObstacleItem, DoorItem)):

                self.selected_item = item

                item.set_selected(True)

                if self.selection_changed_callback:
                    self.selection_changed_callback(item)

            else:

                self.selected_item = None

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
                    name=f"Zone {self.current_floor.zone_count + 1}",
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
                )

                zone.model = zone_model

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
        # -------------------------------------------------

        if self.current_tool == "stair":

            if self.pending_stair is None:

                if self.current_floor.locked:
                    return

                x, y = self.snap(
                    event.scenePos()
                )

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

                self.pending_stair = Staircase(
                    name=f"Stair {self.current_floor.stair_count + 1}",
                    from_position=(
                        x / self.GRID_SIZE,
                        y / self.GRID_SIZE,
                    ),
                    from_floor_id=self.current_floor.id,
                    to_floor_id=destination_floor.id,
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

                self.pending_stair.to_position = (
                    x / self.GRID_SIZE,
                    y / self.GRID_SIZE,
                )

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

        super().mousePressEvent(event)    # =====================================================

    def mouseMoveEvent(self, event):

        if (
            self.current_tool in ("zone", "obstacle")
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
        # floor's model data -- Building/Floor own that.
        for item in list(self.items()):

            if isinstance(
                item,
                (ZoneRectangle, ExitItem, StairItem, CameraItem, DetectorItem, AssemblyPointItem, ObstacleItem, DoorItem),
            ):
                self.removeItem(item)

        self.selected_item = None

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
            )

            rect.model = zone

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