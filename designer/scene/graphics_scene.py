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

from designer.items.camera_item import CameraItem
from designer.items.exit_item import ExitItem
from designer.items.stair_item import StairItem
from designer.items.zone_rectangle import ZoneRectangle

from models.project import Project
from models.camera import Camera
from models.exit import Exit
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

            if isinstance(item, (ZoneRectangle, ExitItem, StairItem, CameraItem)):

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
        # -------------------------------------------------

        if self.current_tool == "stair":

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
                            180,
                            120,
                            40,
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

                stair_model = Staircase(
                    name=f"Stair {self.current_floor.stair_count + 1}",
                    start_point=(
                        x1 / self.GRID_SIZE,
                        y1 / self.GRID_SIZE,
                    ),
                    end_point=(
                        x / self.GRID_SIZE,
                        y / self.GRID_SIZE,
                    ),
                    from_floor_id=self.current_floor.id,
                )

                self.current_floor.add_stair(
                    stair_model
                )

                stair_item = StairItem(
                    x1,
                    y1,
                    x,
                    y,
                    model=stair_model,
                )

                self.addItem(stair_item)

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

        super().mousePressEvent(event)    # =====================================================

    def mouseMoveEvent(self, event):

        if (
            self.current_tool == "zone"
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
            self.current_tool in ("exit", "stair")
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

                    self.current_floor.remove_stair(
                        self.selected_item.model
                    )

                elif isinstance(
                    self.selected_item,
                    CameraItem,
                ):

                    self.current_floor.remove_camera(
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
                (ZoneRectangle, ExitItem, StairItem, CameraItem),
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

        for stair_obj in self.current_floor.stairs:

            x1, y1 = stair_obj.start_point
            x2, y2 = stair_obj.end_point

            stair_item = StairItem(
                x1 * self.GRID_SIZE,
                y1 * self.GRID_SIZE,
                x2 * self.GRID_SIZE,
                y2 * self.GRID_SIZE,
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