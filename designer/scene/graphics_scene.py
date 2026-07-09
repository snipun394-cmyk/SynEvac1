from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QBrush, QKeyEvent, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
)

from designer.items.zone_rectangle import ZoneRectangle

from models.project import Project
from models.building import Building
from models.floor import Floor
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
        # -------------------------------------------------

        self.project = Project(
            name="Untitled Project"
        )

        building = Building(
            id="B001",
            name="Building",
        )

        floor = Floor(
            id="F001",
            name="Ground Floor",
        )

        building.add_floor(floor)

        self.project.set_building(building)

        self.current_floor = floor

        # -------------------------------------------------

        self.current_tool = "select"

        self.floor_plan = None

        self.start_point = None
        self.preview_rect = None
        self.dimension_text = None

        self.selected_zone = None

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

    def load_floor_plan(self, image_path):

        if self.floor_plan:

            self.removeItem(self.floor_plan)

        self.floor_plan = QGraphicsPixmapItem(
            QPixmap(image_path)
        )

        self.floor_plan.setZValue(-100)

        self.addItem(self.floor_plan)

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

            if self.selected_zone:
                self.selected_zone.set_selected(False)

            if isinstance(item, ZoneRectangle):

                self.selected_zone = item

                item.set_selected(True)

                if self.selection_changed_callback:
                    self.selection_changed_callback(item)

            else:

                self.selected_zone = None

                if self.selection_changed_callback:
                    self.selection_changed_callback(None)

            return

        # -------------------------------------------------
        # Zone Tool
        # -------------------------------------------------

        if self.current_tool == "zone":

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

        super().mouseMoveEvent(event)

    # =====================================================

    def keyPressEvent(
        self,
        event: QKeyEvent,
    ):

        if event.key() == Qt.Key.Key_Delete:

            if self.selected_zone:

                if hasattr(
                    self.selected_zone,
                    "model",
                ):

                    self.current_floor.remove_zone(
                        self.selected_zone.model
                    )

                self.removeItem(
                    self.selected_zone
                )

                self.selected_zone = None

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

    def clear_project(self):

        for item in list(self.items()):

            if isinstance(
                item,
                ZoneRectangle,
            ):
                self.removeItem(item)

        self.current_floor.zones.clear()

        self.selected_zone = None

    # =====================================================

    def rebuild_scene(self):

        self.clear_project()

        for zone in self.current_floor.zones:

            rect = ZoneRectangle(
                zone.x * self.GRID_SIZE,
                zone.y * self.GRID_SIZE,
                zone.width * self.GRID_SIZE,
                zone.height * self.GRID_SIZE,
            )

            rect.model = zone

            self.addItem(rect)