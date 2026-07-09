from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QGraphicsView

from designer.scene.graphics_scene import GraphicsScene


class GraphicsView(QGraphicsView):

    MIN_ZOOM = 10
    MAX_ZOOM = 500
    ZOOM_FACTOR = 1.15

    def __init__(self):
        super().__init__()

        self.scene_obj = GraphicsScene()
        self.setScene(self.scene_obj)

        self.status_callback = None

        self.zoom_level = 100

        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        self.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        self.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform
        )

        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
        )

    # =====================================================

    def mouseMoveEvent(self, event):

        scene_pos = self.mapToScene(event.pos())

        if self.status_callback:

            self.status_callback(
                scene_pos.x()
                / self.scene_obj.GRID_SIZE,
                scene_pos.y()
                / self.scene_obj.GRID_SIZE,
            )

        super().mouseMoveEvent(event)

    # =====================================================

    def wheelEvent(self, event):

        if event.angleDelta().y() > 0:

            factor = self.ZOOM_FACTOR

            self.zoom_level = min(
                self.MAX_ZOOM,
                int(self.zoom_level * factor),
            )

        else:

            factor = 1 / self.ZOOM_FACTOR

            self.zoom_level = max(
                self.MIN_ZOOM,
                int(self.zoom_level * factor),
            )

        self.scale(factor, factor)

    # =====================================================

    def load_floor_plan(self, image_path):

        self.scene_obj.load_floor_plan(image_path)

        self.fitInView(
            self.scene_obj.itemsBoundingRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

        self.zoom_level = 100

    # =====================================================

    def set_tool(self, tool):

        self.scene_obj.set_tool(tool)

    # =====================================================

    def reset_view(self):

        self.resetTransform()

        self.zoom_level = 100

    # =====================================================

    def zoom_in(self):

        self.scale(
            self.ZOOM_FACTOR,
            self.ZOOM_FACTOR,
        )

    # =====================================================

    def zoom_out(self):

        self.scale(
            1 / self.ZOOM_FACTOR,
            1 / self.ZOOM_FACTOR,
        )

    # =====================================================

    def current_project(self):

        return self.scene_obj.project

    # =====================================================

    def current_floor(self):

        return self.scene_obj.current_floor