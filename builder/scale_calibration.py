import math

from PyQt6.QtCore import QEvent, QObject, Qt


class ScaleCalibrationError(Exception):

    # Raised by compute_scale_pixels_per_meter() for a degenerate
    # calibration (the two points coincide, or a non-positive
    # real-world distance was entered) -- never silently produces an
    # infinite or zero scale.
    pass


def compute_scale_pixels_per_meter(point_a, point_b, distance_m):

    # Pure math, no Qt/UI dependency -- point_a/point_b are (x, y)
    # pairs in the RAW floor plan IMAGE's own pixel space (see
    # ScaleCalibrationController below for how those are captured).
    # Returns pixels-per-meter, the same unit Floor.floor_plan_scale
    # stores.

    pixel_distance = math.hypot(
        point_b[0] - point_a[0],
        point_b[1] - point_a[1],
    )

    if pixel_distance <= 0:
        raise ScaleCalibrationError(
            "The two calibration points must not be the same point."
        )

    if distance_m is None or distance_m <= 0:
        raise ScaleCalibrationError(
            "The real-world distance must be a positive number."
        )

    return pixel_distance / distance_m


class ScaleCalibrationController(QObject):

    # Captures exactly two left-clicks on a GraphicsView's viewport and
    # reports them back as points in the CURRENT floor plan image's own
    # pixel space -- via a Qt event filter installed only while active,
    # never a new branch in GraphicsScene.mousePressEvent's existing
    # tool-dispatch chain (that 2700+-line method is shared with Studio
    # and is deliberately left untouched by this milestone; see
    # docs/architecture/synevac_builder_feasibility_investigation.md).
    # Consuming the events at the viewport level also means whatever
    # GraphicsScene.current_tool happens to be selected is irrelevant
    # and untouched during calibration.

    def __init__(self, view):
        super().__init__()

        self.view = view

        self.on_points_chosen = None
        self.on_cancelled = None

        self._active = False
        self._first_point_scene = None

    # =====================================================

    @property
    def active(self):

        return self._active

    # =====================================================

    def start(self):

        if self._active:
            return

        self._active = True
        self._first_point_scene = None

        self.view.viewport().installEventFilter(self)
        self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)

    # =====================================================

    def cancel(self):

        if not self._active:
            return

        self._stop()

        if self.on_cancelled:
            self.on_cancelled()

    # =====================================================

    def _stop(self):

        self._active = False
        self._first_point_scene = None

        self.view.viewport().removeEventFilter(self)
        self.view.viewport().unsetCursor()

    # =====================================================

    def eventFilter(self, watched, event):

        if not self._active:
            return False

        if event.type() != QEvent.Type.MouseButtonPress:
            return False

        if event.button() != Qt.MouseButton.LeftButton:
            return False

        scene_point = self.view.mapToScene(event.pos())

        if self._first_point_scene is None:

            self._first_point_scene = scene_point

            return True

        first_scene = self._first_point_scene
        second_scene = scene_point

        self._stop()

        if self.on_points_chosen:
            self.on_points_chosen(first_scene, second_scene)

        return True
