from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from camera_manager.connection_status import CameraConnectionState


BOX_COLOR = QColor(0, 220, 120)
BOX_PEN_WIDTH = 2
FPS_SMOOTHING_WINDOW = 10


class CameraTileWidget(QWidget):

    # One grid cell of command_center/live_camera_grid_panel.py's
    # LiveCameraGridPanel -- a dumb widget, same "MainWindow/Dashboard
    # pushes updates in via one refresh()/show_x() entry point"
    # convention every other Command Center panel already follows.
    # Owns no timer, no RTSP/decoder/detector reference of any kind --
    # purely a consumer of whatever CameraTileData its gateway already
    # computed this cycle (command_center/live_camera_view_gateway.py).
    # Overlay boxes/track ID/confidence are drawn from the SAME
    # RawHumanDetection tuple LiveCameraPipeline.run_cycle() already
    # produced this cycle -- no second detector invocation happens
    # anywhere in this file.

    def __init__(self):
        super().__init__()

        self._frame_timestamps = []  # rolling window for the FPS display

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        self.setLayout(layout)

        self.name_label = QLabel("No camera configured")
        self.name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.name_label)

        self.status_label = QLabel("-")
        layout.addWidget(self.status_label)

        self.video_label = QLabel("No signal")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(240, 180)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label, 1)

        self.detail_label = QLabel("-")
        self.detail_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.detail_label)

    # =====================================================
    # Public entry point
    # =====================================================

    def refresh(self, tile_data) -> None:

        self.name_label.setText(tile_data.name or tile_data.camera_id)

        if not tile_data.configured:

            self.status_label.setText("Not Configured")
            self._clear_video()
            self.detail_label.setText("-")
            self._frame_timestamps.clear()
            return

        status = tile_data.connection_status
        status_text = status.value if isinstance(status, CameraConnectionState) else str(status)
        self.status_label.setText(status_text)

        if status != CameraConnectionState.ONLINE or tile_data.frame is None:

            self._clear_video()
            self.detail_label.setText("-")
            self._frame_timestamps.clear()
            return

        frame = tile_data.frame
        pixmap = self._render_frame(frame, tile_data.detections)

        if pixmap is None:

            self._clear_video()
            self.detail_label.setText("-")
            return

        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        fps = self._update_fps(frame.timestamp)
        resolution = f"{frame.width}x{frame.height}" if frame.width and frame.height else "-"
        fps_text = f"{fps:.1f} fps" if fps is not None else "- fps"

        self.detail_label.setText(f"{resolution}   |   {fps_text}   |   {len(tile_data.detections)} detected")

    # =====================================================
    # Never leave a stale frame displayed once the camera is no longer
    # Online/configured -- same discipline as designer/widgets/
    # live_camera_view_panel.py::_clear_video.
    # =====================================================

    def _clear_video(self) -> None:

        self.video_label.clear()
        self.video_label.setText("No signal")

    # =====================================================
    # Same .copy()-required buffer-safety discipline as designer/widgets/
    # live_camera_view_panel.py::_frame_to_pixmap -- the decoder can
    # reuse/overwrite payload_ref's buffer on its next read(), and QImage
    # does not copy a raw buffer it is constructed from by default.
    # Boxes are drawn on the full-resolution QImage BEFORE the caller
    # scales it to the label -- bounding_box coordinates are absolute
    # pixel-space in the ORIGINAL frame (confirmed against
    # UltralyticsYOLOBackend.infer()'s own box.xyxy), so drawing after a
    # scale would require re-deriving a scale factor for no benefit.
    # Defensively guarded end to end: any unexpected payload/detection
    # shape must never crash the tile, only fall back to "no signal".
    # =====================================================

    def _render_frame(self, frame, detections):

        try:

            payload = frame.payload_ref

            if payload is None:
                return None

            height, width, channels = payload.shape

            if channels != 3:
                return None

            bytes_per_line = width * channels

            image = QImage(
                payload.copy().data, width, height, bytes_per_line,
                QImage.Format.Format_BGR888,
            )

            painter = QPainter(image)
            try:

                pen = QPen(BOX_COLOR)
                pen.setWidth(BOX_PEN_WIDTH)
                painter.setPen(pen)

                for detection in detections:

                    box = detection.bounding_box
                    if box is None:
                        continue

                    x1, y1, x2, y2 = box
                    painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

                    label_parts = []
                    if detection.local_track_id:
                        label_parts.append(f"ID {detection.local_track_id}")
                    label_parts.append(f"{detection.confidence:.0%}")

                    painter.drawText(int(x1), max(0, int(y1) - 4), "  ".join(label_parts))

            finally:
                painter.end()

            return QPixmap.fromImage(image)

        except Exception:

            return None

    # =====================================================
    # A small, local, honest derived metric -- rolling average over the
    # last FPS_SMOOTHING_WINDOW frame timestamps THIS tile has already
    # been handed (never a second read/decode of any kind). Reset
    # whenever the camera stops being visibly Online/configured
    # (refresh() above), so a stale rate never lingers across a
    # disconnect.
    # =====================================================

    def _update_fps(self, timestamp: float):

        self._frame_timestamps.append(timestamp)
        self._frame_timestamps = self._frame_timestamps[-FPS_SMOOTHING_WINDOW:]

        if len(self._frame_timestamps) < 2:
            return None

        span = self._frame_timestamps[-1] - self._frame_timestamps[0]

        if span <= 0:
            return None

        return (len(self._frame_timestamps) - 1) / span
