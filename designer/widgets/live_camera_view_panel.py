from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from camera_manager.connection_status import CameraConnectionState


class LiveCameraViewPanel(QWidget):

    # Live Camera Viewer milestone -- the Studio-facing, READ-ONLY
    # video presentation of the real camera stream already decoded by
    # LiveCameraPipeline (live_camera_pipeline/pipeline.py's own
    # latest_frame() cache). Same "dumb widget, MainWindow pushes
    # updates in" convention as RecommendationPanel/ExecutionPanel: the
    # one public entry point is refresh(camera_name, connection_state,
    # frame), called by MainWindow on the same LiveRuntimeController
    # tick as those two panels. Owns no timer, no RTSP/decoder
    # reference of its own, opens no connection of any kind -- it is
    # purely a consumer of whatever CameraFrame LiveCameraPipeline
    # already read this cycle.
    #
    # No YOLO overlay, no detection/tracking rendering -- video only.

    def __init__(self):

        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.camera_name_label = QLabel("No camera configured")
        layout.addWidget(self.camera_name_label)

        self.status_label = QLabel("-")
        layout.addWidget(self.status_label)

        self.video_label = QLabel("No signal")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label)

    # =====================================================
    # Public entry point
    # =====================================================

    def refresh(self, camera_name, connection_state, frame):

        self.camera_name_label.setText(camera_name or "No camera configured")

        status_text = (
            connection_state.value if isinstance(connection_state, CameraConnectionState)
            else (connection_state or "-")
        )
        self.status_label.setText(f"Status: {status_text}")

        if connection_state != CameraConnectionState.ONLINE or frame is None:

            self._clear_video()
            return

        pixmap = self._frame_to_pixmap(frame)

        if pixmap is None:

            self._clear_video()
            return

        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # =====================================================
    # Never leave a stale frame displayed once the camera is no longer
    # Online -- clearing the label is the only way to guarantee that
    # (a cached QPixmap on the label would otherwise persist).
    # =====================================================

    def _clear_video(self):

        self.video_label.clear()
        self.video_label.setText("No signal")

    # =====================================================
    # Presentation-boundary conversion only -- the same BGR ndarray
    # HumanDetector already reads for inference (opencv_decoder_
    # backend.py's own read()), never reinterpreted anywhere upstream
    # of this widget. Defensively guarded: an unexpected payload_ref
    # shape/type must never crash the panel, only fall back to "no
    # signal".
    # =====================================================

    def _frame_to_pixmap(self, frame):

        try:

            payload = frame.payload_ref

            if payload is None:
                return None

            height, width, channels = payload.shape

            if channels != 3:
                return None

            bytes_per_line = width * channels

            # .copy() is required: the ndarray's underlying buffer can
            # be reused/overwritten by the decoder on its next read()
            # call, and QImage does not copy a raw buffer it is
            # constructed from by default -- without this, the
            # displayed image could become corrupted or reference
            # freed memory.
            image = QImage(
                payload.copy().data, width, height, bytes_per_line,
                QImage.Format.Format_BGR888,
            )

            return QPixmap.fromImage(image)

        except Exception:

            return None
