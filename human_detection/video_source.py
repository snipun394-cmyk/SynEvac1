from pathlib import Path
from typing import List, Tuple, Union

import cv2


# =====================================================
# Real Human Detection Pipeline milestone, Phase 8 -- a small offline
# loader, not a second CameraFrameSource architecture. Investigated per
# this milestone's own instruction: live_camera_pipeline.
# replay_frame_source.ReplayFrameSource already accepts exactly the
# shape this module produces (Sequence[Tuple[float, Any]]) and already
# proves the CameraFrameSource contract end-to-end -- there is no need
# for, and this milestone does not add, a video-specific
# CameraFrameSource. This module's only job is decoding a local video/
# image file into that same (timestamp, payload_ref) shape,
# payload_ref being a decoded frame (an OpenCV/numpy BGR ndarray) --
# the same convention human_detection.yolo_human_detector.
# YOLOHumanDetector already expects from CameraFrame.payload_ref.
#
# Deliberately lives in human_detection/, not live_camera_pipeline/ --
# this is exactly the kind of OpenCV-dependent code tests/
# test_no_cv_dependencies.py forbids inside live_camera_pipeline/
# itself (Phase 1's own investigation finding, re-applied here).
# =====================================================


class VideoSourceError(Exception):

    # Raised when a supplied path does not exist or cannot be opened/
    # decoded at all -- an honest, immediate failure, never a silent
    # empty result masquerading as "the video had zero frames."

    pass


def load_video_frames(path: Union[str, Path], *, fps_override: float = None) -> List[Tuple[float, "object"]]:

    # Decodes every frame of a local video file, in order, into
    # (timestamp_seconds, frame_ndarray) tuples -- ready to pass
    # straight into ReplayFrameSource(camera_id=..., frames=...).
    # Timestamps are derived from the video's own reported FPS (or
    # fps_override, for a file whose container does not report one
    # honestly) -- frame index / fps, starting at 0.0. No network
    # access, no CCTV, no RTSP -- cv2.VideoCapture reads only the local
    # file path given.

    video_path = Path(path)

    if not video_path.exists():
        raise VideoSourceError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise VideoSourceError(f"Could not open video file: {video_path}")

    fps = fps_override or capture.get(cv2.CAP_PROP_FPS) or 1.0

    frames: List[Tuple[float, "object"]] = []

    try:

        index = 0

        while True:

            ok, frame = capture.read()

            if not ok:
                break

            frames.append((index / fps, frame))
            index += 1

    finally:
        capture.release()

    return frames


# =====================================================


def load_image_frame(path: Union[str, Path]) -> "object":

    # Decodes a single local image file into a frame ndarray -- the
    # same payload_ref shape a video frame produces, for a caller that
    # has individual snapshots rather than a video container.

    image_path = Path(path)

    if not image_path.exists():
        raise VideoSourceError(f"Image file not found: {image_path}")

    frame = cv2.imread(str(image_path))

    if frame is None:
        raise VideoSourceError(f"Could not decode image file: {image_path}")

    return frame


# =====================================================


def load_image_frames(paths, *, fps: float = 1.0) -> List[Tuple[float, "object"]]:

    # Decodes an ordered sequence of local image files into the same
    # (timestamp, payload_ref) shape load_video_frames() produces --
    # for a caller whose "prerecorded footage" is a directory of
    # snapshots rather than one video container.

    return [
        (index / fps, load_image_frame(path))
        for index, path in enumerate(paths)
    ]
