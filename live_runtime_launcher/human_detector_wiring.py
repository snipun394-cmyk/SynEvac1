from pathlib import Path
from typing import Tuple, Union

from human_detection.yolo_backend import UltralyticsYOLOBackend
from human_detection.yolo_human_detector import YOLOHumanDetector

from live_camera_pipeline.human_detector import HumanDetector
from live_camera_pipeline.identity_resolver import IdentityResolver, SimulationIdentityResolver


# =====================================================
# Camera 1 Live Human-Detection Integration milestone -- closes the
# SECOND composition gap named (but deliberately left open) by the CP
# PLUS NVR -> SynEvac1 Live Runtime Integration milestone: frame_sources
# alone never builds a camera_pipeline (build_live_runtime()'s own
# existing gate also requires human_detector AND identity_resolver).
# This is the smallest seam that supplies both, reusing the exact real
# production classes docs/architecture/human_detection.md's own
# real-world validation run already proved end-to-end:
# UltralyticsYOLOBackend -> YOLOHumanDetector, paired with
# SimulationIdentityResolver -- the natural, already-existing,
# non-invented choice for a single-camera Live deployment with no
# cross-camera reconciliation configured (see live_camera_pipeline/
# identity_resolver.py's own docstring: "local_track_id IS already the
# global identity in this strategy"). MappingIdentityResolver is NOT
# used here -- it exists specifically to reconcile two cameras seeing
# the SAME person simultaneously, not needed for one camera.
#
# No tracker/behavior_recognizer/cross_camera_identity_resolver/
# world_projector/live_occupant_manager is wired here -- every one of
# those is an OPTIONAL, additive seam on LiveCameraPipeline itself
# (see pipeline.py's own docstring), out of scope for "prove a real
# frame reaches the existing detector and produces a real detection."
# =====================================================


def build_yolo_human_detector(
    weights_path: Union[str, Path], device: str = "cpu",
) -> Tuple[HumanDetector, IdentityResolver]:

    backend = UltralyticsYOLOBackend(weights_path, device=device)

    return YOLOHumanDetector(backend), SimulationIdentityResolver()
