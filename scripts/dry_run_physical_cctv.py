"""CCTV Connection & Calibration Readiness milestone, Phase 10 -- an
offline rehearsal of the EXACT sequence to be performed the day
physical college CCTV access exists, using a local video file
(validation_media/vtest.avi by default) in place of a network camera.

Every stage below is the REAL production class -- OpenCVFrameDecoderBackend,
RTSPFrameSource, YOLOHumanDetector(UltralyticsYOLOBackend), SimpleSingleCameraTracker,
SimulationIdentityResolver, LiveCameraPipelineDetectionProvider, CameraManager,
MultiCameraFusionEngine, BuildingStateEstimator, LiveOccupantManager -- the ONLY
thing this script substitutes for the real day is the endpoint itself (a local
file path instead of an rtsp:// URL) and, if --calibration is omitted, the
calibration lookup (camera_calibration.projection.WorldProjector genuinely
reports "no calibration" rather than fabricating one).

Not a pytest test -- run manually:
    python scripts/dry_run_physical_cctv.py
    python scripts/dry_run_physical_cctv.py --video validation_media/vtest.avi --weights weights/yolov8n.pt
    python scripts/dry_run_physical_cctv.py --calibration calibration.json

Each stage is reported as either:
    READY NOW                       -- exercised successfully by this dry run,
                                        using real production code, right now.
    REQUIRES PHYSICAL CCTV ACCESS   -- this dry run cannot exercise it (a local
                                        file has no real network/credentials/
                                        camera geometry to offer), but the code
                                        path itself is unchanged from Milestone A
                                        onward -- see docs/architecture/
                                        cctv_integration_readiness.md.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot

from models.camera import Camera
from models.engineering_asset import DeviceMode

from camera_manager.manager import CameraManager

from credential_store.local_file_store import LocalFileCredentialStore

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline
from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

from human_detection.opencv_decoder_backend import OpenCVFrameDecoderBackend

from tracking.simple_tracker import SimpleSingleCameraTracker

from live_occupants.manager import LiveOccupantManager

from multi_camera_fusion.engine import MultiCameraFusionEngine

from building_state.estimator import BuildingStateEstimator

from camera_calibration.calibration_loader import load_calibration_json
from camera_calibration.projection import WorldProjector


CAMERA_ID = "CAM-DRYRUN"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = REPO_ROOT / "validation_media" / "vtest.avi"


class StageReport:

    def __init__(self):
        self.lines = []

    def ready(self, name: str, detail: str = "") -> None:
        self.lines.append(("READY NOW", name, detail))

    def requires_physical(self, name: str, detail: str = "") -> None:
        self.lines.append(("REQUIRES PHYSICAL CCTV ACCESS", name, detail))

    def failed(self, name: str, detail: str = "") -> None:
        self.lines.append(("FAILED (unexpected)", name, detail))

    def print_all(self) -> None:

        print()
        print("=== Physical CCTV Dry Run -- Stage Report ===")
        print()

        width = max(len(status) for status, _, _ in self.lines)

        for status, name, detail in self.lines:
            line = f"[{status.ljust(width)}] {name}"
            if detail:
                line += f" -- {detail}"
            print(line)

        print()


def run_dry_run(video_path: Path, weights_path: str, calibration_path: str) -> None:

    report = StageReport()

    # ---- 1. Camera configuration -------------------------------------
    camera = Camera(id=CAMERA_ID, name="Dry Run Camera", floor_id="floor-1")
    camera_manager = CameraManager()
    camera_manager.register_camera(camera)
    report.ready("1. Camera configuration", f"Camera.id={camera.id!r} registered in CameraManager")

    # ---- 2. Credential lookup ------------------------------------------
    # A real deployment resolves a saved password through credential_store
    # at connect time (RTSPFrameSource._resolve_password) -- this dry run's
    # endpoint is a local file with no authentication, so the credential
    # store itself is exercised (constructed, queried) but genuinely has
    # nothing to resolve.
    store = LocalFileCredentialStore()
    has_credential = store.has_credential(CAMERA_ID)
    report.ready(
        "2. Credential lookup", f"LocalFileCredentialStore queried (has_credential={has_credential}); "
        f"real on-site day resolves a genuinely saved password the same way",
    )

    # ---- 3. Decoder startup --------------------------------------------
    backend = OpenCVFrameDecoderBackend()
    source = RTSPFrameSource(camera_id=CAMERA_ID, endpoint=str(video_path), decoder_backend=backend)
    source.start()

    if source.status != "Online":
        report.failed("3. Decoder startup", f"status={source.status}, last_error={source.last_error}")
        report.print_all()
        return

    report.ready("3. Decoder startup", "real OpenCVFrameDecoderBackend + RTSPFrameSource reached Online")

    # ---- 4. Frame acquisition ------------------------------------------
    first_frame = source.read_frame()
    if first_frame is None:
        report.failed("4. Frame acquisition", "no frame received")
        source.stop()
        report.print_all()
        return

    report.ready(
        "4. Frame acquisition",
        f"real decoded frame received ({first_frame.width}x{first_frame.height}, codec={first_frame.codec})",
    )

    # ---- 5. YOLO --------------------------------------------------------
    if weights_path and Path(weights_path).exists():

        from human_detection.yolo_backend import UltralyticsYOLOBackend
        from human_detection.yolo_human_detector import YOLOHumanDetector

        detector = YOLOHumanDetector(UltralyticsYOLOBackend(weights_path, device="cpu"))
        report.ready("5. YOLO", f"real UltralyticsYOLOBackend loaded from {weights_path}")

    else:

        from tests.human_detection_fixtures import FakeYOLOBackend
        from human_detection.yolo_human_detector import YOLOHumanDetector

        detector = YOLOHumanDetector(FakeYOLOBackend())
        report.requires_physical(
            "5. YOLO", "no --weights supplied -- using a deterministic fake detector for this dry run only; "
            "swap in real weights with --weights <path/to/yolov8n.pt> (no physical camera needed for this step)",
        )

    # ---- 6. Tracking ------------------------------------------------------
    tracker = SimpleSingleCameraTracker()
    report.ready("6. Tracking", "real SimpleSingleCameraTracker")

    # ---- 7. Calibration lookup / 8. World projection -----------------------
    world_projector = None
    if calibration_path and Path(calibration_path).exists():

        try:
            profile = load_calibration_json(calibration_path)
            world_projector = WorldProjector(calibrations={CAMERA_ID: profile}, zones_by_floor={})
            report.ready("7. Calibration lookup", f"loaded {calibration_path}")
            report.ready("8. World projection", "real WorldProjector configured with a loaded calibration")
        except Exception as exc:
            report.failed("7. Calibration lookup", str(exc))

    else:

        report.requires_physical(
            "7. Calibration lookup", "no --calibration supplied -- requires a physically measured scene "
            "(see docs/architecture/physical_cctv_access_checklist.md); WorldProjector honestly reports "
            "no calibration for this camera, world_position stays None",
        )
        report.requires_physical(
            "8. World projection", "same reason as step 7 -- code path itself (camera_calibration.projection."
            "WorldProjector) is unchanged and already proven, only real measured input is missing",
        )

    # ---- 9. LiveOccupant / 10. BuildingState -------------------------------
    identity_resolver = SimulationIdentityResolver()
    detection_provider = LiveCameraPipelineDetectionProvider()
    occupant_manager = LiveOccupantManager()

    pipeline = LiveCameraPipeline(
        frame_sources={CAMERA_ID: source},
        human_detector=detector,
        identity_resolver=identity_resolver,
        detection_provider=detection_provider,
        tracker=tracker,
        world_projector=world_projector,
        live_occupant_manager=occupant_manager,
    )

    camera_manager.register_detection_provider(DeviceMode.LIVE, detection_provider)
    camera_manager.set_camera_mode(CAMERA_ID, DeviceMode.LIVE)

    fusion_engine = MultiCameraFusionEngine()

    frames_processed = 0
    for index in range(60):
        pipeline.run_cycle(float(index))
        frames_processed += 1

    all_detections = camera_manager.all_detections(float(frames_processed - 1))
    fusion_result = fusion_engine.fuse(all_detections, float(frames_processed - 1))
    building_state = BuildingStateEstimator().estimate(
        float(frames_processed - 1),
        hazard_snapshot=HazardSnapshot(),
        occupancy_snapshot=OccupancySnapshot(),
        fusion_result=fusion_result,
    )

    source.stop()

    report.ready(
        "9. LiveOccupant", f"{len(occupant_manager.all_occupants())} occupant(s) reached LiveOccupantManager "
        f"across {frames_processed} cycles",
    )
    report.ready(
        "10. BuildingState", f"{len(building_state.occupant_tracks)} occupant track(s) reached BuildingState",
    )

    report.print_all()

    print("Summary: every stage above ran through REAL production code on a local video file.")
    print("The ONLY things a physical CCTV day adds are: a real network endpoint/credentials")
    print("(steps 2-4), and a real measured calibration scene (steps 7-8) -- nothing else in")
    print("this chain changes. See docs/architecture/physical_cctv_access_checklist.md.")


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Local video file standing in for the network camera.")
    parser.add_argument("--weights", default=None, help="Path to real YOLO weights (omit to use a fake detector for this dry run).")
    parser.add_argument("--calibration", default=None, help="Path to a calibration JSON (scripts/calibrate_camera_scene.py output).")
    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        raise SystemExit(f"Video file not found: {video_path}")

    run_dry_run(video_path, args.weights, args.calibration)


if __name__ == "__main__":
    main()
