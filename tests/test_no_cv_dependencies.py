import pathlib
import re
import unittest


# Phase 16: none of these packages may depend on a real computer-vision
# or streaming library today -- that dependency belongs entirely behind
# the not-yet-implemented RTSPFrameSource/HumanDetector seams (Phase 3/4),
# never inside the Camera Asset, CameraManager, MultiCameraFusionEngine,
# BuildingState, Advisory System, or Command Center themselves. Same
# regex-scan-the-source-files convention this codebase already uses per
# package (see tests.test_camera_manager.
# CameraManagerPackageDependencyDirectionTests) -- consolidated here
# once, across every package this milestone's own guard list names,
# rather than duplicated seven times.

FORBIDDEN = r"^\s*(from|import)\s+(cv2|torch|ultralytics|onvif)\b"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

TARGETS = (
    REPO_ROOT / "models" / "camera.py",
    REPO_ROOT / "models" / "engineering_asset.py",
    REPO_ROOT / "camera_manager",
    REPO_ROOT / "multi_camera_fusion",
    REPO_ROOT / "virtual_camera",
    REPO_ROOT / "building_state",
    REPO_ROOT / "advisory_system",
    REPO_ROOT / "command_center",
    REPO_ROOT / "live_camera_pipeline",
    REPO_ROOT / "credential_store",
)


def _python_files(target: pathlib.Path):

    if target.is_file():
        return [target]

    return list(target.glob("*.py"))


class NoComputerVisionDependencyTests(unittest.TestCase):

    def test_no_target_package_imports_a_cv_or_streaming_library(self):

        for target in TARGETS:

            for path in _python_files(target):

                text = path.read_text(encoding="utf-8")

                match = re.search(FORBIDDEN, text, re.MULTILINE)

                self.assertIsNone(
                    match,
                    f"{path.relative_to(REPO_ROOT)} imports "
                    f"{match.group(0).strip() if match else ''!r} -- no computer-vision or "
                    f"RTSP/ONVIF library may be imported before a real camera adapter exists "
                    f"(Pre-CCTV Readiness milestone, Phase 16)",
                )


if __name__ == "__main__":
    unittest.main()
