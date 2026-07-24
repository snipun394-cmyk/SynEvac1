import pathlib
import re
import unittest


# =====================================================
# Real YOLO Person Detection Validation & Productionization milestone,
# Phase 14 -- mechanical architecture-boundary guards, mirroring
# tests/test_tracking_architecture_guards.py's own established pattern.
# human_detection/ must only ever produce RawHumanDetection objects; it
# must never make evacuation decisions, modify BuildingState/hazards,
# broadcast voice, execute building controls, or control FACP/signs.
# =====================================================

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HUMAN_DETECTION_PACKAGE = REPO_ROOT / "human_detection"

FORBIDDEN = (
    r"^\s*(from|import)\s+("
    r"ai_decision|ai_registry|ai_inference|ai_training|ai_explainability|rl_training|"
    r"decision_policy|"
    r"advisory_system|command_center|"
    r"voice_evacuation|speaker_manager|"
    r"building_control|"
    r"facp|"
    r"sign_manager|dynamic_signage|"
    r"hazard|hazard_evolution|fire_growth|smoke_propagation|"
    r"building_state|"
    r"multi_camera_fusion|camera_manager|"
    r"live_system|live_runtime"
    r")\b"
)


class HumanDetectionArchitectureGuardTests(unittest.TestCase):

    def test_human_detection_package_imports_nothing_forbidden(self):

        for path in sorted(HUMAN_DETECTION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(FORBIDDEN, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports "
                f"{match.group(0).strip() if match else ''!r} -- human_detection/ must only "
                f"ever produce RawHumanDetection objects, never reach decision/execution layers.",
            )

    def test_human_detection_only_depends_on_live_camera_pipeline_seams(self):

        # The only cross-package import human_detection/ is allowed at
        # all: live_camera_pipeline.frame_source (CameraFrame) and
        # live_camera_pipeline.human_detector (HumanDetector,
        # RawHumanDetection) -- never .identity_resolver, .pipeline, or
        # .detection_provider (those would be a sign this package is
        # reaching beyond its own detection responsibility).
        #
        # CCTV Connection & Calibration Readiness milestone, Phase 2 --
        # human_detection/opencv_decoder_backend.py is a deliberate,
        # narrow exception: it implements live_camera_pipeline.
        # rtsp_backend.FrameDecoderBackend (the real transport/decode
        # seam, not identity/pipeline/detection-provider), the same
        # "the seam's interface lives in live_camera_pipeline/, the real
        # implementation lives in human_detection/ (the one package
        # allowed to import cv2)" pattern yolo_backend.py already
        # established for YOLOInferenceBackend/human_detector.py. Every
        # OTHER file in this package remains forbidden from importing
        # rtsp_backend/rtsp_frame_source -- only the module whose entire
        # job is decoding actually needs it.
        forbidden_submodule_pattern = (
            r"^\s*(from|import)\s+live_camera_pipeline\.(identity_resolver|pipeline|detection_provider|rtsp_frame_source|rtsp_backend)\b"
        )

        exempted_files = {"opencv_decoder_backend.py"}

        for path in sorted(HUMAN_DETECTION_PACKAGE.glob("*.py")):

            if path.name in exempted_files:
                continue

            text = path.read_text(encoding="utf-8")
            match = re.search(forbidden_submodule_pattern, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"human_detection/ may only depend on live_camera_pipeline.frame_source/human_detector.",
            )

    def test_opencv_decoder_backend_only_imports_the_decoder_seam_not_pipeline_layers(self):

        # The exemption above is narrow: opencv_decoder_backend.py may
        # import live_camera_pipeline.rtsp_backend (the seam it
        # implements) but still must never reach into identity_resolver/
        # pipeline/detection_provider -- those remain a genuine
        # architecture violation for ANY file in this package, no
        # exceptions.
        forbidden = r"^\s*(from|import)\s+live_camera_pipeline\.(identity_resolver|pipeline|detection_provider)\b"

        text = (HUMAN_DETECTION_PACKAGE / "opencv_decoder_backend.py").read_text(encoding="utf-8")
        match = re.search(forbidden, text, re.MULTILINE)

        self.assertIsNone(
            match,
            f"opencv_decoder_backend.py imports {match.group(0).strip() if match else ''!r} -- "
            f"it may only depend on live_camera_pipeline.rtsp_backend, nothing higher-level.",
        )

    def test_human_detection_never_imports_tracking_or_identity_layers(self):

        # Detection, Tracking, Identity, Behavior, Intelligence,
        # Decision, and Execution are kept as separate layers --
        # human_detection/ (Detection) must never reach up into
        # Tracking/Identity/Behavior/Intelligence itself; composition
        # happens one level up (LiveCameraPipeline/live_runtime.factory),
        # never inside this package.
        forbidden = (
            r"^\s*(from|import)\s+("
            r"tracking|cross_camera_identity|behavior_recognition|camera_calibration|"
            r"live_occupants|live_perception|"
            r"crowd_intelligence|evacuation_progress|trajectory_intelligence|"
            r"emergency_response|evacuation_recommendation|evacuation_guidance"
            r")\b"
        )

        for path in sorted(HUMAN_DETECTION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(forbidden, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"Detection/Tracking/Identity/Behavior/Intelligence must stay separate layers.",
            )

    def test_yolo_human_detector_only_ever_produces_raw_human_detection(self):

        # Positive shape check: the one production method (detect())
        # returns a tuple of RawHumanDetection. NOTE: a bare substring
        # scan for "BuildingState" would false-positive on this file's
        # own docstring, which explicitly *disclaims* any BuildingState
        # coupling ("never coupled to ... BuildingState") -- exactly the
        # same false-positive shape a prior milestone already worked
        # around (`.reset(` on an unrelated Gym API), so this check
        # verifies the actual return-type annotation instead of banning
        # a word that legitimately appears in an honest disclaimer.
        text = (HUMAN_DETECTION_PACKAGE / "yolo_human_detector.py").read_text(encoding="utf-8")

        self.assertIn("def detect(self, frame: CameraFrame) -> Tuple[RawHumanDetection, ...]:", text)

    def test_no_operator_action_verbs_anywhere_in_human_detection(self):

        forbidden_calls = (
            r"\.acknowledge\(|\.silence\(|\.reset\(|"
            r"\.broadcast\(|\.announce\(|"
            r"\.execute_control\(|\.confirm\(|\.dispatch\("
        )

        for path in sorted(HUMAN_DETECTION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(forbidden_calls, text)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} calls {match.group(0) if match else ''!r} -- "
                f"human_detection/ only ever detects people, it never executes actions.",
            )


if __name__ == "__main__":
    unittest.main()
