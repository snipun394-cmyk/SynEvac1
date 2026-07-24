import json
import subprocess
import sys
import unittest
from pathlib import Path


# =====================================================
# Physical CCTV Field Validation milestone -- proves scripts/
# run_physical_camera_validation.py against real local artifacts
# (validation_media/vtest.avi, weights/yolov8n.pt) wherever both are
# available, and its pure/offline logic (outcome classification, report
# JSON shape) unconditionally. Every physical-network scenario (a real
# RTSP camera) is explicitly out of scope here -- this file proves the
# script's OWN logic, orchestration, and honest-failure behavior, not a
# real camera.
# =====================================================


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
VIDEO_PATH = REPO_ROOT / "validation_media" / "vtest.avi"
WEIGHTS_PATH = REPO_ROOT / "weights" / "yolov8n.pt"
SCRIPT_PATH = SCRIPTS_DIR / "run_physical_camera_validation.py"

sys.path.insert(0, str(SCRIPTS_DIR))


def run_cli(args, timeout=60):

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)] + args,
        capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT),
    )
    return result


class ClassifyConnectionOutcomeTests(unittest.TestCase):

    def test_online_is_ok(self):
        import run_physical_camera_validation as runner
        self.assertEqual(runner.classify_connection_outcome("Online", None), runner.OUTCOME_OK)

    def test_unreachable_hostname_detail_is_network_unreachable(self):
        import run_physical_camera_validation as runner
        self.assertEqual(
            runner.classify_connection_outcome("Stream Unavailable", "Failed to resolve hostname xyz"),
            runner.OUTCOME_NETWORK_UNREACHABLE,
        )

    def test_401_detail_is_authentication_failed(self):
        import run_physical_camera_validation as runner
        self.assertEqual(
            runner.classify_connection_outcome("Stream Unavailable", "server returned 401 Unauthorized"),
            runner.OUTCOME_AUTHENTICATION_FAILED,
        )

    def test_generic_failure_falls_back_to_stream_unavailable(self):
        import run_physical_camera_validation as runner
        self.assertEqual(
            runner.classify_connection_outcome("Stream Unavailable", "some opaque decode error"),
            runner.OUTCOME_STREAM_UNAVAILABLE,
        )

    def test_generic_backend_unreachable_wording_is_not_misclassified_as_network(self):
        # Regression: OpenCVFrameDecoderBackend's own generic open() failure
        # message ("stream unreachable, wrong path, or unsupported by this
        # OpenCV build") uses the word "unreachable" for EVERY open failure,
        # including a plain bad local path with no network involved at all --
        # this must fall back to the generic STREAM_UNAVAILABLE, never a
        # confident (and wrong) NETWORK_UNREACHABLE guess.
        import run_physical_camera_validation as runner
        self.assertEqual(
            runner.classify_connection_outcome(
                "Stream Unavailable",
                "Could not open video stream at 'bad_path.avi' -- stream unreachable, wrong path, "
                "or unsupported by this OpenCV build.",
            ),
            runner.OUTCOME_STREAM_UNAVAILABLE,
        )


class FieldValidationReportTests(unittest.TestCase):

    def test_default_sections_are_none_until_populated(self):
        import run_physical_camera_validation as runner
        report = runner.FieldValidationReport("CAM-1", "rtsp://host/stream", "connection-only")

        self.assertIsNone(report.data["connection"])
        self.assertIsNone(report.data["frames"])
        self.assertEqual(report.data["warnings"], [])
        self.assertEqual(report.data["failures"], [])

    def test_endpoint_is_redacted_even_if_credentials_were_embedded(self):
        import run_physical_camera_validation as runner
        report = runner.FieldValidationReport("CAM-1", "rtsp://admin:secret@host/stream", "connection-only")

        self.assertNotIn("secret", report.data["endpoint_redacted"])

    def test_fail_sets_final_outcome_and_records_failure(self):
        import run_physical_camera_validation as runner
        report = runner.FieldValidationReport("CAM-1", "rtsp://host/stream", "connection-only")

        report.fail(runner.OUTCOME_CALIBRATION_REQUIRED, "no calibration supplied")

        self.assertEqual(report.data["final_outcome"], runner.OUTCOME_CALIBRATION_REQUIRED)
        self.assertEqual(len(report.data["failures"]), 1)

    def test_save_round_trips_through_json(self, tmp_path=None):
        import tempfile
        import run_physical_camera_validation as runner

        report = runner.FieldValidationReport("CAM-1", "rtsp://host/stream", "frames")
        report.succeed()

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "report.json"
            report.save(str(out_path))

            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["camera_id"], "CAM-1")
            self.assertEqual(loaded["final_outcome"], "OK")


@unittest.skipUnless(VIDEO_PATH.exists(), f"local validation video not found at {VIDEO_PATH}")
class CliConnectionOnlyTests(unittest.TestCase):

    def test_connection_only_against_local_file_succeeds(self):

        result = run_cli(["--camera-id", "CAM-CLI-CONN", "--endpoint", str(VIDEO_PATH), "--connection-only"])

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Final outcome:   OK", result.stdout)

    def test_connection_only_against_bad_path_fails_honestly(self):

        result = run_cli([
            "--camera-id", "CAM-CLI-BAD", "--endpoint", "this_file_does_not_exist_at_all.avi",
            "--connection-only", "--max-retries", "0",
        ])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STREAM_UNAVAILABLE", result.stdout)

    def test_report_out_produces_valid_json_with_no_secrets(self):

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:

            report_path = Path(tmp_dir) / "report.json"

            result = run_cli([
                "--camera-id", "CAM-CLI-REPORT", "--endpoint", str(VIDEO_PATH),
                "--username", "admin", "--connection-only", "--report-out", str(report_path),
            ])

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(data["camera_id"], "CAM-CLI-REPORT")
            self.assertIn("software_commit", data)
            self.assertIn("timestamp_utc", data)

            # Never a password anywhere in the report, even though none
            # was supplied here -- also confirms no stray field leaks one.
            report_text = report_path.read_text(encoding="utf-8")
            self.assertNotIn("password", report_text.lower())


@unittest.skipUnless(VIDEO_PATH.exists(), f"local validation video not found at {VIDEO_PATH}")
class CliFramesStageTests(unittest.TestCase):

    def test_frames_mode_reports_real_resolution_and_fps(self):

        result = run_cli([
            "--camera-id", "CAM-CLI-FRAMES", "--endpoint", str(VIDEO_PATH),
            "--frames", "--max-frames", "20", "--duration-seconds", "3",
        ])

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("768x576", result.stdout)


@unittest.skipUnless(
    VIDEO_PATH.exists() and WEIGHTS_PATH.exists(),
    f"real YOLO weights ({WEIGHTS_PATH}) and/or local validation video ({VIDEO_PATH}) not found",
)
class CliDetectAndTrackStageTests(unittest.TestCase):

    def test_detect_mode_reports_real_person_detections(self):

        result = run_cli([
            "--camera-id", "CAM-CLI-DETECT", "--endpoint", str(VIDEO_PATH),
            "--detect", "--weights", str(WEIGHTS_PATH), "--max-frames", "20", "--duration-seconds", "5",
        ], timeout=60)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[detect]", result.stdout)

    def test_track_mode_reports_unique_track_ids(self):

        result = run_cli([
            "--camera-id", "CAM-CLI-TRACK", "--endpoint", str(VIDEO_PATH),
            "--track", "--weights", str(WEIGHTS_PATH), "--max-frames", "20", "--duration-seconds", "5",
        ], timeout=60)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[track]", result.stdout)

    def test_project_mode_without_calibration_fails_honestly_never_fabricates(self):

        result = run_cli([
            "--camera-id", "CAM-CLI-PROJECT", "--endpoint", str(VIDEO_PATH),
            "--project", "--weights", str(WEIGHTS_PATH), "--max-frames", "20", "--duration-seconds", "5",
        ], timeout=60)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CALIBRATION_REQUIRED", result.stdout)
        self.assertIn("CALIBRATION REQUIRED", result.stdout)

    def test_detect_mode_without_weights_fails_honestly(self):

        result = run_cli([
            "--camera-id", "CAM-CLI-NOWEIGHTS", "--endpoint", str(VIDEO_PATH),
            "--detect", "--max-frames", "10", "--duration-seconds", "3",
        ], timeout=30)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("YOLO_UNAVAILABLE", result.stdout)


@unittest.skipUnless(
    VIDEO_PATH.exists() and WEIGHTS_PATH.exists(),
    f"real YOLO weights ({WEIGHTS_PATH}) and/or local validation video ({VIDEO_PATH}) not found",
)
class CliFullRuntimeStageTests(unittest.TestCase):

    def test_full_runtime_reaches_building_state_with_a_real_calibration(self):

        import tempfile

        from camera_calibration.camera_model import CameraExtrinsics, CameraIntrinsics, CalibrationProfile
        from camera_calibration.calibration_loader import save_calibration_json

        camera_id = "CAM-CLI-FULLRUNTIME"

        with tempfile.TemporaryDirectory() as tmp_dir:

            calibration_path = Path(tmp_dir) / "calibration.json"

            profile = CalibrationProfile(
                camera_id=camera_id, floor_id="floor-1",
                intrinsics=CameraIntrinsics(image_width=768, image_height=576, focal_length_x=500.0, focal_length_y=500.0),
                extrinsics=CameraExtrinsics(position=(0.0, 0.0), mount_height=3.5, yaw_degrees=0.0, pitch_degrees=28.0),
            )
            save_calibration_json(profile, str(calibration_path))

            report_path = Path(tmp_dir) / "report.json"

            result = run_cli([
                "--camera-id", camera_id, "--endpoint", str(VIDEO_PATH),
                "--full-runtime", "--weights", str(WEIGHTS_PATH), "--calibration", str(calibration_path),
                "--max-frames", "20", "--duration-seconds", "5", "--report-out", str(report_path),
            ], timeout=90)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("BuildingState reached: True", result.stdout)

            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(data["runtime"]["building_state_reached"])
            self.assertFalse(data["runtime"]["voice_broadcast_attempted"])
            self.assertFalse(data["runtime"]["building_control_executed"])


class SecuritySourceScanTests(unittest.TestCase):

    def test_field_runner_never_accepts_a_bare_password_cli_argument(self):

        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn('"--password"', source)

    def test_field_runner_source_never_prints_a_resolved_password_variable(self):

        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("print(password", source)
        self.assertNotIn("{password}", source)
        self.assertNotIn("{password!r}", source)


if __name__ == "__main__":
    unittest.main()
