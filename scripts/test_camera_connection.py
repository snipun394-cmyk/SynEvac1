"""CCTV Connection & Calibration Readiness milestone, Phase 5 -- the
physical-access-day diagnostic utility: point this at ONE configured
camera (real RTSP endpoint, or a local video file for a dry run) and
get an honest report of exactly what happened, with zero YOLO required
by default.

Not a pytest test -- run manually, standing in front of the camera/NVR:

    python scripts/test_camera_connection.py --camera-id CAM-001 \\
        --endpoint rtsp://192.168.1.50:554/stream1 --username admin \\
        --credential-ref CAM-001

    python scripts/test_camera_connection.py --camera-id CAM-001 \\
        --endpoint rtsp://192.168.1.50:554/stream1 --username admin \\
        --credential-ref CAM-001 --detect --weights weights/yolov8n.pt

    # Offline dry run against a local file instead of a network camera:
    python scripts/test_camera_connection.py --camera-id CAM-DRYRUN \\
        --endpoint validation_media/vtest.avi

A password, if one is needed, is resolved from `credential_store`
(`--credential-ref`, matching the reference the Property Panel already
captured it under -- see docs/architecture/cctv_integration_readiness.md
§7) or, for a quick ad-hoc test only, typed interactively via
`--prompt-password` (never taken as a bare CLI argument -- a plaintext
password on the command line ends up in shell history, exactly the kind
of leak this milestone's own Phase 12 re-audits against). The resolved
password is NEVER printed, logged, or included in any report line --
only `--credential-ref` (a reference id, not a secret) ever appears in
output, matching ConnectionInfo.__repr__'s own redaction discipline.
"""

import argparse
import getpass
import statistics
import sys
import time as time_module
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from credential_store.local_file_store import LocalFileCredentialStore
from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource, redact_endpoint
from human_detection.opencv_decoder_backend import OpenCVFrameDecoderBackend


def _no_sleep(_seconds):
    # A manual diagnostic run should never silently hang through a real
    # backoff sleep -- reconnect attempts still happen, just without
    # waiting, so the operator sees the outcome immediately.
    pass


class ConnectionDiagnostic:

    # Accumulates exactly what §5 of the milestone spec asks for, purely
    # by observing a real (unmodified) RTSPFrameSource -- no parallel
    # connection-status tracking logic of its own beyond recording what
    # the source itself already reports via status_callback.

    def __init__(self, camera_id: str):

        self.camera_id = camera_id
        self.status_transitions: List[tuple] = []  # (wall_clock, status, sanitized_detail)
        self.frame_timestamps: List[float] = []
        self.first_frame_resolution: Optional[tuple] = None
        self.first_frame_codec: Optional[str] = None
        self.frames_received = 0
        self.detections_by_frame: List[int] = []
        self.reached_online = False

    def on_status_changed(self, camera_id: str, status: str, detail: Optional[str]) -> None:
        self.status_transitions.append((time_module.time(), status, detail))
        if status == "Online":
            self.reached_online = True

    def record_frame(self, frame) -> None:

        self.frames_received += 1
        self.frame_timestamps.append(time_module.time())

        if self.first_frame_resolution is None and frame.width and frame.height:
            self.first_frame_resolution = (frame.width, frame.height)
            self.first_frame_codec = frame.codec

    def measured_fps(self) -> Optional[float]:

        if len(self.frame_timestamps) < 2:
            return None

        deltas = [
            b - a for a, b in zip(self.frame_timestamps, self.frame_timestamps[1:]) if b > a
        ]

        if not deltas:
            return None

        mean_delta = statistics.mean(deltas)
        return (1.0 / mean_delta) if mean_delta > 0 else None

    def reconnect_events(self) -> int:

        # A reconnect is any transition INTO Degraded (a drop was
        # detected and a bounded retry began) -- counted, never assumed;
        # zero of these across a whole run is itself an honest, reportable
        # fact ("stream never dropped during this test"), not a gap.
        return sum(1 for _, status, _ in self.status_transitions if status == "Degraded")

    def print_report(self) -> None:

        print()
        print("=== Camera Connection Diagnostic ===")
        print(f"Camera ID:            {self.camera_id}")

        # The diagnostic always calls RTSPFrameSource.stop() at the end
        # of its own sampling window (so an operator can rerun it
        # immediately without a stale connection) -- stop() always
        # reports STATUS_CONFIGURED regardless of how the test itself
        # went (rtsp_frame_source.py's own documented behavior), so the
        # LAST transition is never itself the honest "how did this test
        # go" answer. reached_online (set the moment ANY status
        # transition this run reported Online) is that answer instead.
        print(f"Connected during test: {'YES' if self.reached_online else 'NO'}")
        print(f"Session ended as:      {self.status_transitions[-1][1] if self.status_transitions else '(never attempted)'}")

        print(f"Status transitions:")
        for wall_clock, status, detail in self.status_transitions:
            stamp = time_module.strftime("%H:%M:%S", time_module.localtime(wall_clock))
            line = f"  [{stamp}] {status}"
            if detail:
                line += f" -- {detail}"
            print(line)

        print(f"Frames received:      {self.frames_received}")

        if self.first_frame_resolution:
            w, h = self.first_frame_resolution
            print(f"Frame resolution:     {w}x{h}")
        else:
            print(f"Frame resolution:     (not reported by this stream)")

        print(f"Frame codec:          {self.first_frame_codec or '(not reported by this stream)'}")

        fps = self.measured_fps()
        print(f"Measured FPS:         {f'{fps:.2f}' if fps is not None else '(not enough frames to measure)'}")

        print(f"Reconnect attempts observed: {self.reconnect_events()}")

        if self.detections_by_frame:
            total = sum(self.detections_by_frame)
            print(f"Detections (--detect): {total} across {len(self.detections_by_frame)} frame(s) checked")

        print()

        if self.reached_online and self.frames_received > 0:
            print("RESULT: Connection OK -- real frames received.")
        elif self.reached_online:
            print("RESULT: Connected, but no frames were received within the test window.")
        else:
            print("RESULT: Connection FAILED -- see status transitions above for the sanitized cause.")


def resolve_password(args) -> Optional[str]:

    if args.prompt_password:
        # getpass never echoes to the terminal and this value is never
        # printed/logged anywhere below.
        return getpass.getpass(f"Password for {args.camera_id} (input hidden): ")

    if args.credential_ref:

        store = LocalFileCredentialStore(Path(args.credential_store_path) if args.credential_store_path else None)
        password = store.get_credential(args.credential_ref)

        if password is None:
            print(
                f"WARNING: no credential saved under reference {args.credential_ref!r} -- "
                f"proceeding with no password. Save one via the Property Panel first, or use "
                f"--prompt-password for a one-off test.",
                file=sys.stderr,
            )

        return password

    return None


def run_diagnostic(args) -> ConnectionDiagnostic:

    password = resolve_password(args)

    diagnostic = ConnectionDiagnostic(args.camera_id)

    backend = OpenCVFrameDecoderBackend(open_timeout_ms=args.open_timeout_ms)

    source = RTSPFrameSource(
        camera_id=args.camera_id,
        endpoint=args.endpoint,
        decoder_backend=backend,
        username=args.username,
        password=password,
        max_retries=args.max_retries,
        sleep_fn=_no_sleep,
        status_callback=diagnostic.on_status_changed,
    )

    print(f"Connecting to {args.camera_id} @ {redact_endpoint(args.endpoint)!r} ...")

    detector = None
    if args.detect:

        from human_detection.yolo_backend import UltralyticsYOLOBackend
        from human_detection.yolo_human_detector import YOLOHumanDetector

        if not args.weights:
            print("ERROR: --detect requires --weights <path/to/yolov8n.pt>", file=sys.stderr)
            sys.exit(2)

        detector = YOLOHumanDetector(UltralyticsYOLOBackend(args.weights, device="cpu"))

    source.start()

    deadline = time_module.monotonic() + args.duration_seconds

    while time_module.monotonic() < deadline and diagnostic.frames_received < args.max_frames:

        frame = source.read_frame()

        if frame is None:
            time_module.sleep(0.01)
            continue

        diagnostic.record_frame(frame)

        if detector is not None:
            detections = detector.detect(frame)
            diagnostic.detections_by_frame.append(len(detections))

    source.stop()

    return diagnostic


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--camera-id", required=True, help="The exact Digital Twin Camera.id to test (copy, don't retype).")
    parser.add_argument("--endpoint", required=True, help="RTSP URL, or a local video file path for a dry run.")
    parser.add_argument("--username", default=None, help="Camera/NVR username, if authentication is required.")
    parser.add_argument("--credential-ref", default=None, help="credential_store reference id to resolve a saved password from.")
    parser.add_argument("--credential-store-path", default=None, help="Override credential_store JSON path (defaults to ~/.synevac/credentials.json).")
    parser.add_argument("--prompt-password", action="store_true", help="Prompt interactively (hidden input) for a password instead of resolving one from credential_store.")
    parser.add_argument("--max-retries", type=int, default=3, help="Bounded reconnect attempts (same policy as production RTSPFrameSource).")
    parser.add_argument("--open-timeout-ms", type=int, default=5000, help="Per-attempt connection timeout in milliseconds.")
    parser.add_argument("--duration-seconds", type=float, default=10.0, help="Maximum wall-clock time to sample frames for.")
    parser.add_argument("--max-frames", type=int, default=150, help="Stop early once this many frames have been received.")
    parser.add_argument("--detect", action="store_true", help="Also run the real YOLO human detector on received frames.")
    parser.add_argument("--weights", default=None, help="Path to a local YOLO .pt weights file (required with --detect).")

    args = parser.parse_args()

    diagnostic = run_diagnostic(args)
    diagnostic.print_report()


if __name__ == "__main__":
    main()
