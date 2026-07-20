"""RTSP Frame Source milestone, Phase 13 performance readiness benchmark.

Benchmarks RTSPFrameSource's own overhead -- frame routing, CameraFrame
construction, and status-update dispatch -- using FakeRTSPBackend and
already-decoded fake frames. There is no real RTSP stream to benchmark
yet (no physical CCTV access, no real FrameDecoderBackend
implementation), so this deliberately measures only the seam this
milestone actually built, exactly as scripts/
benchmark_live_camera_pipeline.py does for the identity-resolution/
fusion/estimation seam.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_rtsp_frame_source.py`) and read the
printed report.

Explicitly NOT measured here (see docs/architecture/
cctv_integration_readiness.md's RTSP section): real decode latency,
real network latency, real frame-drop rate, real stream stability --
every one of those requires an actual physical CCTV stream to measure
honestly, and none exists yet.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

from tests.live_camera_pipeline_fixtures import FakeRTSPBackend


FRAME_COUNT = 5000


def no_delay(_seconds):
    pass


def benchmark_frame_routing() -> dict:

    backend = FakeRTSPBackend()

    for i in range(FRAME_COUNT):
        backend.queue_frame(payload_ref={"i": i}, width=1920, height=1080, codec="H264")

    source = RTSPFrameSource(
        camera_id="CAM-BENCH", endpoint="rtsp://bench/stream",
        decoder_backend=backend, sleep_fn=no_delay,
    )
    source.start()

    start = time.perf_counter()

    read_count = 0
    while True:
        frame = source.read_frame()
        if frame is None:
            break
        read_count += 1

    elapsed_seconds = time.perf_counter() - start

    source.stop()

    return {
        "frame_count": read_count,
        "total_ms": elapsed_seconds * 1000,
        "per_frame_us": (elapsed_seconds * 1_000_000) / max(read_count, 1),
    }


def benchmark_status_updates() -> dict:

    call_count = 5000
    received = []

    def callback(camera_id, status, detail):
        received.append((camera_id, status, detail))

    backend = FakeRTSPBackend()
    source = RTSPFrameSource(
        camera_id="CAM-BENCH", endpoint="rtsp://bench/stream",
        decoder_backend=backend, sleep_fn=no_delay, status_callback=callback,
    )

    start = time.perf_counter()

    for _ in range(call_count):
        source._set_status("Online")  # internal call -- benchmarking dispatch overhead only

    elapsed_seconds = time.perf_counter() - start

    return {
        "call_count": call_count,
        "total_ms": elapsed_seconds * 1000,
        "per_call_us": (elapsed_seconds * 1_000_000) / call_count,
    }


def main():

    routing = benchmark_frame_routing()
    print(
        f"Frame routing: {routing['frame_count']} frames, "
        f"{routing['total_ms']:.3f} ms total, {routing['per_frame_us']:.3f} us/frame"
    )

    status = benchmark_status_updates()
    print(
        f"Status updates: {status['call_count']} calls, "
        f"{status['total_ms']:.3f} ms total, {status['per_call_us']:.3f} us/call"
    )

    print()
    print(
        "NOTE: decode latency, network latency, frame-drop rate, and stream "
        "stability are NOT measured here -- no real RTSP stream/decoder exists "
        "yet to measure them honestly. This benchmark covers only "
        "RTSPFrameSource's own routing/construction/status overhead."
    )


if __name__ == "__main__":
    main()
