from enum import Enum


# =====================================================
# Application Live Runtime Launcher milestone -- Phase 1's own
# investigation (docs/architecture/application_live_runtime_integration.md)
# found `main.py` -> `core.app.SynEvacApp` -> `designer.windows.
# main_window.MainWindow` and `live_runtime.factory.build_live_runtime()`
# were two graphs that never met: the composition root existed and was
# fully tested, but no production entry point ever called it. This
# module defines the vocabulary the application layer built on top of
# that composition root uses -- it introduces no new subsystem of its
# own, only names for states live_runtime.runtime.LiveRuntime and
# live_runtime_launcher.session.LiveRuntimeSession already produce.
# =====================================================


class ApplicationMode(Enum):

    # DESIGNER is authoring-only and never constructs a LiveRuntime --
    # designer.windows.main_window.MainWindow's existing behavior,
    # completely unchanged by this milestone. LIVE and OFFLINE_DEMO both
    # construct a real live_runtime.runtime.LiveRuntime (the SAME
    # composition root every existing test/validation script already
    # uses) from the currently loaded Building; the only difference is
    # which live_runtime.factory function builds it:
    #   LIVE          -> build_live_runtime()          (no output
    #                     providers -- Voice/Signage/Building-Control
    #                     honestly report NO_PROVIDER; no physical
    #                     hardware provider exists yet, see Phase 9)
    #   OFFLINE_DEMO  -> build_offline_demo_runtime()   (Simulation*
    #                     providers defaulted in, Phase 3's own named
    #                     "OFFLINE LIVE DEMO MODE" entry point)

    DESIGNER = "DESIGNER"
    LIVE = "LIVE"
    OFFLINE_DEMO = "OFFLINE_DEMO"


class RuntimeLifecycleState(Enum):

    # Honest operator-facing lifecycle state for a LiveRuntimeSession.
    # STOPPED/RUNNING mirror live_runtime.runtime.LiveRuntime's own
    # is_running boolean, the ground truth this class reads -- never a
    # second, independently-tracked notion of "running." STARTING/
    # FAILED/DEGRADED are this layer's own additions (LiveRuntime.start()
    # itself has no transient or failed state -- it either succeeds or
    # raises). DEGRADED is only ever reported when a configured
    # CameraFrameSource genuinely failed to reach Online -- never
    # fabricated. Reachable under ApplicationMode.LIVE since the CP
    # PLUS NVR -> SynEvac1 Live Runtime Integration milestone (see
    # session.py's own build_rtsp_frame_sources() wiring); still
    # unreachable under OFFLINE_DEMO, which never configures a real
    # RTSPFrameSource.

    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
