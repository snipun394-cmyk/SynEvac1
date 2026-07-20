import re
import time
from typing import Callable, Optional

from credential_store.store import CredentialStore

from live_camera_pipeline.frame_source import CameraFrame, CameraFrameSource
from live_camera_pipeline.rtsp_backend import FrameDecoderBackend, FrameDecoderError


# RTSP Frame Source milestone: the production, network-capable
# CameraFrameSource -- the one piece docs/architecture/
# cctv_integration_readiness.md Sec 18.4 named as genuinely missing
# code. Everything below it in the chain (CameraManager,
# MultiCameraFusionEngine, BuildingStateEstimator, ...) already works
# unchanged; this file is the whole of what's new.
#
# Status vocabulary: deliberately five PLAIN STRINGS -- "Configured",
# "Connecting", "Online", "Degraded", "Stream Unavailable" -- chosen to
# match camera_manager.connection_status.CameraConnectionState's own
# `.value` strings exactly, character for character. This is not a
# competing enum: live_camera_pipeline/ is forbidden from importing
# camera_manager at all (tests/test_live_camera_pipeline.py's own
# LiveCameraPipelineDependencyDirectionTests), so this module cannot
# import CameraConnectionState to reuse it directly. A caller wiring
# RTSPFrameSource's status_callback into a real CameraManager converts
# with a one-line `CameraConnectionState(status)` at that boundary
# (see tests/test_rtsp_camera_manager_status_integration.py) -- the
# existing vocabulary is reused in substance, just not by import.
#
# "RECONNECTING" (the milestone spec's own conceptual state) maps onto
# DEGRADED here: a source that was previously ONLINE and is now
# retrying after a drop is, by definition, a camera that used to work
# and currently doesn't fully -- exactly what DEGRADED already means
# in CameraConnectionState's own docstring. "FAILED" maps onto
# STREAM_UNAVAILABLE (retries exhausted, nothing being served right
# now, but not a permanent/fatal condition -- calling start() again
# retries). OFFLINE is deliberately never produced by this class itself
# (nothing here needs a sixth state distinct from the five already
# used) -- it remains available for other callers of CameraManager.
STATUS_CONFIGURED = "Configured"
STATUS_CONNECTING = "Connecting"
STATUS_ONLINE = "Online"
STATUS_DEGRADED = "Degraded"
STATUS_STREAM_UNAVAILABLE = "Stream Unavailable"


_CREDENTIAL_IN_ENDPOINT = re.compile(r"(://)([^/@:\s]+):([^/@:\s]+)@")


def redact_endpoint(text: Optional[str]) -> str:

    # Strips an inline rtsp://user:pass@host credential down to
    # rtsp://***:***@host -- applied to the endpoint itself (repr) and
    # to any exception/backend-error text that might happen to echo the
    # endpoint back (Phase 9's "never appear unredacted" requirement).
    # ConnectionInfo (models/engineering_asset.py) never actually
    # embeds credentials inline in rtsp_address -- username/password are
    # separate fields there -- but a raw endpoint string handed directly
    # to RTSPFrameSource (or a decoder backend's own error message) is
    # not guaranteed to follow that convention, so this guards the
    # general case regardless of where the string came from.

    if not text:
        return text or ""

    return _CREDENTIAL_IN_ENDPOINT.sub(r"\1***:***@", text)


class RTSPFrameSource(CameraFrameSource):

    # The production CameraFrameSource that will eventually carry a
    # real physical CCTV stream -- today, always driven by an injected
    # FrameDecoderBackend (live_camera_pipeline/rtsp_backend.py), never
    # a concrete decode/transport library directly (none is imported
    # anywhere in this file -- tests/test_no_cv_dependencies.py
    # enforces this). No production FrameDecoderBackend implementation
    # exists yet (writing one needs a real decode library, explicitly
    # out of scope for this milestone) -- decoder_backend is therefore
    # a required argument, not optional-with-a-real-default, because
    # there is nothing honest to default it to.
    #
    # Constructor performs ZERO network I/O -- every field here is a
    # plain assignment. A Digital Twin Camera Asset can be configured
    # for Live mode, including a full RTSP endpoint and credential
    # reference, with no risk of ever touching the physical network
    # until start() is explicitly called (Phase 2's own hard
    # requirement, re-proven in tests/test_rtsp_frame_source.py).

    def __init__(
        self,
        camera_id: str,
        endpoint: str,
        decoder_backend: FrameDecoderBackend,
        username: Optional[str] = None,
        password: Optional[str] = None,
        credential_ref: Optional[str] = None,
        credential_store: Optional[CredentialStore] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.time,
        status_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):

        self.camera_id = camera_id
        self.endpoint = endpoint
        self.username = username
        self.credential_ref = credential_ref

        self._password = password
        self._credential_store = credential_store

        self._decoder = decoder_backend

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

        self._sleep = sleep_fn
        self._clock = clock_fn
        self._status_callback = status_callback

        self._running = False
        self._connected = False
        self._status = STATUS_CONFIGURED
        self._last_error: Optional[str] = None
        self._next_index = 0

    # =====================================================
    # Status (Phase 6/8 -- read here, pushed out via status_callback)
    # =====================================================

    @property
    def status(self) -> str:
        return self._status

    @property
    def last_error(self) -> Optional[str]:

        # Always already sanitized (see _sanitize_error) -- safe to
        # surface directly in a UI/log without further redaction.
        return self._last_error

    def _set_status(self, status: str, detail: Optional[str] = None) -> None:

        self._status = status
        self._last_error = detail

        if self._status_callback is not None:

            # A caller's callback (e.g. wiring into CameraManager.
            # set_connection_status) is not this class's responsibility
            # to protect against -- but it must never be allowed to
            # corrupt this source's own state or crash a connect/read
            # cycle over a bug in unrelated UI code.
            try:
                self._status_callback(self.camera_id, status, detail)
            except Exception:
                pass

    # =====================================================
    # CameraFrameSource interface
    # =====================================================

    @property
    def is_running(self) -> bool:

        # True from start() until stop() -- a session-active flag, not
        # a stream-health flag (docs/architecture/
        # cctv_integration_readiness.md Sec 18.1's own established
        # convention: "is_running=True only means not stopped"). A
        # source can be is_running=True and STATUS_STREAM_UNAVAILABLE
        # at the same time -- that is the honest "still configured to
        # run, currently unable to" state, distinct from "stopped".
        return self._running

    # =====================================================

    def start(self) -> None:

        if self._connected:
            # Already connected -- idempotent no-op, never reopens a
            # live connection just because start() was called again.
            return

        self._running = True
        self._connect_with_retries(initial=True)

    # =====================================================

    def stop(self) -> None:

        # Safe to call repeatedly, safe to call before start(), safe to
        # call mid-reconnect (the retry loop checks self._running after
        # every sleep and aborts as soon as it sees this flip).
        self._running = False

        if self._decoder.is_open:
            try:
                self._decoder.close()
            except Exception:
                # Closing must never raise into the caller -- a decoder
                # that fails to release cleanly is still "stopped" from
                # RTSPFrameSource's own point of view.
                pass

        self._connected = False
        self._set_status(STATUS_CONFIGURED)

    # =====================================================

    def read_frame(self) -> Optional[CameraFrame]:

        if not self._running:
            return None

        if not self._connected:
            # Never fabricate a frame, and never replay a stale one,
            # while genuinely disconnected (initial connect never
            # succeeded, or a previous drop's reconnect exhausted its
            # retries) -- the honest answer is no frame this cycle.
            return None

        try:
            decoded = self._decoder.read()
        except Exception as exc:
            self._handle_drop(exc)
            return None

        if decoded is None:
            return None

        frame = CameraFrame(
            camera_id=self.camera_id,
            timestamp=self._clock(),
            frame_sequence=self._next_index,
            payload_ref=decoded.payload_ref,
            width=decoded.width,
            height=decoded.height,
            codec=decoded.codec,
        )

        self._next_index += 1

        return frame

    # =====================================================
    # Connect / reconnect (Phase 7)
    # =====================================================

    def _handle_drop(self, exc: Exception) -> None:

        # A read() failure mid-stream is a drop, not a hard stop --
        # close whatever the backend still holds, mark disconnected,
        # and attempt a bounded reconnect right away so the *next*
        # read_frame() call (not this one -- this one honestly returns
        # None, never a stale/fabricated frame for the cycle the drop
        # was detected in) has a chance of finding a fresh connection.

        try:
            self._decoder.close()
        except Exception:
            pass

        self._connected = False

        if not self._running:
            return

        self._connect_with_retries(initial=False)

    # =====================================================

    def _connect_with_retries(self, initial: bool) -> None:

        # Bounded, configurable retry with exponential backoff --
        # deliberately NOT an infinite loop (Phase 7's own hard
        # requirement). `initial=True` is the first-ever connection
        # attempt (start()); `initial=False` is a reconnect after a
        # detected drop (_handle_drop) -- the only difference is which
        # status ("Connecting" vs "Degraded") is reported while trying,
        # matching CameraConnectionState's own CONNECTING/DEGRADED
        # distinction.

        trying_status = STATUS_CONNECTING if initial else STATUS_DEGRADED

        attempts = self.max_retries + 1

        for attempt in range(attempts):

            if not self._running:
                # stop() (directly, or a disabled-camera supervisor
                # translating "disabled" into "stopped") happened
                # before this attempt even started.
                return

            self._set_status(trying_status)

            password = None

            try:
                password = self._resolve_password()
                self._decoder.open(self.endpoint, self.username, password)

            except Exception as exc:

                detail = self._sanitize_error(exc, secret=password)
                self._connected = False

                is_last_attempt = attempt == attempts - 1

                if is_last_attempt:
                    self._set_status(STATUS_STREAM_UNAVAILABLE, detail=detail)
                    return

                delay = self.retry_delay * (self.backoff_factor ** attempt)
                self._sleep(delay)

                if not self._running:
                    # stop() happened during the backoff sleep itself --
                    # stop() already set the correct terminal status;
                    # do not overwrite it with another attempt.
                    return

                continue

            self._connected = True
            self._set_status(STATUS_ONLINE)
            return

    # =====================================================
    # Credential resolution (Phase 9)
    # =====================================================

    def _resolve_password(self) -> Optional[str]:

        # Resolved at connect time, every time -- never cached onto
        # self, never read from anywhere but the CredentialStore (or,
        # for tests/manual use, a directly-supplied password) -- the
        # same "resolve at connect time, not from Camera.connection.
        # password directly" discipline docs/architecture/
        # cctv_integration_readiness.md Sec 7 already establishes for
        # every Live adapter.

        if self.credential_ref is None:
            return self._password

        if self._credential_store is None:

            raise FrameDecoderError(
                f"No credential store configured to resolve "
                f"credential_ref={self.credential_ref!r}"
            )

        password = self._credential_store.get_credential(self.credential_ref)

        if password is None:

            raise FrameDecoderError(
                f"No credential saved for reference {self.credential_ref!r}"
            )

        return password

    # =====================================================
    # Redaction (Phase 9)
    # =====================================================

    def _sanitize_error(self, exc: Exception, secret: Optional[str] = None) -> str:

        message = redact_endpoint(str(exc))

        for value in (secret, self._password):

            if value:
                message = message.replace(value, "***")

        return message

    # =====================================================

    def __repr__(self) -> str:

        # Same "never print the real secret" discipline as
        # models.engineering_asset.ConnectionInfo.__repr__ -- password
        # is never included even redacted-inline, only whether one is
        # set at all.

        password_display = "<redacted>" if self._password else "<unset>"

        return (
            f"RTSPFrameSource(camera_id={self.camera_id!r}, "
            f"endpoint={redact_endpoint(self.endpoint)!r}, "
            f"username={self.username!r}, "
            f"password={password_display}, "
            f"credential_ref={self.credential_ref!r}, "
            f"status={self._status!r})"
        )
