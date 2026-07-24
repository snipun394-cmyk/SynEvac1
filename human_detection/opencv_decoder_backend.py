from typing import Optional
from urllib.parse import urlsplit, urlunsplit


from live_camera_pipeline.rtsp_backend import DecodedFrame, FrameDecoderBackend, FrameDecoderError


# =====================================================
# CCTV Connection & Calibration Readiness milestone, Phase 2 -- the one
# concrete, production FrameDecoderBackend implementation that
# live_camera_pipeline.rtsp_frame_source.RTSPFrameSource has needed
# since it was built (docs/architecture/cctv_integration_readiness.md
# §19.11 named this as the one remaining piece). Lives in
# human_detection/, not live_camera_pipeline/ -- the same "this package
# is the one place cv2/torch/ultralytics may be imported" convention
# human_detection/yolo_backend.py and human_detection/video_source.py
# already established (tests/test_no_cv_dependencies.py's TARGETS list
# does not include human_detection/). Importing FrameDecoderBackend
# FROM live_camera_pipeline is the correct direction -- human_detection
# depending on a live_camera_pipeline INTERFACE, never the reverse.
#
# Deliberately satisfies FrameDecoderBackend and nothing more: this is
# not a general video toolkit, only the smallest real implementation of
# the seam RTSPFrameSource already defines. RTSPFrameSource itself
# (connect/reconnect/backoff/status/credential resolution/redaction)
# does not change at all -- this class only ever needs to open a
# stream, hand back one decoded frame at a time, and close cleanly.
# =====================================================


def build_authenticated_url(endpoint: str, username: Optional[str], password: Optional[str]) -> str:

    # Embeds username/password into an rtsp(s):// URL's netloc, the
    # conventional way most RTSP servers/NVRs expect inline
    # authentication (rtsp://user:pass@host:port/path) -- OpenCV's own
    # FFMPEG-backed VideoCapture reads credentials this way, not via a
    # separate parameter. A non-RTSP endpoint (a plain local file path,
    # used by every offline test and by Phase 3's own vtest.avi proof)
    # has no URL scheme/netloc to embed into at all -- returned
    # unchanged, exactly as a local file path must be.
    #
    # Never called with the resulting value logged/repr'd anywhere --
    # the caller (open(), below) uses this URL for exactly one
    # cv2.VideoCapture(...) call and lets it fall out of scope
    # immediately after; any failure raised past this point is
    # re-sanitized through _redact(endpoint) (the credential-free
    # original), never through this authenticated URL.

    if username is None and password is None:
        return endpoint

    parts = urlsplit(endpoint)

    if parts.scheme not in ("rtsp", "rtsps"):
        # Not a network stream URL at all (a local file path, e.g.
        # "validation_media/vtest.avi") -- credentials have nowhere
        # honest to go; ignored rather than mangling the path.
        return endpoint

    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"

    user = username or ""
    pw = password or ""
    credential_prefix = f"{user}:{pw}@" if (user or pw) else ""

    netloc = f"{credential_prefix}{host}"

    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _redact(endpoint: str) -> str:

    # A plain, credential-free reflection of the endpoint this backend
    # was given -- used in every error message this class raises, so a
    # log/status line never needs to guess whether build_authenticated_url()
    # was ever called. Deliberately independent of live_camera_pipeline.
    # rtsp_frame_source.redact_endpoint() (this module must not import
    # anything from that file beyond the FrameDecoderBackend/DecodedFrame/
    # FrameDecoderError contract itself) -- the endpoint this class
    # receives never has credentials embedded in the first place
    # (RTSPFrameSource always passes its own plain self.endpoint,
    # username, password as three separate arguments), so no stripping
    # is actually required; this exists only so a future caller that
    # DOES hand in a credentialed endpoint string still gets a safe
    # message rather than an accidental leak.

    parts = urlsplit(endpoint)

    if "@" not in parts.netloc:
        return endpoint

    host = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


class OpenCVFrameDecoderBackend(FrameDecoderBackend):

    # Wraps cv2.VideoCapture behind the exact FrameDecoderBackend shape.
    # Constructor performs ZERO I/O (no cv2 import even happens at
    # construction -- imported lazily inside open(), the same "importing
    # this module never requires the real library to be installed unless
    # a real backend is actually used" discipline human_detection.
    # yolo_backend.UltralyticsYOLOBackend._ensure_loaded() already
    # established one milestone ago).
    #
    # One instance is single-camera, single-stream -- exactly the
    # granularity RTSPFrameSource itself already assumes (one
    # decoder_backend per RTSPFrameSource, one RTSPFrameSource per
    # camera_id). Re-opening after close() is supported (open() simply
    # creates a fresh cv2.VideoCapture) -- this is what lets
    # RTSPFrameSource's own bounded reconnect loop (rtsp_frame_source.py
    # _connect_with_retries) work unmodified against a real backend.

    def __init__(self, *, api_preference: Optional[int] = None, open_timeout_ms: Optional[int] = None):

        # api_preference: an optional cv2.CAP_* constant (e.g.
        # cv2.CAP_FFMPEG) -- left as a plain int (never a bound cv2
        # constant at the constructor-argument level) so importing this
        # module still costs nothing when cv2 is not installed. None
        # (the default) lets OpenCV auto-detect, its own normal
        # behavior. open_timeout_ms, if supplied, is applied via
        # cv2.CAP_PROP_OPEN_TIMEOUT_MSEC once the capture exists --
        # bounds how long a single open() call can hang against an
        # unreachable host, rather than blocking forever (Phase 2's own
        # "never silently retries forever" requirement, applied one
        # level down: a single open() attempt must not hang forever
        # either).

        self._api_preference = api_preference
        self._open_timeout_ms = open_timeout_ms

        self._capture = None
        self._endpoint_for_errors: Optional[str] = None

    # =====================================================

    @property
    def is_open(self) -> bool:

        return self._capture is not None and bool(self._capture.isOpened())

    # =====================================================

    def open(self, endpoint: str, username: Optional[str], password: Optional[str]) -> None:

        # No hidden retry here -- exactly one connection attempt.
        # RTSPFrameSource._connect_with_retries is the ONLY place that
        # decides whether/when to try again (bounded, with backoff) --
        # duplicating that policy here would give this codebase two
        # independent, uncoordinated retry loops, which is exactly what
        # Phase 2's "never silently retries forever" instruction warns
        # against.

        import cv2

        self._endpoint_for_errors = endpoint

        # A stale handle from a previous failed/dropped attempt must
        # never leak past this call -- always start from a clean state.
        self._release_quietly()

        url = build_authenticated_url(endpoint, username, password)

        # CAP_PROP_OPEN_TIMEOUT_MSEC/CAP_PROP_READ_TIMEOUT_MSEC must be
        # supplied as constructor `params`, NOT via a post-construction
        # capture.set() call -- cv2.VideoCapture(url, ...) performs the
        # actual connect synchronously INSIDE the constructor itself, so
        # a property set afterward is already too late to bound an
        # unreachable host's connect time (verified directly: an
        # unroutable RTSP host otherwise blocks for OS-level TCP
        # retransmission timeouts, tens of seconds to minutes -- exactly
        # the "must not hang forever" failure this Phase 2 requirement
        # exists to prevent).
        #
        # Passing timeout params also REQUIRES pinning an explicit
        # apiPreference (verified directly: cv2.CAP_ANY -- the default
        # OpenCV would otherwise pick -- probes multiple backend
        # candidates in turn when params are present, and any candidate
        # that does not understand these two properties simply ignores
        # them and blocks for its own much longer default timeout,
        # silently defeating the bound this code exists to enforce).
        # CAP_FFMPEG is the one backend that honors both properties for
        # RTSP and reads local video files equally well -- the default
        # unless a caller explicitly overrides api_preference.
        params = []
        if self._open_timeout_ms is not None:
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._open_timeout_ms])
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._open_timeout_ms])

        if self._api_preference is not None:
            api_preference = self._api_preference
        elif params:
            api_preference = cv2.CAP_FFMPEG
        else:
            api_preference = None

        try:

            if params:
                capture = cv2.VideoCapture(url, api_preference, params)
            elif api_preference is not None:
                capture = cv2.VideoCapture(url, api_preference)
            else:
                capture = cv2.VideoCapture(url)

        except Exception as exc:

            raise FrameDecoderError(
                f"Failed to open video stream at {_redact(endpoint)!r}: {exc}"
            ) from None

        if not capture.isOpened():

            capture.release()

            raise FrameDecoderError(
                f"Could not open video stream at {_redact(endpoint)!r} -- "
                f"stream unreachable, wrong path, or unsupported by this OpenCV build."
            )

        self._capture = capture

    # =====================================================

    def read(self) -> Optional[DecodedFrame]:

        if self._capture is None:

            raise FrameDecoderError("read() called before a successful open() -- no active capture.")

        if not self._capture.isOpened():

            # The capture was open a moment ago and is not anymore --
            # a genuine stream drop, not "no frame this cycle yet".
            # RTSPFrameSource._handle_drop treats any raised exception
            # from read() exactly this way already.
            raise FrameDecoderError(
                f"Video capture for {_redact(self._endpoint_for_errors or '')!r} closed unexpectedly."
            )

        import cv2

        ok, frame = self._capture.read()

        if not ok or frame is None:
            # Honest "no frame available this call" -- true both for a
            # transient decode miss on a live stream and for having
            # reached the end of a finite local file. Never fabricated,
            # never raised: RTSPFrameSource.read_frame() already treats
            # None as "nothing this cycle," exactly the right meaning
            # here.
            return None

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        codec = self._decode_fourcc(self._capture.get(cv2.CAP_PROP_FOURCC))

        return DecodedFrame(payload_ref=frame, width=width, height=height, codec=codec)

    # =====================================================

    def close(self) -> None:

        # Must never raise (FrameDecoderBackend's own contract,
        # rtsp_backend.py:84-88) -- RTSPFrameSource.stop()/_handle_drop
        # both call this defensively and would otherwise have to guard
        # every call site themselves.
        self._release_quietly()

    # =====================================================

    def _release_quietly(self) -> None:

        if self._capture is not None:

            try:
                self._capture.release()
            except Exception:
                pass

        self._capture = None

    # =====================================================

    def _decode_fourcc(self, raw_fourcc: float) -> Optional[str]:

        try:

            code = int(raw_fourcc)

            if code == 0:
                return None

            chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
            stripped = chars.strip().strip("\x00")

            return stripped or None

        except Exception:
            # Codec reporting is diagnostic-only -- never worth failing
            # a real frame read over.
            return None
