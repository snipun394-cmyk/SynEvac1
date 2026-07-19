from enum import Enum


class CameraConnectionState(Enum):

    # Runtime-only signal, distinct from Camera.active
    # (models/engineering_asset.py) on purpose:
    #
    #   Camera.active   -- configuration. "Is this asset enabled for
    #                       use at all?" Set by a user, persisted with
    #                       the project, meaningful in every mode.
    #   CameraConnectionState -- runtime. "Is the live physical stream
    #                       actually reachable right now?" Never
    #                       persisted (see CameraManager.__init__),
    #                       meaningless in Simulation/Replay, only
    #                       relevant once a Live adapter exists.
    #
    # A disabled-but-configured camera and an enabled-but-unreachable
    # camera are different problems; conflating them (as
    # CameraStatus.active / building_state/estimator.py's
    # offline_camera_ids derivation currently does) is a known,
    # documented gap this milestone does not fix -- see
    # docs/architecture/cctv_integration_readiness.md.

    CONFIGURED = "Configured"
    CONNECTING = "Connecting"
    ONLINE = "Online"
    OFFLINE = "Offline"
    DEGRADED = "Degraded"
    STREAM_UNAVAILABLE = "Stream Unavailable"
