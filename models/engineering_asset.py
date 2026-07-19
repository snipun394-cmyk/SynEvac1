from dataclasses import dataclass, field
from typing import Optional, Tuple

from models.base_object import BaseObject


class DeviceMode:

    # Where a device's detections currently originate. The device
    # object itself (Camera today; Smoke/Heat Detector, Manual Call
    # Point, Voice Evacuation Speaker, Fire Alarm Panel in the future)
    # never changes shape across these -- only the (not-yet-built)
    # source feeding it does:
    #
    #   Simulation Mode -> Virtual Camera / simulated sensor reading
    #   Replay Mode     -> Recorded video / logged sensor reading
    #   Live Mode       -> Real CCTV / real hardware over the
    #                      Connection Info placeholders below
    #
    # A plain string constant, not an Enum: it round-trips through
    # to_dict()/from_dict() as a bare JSON string with no extra
    # decoding step, same convention Detector.DETECTOR_TYPES already
    # uses for its own closed set of string values.

    SIMULATION = "Simulation"
    REPLAY = "Replay"
    LIVE = "Live"

    ALL = (SIMULATION, REPLAY, LIVE)


@dataclass
class ConnectionInfo:

    # Real-world connection placeholders only -- never read, never
    # connected to, by anything in this codebase today. Storing them
    # now means a user can pre-configure a not-yet-installed device's
    # real-world address ahead of time, and it survives every save/
    # load until a future Live Mode integration reads it for the
    # first time. No validation is performed here (a malformed RTSP
    # URL or IP is not this class's concern) -- it is a plain,
    # inert value holder.
    #
    # `password` is deliberately NOT part of to_dict()'s output --
    # see credential_store/project_credentials.py. It stays as an
    # in-memory-only field so the Property Panel still has somewhere
    # to hold a freshly typed password before it gets captured into
    # the CredentialStore; `credential_ref` is what actually
    # round-trips through project save/load. from_dict() still
    # tolerates a legacy "password" key (old project files saved
    # before this change) purely so those files keep loading --
    # capture_and_clear_camera_credentials() is what migrates that
    # legacy plaintext into the store and clears it from memory.

    rtsp_address: str = ""
    ip_address: str = ""
    username: str = ""
    password: str = ""
    credential_ref: Optional[str] = None

    def to_dict(self):

        return {
            "rtsp_address": self.rtsp_address,
            "ip_address": self.ip_address,
            "username": self.username,
            "credential_ref": self.credential_ref,
        }

    @classmethod
    def from_dict(cls, data):

        if not data:
            return cls()

        return cls(
            rtsp_address=data.get("rtsp_address", ""),
            ip_address=data.get("ip_address", ""),
            username=data.get("username", ""),
            # Legacy-only: old project files saved the real password
            # here. Loaded into memory so nothing crashes and nothing
            # is silently lost, then migrated out by
            # capture_and_clear_camera_credentials().
            password=data.get("password", ""),
            credential_ref=data.get("credential_ref"),
        )

    def __repr__(self):

        # Never print the real password, whether it holds a freshly
        # typed value or a legacy-loaded one -- repr() is exactly the
        # kind of thing that ends up in a log line or a debug panel.
        password_display = "<redacted>" if self.password else "<unset>"

        return (
            f"ConnectionInfo(rtsp_address={self.rtsp_address!r}, "
            f"ip_address={self.ip_address!r}, "
            f"username={self.username!r}, "
            f"password={password_display}, "
            f"credential_ref={self.credential_ref!r})"
        )


@dataclass
class EngineeringAsset(BaseObject):

    # Shared shape for every physical device the Digital Twin (the
    # Building Designer) places inside a building -- Camera is the
    # first concrete one; Smoke/Heat Detector, Manual Call Point,
    # Voice Evacuation Speaker, and Fire Alarm Panel are future
    # candidates that fit the same shape without this class changing.
    #
    # Deliberately NOT retrofitted onto Detector in this change --
    # Detector's own fields already match this shape 1:1
    # (position/floor_id/mount_height/active), so rebasing it here
    # later is a zero-behavior-change move, not something that needs
    # doing now. Nothing about this class depends on that happening.
    #
    # This is a plain data container, same as BaseObject itself --
    # no behavior beyond to_dict()/from_dict() support. Coverage
    # geometry, rendering, and everything device-specific stay on
    # the concrete subclass (see Camera.coverage_polygon()).

    floor_id: str = ""

    # Zones this device is considered to cover/serve. Plain id
    # references, same convention as Staircase.from_zone_id/
    # Door.zone_a_id -- never resolved or validated here, only ever
    # set explicitly and interpreted by whatever consumes this device
    # (e.g. a future GroundTruthCameraProvider zone-assignment
    # override). A tuple so equality/hashing behave like every other
    # immutable-by-convention collection field in this codebase
    # (compare pathfinding.route.Route's tuple-of-nodes); mutation
    # always goes through reassignment, never in-place append.
    zone_ids: Tuple[str, ...] = field(default_factory=tuple)

    position: tuple = (0.0, 0.0)
    mount_height: float = 3.0

    active: bool = True

    # Simulation / Replay / Live -- see DeviceMode. Only Simulation is
    # meaningful today (nothing compares this value against anything
    # yet); stored so a future Replay/Live integration has somewhere
    # authoritative to read the user's choice from without any change
    # to this class or its subclasses.
    mode: str = DeviceMode.SIMULATION

    connection: ConnectionInfo = field(default_factory=ConnectionInfo)

    # =====================================================
    # Subclasses call these from their own to_dict()/from_dict()
    # rather than inheriting a template-method to_dict() directly --
    # every existing model in this codebase (Camera, Detector, Door,
    # ...) builds its own concrete to_dict()/from_dict(), and a
    # subclass still needs to call cls(**kwargs) itself to end up
    # with the *concrete* type, not an EngineeringAsset. Splitting
    # the shared portion out as a plain dict-in/dict-out helper
    # avoids fighting a multi-level dataclass constructor chain for
    # that one reason.
    # =====================================================

    def _asset_dict(self):

        data = super().to_dict()

        data.update({
            "floor_id": self.floor_id,
            "zone_ids": list(self.zone_ids),
            "position": self.position,
            "mount_height": self.mount_height,
            "active": self.active,
            "mode": self.mode,
            "connection": self.connection.to_dict(),
        })

        return data

    # =====================================================

    @classmethod
    def _asset_kwargs(cls, data):

        return dict(

            id=data["id"],

            name=data.get(
                "name",
                "",
            ),

            properties=data.get(
                "properties",
                {},
            ),

            created_at=data.get(
                "created_at",
                "",
            ),

            modified_at=data.get(
                "modified_at",
                "",
            ),

            floor_id=data.get(
                "floor_id",
                "",
            ),

            zone_ids=tuple(
                data.get(
                    "zone_ids",
                    (),
                )
            ),

            position=tuple(
                data.get(
                    "position",
                    (0.0, 0.0),
                )
            ),

            mount_height=data.get(
                "mount_height",
                3.0,
            ),

            active=data.get(
                "active",
                True,
            ),

            mode=data.get(
                "mode",
                DeviceMode.SIMULATION,
            ),

            connection=ConnectionInfo.from_dict(
                data.get("connection")
            ),
        )
