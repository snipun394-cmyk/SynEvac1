import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CameraIntrinsics:

    # Camera Calibration & World Coordinate Projection milestone,
    # Phase 3 -- a standard pinhole-camera intrinsic model: image
    # resolution (pixels), focal length (pixels, one value per axis --
    # a real lens is rarely perfectly square-pixeled, so fx/fy are
    # independent), and principal point (pixels -- where the optical
    # axis actually pierces the image plane; not always exactly the
    # image center for a real lens, so it defaults to the center only
    # when not supplied).
    #
    # Deliberately its own type, never stored on models.camera.Camera
    # (Phase 3's own "store calibration separately from Camera Assets"
    # requirement) -- Camera.resolution is a display-only string
    # ("1920x1080", never parsed -- see models/camera.py's own
    # docstring), and Camera.horizontal_fov describes the 2D coverage-
    # polygon wedge Visibility uses, not a real lens's focal length.
    # from_horizontal_fov() below is the one deliberate bridge between
    # the two, for a caller with only a coverage-style FOV angle and no
    # real lens datasheet.

    image_width: int
    image_height: int

    focal_length_x: float
    focal_length_y: float

    principal_point_x: Optional[float] = None
    principal_point_y: Optional[float] = None

    def __post_init__(self):

        if self.principal_point_x is None:
            object.__setattr__(self, "principal_point_x", self.image_width / 2.0)

        if self.principal_point_y is None:
            object.__setattr__(self, "principal_point_y", self.image_height / 2.0)

    # =====================================================

    @classmethod
    def from_horizontal_fov(
        cls,
        image_width: int,
        image_height: int,
        horizontal_fov_degrees: float,
    ) -> "CameraIntrinsics":

        # Derives an equivalent focal length from a horizontal FOV
        # angle -- the standard pinhole relationship fx = (width/2) /
        # tan(fov/2). Assumes square pixels (fy = fx) since a FOV-only
        # description carries no separate vertical-axis information --
        # an honest, explicitly-stated assumption, not a silent one.
        # This is the bridge a caller uses to build a first calibration
        # straight from an existing models.camera.Camera.horizontal_fov
        # (see calibration_loader.calibration_from_camera()), before any
        # real lens datasheet/OpenCV calibration is available.

        half_fov_radians = math.radians(horizontal_fov_degrees) / 2.0
        focal_length = (image_width / 2.0) / math.tan(half_fov_radians)

        return cls(
            image_width=image_width, image_height=image_height,
            focal_length_x=focal_length, focal_length_y=focal_length,
        )


@dataclass(frozen=True)
class CameraExtrinsics:

    # Phase 3 -- where the camera physically is and how it's aimed.
    # position/mount_height deliberately reuse models.engineering_asset.
    # EngineeringAsset's own existing fields/units/convention exactly
    # (position: (x, y) floor-plan meters; mount_height: meters above
    # THIS floor's own surface, default 3.0 -- the exact field real
    # Camera Assets already carry) -- never redefined or duplicated
    # here, only referenced by value. yaw_degrees likewise reuses
    # models.camera.Camera.rotation's own exact convention ("0 degrees
    # points along +x, increasing clockwise").
    #
    # pitch_degrees and roll_degrees are the two genuinely NEW angles
    # this milestone introduces -- nothing in this codebase modeled a
    # camera's vertical tilt or roll before now. pitch_degrees is
    # POSITIVE when the camera is tilted DOWNWARD from perfectly
    # horizontal (0 degrees = level, 90 degrees = looking straight down
    # at the floor) -- the natural convention for a ceiling/wall-
    # mounted security camera, which must look down to see a floor at
    # all. roll_degrees rotates the image around the camera's own
    # optical (forward) axis -- typically ~0 for a normally-mounted
    # camera, included for full generality (Phase 3's own explicit
    # ask).

    position: Tuple[float, float]
    mount_height: float
    yaw_degrees: float
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0


@dataclass(frozen=True)
class CalibrationProfile:

    # One camera's complete calibration -- intrinsics + extrinsics +
    # which camera/floor it belongs to. Intentionally NOT attached to
    # models.camera.Camera itself (Phase 3's own requirement) --
    # cross_camera_identity.identity_registry.IdentityRegistry already
    # established this codebase's own "a NEW registry, keyed by id,
    # owns this new concern separately from the Digital Twin asset
    # itself" convention; camera_calibration.calibration.
    # CalibrationRegistry (calibration.py) is the same pattern applied
    # here.

    camera_id: str
    floor_id: str

    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
