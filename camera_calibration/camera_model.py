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

        # Real Camera Calibration & World-Coordinate Validation
        # milestone, Phase 14 -- a genuine gap this milestone's own
        # failure-mode investigation surfaced: without this check, a
        # FOV of 0 degrees raised an opaque ZeroDivisionError, a FOV of
        # 180 degrees (geometrically meaningless for any pinhole model
        # -- tan(90 degrees) diverges) silently returned a near-zero
        # focal length, and a negative FOV silently returned a
        # negative focal length -- three different ways to end up with
        # nonsense CameraIntrinsics with no clear signal that the input
        # itself was invalid. A pinhole model's horizontal FOV is only
        # geometrically meaningful strictly between 0 and 180 degrees.
        if not (0.0 < horizontal_fov_degrees < 180.0):
            raise ValueError(
                f"horizontal_fov_degrees must be strictly between 0 and 180 (got {horizontal_fov_degrees})."
            )

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
class CalibrationQuality:

    # Real Camera Calibration & World-Coordinate Validation milestone,
    # Phase 12 -- the one thing CalibrationProfile could not honestly
    # say before this milestone: whether its own numbers have ever been
    # checked against a measured real-world point at all. Deliberately
    # NOT derived from FOV/geometry alone (Phase 12's own explicit "do
    # not invent confidence from FOV alone" instruction) -- the only
    # way this ever gets attached to a CalibrationProfile is a caller
    # actually running camera_calibration.validation.validate_calibration()
    # against real measured ReferencePoints and choosing to record the
    # result. None on CalibrationProfile.quality (the default) means
    # exactly "never validated" -- distinguishable at a glance from a
    # profile that HAS been checked, however good or bad that check
    # turned out to be.

    reference_point_count: int
    validated_point_count: int
    mean_error_m: Optional[float]
    median_error_m: Optional[float]
    max_error_m: Optional[float]
    rmse_m: Optional[float]
    validation_timestamp: str


# CCTV Connection & Calibration Readiness milestone, Phase 7/8 -- the
# runtime policy this milestone's own investigation found missing:
# downstream code (WorldProjector.project(), and everything it feeds --
# RawHumanDetection/Detection/LiveOccupant) previously had NO way to
# tell an UNVALIDATED calibration's world_position apart from a
# VALIDATED one, or from "no calibration at all" -- all three produced
# an identical-looking (x, y) tuple. Deliberately three plain string
# constants, not a new enum type living in a different package (the
# same "plain strings for cross-package status vocabulary, no forced
# import" convention live_camera_pipeline.rtsp_frame_source.
# RTSPFrameSource.status already established for STATUS_ONLINE etc.).
#
# This does NOT introduce a pass/fail accuracy threshold -- no
# engineering justification exists yet for one (see docs/architecture/
# camera_calibration_and_world_projection.md §7). It only makes the
# existing, already-computed CalibrationQuality distinction (§8 there)
# reach every consumer of a world_position, not just a human reading
# the Designer's own calibration status label.
WORLD_POSITION_PROVENANCE_NONE = "no_calibration"
WORLD_POSITION_PROVENANCE_UNVALIDATED = "unvalidated"
WORLD_POSITION_PROVENANCE_VALIDATED = "validated"


def calibration_provenance(profile: "Optional[CalibrationProfile]") -> str:

    # The one place this three-way decision is made -- WorldProjector
    # (projection.py) is the only caller, so every world_position this
    # codebase ever produces is honestly labeled by the SAME rule
    # CalibrationQuality's own docstring already establishes: quality is
    # None means never validated (CONFIGURED -- UNVALIDATED, or NOT
    # CONFIGURED if there is no profile at all); quality.rmse_m is not
    # None means genuinely checked against measured reference points
    # (VALIDATED). A quality object that exists but whose rmse_m is
    # still None (validation attempted, but zero reference points
    # projected) is honestly UNVALIDATED, not VALIDATED -- an attempted-
    # but-failed check earns no more trust than never having checked at
    # all.

    if profile is None:
        return WORLD_POSITION_PROVENANCE_NONE

    if profile.quality is not None and profile.quality.rmse_m is not None:
        return WORLD_POSITION_PROVENANCE_VALIDATED

    return WORLD_POSITION_PROVENANCE_UNVALIDATED


def calibration_status_text(profile: "Optional[CalibrationProfile]") -> str:

    # CCTV Connection & Calibration Readiness milestone, Phase 9 -- the
    # ONE formatter for this exact four-way status message, factored out
    # of designer/widgets/property_panel.py where it previously existed
    # as two near-identical, independently-maintained copies (its own
    # calibration status label, and its calibration dialog's own status
    # label). designer.widgets.camera_manager_panel.CameraManagerPanel's
    # new Calibration column (Phase 9) reuses this exact same function --
    # a camera's calibration status must read identically wherever it is
    # shown, never subtly reworded per screen. Deliberately still no
    # invented accuracy threshold/PASS-FAIL verdict (§7 of docs/
    # architecture/camera_calibration_and_world_projection.md) -- RMSE
    # is reported, never judged.

    if profile is None:
        return "NOT CONFIGURED"

    if profile.quality is None:
        return "CONFIGURED -- UNVALIDATED"

    if profile.quality.rmse_m is None:
        return "CONFIGURED -- VALIDATION ATTEMPTED, NO POINTS PROJECTED"

    return f"VALIDATED -- RMSE: {profile.quality.rmse_m:.3f} m"


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
    #
    # `quality` (Real Camera Calibration & World-Coordinate Validation
    # milestone) is purely additive -- optional, defaults to None
    # ("never validated"), and touches no existing call site (every
    # existing construction of this class uses keyword arguments, see
    # camera_calibration/calibration_loader.py). A profile built by
    # calibration_from_camera()/calibration_from_dict() alone, with no
    # validation step ever run, stays quality=None forever -- exactly
    # the honest "CONFIGURED -- UNVALIDATED" state Phase 12/13 require,
    # never silently upgraded to "validated" by anything other than a
    # genuine validate_calibration() call.

    camera_id: str
    floor_id: str

    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics

    quality: Optional[CalibrationQuality] = None
