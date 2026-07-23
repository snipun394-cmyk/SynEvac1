from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from scipy.optimize import minimize

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.validation import ReferencePoint, project_pixel_point


# =====================================================
# Real Camera Calibration & World-Coordinate Validation milestone,
# Phase 2/3 -- the practical calibration workflow this milestone
# actually needed: NOT automatic/OpenCV calibration (still explicitly
# out of scope, camera_calibration/calibration_loader.py's own Phase 5
# comment stands unchanged), and NOT asking an operator to somehow know
# their camera's exact yaw/pitch/roll in degrees by eye either -- that
# is rarely something anyone can measure directly on a real, already-
# mounted CCTV camera.
#
# What a real site visit CAN produce honestly: a tape-measure/laser-
# measure height for the camera mount, the camera's rough floor-plan
# position, the image resolution, and a handful of (pixel, world)
# correspondences -- e.g. "this floor tile corner is 4.2m east of the
# camera's own footprint, and appears at pixel (512, 780) in a paused
# frame." This module fits yaw_degrees/pitch_degrees (roll_degrees is
# held at the caller-supplied value, normally 0.0 -- a normally-mounted
# camera has negligible roll, and fitting three angles from as few as
# 2-3 correspondences is poorly conditioned) to MINIMIZE the same
# metric camera_calibration.validation.validate_calibration() reports
# -- squared world-space reprojection error -- using the exact same
# forward model (pixel_ray_direction + intersect_ray_with_floor) this
# package already established, never a second/competing projection
# algorithm. `scipy` is already a project dependency (transitively
# required by the already-installed `scikit-learn`); this is a plain
# least-squares fit, not a new framework.
# =====================================================


class CalibrationSolveError(Exception):

    # Raised when a solve is not honestly attempted at all -- too few
    # correspondences to fit two free angles, or the optimizer's own
    # result is degenerate (every correspondence unprojectable at the
    # fitted pose). Never returns a CalibrationProfile silently built
    # from a failed fit.

    pass


MINIMUM_CORRESPONDENCES = 3  # 2 free angles (yaw, pitch) -- at least 3 points to be even loosely over-determined


@dataclass(frozen=True)
class SolvedCalibration:

    profile: CalibrationProfile
    residual_rmse_m: float  # fit quality against the SAME correspondences used to solve -- not a held-out validation


def solve_calibration_from_correspondences(
    camera_id: str,
    floor_id: str,
    intrinsics: CameraIntrinsics,
    camera_position: Tuple[float, float],
    mount_height: float,
    correspondences: Sequence[ReferencePoint],
    *,
    roll_degrees: float = 0.0,
    initial_yaw_degrees: float = 0.0,
    initial_pitch_degrees: float = 30.0,
) -> SolvedCalibration:

    # The practical Phase 2/3 workflow in one function: known mounting
    # height + known/measured camera floor-plan position + known image
    # resolution/FOV (already expressed as `intrinsics`, e.g. via
    # CameraIntrinsics.from_horizontal_fov()) + a handful of measured
    # (pixel, world) floor points -> a fitted CalibrationProfile.
    # yaw_degrees/pitch_degrees are the two angles solved for (an
    # operator rarely knows either precisely on a real mounted camera);
    # mount_height/position are taken as GIVEN, measured inputs, not
    # fitted -- a tape-measure height reading is already more reliable
    # than anything a 2-angle fit over a handful of points could
    # recover for a THIRD unknown simultaneously (poorly conditioned
    # with few correspondences, and this milestone explicitly stops
    # short of a full solvePnP-style joint solve -- see calibration_
    # loader.py's own "future OpenCV calibration" extension point).

    if len(correspondences) < MINIMUM_CORRESPONDENCES:
        raise CalibrationSolveError(
            f"At least {MINIMUM_CORRESPONDENCES} (pixel, world) correspondences are required to fit "
            f"yaw/pitch (got {len(correspondences)})."
        )

    def _residuals(angles) -> float:

        yaw_degrees, pitch_degrees = angles

        extrinsics = CameraExtrinsics(
            position=camera_position, mount_height=mount_height,
            yaw_degrees=yaw_degrees, pitch_degrees=pitch_degrees, roll_degrees=roll_degrees,
        )
        candidate = CalibrationProfile(
            camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics,
        )

        total_squared_error = 0.0

        for reference in correspondences:

            projected = project_pixel_point(reference.pixel, candidate)

            if projected is None:
                # A candidate pose that cannot even reach the floor for
                # this pixel is a bad fit -- a large, finite penalty
                # (never inf/nan, which would break the optimizer's own
                # gradient-free search) keeps the solver moving away
                # from geometrically-impossible poses instead of
                # silently ignoring their contribution to error.
                total_squared_error += 10_000.0 ** 2
                continue

            dx = projected[0] - reference.world[0]
            dy = projected[1] - reference.world[1]
            total_squared_error += dx * dx + dy * dy

        return total_squared_error

    result = minimize(
        _residuals,
        x0=(initial_yaw_degrees, initial_pitch_degrees),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000},
    )

    fitted_yaw, fitted_pitch = result.x

    extrinsics = CameraExtrinsics(
        position=camera_position, mount_height=mount_height,
        yaw_degrees=float(fitted_yaw), pitch_degrees=float(fitted_pitch), roll_degrees=roll_degrees,
    )
    profile = CalibrationProfile(camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics)

    projectable_errors = []
    for reference in correspondences:

        projected = project_pixel_point(reference.pixel, profile)
        if projected is None:
            continue

        dx = projected[0] - reference.world[0]
        dy = projected[1] - reference.world[1]
        projectable_errors.append(dx * dx + dy * dy)

    if not projectable_errors:
        raise CalibrationSolveError(
            "The fitted pose cannot project any of the supplied correspondences onto the floor plane -- "
            "this usually means the measured mount_height/positions/pixels are inconsistent with each other."
        )

    residual_rmse = (sum(projectable_errors) / len(projectable_errors)) ** 0.5

    return SolvedCalibration(profile=profile, residual_rmse_m=residual_rmse)
