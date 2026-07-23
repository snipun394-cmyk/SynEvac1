import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence, Tuple

from camera_calibration.camera_model import CalibrationProfile
from camera_calibration.geometry import intersect_ray_with_floor, pixel_ray_direction


# =====================================================
# Real Camera Calibration & World-Coordinate Validation milestone,
# Phase 4 -- "a calibration is useless unless we can measure its error."
# Pure geometry + statistics, same discipline as geometry.py/
# projection.py: no camera_id lookup, no Zone/registry dependency, no
# fabricated results. project_pixel_point() is the one new primitive
# this module adds to camera_calibration itself -- projection.py's own
# WorldProjector.project() only ever projects a DETECTION's bounding
# box (bottom-center ground-contact assumption); validating a
# calibration needs the more general "project THIS EXACT pixel" (a
# marked floor-tile corner, a tape-measure mark's photographed pixel,
# ...), which has no bounding box at all. Reuses the exact same
# pixel_ray_direction()/intersect_ray_with_floor() ray-cast this
# package already established -- no second projection algorithm.
# =====================================================


def project_pixel_point(
    pixel: Tuple[float, float],
    calibration: CalibrationProfile,
) -> Optional[Tuple[float, float]]:

    # The same ray-cast WorldProjector._ground_contact_point() feeds,
    # applied directly to an already-known pixel (no bounding box, no
    # "assume feet touch the floor" step -- the caller already knows
    # exactly which pixel corresponds to a real, measured floor point).
    # None -- never a fabricated position -- whenever the ray is
    # geometrically incapable of reaching the floor plane (see
    # intersect_ray_with_floor's own docstring), the same honest
    # degradation every other projection path in this package uses.

    direction = pixel_ray_direction(
        pixel[0], pixel[1], calibration.intrinsics, calibration.extrinsics,
    )
    origin = (
        calibration.extrinsics.position[0],
        calibration.extrinsics.position[1],
        calibration.extrinsics.mount_height,
    )

    return intersect_ray_with_floor(origin, direction)


def resolution_mismatch(
    calibration: CalibrationProfile,
    frame_width: Optional[int],
    frame_height: Optional[int],
) -> Optional[Tuple[int, int]]:

    # Real Camera Calibration & World-Coordinate Validation milestone,
    # Phase 14 -- a genuine, previously-undetectable failure mode this
    # milestone's own investigation surfaced: WorldProjector.project()
    # only ever receives a bare bounding box, never the frame's own
    # actual resolution, so a CalibrationProfile fitted/entered for one
    # resolution (e.g. 1920x1080) silently produces a WRONG-but-
    # plausible-looking world position if it is later used against
    # detections from a differently-configured camera stream (e.g.
    # 640x480) -- every pixel coordinate is simply reinterpreted
    # against the wrong principal point/focal length, with no error
    # anywhere. This is NOT fixed by changing WorldProjector's own
    # signature (that would be exactly the "redesign the pipeline"
    # this milestone was told not to do) -- it is instead exposed as a
    # standalone, opt-in diagnostic a caller (Designer's calibration
    # dialog, an operator's own pre-flight check) can run against a
    # real live_camera_pipeline.frame_source.CameraFrame's own
    # width/height fields (already populated by a real RTSP/decoded
    # source, see frame_source.py) whenever it wants to catch this
    # honestly instead of silently trusting a stale calibration.
    #
    # Returns None when there is nothing to compare (frame_width/height
    # unknown -- e.g. a ReplayFrameSource/synthetic test frame that
    # never populates them) or when they match; otherwise returns the
    # calibration's own (image_width, image_height) so a caller can
    # report exactly what mismatched against what.

    if frame_width is None or frame_height is None:
        return None

    if calibration.intrinsics.image_width == frame_width and calibration.intrinsics.image_height == frame_height:
        return None

    return (calibration.intrinsics.image_width, calibration.intrinsics.image_height)


@dataclass(frozen=True)
class ReferencePoint:

    # One ground-truth correspondence: a pixel a human operator marked
    # (or clicked) in the image, paired with that same physical spot's
    # already-known real-world floor-plan position (measured with a
    # tape measure/laser measure, or read off an as-built floor plan --
    # never guessed). label is optional provenance only (e.g. "tile
    # corner NE", "doorway threshold") -- never interpreted, purely for
    # a human reading a validation report to know which physical mark
    # each row refers to.

    pixel: Tuple[float, float]
    world: Tuple[float, float]
    label: Optional[str] = None


@dataclass(frozen=True)
class PointValidationResult:

    reference: ReferencePoint
    projected_world: Optional[Tuple[float, float]]
    error_m: Optional[float]  # None whenever projection itself failed -- never a fabricated 0.0/None-as-zero


@dataclass(frozen=True)
class CalibrationValidationReport:

    # Camera Calibration & World-Coordinate Validation milestone, Phase
    # 4/12 -- the one honest, quantitative answer to "is this
    # calibration any good", built ONLY from points a caller actually
    # supplied measured ground truth for. reference_point_count is the
    # TOTAL points supplied (including any that failed to project at
    # all -- a calibration that cannot even project half its own
    # reference points is itself a genuine finding, not something to
    # silently exclude from the count); every error statistic below is
    # computed only over the subset that DID project (an unprojectable
    # point contributes no numeric error, never a fabricated one).

    camera_id: str
    reference_point_count: int
    point_results: Tuple[PointValidationResult, ...]

    mean_error_m: Optional[float]
    median_error_m: Optional[float]
    max_error_m: Optional[float]
    rmse_m: Optional[float]

    validated_point_count: int  # how many of reference_point_count actually produced a projected position
    validation_timestamp: str  # ISO-8601 UTC, when this report was computed


def validate_calibration(
    calibration: CalibrationProfile,
    reference_points: Sequence[ReferencePoint],
) -> CalibrationValidationReport:

    # Projects every supplied reference point's own pixel through THIS
    # calibration and compares against that point's already-known real
    # world position -- the only way to honestly measure calibration
    # error (never inferred from FOV/geometry alone, Phase 12's own
    # explicit "do not invent confidence from FOV alone" instruction).

    point_results = []
    errors = []

    for reference in reference_points:

        projected = project_pixel_point(reference.pixel, calibration)

        if projected is None:
            point_results.append(
                PointValidationResult(reference=reference, projected_world=None, error_m=None)
            )
            continue

        dx = projected[0] - reference.world[0]
        dy = projected[1] - reference.world[1]
        error = math.sqrt(dx * dx + dy * dy)

        point_results.append(
            PointValidationResult(reference=reference, projected_world=projected, error_m=error)
        )
        errors.append(error)

    mean_error = statistics.mean(errors) if errors else None
    median_error = statistics.median(errors) if errors else None
    max_error = max(errors) if errors else None
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None

    return CalibrationValidationReport(
        camera_id=calibration.camera_id,
        reference_point_count=len(reference_points),
        point_results=tuple(point_results),
        mean_error_m=mean_error,
        median_error_m=median_error,
        max_error_m=max_error,
        rmse_m=rmse,
        validated_point_count=len(errors),
        validation_timestamp=datetime.now(timezone.utc).isoformat(),
    )
