"""Real Camera Calibration & World-Coordinate Validation milestone --
the practical, minimum-viable calibration workflow a SynEvac operator
runs for one real, already-mounted, fixed CCTV camera. Reuses the
EXISTING camera_calibration architecture end to end (CameraIntrinsics/
CameraExtrinsics/CalibrationProfile/CalibrationRegistry/
calibration_loader) -- this script contains no competing calibration
type and no automatic/OpenCV calibration (still explicitly out of
scope, see camera_calibration/calibration_loader.py's own Phase 5
comment).

Two supported inputs, both reusing a single JSON "scene" description
(see docs/architecture/camera_calibration_and_world_projection.md for
the full field reference and a worked example):

  1. MANUAL -- the operator already knows (or measured) yaw_degrees AND
     pitch_degrees directly. The scene file's own `yaw_degrees`/
     `pitch_degrees` are used as-is; no fitting happens.

  2. CORRESPONDENCE-FITTED -- the operator instead supplies >= 3
     (pixel, world) correspondences (a paused frame's own pixel
     coordinates, paired with that same physical floor spot's already-
     measured real-world position) and yaw_degrees/pitch_degrees are
     omitted (or null) -- camera_calibration.calibration_solver fits
     them to minimize reprojection error against exactly those points.

Either way, mount_height/camera_position/image resolution/FOV are
ALWAYS caller-supplied, measured inputs -- never fitted, never
guessed (Phase 2's own "minimum practical manual/semi-manual approach"
boundary: a tape-measure height reading is more reliable than anything
a handful of correspondences could recover for a third simultaneously-
fitted unknown).

If the scene file also supplies `validation_points` (a SEPARATE,
held-out set of correspondences -- never the same points used to fit),
the resulting calibration is validated against them and a
CalibrationQuality is attached, reported, and saved alongside the
profile (Phase 4/12). Omitting `validation_points` leaves the saved
calibration honestly `quality=None` ("CONFIGURED -- UNVALIDATED",
Phase 13's own status vocabulary) -- never fabricated as validated.

Usage:
    python scripts/calibrate_camera_scene.py scene.json --out calibration.json
    python scripts/calibrate_camera_scene.py scene.json --out calibration.json --validate-only

A lightweight, optional point-picking helper (NOT a second Designer
application -- a single cv2 window, click to record a pixel, no other
UI) is also provided for building the scene file's own `correspondences`
pixel coordinates directly off a paused real frame:
    python scripts/calibrate_camera_scene.py --pick-points frame.png --points-out clicked_pixels.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_calibration.camera_model import CalibrationProfile, CalibrationQuality, CameraExtrinsics, CameraIntrinsics
from camera_calibration.calibration_loader import save_calibration_json
from camera_calibration.calibration_solver import solve_calibration_from_correspondences
from camera_calibration.validation import ReferencePoint, validate_calibration


def _build_intrinsics(scene: dict) -> CameraIntrinsics:

    image_width = scene["image_width"]
    image_height = scene["image_height"]

    if "horizontal_fov_degrees" in scene:
        return CameraIntrinsics.from_horizontal_fov(
            image_width=image_width, image_height=image_height,
            horizontal_fov_degrees=scene["horizontal_fov_degrees"],
        )

    return CameraIntrinsics(
        image_width=image_width, image_height=image_height,
        focal_length_x=scene["focal_length_x"], focal_length_y=scene["focal_length_y"],
    )


def _parse_reference_points(entries) -> list:

    return [
        ReferencePoint(pixel=tuple(entry["pixel"]), world=tuple(entry["world"]), label=entry.get("label"))
        for entry in entries
    ]


def _report_validation(report) -> None:

    print(f"  reference points supplied: {report.reference_point_count}")
    print(f"  reference points that projected onto the floor plane: {report.validated_point_count}")

    if report.mean_error_m is None:
        print("  NO POINTS PROJECTED -- calibration cannot be validated against these points at all.")
        return

    print(f"  mean error:   {report.mean_error_m:.4f} m")
    print(f"  median error: {report.median_error_m:.4f} m")
    print(f"  max error:    {report.max_error_m:.4f} m")
    print(f"  RMSE:         {report.rmse_m:.4f} m")

    for point_result in report.point_results:
        label = point_result.reference.label or "(unlabeled)"
        if point_result.error_m is None:
            print(f"    [{label}] pixel={point_result.reference.pixel} -- DID NOT PROJECT onto the floor plane")
        else:
            print(f"    [{label}] pixel={point_result.reference.pixel} error={point_result.error_m:.4f} m")


def run_calibration(scene_path: str, out_path: str, validate_only: bool) -> None:

    scene = json.loads(Path(scene_path).read_text(encoding="utf-8"))

    camera_id = scene["camera_id"]
    floor_id = scene["floor_id"]
    intrinsics = _build_intrinsics(scene)
    camera_position = tuple(scene["camera_position"])
    mount_height = scene["mount_height"]
    roll_degrees = scene.get("roll_degrees", 0.0)

    manual_yaw = scene.get("yaw_degrees")
    manual_pitch = scene.get("pitch_degrees")

    if manual_yaw is not None and manual_pitch is not None:

        print("=== MANUAL calibration (yaw/pitch supplied directly) ===")
        extrinsics = CameraExtrinsics(
            position=camera_position, mount_height=mount_height,
            yaw_degrees=manual_yaw, pitch_degrees=manual_pitch, roll_degrees=roll_degrees,
        )
        profile = CalibrationProfile(camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics)

    else:

        correspondences = _parse_reference_points(scene["correspondences"])

        print(f"=== CORRESPONDENCE-FITTED calibration ({len(correspondences)} points) ===")
        solved = solve_calibration_from_correspondences(
            camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics,
            camera_position=camera_position, mount_height=mount_height,
            correspondences=correspondences, roll_degrees=roll_degrees,
        )
        profile = solved.profile

        print(f"  fitted yaw_degrees:   {profile.extrinsics.yaw_degrees:.3f}")
        print(f"  fitted pitch_degrees: {profile.extrinsics.pitch_degrees:.3f}")
        print(f"  fit residual RMSE against the SAME points used to fit: {solved.residual_rmse_m:.4f} m "
              f"(not a held-out validation -- see below)")

    validation_points = scene.get("validation_points")

    if validation_points:

        print("\n=== VALIDATION against held-out reference points ===")
        held_out = _parse_reference_points(validation_points)
        report = validate_calibration(profile, held_out)
        _report_validation(report)

        quality = CalibrationQuality(
            reference_point_count=report.reference_point_count,
            validated_point_count=report.validated_point_count,
            mean_error_m=report.mean_error_m,
            median_error_m=report.median_error_m,
            max_error_m=report.max_error_m,
            rmse_m=report.rmse_m,
            validation_timestamp=report.validation_timestamp,
        )
        profile = CalibrationProfile(
            camera_id=profile.camera_id, floor_id=profile.floor_id,
            intrinsics=profile.intrinsics, extrinsics=profile.extrinsics, quality=quality,
        )
        print(f"\nCALIBRATION: VALIDATED -- RMSE: {report.rmse_m:.4f} m"
              if report.rmse_m is not None else "\nCALIBRATION: VALIDATION ATTEMPTED BUT NO POINTS PROJECTED")

    else:

        print("\nNo `validation_points` supplied in the scene file -- saving as "
              "CALIBRATION: CONFIGURED -- UNVALIDATED (never fabricated as validated).")

    if validate_only:
        print("\n--validate-only: nothing saved.")
        return

    save_calibration_json(profile, out_path)
    print(f"\nSaved calibration to: {out_path}")


# =====================================================
# Optional lightweight point-picking helper (Phase 3 -- "if a visual
# point-selection tool is useful, keep it lightweight"). A single cv2
# window, click to record a pixel and its on-screen order; no
# annotation UI, no second Designer application.
# =====================================================


def pick_points(image_path: str, points_out: str) -> None:

    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"Could not decode image: {image_path}")

    clicked = []

    def _on_click(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append([x, y])
            cv2.circle(image, (x, y), 4, (0, 200, 0), -1)
            cv2.putText(image, str(len(clicked)), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            cv2.imshow("calibrate_camera_scene -- click floor reference points, press q when done", image)

    cv2.imshow("calibrate_camera_scene -- click floor reference points, press q when done", image)
    cv2.setMouseCallback("calibrate_camera_scene -- click floor reference points, press q when done", _on_click)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    Path(points_out).write_text(json.dumps({"clicked_pixels": clicked}, indent=2), encoding="utf-8")
    print(f"Recorded {len(clicked)} clicked pixel(s) to {points_out} -- pair each, in order, with its "
          f"measured real-world (x, y) to build a scene file's `correspondences` list.")


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", nargs="?", default=None, help="Path to a scene description JSON file.")
    parser.add_argument("--out", default=None, help="Path to write the resulting calibration JSON to.")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Solve/validate and print the report, but do not save a calibration file.",
    )
    parser.add_argument(
        "--pick-points", default=None, metavar="IMAGE",
        help="Open a lightweight point-picking window on IMAGE instead of running calibration.",
    )
    parser.add_argument("--points-out", default="clicked_pixels.json", help="Where --pick-points writes clicked pixel coordinates.")
    args = parser.parse_args()

    if args.pick_points is not None:
        pick_points(args.pick_points, args.points_out)
        return

    if args.scene is None:
        parser.error("a scene JSON file is required unless --pick-points is used")

    if args.out is None and not args.validate_only:
        parser.error("--out is required unless --validate-only is used")

    run_calibration(args.scene, args.out, args.validate_only)


if __name__ == "__main__":
    main()
