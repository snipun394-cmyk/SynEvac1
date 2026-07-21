import json
from pathlib import Path
from typing import Union

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics


# Camera Calibration & World Coordinate Projection milestone, Phase 5 --
# supports MANUAL calibration entry (direct construction, already
# possible via camera_model.py's own dataclasses), JSON (this module),
# and a documented extension point for a FUTURE OpenCV-based routine
# (cv2.calibrateCamera()/solvePnP(), producing a real camera matrix +
# distortion coefficients + rotation/translation vectors) -- explicitly
# NOT implemented here (Phase 5's own "do NOT implement automatic
# calibration" instruction). A future loader would convert an OpenCV
# camera matrix (fx, fy, cx, cy) and a solvePnP rotation/translation
# result directly into a CameraIntrinsics/CameraExtrinsics pair --
# both already shaped to receive that data (independent fx/fy,
# explicit principal point, explicit position/mount_height/yaw/pitch/
# roll) without any change to this milestone's own types.


class CalibrationLoadError(Exception):

    # Raised for malformed/incomplete calibration data -- an honest,
    # immediate failure, never a silently-incomplete CalibrationProfile.

    pass


def calibration_from_camera(
    camera,
    pitch_degrees: float = 0.0,
    roll_degrees: float = 0.0,
    image_width: int = 1920,
    image_height: int = 1080,
) -> CalibrationProfile:

    # The manual-entry convenience path Phase 5 names first: reuses an
    # EXISTING models.camera.Camera's own position/mount_height/
    # rotation/horizontal_fov/floor_id/id exactly as-is (never
    # duplicating or re-entering them) -- a caller only ever supplies
    # the two genuinely new angles this milestone introduces
    # (pitch/roll, since Camera has no field for either) and, if the
    # Camera's own `resolution` string is not a real, usable pixel size
    # (it is display-only and never validated -- see models/camera.py),
    # an explicit image_width/image_height for the intrinsics this
    # calibration needs.

    intrinsics = CameraIntrinsics.from_horizontal_fov(
        image_width=image_width, image_height=image_height, horizontal_fov_degrees=camera.horizontal_fov,
    )

    extrinsics = CameraExtrinsics(
        position=tuple(camera.position), mount_height=camera.mount_height,
        yaw_degrees=camera.rotation, pitch_degrees=pitch_degrees, roll_degrees=roll_degrees,
    )

    return CalibrationProfile(camera_id=camera.id, floor_id=camera.floor_id, intrinsics=intrinsics, extrinsics=extrinsics)


# =====================================================
# Dict / JSON (Phase 5)
# =====================================================


def calibration_from_dict(data: dict) -> CalibrationProfile:

    try:

        camera_id = data["camera_id"]
        floor_id = data["floor_id"]

        intrinsics_data = data["intrinsics"]
        extrinsics_data = data["extrinsics"]

        if "horizontal_fov_degrees" in intrinsics_data:

            intrinsics = CameraIntrinsics.from_horizontal_fov(
                image_width=intrinsics_data["image_width"],
                image_height=intrinsics_data["image_height"],
                horizontal_fov_degrees=intrinsics_data["horizontal_fov_degrees"],
            )

        else:

            intrinsics = CameraIntrinsics(
                image_width=intrinsics_data["image_width"],
                image_height=intrinsics_data["image_height"],
                focal_length_x=intrinsics_data["focal_length_x"],
                focal_length_y=intrinsics_data["focal_length_y"],
                principal_point_x=intrinsics_data.get("principal_point_x"),
                principal_point_y=intrinsics_data.get("principal_point_y"),
            )

        extrinsics = CameraExtrinsics(
            position=tuple(extrinsics_data["position"]),
            mount_height=extrinsics_data["mount_height"],
            yaw_degrees=extrinsics_data["yaw_degrees"],
            pitch_degrees=extrinsics_data.get("pitch_degrees", 0.0),
            roll_degrees=extrinsics_data.get("roll_degrees", 0.0),
        )

    except KeyError as exc:

        raise CalibrationLoadError(f"Missing required calibration field: {exc}") from exc

    return CalibrationProfile(camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics)


def calibration_to_dict(profile: CalibrationProfile) -> dict:

    return {
        "camera_id": profile.camera_id,
        "floor_id": profile.floor_id,
        "intrinsics": {
            "image_width": profile.intrinsics.image_width,
            "image_height": profile.intrinsics.image_height,
            "focal_length_x": profile.intrinsics.focal_length_x,
            "focal_length_y": profile.intrinsics.focal_length_y,
            "principal_point_x": profile.intrinsics.principal_point_x,
            "principal_point_y": profile.intrinsics.principal_point_y,
        },
        "extrinsics": {
            "position": list(profile.extrinsics.position),
            "mount_height": profile.extrinsics.mount_height,
            "yaw_degrees": profile.extrinsics.yaw_degrees,
            "pitch_degrees": profile.extrinsics.pitch_degrees,
            "roll_degrees": profile.extrinsics.roll_degrees,
        },
    }


def load_calibration_json(path: Union[str, Path]) -> CalibrationProfile:

    json_path = Path(path)

    if not json_path.exists():
        raise CalibrationLoadError(f"Calibration file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:

        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise CalibrationLoadError(f"Malformed calibration JSON at {json_path}: {exc}") from exc

    return calibration_from_dict(data)


def save_calibration_json(profile: CalibrationProfile, path: Union[str, Path]) -> None:

    json_path = Path(path)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(calibration_to_dict(profile), handle, indent=2)
