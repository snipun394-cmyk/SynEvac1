import math

from dataclasses import dataclass

from models.engineering_asset import EngineeringAsset


@dataclass
class Camera(EngineeringAsset):

    rotation: float = 0.0

    horizontal_fov: float = 90.0

    # Doubles as both "how far the coverage sector is drawn" and
    # "detection range" -- there is only one distance an engineering
    # camera asset needs, and giving it a second name/field here
    # would just be two words for the same number. Reused as-is by
    # GroundTruthCameraProvider (see perception/providers/
    # ground_truth_camera_provider.py) as the camera's effective
    # detection range.
    max_range: float = 25.0

    # Logical-only until a real Live Mode integration exists (see
    # EngineeringAsset.mode / ConnectionInfo) -- never parsed,
    # validated, or used to configure an actual video pipeline.
    resolution: str = "1920x1080"
    fps: int = 15

    # =====================================================

    def __post_init__(self):

        self.object_type = "Camera"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def rotate(self, angle):

        self.rotation = angle

    # =====================================================
    # Derived Coverage Geometry
    #
    # Never stored -- always recomputed from Position, Rotation,
    # Horizontal FOV and Max Range. A sector (fan) polygon: the
    # camera position, followed by points along the arc from
    # -fov/2 to +fov/2 around the facing direction. 0 degrees
    # points along +x, increasing clockwise -- matching Qt's own
    # rotation convention so the model and the graphics item
    # agree without either depending on the other.
    # =====================================================

    def coverage_polygon(self, segments=24):

        cx, cy = self.position

        half_fov = self.horizontal_fov / 2

        points = [(cx, cy)]

        for i in range(segments + 1):

            angle_degrees = (
                self.rotation
                - half_fov
                + (self.horizontal_fov * i / segments)
            )

            angle_radians = math.radians(angle_degrees)

            points.append(
                (
                    cx + self.max_range * math.cos(angle_radians),
                    cy + self.max_range * math.sin(angle_radians),
                )
            )

        return points

    # =====================================================

    def to_dict(self):

        data = self._asset_dict()

        data.update({

            "rotation": self.rotation,

            "horizontal_fov": self.horizontal_fov,

            "max_range": self.max_range,

            "resolution": self.resolution,

            "fps": self.fps,

        })

        return data

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        kwargs = cls._asset_kwargs(data)

        kwargs.update(

            rotation=data.get(
                "rotation",
                0.0,
            ),

            horizontal_fov=data.get(
                "horizontal_fov",
                90.0,
            ),

            max_range=data.get(
                "max_range",
                25.0,
            ),

            resolution=data.get(
                "resolution",
                "1920x1080",
            ),

            fps=data.get(
                "fps",
                15,
            ),
        )

        return cls(**kwargs)
