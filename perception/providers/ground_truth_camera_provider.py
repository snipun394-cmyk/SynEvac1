from typing import List, Tuple

from models.camera import Camera
from models.zone import Zone

from occupancy.provider import OccupancyProvider

from perception.models.camera_observation import CameraFrameObservation
from perception.providers.camera_provider import CameraProvider


def _point_in_polygon(point: Tuple[float, float], polygon_points: List[Tuple[float, float]]) -> bool:

    # Standard ray-casting point-in-polygon test -- no third-party
    # geometry dependency, and Camera.coverage_polygon() already
    # returns a plain, simple (non-self-intersecting) list of (x, y)
    # points starting at the camera's own position, which is exactly
    # what this algorithm expects. Kept private to this module: the
    # only caller is GroundTruthCameraProvider's own zone-coverage
    # resolution below.

    x, y = point

    if len(polygon_points) < 3:
        return False

    inside = False
    j = len(polygon_points) - 1

    for i in range(len(polygon_points)):

        xi, yi = polygon_points[i]
        xj, yj = polygon_points[j]

        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside

        j = i

    return inside


class GroundTruthCameraProvider(CameraProvider):

    # The Ground Truth adapter for Camera-shaped devices -- answers
    # "what would this camera observe" by intersecting its own
    # engineering coverage_polygon() against the Building's Zone
    # geometry, then reading whatever Ground Truth occupancy already
    # exists for the zones it geometrically covers. This is
    # deliberately NOT Occupancy Estimation: no fusion across multiple
    # cameras, no noise model, no false-negative rate, and no
    # interpretation of what a count "means" -- it only answers what a
    # perfect sensor with this exact footprint would see, given Ground
    # Truth. A future Occupancy Estimation stage is what turns this
    # (and other cameras', and IoT localization's) raw observations
    # into one fused, noise-modeled ObservedOccupancy per node -- that
    # stage does not exist yet, and this class does not anticipate it.
    #
    # Consumes Ground Truth exclusively through OccupancyProvider.
    # snapshot_at(time) -- the same seam AIDecisionEngine/a future
    # Occupancy Estimation stage would use -- never simulator/sandbox
    # internals. Camera/Zone are Building Model (Designer) objects,
    # not simulator runtime state; nothing in this class imports
    # simulator or sandbox.
    #
    # Coverage is resolved at zone granularity, not per-occupant: a
    # zone counts as "covered" when its center point falls inside the
    # camera's coverage_polygon(), and this class reports that whole
    # zone's Ground Truth occupant_count as visible. Sub-zone partial
    # coverage (a camera that only covers half a large Zone) cannot be
    # modeled this way, so a zone is treated as either fully visible
    # or not visible at all.
    #
    # V1 IMPLEMENTATION LIMITATION -- not a design choice, a stopgap:
    # zone-level resolution is used here specifically because the
    # simulator does not yet expose individual occupant positions
    # through any public interface. Ground Truth occupancy (via
    # OccupancyProvider/OccupancySnapshot) is aggregated per Node.id
    # (per Zone) by design (see occupancy/snapshot.py) -- there is no
    # per-occupant position feed this class could read even if it
    # wanted to test individual occupants against coverage geometry
    # directly. This is a limitation of what Ground Truth currently
    # exposes, not of Camera's own coverage_polygon() geometry, which
    # already supports much finer-grained testing than this class
    # currently uses it for.
    #
    # TODO(future version): once per-occupant Ground Truth positions
    # are available through a public interface, replace zone-level
    # aggregation with true per-occupant visibility testing -- check
    # each occupant's individual position against the camera's full
    # configurable field of view (max_range, horizontal_fov,
    # rotation/orientation, and future occlusion support -- e.g. an
    # occupant standing behind a wall or Obstacle within an otherwise
    # covered sector should not count as visible), so a camera reports
    # only occupants actually visible within its coverage, not every
    # occupant anywhere in a zone whose center merely falls inside the
    # coverage polygon. Not implemented here -- V1 intentionally stops
    # at zone-level aggregation.

    def __init__(
        self,
        cameras: List[Camera],
        zones: List[Zone],
        occupancy_provider: OccupancyProvider,
    ):

        self._cameras_by_id = {camera.id: camera for camera in cameras}
        self._zones = list(zones)
        self._occupancy_provider = occupancy_provider

    # =====================================================

    def frame_observation_at(self, camera_id: str, time: float) -> CameraFrameObservation:

        # camera_id not registered with this provider is a caller
        # error (asking about a device this provider was never told
        # about), not an absent-reading situation -- raises, same as
        # a dict lookup would, rather than silently returning an empty
        # observation that could be confused with "camera exists but
        # currently sees nothing".
        camera = self._cameras_by_id[camera_id]

        if not camera.active:
            return CameraFrameObservation(camera_id=camera_id, timestamp=time)

        covered_zone_ids = self._covered_zone_ids(camera)

        if not covered_zone_ids:
            return CameraFrameObservation(camera_id=camera_id, timestamp=time)

        snapshot = self._occupancy_provider.snapshot_at(time)

        known_counts = [
            count
            for count in (
                snapshot.observation_at(zone_id).occupant_count
                for zone_id in covered_zone_ids
            )
            if count is not None
        ]

        if not known_counts:
            # Every covered zone has no Ground Truth reading at all --
            # "no reading available", never "zero people" (same
            # convention OccupancySnapshot/OccupancyObservation
            # already document).
            return CameraFrameObservation(camera_id=camera_id, timestamp=time)

        return CameraFrameObservation(
            camera_id=camera_id,
            timestamp=time,
            estimated_occupant_count=sum(known_counts),
            # A fixed, documented constant, not a computed probability
            # -- this adapter reports Ground Truth occupancy for
            # whatever falls inside its own coverage geometry with
            # certainty; it makes no claim about real-world camera/
            # vision-model accuracy, which belongs to a future,
            # separate sensor-fidelity model.
            confidence=1.0,
        )

    # =====================================================

    def _covered_zone_ids(self, camera: Camera) -> List[str]:

        polygon = camera.coverage_polygon()

        return [
            zone.id
            for zone in self._zones
            if zone.floor_id == camera.floor_id and _point_in_polygon(zone.center, polygon)
        ]
