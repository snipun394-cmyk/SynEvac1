from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

from visibility.geometry import point_in_polygon

from camera_calibration.asset_lookup import locate_asset
from camera_calibration.camera_model import CalibrationProfile, WORLD_POSITION_PROVENANCE_NONE, calibration_provenance
from camera_calibration.geometry import intersect_ray_with_floor, pixel_ray_direction


DEFAULT_GRAZING_ANGLE_RATIO = 0.15  # minimum |direction.z| / |direction| for full-confidence projection


@dataclass(frozen=True)
class WorldProjection:

    # Camera Calibration & World Coordinate Projection milestone,
    # Phase 4/7 -- one bounding box's complete projection result, as of
    # one project() call. Every field is Optional and NEVER fabricated
    # -- a missing calibration, a geometrically-impossible ray (see
    # camera_calibration.geometry.intersect_ray_with_floor), or a point
    # that lands outside every known Zone all honestly produce None for
    # the fields that genuinely cannot be determined, never a guessed
    # stand-in.

    world_position: Optional[Tuple[float, float]]
    floor_id: Optional[str]
    zone_id: Optional[str]
    projection_confidence: Optional[float]

    # Observable Asset Perception Framework milestone -- the GENERIC
    # spatial-asset match, alongside zone_id, never instead of it and
    # never merged into it (the prior audit's own explicit "do NOT place
    # Stair IDs into zone IDs" requirement, generalized: no observable
    # asset id ever gets merged into zone_id either). A world point can
    # honestly match a Zone, an observable asset, both (overlapping
    # authored geometry -- e.g. a "Stair Lobby" Zone drawn over part of a
    # Staircase's landing region), or neither; each field is resolved
    # independently from the SAME already-computed world_position/
    # floor_id, never a second projection. asset_id/asset_type are None
    # whenever no registered asset's observable region contains this
    # point (the overwhelmingly common case). asset_localization_ambiguous
    # is True only when more than one asset's region matched -- asset_id/
    # asset_type stay None in that case too (never an arbitrary pick), so
    # a caller can distinguish "definitely not on any observable asset"
    # from "matched more than one, genuinely unresolved."
    asset_id: Optional[str] = None
    asset_type: Optional[str] = None
    asset_localization_ambiguous: bool = False

    # Observable Stair Perception milestone -- kept as a THIN, backward-
    # compatible convenience view over asset_id/asset_type above, never a
    # second independent computation: stair_id is simply asset_id
    # whenever asset_type == "Stair", None otherwise (see project()
    # below). Existing callers/tests written against this field before
    # the Observable Asset Perception Framework milestone continue to
    # work completely unchanged. A future Door/Exit/etc. integration
    # reads asset_id/asset_type directly instead -- no analogous
    # `door_id` field is added here, and none should be; that would be
    # exactly the per-type field duplication this milestone exists to
    # prevent.
    stair_id: Optional[str] = None
    stair_localization_ambiguous: bool = False

    # CCTV Connection & Calibration Readiness milestone, Phase 7/8 --
    # additive, always set (never left to a default -- every return
    # path in project() below supplies it explicitly). One of
    # camera_calibration.camera_model.WORLD_POSITION_PROVENANCE_{NONE,
    # UNVALIDATED,VALIDATED}. This is what lets a downstream consumer
    # (LiveOccupant, Command Center) tell "no honest basis to project
    # at all" apart from "projected using a calibration nobody has ever
    # checked against a measured point" apart from "projected using a
    # calibration with a genuinely measured RMSE" -- three previously-
    # indistinguishable cases that produced identical-looking
    # world_position tuples before this milestone.
    provenance: str = WORLD_POSITION_PROVENANCE_NONE


class WorldProjector:

    # The Phase 4 pipeline in one class: bounding box -> ground contact
    # point -> world coordinate -> floor coordinate -> zone lookup.
    # Stateless/per-call -- holds only its own configuration
    # (calibrations, zones), never any per-track history (that remains
    # entirely behavior_recognition's job -- Phase 11's "Projection
    # depends only on geometry, camera calibration, building geometry"
    # deliberately excludes time/tracking).

    def __init__(
        self,
        calibrations: Mapping[str, CalibrationProfile],
        zones_by_floor: Mapping[str, Sequence[object]],
        grazing_angle_ratio: float = DEFAULT_GRAZING_ANGLE_RATIO,
        stairs_by_floor: Optional[Mapping[str, Sequence[object]]] = None,
        assets_by_floor: Optional[Mapping[str, Sequence[Tuple[str, object]]]] = None,
    ):

        # zones_by_floor: floor_id -> a sequence of zone-shaped objects
        # (models.zone.Zone, or anything exposing the same `.id`,
        # `.floor_id`, `.contains(x, y)`, `.polygon` shape -- "building
        # geometry", Phase 11's own allowed dependency). Never mutated,
        # never re-derived from a Building here -- a caller supplies
        # whatever it already has.

        # Observable Asset Perception Framework milestone -- the
        # GENERIC spatial-asset registry this projector consults, floor_id
        # -> a sequence of (asset_type, asset-shaped-object) pairs (see
        # camera_calibration.asset_lookup.build_assets_by_floor(), which
        # produces exactly this shape from a Building + a sequence of
        # registered ObservableAssetKind). `stairs_by_floor` (Observable
        # Stair Perception milestone) is kept as a backward-compatible
        # convenience alias for the common single-kind case -- passing it
        # is exactly equivalent to passing `assets_by_floor` pre-tagged
        # "Stair", and an existing caller that only ever knew about
        # stairs_by_floor keeps working completely unchanged. If BOTH are
        # supplied they are merged (stairs_by_floor's entries added
        # alongside assets_by_floor's own). Neither is required -- an
        # existing caller that supplies neither keeps this projector's
        # exact pre-Observable-Stair-Perception-milestone behavior (every
        # asset/stair field simply stays at its honest None/False
        # default). Zone lookup and asset lookup are deliberately two
        # independent dicts/lookups reusing the SAME zone-lookup pattern
        # (Phase 6 of the prior milestone's own "do not duplicate
        # projection math" -- both consult the SAME already-computed
        # world_position/floor_id, neither re-projects anything) -- Zone
        # and observable-asset identity are never conflated into one
        # dict.

        self._calibrations = dict(calibrations)
        self._zones_by_floor = {floor_id: tuple(zones) for floor_id, zones in zones_by_floor.items()}

        merged_assets_by_floor: Dict[str, list] = {}

        for floor_id, assets in (assets_by_floor or {}).items():
            merged_assets_by_floor.setdefault(floor_id, []).extend(assets)

        for floor_id, stairs in (stairs_by_floor or {}).items():
            merged_assets_by_floor.setdefault(floor_id, []).extend(("Stair", stair) for stair in stairs)

        self._assets_by_floor = {
            floor_id: tuple(assets) for floor_id, assets in merged_assets_by_floor.items()
        }

        self.grazing_angle_ratio = grazing_angle_ratio

    # =====================================================

    def project(
        self,
        camera_id: str,
        bounding_box: Optional[Tuple[float, float, float, float]],
        detection_confidence: float,
    ) -> WorldProjection:

        calibration = self._calibrations.get(camera_id)
        provenance = calibration_provenance(calibration)

        if calibration is None or bounding_box is None:
            # No honest basis to project at all -- an uncalibrated
            # camera, or a detection with no geometry (Phase 7's own
            # "never fabricate values").
            return WorldProjection(
                world_position=None, floor_id=None, zone_id=None, projection_confidence=None,
                provenance=WORLD_POSITION_PROVENANCE_NONE,
            )

        ground_pixel = self._ground_contact_point(bounding_box)

        direction = pixel_ray_direction(
            ground_pixel[0], ground_pixel[1], calibration.intrinsics, calibration.extrinsics,
        )
        origin = (
            calibration.extrinsics.position[0],
            calibration.extrinsics.position[1],
            calibration.extrinsics.mount_height,
        )

        world_position = intersect_ray_with_floor(origin, direction)

        if world_position is None:
            # Geometrically impossible (e.g. an implausible pitch/box
            # combination that never reaches the floor) -- still
            # honestly reports which floor this camera itself belongs
            # to (that much is certain regardless of the ray), but no
            # position/zone/confidence.
            return WorldProjection(
                world_position=None, floor_id=calibration.floor_id, zone_id=None, projection_confidence=None,
                provenance=provenance,
            )

        zone_id = self._lookup_zone(calibration.floor_id, world_position)
        asset_match = self._lookup_asset(calibration.floor_id, world_position)
        confidence = self._projection_confidence(direction, detection_confidence)

        # stair_id/stair_localization_ambiguous are a thin, backward-
        # compatible view over the generic match -- see WorldProjection's
        # own docstring.
        stair_id = asset_match.asset_id if asset_match.asset_type == "Stair" else None

        return WorldProjection(
            world_position=world_position, floor_id=calibration.floor_id,
            zone_id=zone_id, projection_confidence=confidence,
            asset_id=asset_match.asset_id, asset_type=asset_match.asset_type,
            asset_localization_ambiguous=asset_match.ambiguous,
            stair_id=stair_id, stair_localization_ambiguous=asset_match.ambiguous,
            provenance=provenance,
        )

    # =====================================================

    def _ground_contact_point(self, bounding_box: Tuple[float, float, float, float]) -> Tuple[float, float]:

        # "Assume feet touch the floor" (Phase 4's own explicit
        # instruction, and explicitly NOT a 3D skeleton/pose estimate):
        # the bottom-center of the bounding box, in the same
        # (x1, y1, x2, y2) image-pixel-space convention human_detection.
        # yolo_backend.BoundingBoxDetection already establishes (y
        # increasing downward, y2 = the bottom edge).

        x1, y1, x2, y2 = bounding_box

        return ((x1 + x2) / 2.0, y2)

    # =====================================================

    def _lookup_zone(self, floor_id: str, world_position: Tuple[float, float]) -> Optional[str]:

        zones = self._zones_by_floor.get(floor_id, ())

        x, y = world_position

        for zone in zones:

            if zone.polygon:

                if point_in_polygon((x, y), zone.polygon):
                    return zone.id

            elif zone.contains(x, y):

                return zone.id

        return None

    # =====================================================

    def _lookup_asset(self, floor_id: str, world_position: Tuple[float, float]):

        candidates = self._assets_by_floor.get(floor_id, ())

        return locate_asset(candidates, floor_id, world_position)

    # =====================================================
    # Observable Asset Perception Framework milestone -- which floors
    # this projector actually has a calibrated camera on this cycle,
    # straight from the calibrations it was constructed with (never a
    # second, independent camera query) -- the necessary-condition input
    # camera_calibration.asset_lookup.covered_asset_ids() needs to
    # distinguish OBSERVED (possibly zero) from UNKNOWN asset occupancy.
    # =====================================================

    def calibrated_floor_ids(self) -> FrozenSet[str]:

        return frozenset(calibration.floor_id for calibration in self._calibrations.values())

    # =====================================================

    def _projection_confidence(self, direction, detection_confidence: float) -> Optional[float]:

        # A ray nearly parallel to the floor (direction.z close to 0)
        # is numerically unstable: a tiny pixel error near the horizon
        # translates into a huge world-position error, a well-known
        # real limitation of single-camera ground-plane projection --
        # this is honestly reflected as reduced confidence, never
        # hidden. Scales detection_confidence down as the ray
        # approaches horizontal, reaching full confidence once the
        # ray's own downward steepness clears grazing_angle_ratio.

        magnitude = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5

        if magnitude == 0.0:
            return None

        steepness = abs(direction[2]) / magnitude

        scale = min(1.0, steepness / self.grazing_angle_ratio) if self.grazing_angle_ratio > 0 else 1.0

        return detection_confidence * scale


def nearest_navigation_node(navigation_graph, floor_id: str, world_position: Tuple[float, float]) -> Optional[str]:

    # Phase 4's own motivating question -- "which navigation node are
    # they closest to" -- answered as a light, standalone utility
    # (never a Detection field; Phase 7 names world_position/zone_id/
    # floor_id/world_velocity/projection_confidence as the fields to
    # add, not this). Duck-typed against navigation.graph.NavigationGraph's
    # own real shape (`.nodes`, a dict of Node objects each exposing
    # `.floor_id`/`.node_type`/`.reference`) without importing that
    # package at module scope -- "building geometry" (Phase 11's own
    # allowed dependency), consulted only when a caller actually passes
    # a real graph.
    #
    # Only Zone-type nodes are considered (a Node.ZONE's own `.reference`
    # is a models.zone.Zone, whose `.center` this function reuses
    # directly) -- Outside/AssemblyPoint nodes have no comparable
    # "occupiable position" to measure distance to in the same sense.

    best_node_id = None
    best_distance = None

    x, y = world_position

    for node in navigation_graph.nodes.values():

        if node.floor_id != floor_id:
            continue

        if node.node_type != "Zone":
            continue

        zone = node.reference

        if zone is None:
            continue

        zx, zy = zone.center
        distance = ((zx - x) ** 2 + (zy - y) ** 2) ** 0.5

        if best_distance is None or distance < best_distance or (distance == best_distance and node.id < best_node_id):
            best_distance = distance
            best_node_id = node.id

    return best_node_id
