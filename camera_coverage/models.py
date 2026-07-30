from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import FrozenSet, Mapping, Optional, Tuple
from uuid import uuid4


# =====================================================
# Camera Coverage Intelligence & Observable Asset Mapping milestone --
# cameras become first-class perception devices: not just "a Camera
# Asset exists on floor X", but "Camera C3 observes Stair S1, fully".
#
# This package answers exactly two questions, in both directions:
#   Camera            -> which observable assets does it cover?
#   Observable Asset   -> which cameras cover it?
#
# It is deliberately pure metadata/intelligence -- no occupancy count,
# no congestion score, no safety judgement of any kind lives here (same
# narrow-scope discipline observable_assets.models already establishes
# for occupancy truth; see that module's own docstring). Real occupancy
# stays exactly where the Observable Asset Perception Framework put it:
# observable_assets.models.ObservableAssetSnapshot, asset-id keyed, one
# entry per asset regardless of how many cameras see it. This package
# never duplicates or re-derives that count -- see CameraCoverageSnapshot's
# own docstring on why multi-camera coverage cannot cause double counting
# by construction.
# =====================================================


class CoverageState(Enum):

    # Four honest outcomes for "does this camera cover this observable
    # asset", never a fifth silently-invented one. Determined purely
    # from geometry (this camera's calibrated coverage sector vs. the
    # asset's own authored observable region) -- see
    # camera_coverage.discovery for exactly how each is computed.

    # Every corner of the asset's observable region lies inside this
    # camera's calibrated coverage sector -- the whole asset footprint
    # is within view.
    FULLY_VISIBLE = auto()

    # The asset's observable region and this camera's calibrated
    # coverage sector genuinely overlap, but not completely -- part of
    # the asset footprint is outside the sector (out of frame, out of
    # range, or beyond the FOV edge).
    PARTIALLY_VISIBLE = auto()

    # Both geometries are known (camera is calibrated, asset has an
    # authored observable region on this floor) and they provably do
    # not overlap at all.
    NOT_VISIBLE = auto()

    # No honest geometric basis to say anything at all -- the camera
    # has no calibration profile, its calibration cannot yield a usable
    # coverage sector, or the asset has no observable region authored
    # on the camera's floor. Never FULLY/PARTIALLY/NOT_VISIBLE by
    # default; a consumer must always check `state`, never assume
    # coverage from an entry's mere presence.
    UNKNOWN = auto()


# The two states that honestly mean "this camera contributes to seeing
# this asset, to some degree" -- the single place that distinction is
# made, so no caller re-derives it with a different (state != NOT_VISIBLE
# and state != UNKNOWN) spelling.
DETERMINED_COVERAGE_STATES = (CoverageState.FULLY_VISIBLE, CoverageState.PARTIALLY_VISIBLE)


@dataclass(frozen=True)
class AssetCoverage:

    # One (camera, observable asset) pair's coverage fact -- lives inside
    # a CameraCoverage, keyed by asset_id. `asset_type` is a plain string,
    # same "never a closed enum this package would need editing to
    # extend" convention observable_assets.models.AssetObservation
    # already establishes for the same reason.

    asset_id: str
    asset_type: str

    state: CoverageState = CoverageState.UNKNOWN

    # The asset's own observable-region footprint (its four corners, in
    # this floor's local coordinate space) -- present only when
    # `state` is FULLY_VISIBLE or PARTIALLY_VISIBLE ("coverage geometry
    # if appropriate": there is nothing meaningful to show for
    # NOT_VISIBLE/UNKNOWN, and fabricating a polygon for either would
    # violate this codebase's own "never invent geometry" discipline --
    # see models.staircase.StairObservableRegion's own docstring). This
    # is never new geometry -- it is exactly the asset's own already-
    # authored region, restated as corners for a caller that wants to
    # draw or reason about it without re-deriving from center/width/depth.
    region_polygon: Optional[Tuple[Tuple[float, float], ...]] = None

    # A short, human/diagnostic-oriented note on WHY this state holds --
    # optional, never itself a source of coverage truth. Same "provenance
    # for a person debugging a deployment, not a machine-decision input"
    # discipline as observable_assets.models.AssetObservation.provenance.
    provenance: Optional[str] = None

    @property
    def is_covered(self) -> bool:

        return self.state in DETERMINED_COVERAGE_STATES


@dataclass(frozen=True)
class CameraCoverage:

    # One camera's complete coverage picture: which observable assets it
    # covers, and to what degree, plus the camera's own derived coverage
    # sector (its calibrated FOV wedge, out to its detection range --
    # None whenever no calibration was available to derive one).

    camera_id: str
    floor_id: Optional[str] = None
    timestamp: float = 0.0

    # This camera's own calibrated coverage sector, in floor-plan world
    # coordinates -- the SAME geometry every asset's coverage state below
    # was tested against, exposed once per camera (never repeated per
    # asset entry, since it does not depend on which asset is being
    # asked about). None when this camera has no calibration profile, or
    # its calibration cannot yield a usable sector (see
    # camera_coverage.discovery.compute_camera_coverage()).
    sector_polygon: Optional[Tuple[Tuple[float, float], ...]] = None

    assets: Mapping[str, AssetCoverage] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "assets", MappingProxyType(dict(self.assets)))

    # =====================================================
    # Total accessor -- an asset_id this CameraCoverage never even
    # considered (e.g. an asset on a different floor) honestly defaults
    # to UNKNOWN, same "absent means no reading" convention
    # ObservableAssetSnapshot.observation_for() already establishes.
    # =====================================================

    def coverage_for(self, asset_id: str) -> AssetCoverage:

        return self.assets.get(asset_id, AssetCoverage(asset_id=asset_id, asset_type=""))

    def covered_asset_ids(self) -> FrozenSet[str]:

        return frozenset(
            asset_coverage.asset_id for asset_coverage in self.assets.values() if asset_coverage.is_covered
        )


@dataclass(frozen=True)
class CameraCoverageSnapshot:

    # The building-wide "Camera <-> Observable Asset" picture for one
    # point in time -- every calibrated (and attempted-but-uncalibrated)
    # camera's own CameraCoverage, keyed by camera_id. Mirrors
    # observable_assets.models.ObservableAssetSnapshot's own shape
    # (snapshot_id/timestamp, a Mapping keyed by id, MappingProxyType-
    # wrapped, total accessors that never raise).
    #
    # Multi-camera coverage without duplicate occupancy (Phase 5): this
    # type carries ZERO occupancy counts anywhere in it -- only
    # visibility metadata (which cameras see which assets, and how
    # completely). `cameras_observing()` below can honestly return
    # several camera_ids for the same asset_id with no double-counting
    # risk whatsoever, because there is no count here to double: the
    # one and only occupant count for an asset lives in
    # observable_assets.models.ObservableAssetSnapshot, computed once per
    # asset_id regardless of how many cameras cover it. A future
    # consumer wanting "N cameras confirm M people on Stair S1" reads
    # BOTH snapshots and combines them itself; neither snapshot needs to
    # know about the other's existence.

    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0

    by_camera: Mapping[str, CameraCoverage] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "by_camera", MappingProxyType(dict(self.by_camera)))

    # =====================================================

    def coverage_for_camera(self, camera_id: str) -> CameraCoverage:

        return self.by_camera.get(camera_id, CameraCoverage(camera_id=camera_id))

    # =====================================================
    # The two questions this whole package exists to answer -- see
    # docs/architecture/camera_coverage.md Sec "Command Center / future
    # consumers".
    # =====================================================

    def cameras_observing(
        self, asset_id: str, states: Tuple[CoverageState, ...] = DETERMINED_COVERAGE_STATES,
    ) -> Tuple[str, ...]:

        # "Which cameras observe Stair S1?" -- every camera_id whose own
        # coverage of this asset_id falls in `states` (FULLY_VISIBLE or
        # PARTIALLY_VISIBLE by default; a caller wanting only fully-
        # visible cameras passes states=(CoverageState.FULLY_VISIBLE,)).
        # Sorted for a stable, deterministic return order.

        return tuple(sorted(
            camera_id for camera_id, coverage in self.by_camera.items()
            if coverage.coverage_for(asset_id).state in states
        ))

    def assets_observed_by(
        self, camera_id: str, states: Tuple[CoverageState, ...] = DETERMINED_COVERAGE_STATES,
    ) -> Tuple[str, ...]:

        # "Which observable assets does Camera C3 cover?" -- the mirror
        # image of cameras_observing() above.

        coverage = self.coverage_for_camera(camera_id)

        return tuple(sorted(
            asset_coverage.asset_id for asset_coverage in coverage.assets.values()
            if asset_coverage.state in states
        ))
