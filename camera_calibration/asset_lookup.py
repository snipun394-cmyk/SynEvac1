from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple


# =====================================================
# Observable Asset Perception Framework milestone -- the generic "world
# point -> observable engineering asset" lookup engine. Generalized
# directly from the Observable Stair Perception milestone's own
# camera_calibration.stair_lookup.StairMatch/locate_stair/covered_stair_ids,
# which turned out to have ZERO Stair-specific logic in their actual
# bodies -- they only ever duck-typed against `.id` and
# `.observable_region_for_floor()`/`.contains_world_point()`, exactly the
# same contract any future observable asset type (Door, Exit, Assembly
# Point, Refuge Area, Lift Lobby, Escalator, Corridor, Fire Door, Hose
# Reel Area, Hydrant Area, ...) would also need to satisfy. This module
# is that logic, generalized and given honest names; camera_calibration.
# stair_lookup.py now holds only what is GENUINELY Stair-specific (how to
# find every Staircase in a Building) and registers Stair as this
# framework's first concrete kind.
#
# This module itself knows NOTHING about Stair, Door, or any other
# concrete asset type -- it is pure framework, imported by
# camera_calibration.stair_lookup (never the other way around).
# =====================================================


@dataclass(frozen=True)
class AssetMatch:

    # The ONE honest result shape for an observable-asset spatial lookup.
    # Exactly one of three outcomes, never a fourth silently-invented one:
    #   asset_id set, ambiguous=False   -- exactly one asset matched
    #   asset_id None,  ambiguous=False -- no asset matched at all
    #   asset_id None,  ambiguous=True  -- more than one asset matched;
    #     never arbitrarily resolved to "the first one" (the prior
    #     audit's own explicit "never choose arbitrarily between
    #     ambiguous matches" requirement) -- stays UNRESOLVED for the
    #     caller to see. asset_type is None in this case too, for the
    #     same reason.

    asset_id: Optional[str]
    asset_type: Optional[str] = None
    ambiguous: bool = False


def locate_asset(
    candidates: Sequence[Tuple[str, object]], floor_id: str, world_position: Tuple[float, float],
) -> AssetMatch:

    # candidates: a sequence of (asset_type, asset-shaped-object) pairs
    # -- each object exposing `.id` and `.contains_world_point(floor_id,
    # world_position)` (models.staircase.Staircase's own real shape
    # today -- duck-typed, same convention camera_calibration.projection.
    # WorldProjector._lookup_zone already uses for zone-shaped objects).
    # Matching is deliberately type-agnostic: a point ambiguously
    # matching a Stair AND a (hypothetical, future) Door is exactly as
    # unresolved as matching two Stairs -- asset TYPE is only ever a
    # label on an already-resolved match, never part of the matching
    # rule itself.
    #
    # Handles every case the prior audit named without special-casing
    # any of them:
    #   - no matching asset               -> matches == [], AssetMatch(None, None, False)
    #   - exactly one matching asset      -> AssetMatch(asset_id, asset_type, False)
    #   - overlapping/ambiguous regions   -> AssetMatch(None, None, True)
    #   - missing geometry (region=None)  -> contains_world_point() itself
    #                                        returns False, so that asset
    #                                        never enters `matches` at all
    #   - deleted asset reference         -> not possible by construction;
    #                                        `candidates` is always whatever
    #                                        the caller's CURRENT Floor/
    #                                        Building state contains this
    #                                        cycle, never a cached/stale list
    #   - wrong floor                     -> contains_world_point() itself
    #                                        returns False for a floor_id
    #                                        that is neither end of that
    #                                        asset (or whose matching side
    #                                        has no region authored)
    #   - boundary points                 -> the underlying region's own
    #                                        contains() is inclusive on
    #                                        every edge (models.staircase.
    #                                        StairObservableRegion.contains(),
    #                                        models.zone.Zone.contains())

    matches = [
        (asset_type, asset.id) for asset_type, asset in candidates
        if asset.contains_world_point(floor_id, world_position)
    ]

    if len(matches) == 0:
        return AssetMatch(asset_id=None, asset_type=None, ambiguous=False)

    if len(matches) == 1:
        asset_type, asset_id = matches[0]
        return AssetMatch(asset_id=asset_id, asset_type=asset_type, ambiguous=False)

    return AssetMatch(asset_id=None, asset_type=None, ambiguous=True)


@dataclass(frozen=True)
class ObservableAssetKind:

    # A REGISTRATION record for one concrete observable asset type --
    # the generic framework's only notion of "a kind of asset exists" is
    # this: a type label, plus a callable that knows how to find every
    # instance of that type per floor in a Building. Registering a
    # future Door kind requires nothing more than one more instance of
    # this dataclass (see docs/architecture/observable_asset_perception.md
    # Sec 8) -- no change to locate_asset(), build_assets_by_floor(), or
    # covered_asset_ids() below.

    asset_type: str
    build_by_floor: Callable[[object], Mapping[str, Sequence[object]]]


def build_assets_by_floor(
    building, kinds: Sequence[ObservableAssetKind],
) -> Dict[str, Tuple[Tuple[str, object], ...]]:

    # Merges every registered kind's own per-floor mapping into ONE
    # combined per-floor candidate list, each entry tagged with its
    # asset_type -- the shape locate_asset()/camera_calibration.
    # projection.WorldProjector both consume. Iterates `kinds` in the
    # order supplied, calling each kind's own build_by_floor(building)
    # exactly once -- never re-deriving asset geometry itself, and never
    # assuming anything about what a kind's own builder actually reads
    # from Building (camera_calibration.stair_lookup.build_stairs_by_floor()
    # reads floor.stairs; a future Door kind's builder would read
    # floor.doors instead -- this function does not need to know either).

    by_floor: Dict[str, list] = {}

    for kind in kinds:

        for floor_id, assets in kind.build_by_floor(building).items():

            by_floor.setdefault(floor_id, [])

            for asset in assets:
                by_floor[floor_id].append((kind.asset_type, asset))

    return {floor_id: tuple(assets) for floor_id, assets in by_floor.items()}


def covered_asset_ids(
    assets_by_floor: Mapping[str, Sequence[Tuple[str, object]]], calibrated_floor_ids: FrozenSet[str],
) -> FrozenSet[str]:

    # The honest, GEOMETRICALLY DERIVED answer to "is asset X actually
    # observable this cycle," never a manually-authored Camera-to-asset
    # assignment (the prior audit found models.camera.Camera.zone_ids/
    # EngineeringAsset.zone_ids to be cosmetic/non-authoritative for the
    # real live pipeline -- this deliberately does not repeat that
    # mistake for any observable asset type). An asset_id is "covered"
    # this cycle if and only if:
    #   (a) it has an observable region authored for at least one of its
    #       floor sides (otherwise no detection could ever match it,
    #       regardless of any camera), AND
    #   (b) that same floor has at least one calibrated camera (otherwise
    #       no detection could ever be PRODUCED there this cycle).
    # This does NOT attempt real camera-FOV/frustum coverage geometry
    # (out of scope, and not something the existing live calibration
    # path computes for Zones either) -- it is the smallest honest
    # necessary condition: an asset with no region, or a region on a
    # floor with zero calibrated cameras, can categorically never
    # receive a real detection, so it must never be reported OBSERVED
    # (even at zero) -- see observable_assets.facts.
    # compute_asset_occupancy_snapshot(), which turns this into the
    # OBSERVED/UNKNOWN distinction the prior audit's Phase 18 requires.
    #
    # Duck-typed against `.observable_region_for_floor(floor_id)` --
    # ANY asset-shaped object exposing that method works here, not only
    # Staircase (models.staircase.Staircase is simply the first, and
    # today only, real implementation of that contract).

    covered = set()

    for floor_id, assets in assets_by_floor.items():

        if floor_id not in calibrated_floor_ids:
            continue

        for _asset_type, asset in assets:

            region = asset.observable_region_for_floor(floor_id)

            if region is not None:
                covered.add(asset.id)

    return frozenset(covered)
