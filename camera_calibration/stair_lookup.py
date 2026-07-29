from typing import Dict, Tuple

from camera_calibration.asset_lookup import ObservableAssetKind


# =====================================================
# Observable Asset Perception Framework milestone -- this module used to
# hold the ENTIRE spatial-lookup mechanism for Stair (StairMatch,
# locate_stair, covered_stair_ids). A Phase 1 audit of that code found
# none of it was actually Stair-specific -- every line duck-typed against
# `.id`/`.observable_region_for_floor()`/`.contains_world_point()`,
# never against anything a Staircase alone has (from_floor_id,
# to_floor_id, travel_distance, ...). That logic has moved, unchanged in
# behavior, to the generic camera_calibration.asset_lookup module
# (locate_asset/covered_asset_ids) so a future asset type reuses it
# directly instead of copy-pasting a Door-named twin.
#
# What remains HERE is the one genuinely Stair-specific piece:
# build_stairs_by_floor() knows that a Staircase lives in Floor.stairs
# and spans from_floor_id/to_floor_id -- a future Door kind would need
# its own, differently-shaped builder reading Floor.doors instead, which
# is exactly why this stays a per-kind adapter rather than framework
# code. STAIR_ASSET_KIND is the registration record that plugs this
# adapter into the generic framework; DEFAULT_OBSERVABLE_ASSET_KINDS is
# the (currently Stair-only) list a caller passes to camera_calibration.
# asset_lookup.build_assets_by_floor().
# =====================================================


def build_stairs_by_floor(building) -> Dict[str, Tuple[object, ...]]:

    # A Staircase is relevant to spatial lookup on BOTH floors it
    # connects (its from_observable_region belongs to from_floor_id, its
    # to_observable_region belongs to to_floor_id) -- Staircase.
    # contains_world_point() itself picks the correct side once the
    # caller has already narrowed to "which stairs are even worth
    # checking on this floor," which is exactly what this helper
    # precomputes. Mirrors the exact `zones_by_floor[floor.id] =
    # list(floor.zones)` construction convention scripts.
    # run_physical_camera_validation.py already establishes for zones --
    # a caller-side helper, never something WorldProjector re-derives
    # from a Building itself.

    by_floor: Dict[str, list] = {}

    for floor in building.ordered_floors():

        for stair in floor.stairs:

            by_floor.setdefault(stair.from_floor_id, []).append(stair)

            if stair.to_floor_id and stair.to_floor_id != stair.from_floor_id:
                by_floor.setdefault(stair.to_floor_id, []).append(stair)

    return {floor_id: tuple(stairs) for floor_id, stairs in by_floor.items()}


# The framework's first (and, as of this milestone, only) registrant --
# see camera_calibration.asset_lookup.ObservableAssetKind's own docstring
# for what registering a future kind (Door, Exit, ...) would look like.
STAIR_ASSET_KIND = ObservableAssetKind(asset_type="Stair", build_by_floor=build_stairs_by_floor)

DEFAULT_OBSERVABLE_ASSET_KINDS = (STAIR_ASSET_KIND,)
