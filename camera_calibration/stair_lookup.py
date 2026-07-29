from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Optional, Sequence, Tuple


# Observable Stair Perception milestone -- the smallest "world position ->
# traversal asset membership" lookup this milestone needs, kept a separate,
# free-function module rather than folded directly into camera_calibration.
# projection.WorldProjector's own body (Phase 6's own "preserve separation
# of concerns" instruction): WorldProjector calls this, but the matching
# rule itself is independently testable and reusable without constructing a
# WorldProjector at all. Mirrors models.staircase.Staircase.
# contains_world_point()'s own "duck-typed, caller-supplied objects, never
# re-derived from a Building here" convention every other geometry helper
# in this codebase (WorldProjector._lookup_zone, is_near_exit) already
# follows.


@dataclass(frozen=True)
class StairMatch:

    # The ONE honest result shape for a stair spatial lookup. Exactly one
    # of three outcomes, never a fourth silently-invented one:
    #   stair_id set, ambiguous=False   -- exactly one Staircase matched
    #   stair_id None,  ambiguous=False -- no Staircase matched at all
    #   stair_id None,  ambiguous=True  -- more than one Staircase matched;
    #     never arbitrarily resolved to "the first one" (Phase 5's own
    #     explicit "never choose arbitrarily between ambiguous matches"
    #     requirement) -- stays UNRESOLVED for the caller to see.

    stair_id: Optional[str]
    ambiguous: bool = False


def locate_stair(stairs: Sequence[object], floor_id: str, world_position: Tuple[float, float]) -> StairMatch:

    # stairs: any sequence of Staircase-shaped objects exposing `.id` and
    # `.contains_world_point(floor_id, world_position)` (models.staircase.
    # Staircase's own real shape -- duck-typed, same convention
    # WorldProjector._lookup_zone already uses for zone-shaped objects).
    #
    # Handles every case Phase 5 names without special-casing any of them:
    #   - no matching Stair               -> matches == [], StairMatch(None, False)
    #   - exactly one matching Stair      -> StairMatch(stair_id, False)
    #   - overlapping/ambiguous regions   -> StairMatch(None, True)
    #   - missing geometry (region=None)  -> contains_world_point() itself
    #                                        returns False, so that stair
    #                                        never enters `matches` at all
    #   - deleted Stair reference         -> not possible by construction;
    #                                        `stairs` is always whatever the
    #                                        caller's CURRENT Floor/Building
    #                                        state contains this cycle, never
    #                                        a cached/stale id list
    #   - wrong floor                     -> contains_world_point() itself
    #                                        returns False for a floor_id
    #                                        that is neither end of that
    #                                        Staircase (or whose matching
    #                                        side has no region authored)
    #   - boundary points                 -> StairObservableRegion.contains()
    #                                        is inclusive on every edge, same
    #                                        convention models.zone.Zone.
    #                                        contains() already uses

    matches = [stair.id for stair in stairs if stair.contains_world_point(floor_id, world_position)]

    if len(matches) == 0:
        return StairMatch(stair_id=None, ambiguous=False)

    if len(matches) == 1:
        return StairMatch(stair_id=matches[0], ambiguous=False)

    return StairMatch(stair_id=None, ambiguous=True)


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


def covered_stair_ids(stairs_by_floor: Mapping[str, Sequence[object]], calibrated_floor_ids: FrozenSet[str]) -> FrozenSet[str]:

    # Observable Stair Perception milestone, Phase 19 -- the honest,
    # GEOMETRICALLY DERIVED answer to "is Stair S1 actually observable
    # this cycle," never a manually-authored Camera-to-Stair assignment
    # (the prior audit found models.camera.Camera.zone_ids/EngineeringAsset.
    # zone_ids to be cosmetic/non-authoritative for the real live pipeline
    # -- this deliberately does not repeat that mistake for Stair). A
    # stair_id is "covered" this cycle if and only if:
    #   (a) it has an observable region authored for at least one of its
    #       two floor sides (otherwise no detection could ever match it,
    #       regardless of any camera), AND
    #   (b) that same floor has at least one calibrated camera (otherwise
    #       no detection could ever be PRODUCED there this cycle).
    # This does NOT attempt real camera-FOV/frustum coverage geometry
    # (out of scope for this milestone, and not something the existing
    # live calibration path computes for Zones either) -- it is the
    # smallest honest necessary condition: a stair with no region, or a
    # region on a floor with zero calibrated cameras, can categorically
    # never receive a real detection, so it must never be reported
    # OBSERVED (even at zero) -- see stair_perception.facts.
    # compute_stair_occupancy_snapshot(), which turns this into the
    # OBSERVED/UNKNOWN distinction Phase 18 requires.

    covered = set()

    for floor_id, stairs in stairs_by_floor.items():

        if floor_id not in calibrated_floor_ids:
            continue

        for stair in stairs:

            region = stair.observable_region_for_floor(floor_id)

            if region is not None:
                covered.add(stair.id)

    return frozenset(covered)
