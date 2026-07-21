from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from live_occupants.occupant import LiveOccupant


Point = Tuple[float, float]


# =====================================================
# Phase 5 -- crowd movement/flow, and specifically "is this occupant
# approaching this Door/Exit/Stair". Pure geometry + time, exactly the
# same "depends only on geometry and time" discipline behavior_recognition.
# metrics.compute_world_metrics() already documents for itself -- no AI,
# no camera/tracking package imported here at all.
# =====================================================


def _distance_point_to_segment(point: Point, seg_start: Point, seg_end: Point) -> float:

    # Identical formula to live_occupants.lifecycle._distance_point_to_
    # segment -- duplicated rather than imported, since that is a private,
    # module-local helper of a different package never meant to be
    # imported across a package boundary (this package's own dependency
    # surface stays minimal, matching live_occupants' own "depends only
    # on Geometry" precedent one layer over).

    px, py = point
    ax, ay = seg_start
    bx, by = seg_end

    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy

    if length_squared == 0.0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5

    t = ((px - ax) * dx + (py - ay) * dy) / length_squared
    t = max(0.0, min(1.0, t))

    closest_x = ax + t * dx
    closest_y = ay + t * dy

    return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5


@dataclass(frozen=True)
class AssetSide:

    # One traversable "face" of an asset, on ONE floor. A Door/Exit has
    # exactly one side (its own start_point/end_point, a real segment,
    # on floor_id). A Staircase has TWO sides -- Sec 8/Phase 15's own
    # "multi-floor stair scenario" -- each a degenerate (zero-length)
    # segment at from_position/to_position on from_floor_id/to_floor_id
    # respectively (_distance_point_to_segment already handles a
    # zero-length segment correctly, as plain point distance).

    floor_id: str
    segment_start: Point
    segment_end: Point


def door_sides(door) -> Tuple[AssetSide, ...]:

    return (AssetSide(floor_id=door.floor_id, segment_start=tuple(door.start_point), segment_end=tuple(door.end_point)),)


def exit_sides(exit_obj) -> Tuple[AssetSide, ...]:

    return (AssetSide(floor_id=exit_obj.floor_id, segment_start=tuple(exit_obj.start_point), segment_end=tuple(exit_obj.end_point)),)


def stair_sides(stair) -> Tuple[AssetSide, ...]:

    return (
        AssetSide(floor_id=stair.from_floor_id, segment_start=tuple(stair.from_position), segment_end=tuple(stair.from_position)),
        AssetSide(floor_id=stair.to_floor_id, segment_start=tuple(stair.to_position), segment_end=tuple(stair.to_position)),
    )


# =====================================================


def distance_to_asset(world_position: Point, floor_id: Optional[str], sides: Sequence[AssetSide]) -> Optional[float]:

    # None whenever this occupant is not even on a floor this asset has
    # a side on -- never a fabricated cross-floor distance.

    matching = [side for side in sides if side.floor_id == floor_id]

    if not matching:
        return None

    return min(_distance_point_to_segment(world_position, side.segment_start, side.segment_end) for side in matching)


# =====================================================


@dataclass(frozen=True)
class ApproachEvidence:

    # Phase 5's own required evidence shape -- `is_approaching` is True
    # only when real, deterministic evidence supports it (distance
    # decreasing across the occupant's own last two recorded position
    # samples -- Phase 5's own "distance decreasing over time" criterion).
    # `approach_speed` is the closing speed implied by that same pair of
    # samples (positive = closing in, may be negative -- never clamped),
    # None when there are not yet two samples to compare.
    #
    # Only the occupant's own MOST RECENT two position samples are used
    # (never the full history window) -- a deliberate simplification: a
    # LiveOccupant's position_samples carry no per-sample floor_id, so
    # comparing across a floor TRANSITION (a discrete stair-crossing
    # event, not a lingering approach) could otherwise mix two different
    # floors' coordinate spaces. Restricting to the last two samples
    # means this is wrong for at most the single cycle a floor change
    # itself occurs in, and self-corrects immediately afterward.

    is_approaching: bool
    approach_speed: Optional[float]
    current_distance: float


def evaluate_approach(
    current_world_position: Point,
    current_floor_id: Optional[str],
    position_samples,
    sides: Sequence[AssetSide],
) -> Optional[ApproachEvidence]:

    current_distance = distance_to_asset(current_world_position, current_floor_id, sides)

    if current_distance is None:
        return None

    if len(position_samples) < 2:
        return ApproachEvidence(is_approaching=False, approach_speed=None, current_distance=current_distance)

    previous_sample = position_samples[-2]
    previous_distance = distance_to_asset(previous_sample.world_position, current_floor_id, sides)

    if previous_distance is None:
        return ApproachEvidence(is_approaching=False, approach_speed=None, current_distance=current_distance)

    last_sample = position_samples[-1]
    dt = last_sample.timestamp - previous_sample.timestamp

    approach_speed = ((previous_distance - current_distance) / dt) if dt > 0 else None

    is_approaching = current_distance < previous_distance

    return ApproachEvidence(is_approaching=is_approaching, approach_speed=approach_speed, current_distance=current_distance)


# =====================================================


def occupants_near_asset(
    occupants: Sequence[LiveOccupant], sides: Sequence[AssetSide], approach_region_depth: float,
) -> List[Tuple[LiveOccupant, float]]:

    # "Find occupants inside region" (Phase 7) -- only occupants with a
    # KNOWN world_position on a floor this asset has a side on are ever
    # candidates; an occupant with no usable position is honestly
    # excluded, never assigned an arbitrary coordinate (Phase 13).

    nearby = []

    for occupant in occupants:

        if occupant.world_position is None:
            continue

        distance = distance_to_asset(occupant.world_position, occupant.current_floor_id, sides)

        if distance is not None and distance <= approach_region_depth:
            nearby.append((occupant, distance))

    return nearby
