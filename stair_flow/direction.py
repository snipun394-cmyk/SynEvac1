from typing import Optional

from stair_flow.models import TrafficDirection


# =====================================================
# Live Stair Flow & Movement Direction Intelligence milestone, Phase 7 --
# the ONE place direction is derived, grounded in building/floor
# TOPOLOGY, never in screen-space movement or increasing/decreasing
# image Y (this milestone's own explicit prohibition).
#
# Phase 1's audit finding: models.building.Building.floor_elevation()
# already derives a GENUINE vertical ordering -- accumulated floor
# `height`, walked in `ordered_floors()`'s own display_order sequence,
# never assumed from from_floor_id/to_floor_id naming (Building's own
# docstring: "display_order, not list position, is the source of
# order"). models.staircase.Staircase.vertical_height() already reuses
# this exact same function for an unrelated purpose (travel distance),
# confirming it is the established, authoritative source for "which
# floor is physically higher" -- this module adds no new elevation
# concept, it only compares two already-authoritative numbers.
#
# Staircase.from_floor_id is NOT assumed to mean "bottom" (this
# milestone's own explicit warning) -- entered_floor_id is compared
# against BOTH from_floor_id and to_floor_id, and the actual elevations
# of whichever two floors this Staircase really connects are what decide
# UP vs DOWN, never field naming.
# =====================================================


def derive_direction(building, staircase, entered_floor_id: Optional[str]) -> TrafficDirection:

    # entered_floor_id: the floor_id this occupant was confirmed to be
    # on at the exact instant current_stair_id became this staircase's
    # id (see stair_flow.events.extract_stair_flow_events()'s own
    # co-timed PositionSample cross-reference) -- the one piece of
    # GENUINE evidence this derivation requires. None (no co-timed
    # position sample exists) honestly yields UNKNOWN, never a guess.

    if entered_floor_id is None:
        return TrafficDirection.UNKNOWN

    if entered_floor_id == staircase.from_floor_id:
        other_floor_id = staircase.to_floor_id
    elif entered_floor_id == staircase.to_floor_id:
        other_floor_id = staircase.from_floor_id
    else:
        # The occupant's confirmed floor at entry is neither end of THIS
        # staircase -- should not normally happen (WorldProjection only
        # ever resolves a stair_id on one of its own two floor sides),
        # but never assumed/guessed if it does.
        return TrafficDirection.UNKNOWN

    if not other_floor_id:
        return TrafficDirection.UNKNOWN

    entered_floor = building.get_floor(entered_floor_id)
    other_floor = building.get_floor(other_floor_id)

    if entered_floor is None or other_floor is None:
        return TrafficDirection.UNKNOWN

    entered_elevation = building.floor_elevation(entered_floor)
    other_elevation = building.floor_elevation(other_floor)

    if other_elevation > entered_elevation:
        # Heading toward the higher-elevation end -- UP.
        return TrafficDirection.UP

    if other_elevation < entered_elevation:
        return TrafficDirection.DOWN

    # Equal elevation -- a degenerate/misconfigured Staircase (both ends
    # on the same physical level). Never fabricated as UP or DOWN.
    return TrafficDirection.UNKNOWN
