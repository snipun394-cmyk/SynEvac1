from dataclasses import dataclass, field
from typing import Tuple

from live_occupants.occupant import LiveOccupant


# =====================================================
# Expose Real Live Camera Occupant State In SynEvac UI milestone -- the
# ONE new presentation-facing snapshot type this milestone adds. Not a
# new state model: every field is either a direct reference to already-
# computed LiveOccupant objects (never copied/re-shaped) or a single
# pre-computed integer, so no UI consumer ever needs to re-derive
# "which occupants currently count" itself (that decision belongs
# exclusively to live_occupants.manager.LiveOccupantManager.
# active_occupants()/canonical_occupancy() -- see docs/architecture/
# canonical_live_occupancy.md and this milestone's own audit,
# SINGLE_PERSON_TRACK_CHURN_HEADCOUNT_SAFE).
# =====================================================


@dataclass(frozen=True)
class LiveOccupantsSnapshot:

    timestamp: float

    # Every occupant LiveOccupantManager currently holds, historical
    # (TEMPORARILY_LOST/EXITED) and current (NEW/ACTIVE) alike -- an
    # operator-facing lifecycle view intentionally shows both, exactly
    # what LiveOccupantManager.all_occupants() already returns. Never
    # itself the headcount source -- see current_occupant_count below.
    occupants: Tuple[LiveOccupant, ...] = field(default_factory=tuple)

    # The ONE correct "how many people right now" answer -- read
    # straight off LiveOccupantManager.canonical_occupancy(time).
    # total_observed_count, never re-derived from len(occupants) (which
    # would incorrectly include historical/lost tracks).
    current_occupant_count: int = 0
