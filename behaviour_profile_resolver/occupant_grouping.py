import hashlib
import random

from dataclasses import dataclass
from typing import Dict, Optional

from behaviour_profile_resolver.category import OccupantCategory, occupant_category
from behaviour_profile_resolver.occupant_attributes import derive_occupant_attributes


# =====================================================
# Occupant Attributes Phase 3 -- Social Groups. Families, Friends,
# Coworkers, School Groups, Hospital Staff, and Visitors Together --
# civilian-only (firefighters are never grouped, matching the existing
# "keep firefighter logic separate from civilian behavior logic" rule
# behaviour_profile_resolver.firefighter_registrar/behavior_library.
# firefighter_strategies already establish), and, like occupant_
# attributes.py, deliberately NOT stored on Scenario -- assign_occupant_
# groups() is a pure function of already-fixed Scenario data (occupants
# + metadata.seed), so behaviour_profile_resolver (registration),
# ground_truth (outcome summaries), and dataset_builder (feature
# export) each call it independently against the same Scenario and get
# bit-identical results, with zero shared mutable state.
#
# Distinct concept from human_decision_engine.groups.Group/GroupRegistry
# -- that module forms *dynamic, decision-time* assist/follow/rescue-
# escort groupings from runtime pairing decisions (opt-in, used only by
# behaviour_profile_resolver.dynamic_registrar). This module forms
# *pre-existing social units* (a family that arrived together) purely
# from Scenario placement + seed, independent of any runtime decision.
# Neither module imports the other; GroupAssignment.group_id is a
# separate namespace from human_decision_engine.groups.Group.group_id.
# =====================================================

_MIN_GROUP_SIZE = 2
_MAX_GROUP_SIZE = 4

# Fraction of civilian occupants, per zone, who travel with a social
# group at all -- not everyone does; a lone business traveler or a
# single visitor is just as realistic as a family of four. Illustrative,
# same "disclosed starting point, not measured data" honesty every
# other constant in this feature carries.
_GROUPED_FRACTION = 0.65


@dataclass(frozen=True)
class GroupAssignment:

    group_id: str
    group_type: str
    is_leader: bool

    # The group's own leader's occupant_id -- None for the leader's own
    # entry (a leader doesn't follow themselves), always set for every
    # follower entry. What behaviour_profile_resolver.registrar reads
    # to populate the "social_leader_occupant_id" trait
    # behavior_library.attribute_aware_strategies.SocialGroupAware*
    # consumes.
    leader_occupant_id: Optional[str] = None


# =====================================================


def _derive_grouping_seed(scenario_seed: int) -> int:

    # Same sha256-of-a-stable-key convention occupant_attributes.
    # _derive_attribute_seed()/registrar._derive_occupant_seed() already
    # establish -- one shared stream for the whole grouping pass (group
    # *formation* needs to see every occupant in a zone at once, unlike
    # occupant_attributes' own strictly-per-occupant derivation).

    key = f"{scenario_seed}|behaviour_profile_resolver.occupant_grouping|groups"
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big")


def _group_type_for(categories) -> str:

    category_set = set(categories)

    if OccupantCategory.CHILD in category_set and category_set - {OccupantCategory.CHILD}:
        return "Family"

    if category_set == {OccupantCategory.CHILD}:
        return "School Group"

    if category_set == {OccupantCategory.STAFF}:
        return "Hospital Staff"

    if category_set == {OccupantCategory.VISITOR}:
        return "Visitors Together"

    return "Friends"


def assign_occupant_groups(scenario) -> Dict[str, GroupAssignment]:

    rng = random.Random(_derive_grouping_seed(scenario.metadata.seed))

    occupants_by_zone: Dict[str, list] = {}
    for occupant in scenario.occupants:
        occupants_by_zone.setdefault(occupant.zone_id, []).append(occupant)

    assignments: Dict[str, GroupAssignment] = {}
    group_index = 0

    for zone_id in sorted(occupants_by_zone):

        zone_occupants = sorted(occupants_by_zone[zone_id], key=lambda o: o.occupant_id)
        rng.shuffle(zone_occupants)

        sociable = [occupant for occupant in zone_occupants if rng.random() < _GROUPED_FRACTION]

        index = 0
        while index + _MIN_GROUP_SIZE <= len(sociable):

            # The while condition itself already guarantees a lone
            # leftover of 1 is never turned into its own group -- once
            # fewer than _MIN_GROUP_SIZE occupants remain, the loop
            # simply stops and that occupant stays ungrouped (exactly
            # "not everyone travels in a group," the same intent
            # _GROUPED_FRACTION already expresses at the individual
            # level). size is always clamped to `remaining` itself, so
            # it can never exceed _MAX_GROUP_SIZE.
            remaining = len(sociable) - index
            size = rng.randint(_MIN_GROUP_SIZE, min(_MAX_GROUP_SIZE, remaining))

            members = sociable[index:index + size]
            index += size

            group_id = f"group-{zone_id}-{group_index}"
            group_index += 1

            categories = [occupant_category(member.behaviour_profile_id) for member in members]
            group_type = _group_type_for(categories)

            leader = max(
                members,
                key=lambda member: (
                    derive_occupant_attributes(
                        scenario.metadata.seed, member.occupant_id, member.behaviour_profile_id,
                    ).leadership,
                    member.occupant_id,
                ),
            )

            for member in members:

                is_leader = member is leader

                assignments[member.occupant_id] = GroupAssignment(
                    group_id=group_id,
                    group_type=group_type,
                    is_leader=is_leader,
                    leader_occupant_id=None if is_leader else leader.occupant_id,
                )

    return assignments
