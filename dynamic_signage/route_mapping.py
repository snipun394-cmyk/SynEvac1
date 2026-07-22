from dataclasses import dataclass
from typing import Optional, Tuple

from evacuation_guidance.models import EvacuationGuidancePlan, NavigationStep, NavigationStepType

from dynamic_signage.models import TargetAssetType


# =====================================================
# Live Dynamic Evacuation Signage milestone, Phase 5 -- turns an
# EvacuationGuidancePlan's already-computed ordered_navigation_steps
# into "the next MEANINGFUL navigation decision after a given zone",
# and resolves that decision's real Door/Stair/Exit engineering asset
# and position. Never points a sign at every graph node -- only at a
# genuine decision point (Door/Stair/Exit), reusing evacuation_
# guidance.instruction_builder's own classification of which step types
# are "meaningful" (PASS_THROUGH_DOOR/USE_STAIR/CONTINUE_TO_EXIT) rather
# than inventing a second one.
# =====================================================


_MEANINGFUL_STEP_TYPES = (
    NavigationStepType.PASS_THROUGH_DOOR,
    NavigationStepType.USE_STAIR,
    NavigationStepType.CONTINUE_TO_EXIT,
)


@dataclass(frozen=True)
class MeaningfulStep:

    route_step_index: int
    step: NavigationStep
    departure_zone_id: str


def meaningful_steps(plan: EvacuationGuidancePlan) -> Tuple[MeaningfulStep, ...]:

    # Walks ordered_navigation_steps maintaining the "current zone"
    # context (updated by LEAVE_ZONE/ENTER_ZONE steps) -- USE_STAIR
    # steps carry no zone_id of their own (see evacuation_guidance.
    # instruction_builder.build_navigation_steps), so the departure
    # zone for one is only recoverable by tracking this context, never
    # by re-deriving it from scratch.

    result = []
    current_zone_id: Optional[str] = None

    for index, step in enumerate(plan.ordered_navigation_steps):

        if step.step_type in (NavigationStepType.LEAVE_ZONE, NavigationStepType.ENTER_ZONE):
            current_zone_id = step.zone_id
            continue

        if step.step_type in _MEANINGFUL_STEP_TYPES and current_zone_id is not None:
            result.append(MeaningfulStep(route_step_index=index, step=step, departure_zone_id=current_zone_id))

    return tuple(result)


def next_meaningful_step_for_zone(plan: EvacuationGuidancePlan, zone_id: str) -> Optional[MeaningfulStep]:

    # The FIRST meaningful decision whose departure zone is this one --
    # a simple acyclic evacuation route only ever departs a given zone
    # once, so "first" is also "only" in practice; "first" is still the
    # deterministic, documented tie-break rule if that assumption is
    # ever violated by a future graph shape.

    for entry in meaningful_steps(plan):

        if entry.departure_zone_id == zone_id:
            return entry

    return None


# =====================================================


@dataclass(frozen=True)
class TargetInfo:

    asset_id: str
    asset_type: str
    position: Tuple[float, float]
    floor_id: Optional[str]


def _find_door(building, door_id: str):

    for floor in building.ordered_floors():
        for door in floor.doors:
            if door.id == door_id:
                return door

    return None


def _find_exit(building, exit_id: str):

    for floor in building.ordered_floors():
        for exit_obj in floor.exits:
            if exit_obj.id == exit_id:
                return exit_obj

    return None


def _find_stair(building, stair_id: str):

    for floor in building.ordered_floors():
        for stair in floor.stairs:
            if stair.id == stair_id:
                return stair

    return None


def resolve_target(building, meaningful_step: MeaningfulStep) -> Optional[TargetInfo]:

    # Never fabricates a position -- returns None whenever the
    # underlying engineering asset cannot be found (Phase 4/9's own
    # "missing target geometry -> unavailable" requirement), which the
    # planner turns into an honest UNAVAILABLE instruction rather than a
    # guessed direction.

    step = meaningful_step.step

    if step.step_type == NavigationStepType.PASS_THROUGH_DOOR:

        door = _find_door(building, step.door_id)

        if door is None:
            return None

        return TargetInfo(asset_id=door.id, asset_type=TargetAssetType.DOOR, position=door.center, floor_id=door.floor_id)

    if step.step_type == NavigationStepType.USE_STAIR:

        stair = _find_stair(building, step.stair_id)

        if stair is None:
            return None

        # A Staircase spans two floors with its own position on each
        # (models.staircase.Staircase.from_position/to_position) --
        # the departure floor (step.from_floor_id) determines which end
        # is actually on this sign's own floor.
        if stair.from_floor_id == step.from_floor_id:
            position, floor_id = stair.from_position, stair.from_floor_id
        elif stair.to_floor_id == step.from_floor_id:
            position, floor_id = stair.to_position, stair.to_floor_id
        else:
            return None

        return TargetInfo(asset_id=stair.id, asset_type=TargetAssetType.STAIR, position=position, floor_id=floor_id)

    if step.step_type == NavigationStepType.CONTINUE_TO_EXIT:

        exit_obj = _find_exit(building, step.exit_id)

        if exit_obj is None:
            return None

        return TargetInfo(asset_id=exit_obj.id, asset_type=TargetAssetType.EXIT, position=exit_obj.center, floor_id=exit_obj.floor_id)

    return None
