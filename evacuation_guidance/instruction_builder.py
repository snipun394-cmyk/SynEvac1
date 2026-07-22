from typing import Tuple

from navigation.edge import Edge
from navigation.node import Node

from evacuation_guidance.models import NavigationStep, NavigationStepType


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone, Phase 7/8/9
# -- structured navigation steps AND the deterministic, no-LLM
# instructions derived from them. Text is DERIVED from structure, never
# the reverse (Phase 7).
#
# Phase 9's own "do not over-instruct" requirement is satisfied
# structurally: only PASS_THROUGH_DOOR/USE_STAIR/CONTINUE_TO_EXIT steps
# ever produce an instruction sentence -- LEAVE_ZONE/ENTER_ZONE/
# EXIT_BUILDING are retained in the FULL structured step list (so the
# complete route is always reconstructable, Phase 25 test 43) but never
# separately narrated ("Leave Z1, enter Z2, leave Z2..." never happens
# because this package never generates a sentence for those step
# types). Since every Edge in this graph model is already a meaningful
# decision point (Door/Stair/Exit -- there is no lower-granularity
# "walk down the hallway" edge type to begin with), one instruction per
# edge is already the correct, non-redundant granularity.
#
# No left/right/north/south/landmark/distance fabrication anywhere
# (Phase 8) -- the only "direction" ever stated is ascend/descend for a
# Stair, and that is genuinely derived from Floor.display_order (a real,
# stored building attribute), never invented.
# =====================================================


def build_navigation_steps(route) -> Tuple[NavigationStep, ...]:

    if not route.nodes:
        return ()

    steps = [NavigationStep(NavigationStepType.LEAVE_ZONE, zone_id=route.nodes[0].id, floor_id=route.nodes[0].floor_id)]

    for index, edge in enumerate(route.edges):

        from_node = route.nodes[index]
        to_node = route.nodes[index + 1]

        if edge.edge_type == Edge.DOOR:

            steps.append(NavigationStep(NavigationStepType.PASS_THROUGH_DOOR, door_id=edge.id, zone_id=from_node.id))

            if to_node.node_type == Node.ZONE:
                steps.append(NavigationStep(NavigationStepType.ENTER_ZONE, zone_id=to_node.id, floor_id=to_node.floor_id))

        elif edge.edge_type == Edge.STAIR:

            steps.append(NavigationStep(
                NavigationStepType.USE_STAIR, stair_id=edge.id,
                from_floor_id=from_node.floor_id, to_floor_id=to_node.floor_id,
            ))

            if to_node.node_type == Node.ZONE:
                steps.append(NavigationStep(NavigationStepType.ENTER_ZONE, zone_id=to_node.id, floor_id=to_node.floor_id))

        elif edge.edge_type == Edge.EXIT:

            steps.append(NavigationStep(NavigationStepType.CONTINUE_TO_EXIT, exit_id=edge.id, zone_id=from_node.id))
            steps.append(NavigationStep(NavigationStepType.EXIT_BUILDING, exit_id=edge.id))

    return tuple(steps)


# =====================================================


def asset_label(edge) -> str:

    type_label = {Edge.DOOR: "Door", Edge.EXIT: "Exit", Edge.STAIR: "Stair"}[edge.edge_type]
    reference_name = getattr(edge.reference, "name", None)
    raw = reference_name if reference_name else edge.id

    if type_label.lower() in raw.lower():
        return raw

    return f"{type_label} {raw}"


def _zone_label(node) -> str:

    return node.name if node.name else node.id


def _floor_label(building, floor_id):

    floor = building.get_floor(floor_id) if floor_id is not None else None

    if floor is None:
        return floor_id or "an unknown floor"

    return floor.name if floor.name else f"Floor {floor.display_order}"


def build_instructions(route, building) -> Tuple[str, ...]:

    instructions = []

    for index, edge in enumerate(route.edges):

        to_node = route.nodes[index + 1]

        if edge.edge_type == Edge.DOOR:

            label = asset_label(edge)

            if to_node.node_type == Node.ZONE:
                instructions.append(f"Proceed through {label} into {_zone_label(to_node)}.")
            else:
                instructions.append(f"Proceed through {label}.")

        elif edge.edge_type == Edge.STAIR:

            from_node = route.nodes[index]
            from_floor = building.get_floor(from_node.floor_id)
            to_floor = building.get_floor(to_node.floor_id)

            if from_floor is not None and to_floor is not None:
                verb = "descend" if to_floor.display_order < from_floor.display_order else "ascend"
            else:
                verb = "proceed"

            instructions.append(f"Use {asset_label(edge)} to {verb} to {_floor_label(building, to_node.floor_id)}.")

        elif edge.edge_type == Edge.EXIT:

            instructions.append(f"Continue to {asset_label(edge)}.")

    return tuple(instructions)
