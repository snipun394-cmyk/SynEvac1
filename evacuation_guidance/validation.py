from typing import Tuple

from navigation.node import Node

from evacuation_guidance.models import GuidanceInconsistency


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone, Phase 10
# -- an INDEPENDENT re-verification of a Route route_planner.py already
# built, deliberately not trusting construction alone. Checks:
#
#   1. recommended exit still safe (its own zone not hazard-excluded)
#   2. every route edge traversable
#   3. every zone on the route not hazard-excluded (covers stairs/doors
#      leading through an unsafe zone, not just the terminal exit zone)
#   4. route terminates at the intended exit
#   5. route starts from the intended zone
#
# Returns (is_valid, inconsistency_codes) -- never silently substitutes
# an unsafe route; any failed check produces a visible code (Phase 23),
# never a fabricated pass.
# =====================================================


def validate_route(route, zone_id: str, exit_id: str, excluded_zone_ids: frozenset) -> Tuple[bool, Tuple[str, ...]]:

    codes = []

    if not route.nodes or route.nodes[0].id != zone_id:
        codes.append(GuidanceInconsistency.RECOMMENDATION_ROUTE_MISMATCH)

    if not route.edges or route.edges[-1].id != exit_id:
        codes.append(GuidanceInconsistency.RECOMMENDATION_ROUTE_MISMATCH)

    if any(not edge.traversable for edge in route.edges):
        codes.append(GuidanceInconsistency.ROUTE_BECAME_UNSAFE)

    if any(node.id in excluded_zone_ids for node in route.nodes if node.node_type != Node.OUTSIDE):
        codes.append(GuidanceInconsistency.ROUTE_BECAME_UNSAFE)

    # Deduplicate while preserving first-seen order -- a route can
    # legitimately trip both the traversability and hazard checks at
    # once, but each code should still only ever appear once.
    deduped = tuple(dict.fromkeys(codes))

    return not deduped, deduped
