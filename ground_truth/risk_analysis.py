from typing import Any, Dict, List, Optional

from simulator.capacity import derive_stair_capacity
from simulator.occupant import OccupantState


# =====================================================
# Risk scores -- normalized to [0.0, 1.0] wherever computable, built
# only from components that are themselves already 0-1 or a genuine
# ratio of two known counts. A None risk_score inside an entry means
# this run's artifacts gave no honest basis for one (e.g. a zone with
# no occupants and no hazard readings) -- never a guessed value.
# =====================================================


def compute_zone_risk_scores(scenario, movement_result, tick_results, zones) -> List[Dict[str, Any]]:

    timelines = movement_result.occupants
    rows = []

    for zone in zones:

        origin_occupants = [
            occupant for occupant in scenario.occupants if occupant.zone_id == zone.id
        ]

        trapped_fraction = None
        if origin_occupants:

            evacuated = sum(
                1
                for occupant in origin_occupants
                if occupant.occupant_id in timelines
                and timelines[occupant.occupant_id].state == OccupantState.ARRIVED
            )

            trapped_fraction = 1.0 - (evacuated / len(origin_occupants))

        peak_hazard = _zone_peak_hazard_score(zone, tick_results)

        # Mean of whichever normalized components this run actually
        # gives us -- both trapped_fraction and peak_hazard are already
        # in [0.0, 1.0], so their mean is too.
        components = [value for value in (trapped_fraction, peak_hazard) if value is not None]
        risk_score = sum(components) / len(components) if components else None

        rows.append({"zone_id": zone.id, "risk_score": risk_score})

    return rows


# =====================================================


def compute_exit_risk_scores(movement_result, exits) -> List[Dict[str, Any]]:

    rows = []

    for exit_obj in exits:

        risk_score = None

        if exit_obj.capacity:
            peak = movement_result.peak_edge_occupancy.get(exit_obj.id, 0)
            risk_score = min(peak / exit_obj.capacity, 1.0)

        rows.append({"exit_id": exit_obj.id, "risk_score": risk_score})

    return rows


# =====================================================


def compute_stair_risk_scores(movement_result, stairs) -> List[Dict[str, Any]]:

    # Blends two independent, already-available signals the same way
    # compute_zone_risk_scores blends trapped_fraction/peak_hazard,
    # except by max() rather than mean() -- either "most crossings had
    # to queue" or "peak occupancy exceeded the derived stair
    # capacity" is, on its own, enough to call a stair high-risk, so
    # the worse of the two wins rather than being averaged away.

    rows = []

    for stair in stairs:

        total = 0
        queued = 0
        walking_distance = 0.0

        for timeline in movement_result.occupants.values():
            for step in timeline.steps:
                if step.edge.id == stair.id:
                    total += 1
                    if step.queue_wait_time > 0:
                        queued += 1
                    walking_distance = step.edge.walking_distance or 0.0

        queue_fraction = (queued / total) if total > 0 else None

        capacity = derive_stair_capacity(stair.width, walking_distance)
        peak = movement_result.peak_edge_occupancy.get(stair.id, 0)
        capacity_fraction = min(peak / capacity, 1.0) if capacity > 0 else None

        components = [value for value in (queue_fraction, capacity_fraction) if value is not None]
        risk_score = max(components) if components else None

        rows.append({"stair_id": stair.id, "risk_score": risk_score})

    return rows


# =====================================================


def _zone_peak_hazard_score(zone, tick_results) -> Optional[float]:

    if not tick_results:
        return None

    return max(tick.hazard_snapshot.node_state(zone.id).hazard_score for tick in tick_results)


# =====================================================
# Fire -- spread order/first/worst zone read straight off each tick's
# already-computed HazardSnapshot; highest-risk floor is just an
# average of this run's own zone_risk_scores grouped by Zone.floor_id.
# =====================================================


def compute_fire_summary(tick_results, zones, zone_risk_scores) -> Dict[str, Any]:

    first_hazardous_zone = None
    hazard_spread_order: List[str] = []
    maximum_hazard_zone = None

    if tick_results:

        zone_order = {zone.id: index for index, zone in enumerate(zones)}
        first_hazard_time: Dict[str, float] = {}

        for tick in tick_results:
            for zone in zones:

                if zone.id in first_hazard_time:
                    continue

                score = tick.hazard_snapshot.node_state(zone.id).hazard_score

                if score > 0:
                    first_hazard_time[zone.id] = tick.time

        if first_hazard_time:

            ordered = sorted(
                first_hazard_time.items(),
                key=lambda item: (item[1], zone_order[item[0]]),
            )
            hazard_spread_order = [zone_id for zone_id, _ in ordered]
            first_hazardous_zone = hazard_spread_order[0]

        peak_scores = {
            zone.id: max(
                (tick.hazard_snapshot.node_state(zone.id).hazard_score for tick in tick_results),
                default=0.0,
            )
            for zone in zones
        }

        if peak_scores:

            best_zone_id = max(peak_scores, key=peak_scores.get)

            # A zone that never registered any hazard is not a
            # "maximum hazard zone" -- there is nothing to report.
            if peak_scores[best_zone_id] > 0:
                maximum_hazard_zone = best_zone_id

    highest_risk_floor = _highest_risk_floor(zones, zone_risk_scores)

    return {
        "first_hazardous_zone": first_hazardous_zone,
        "hazard_spread_order": hazard_spread_order,
        "maximum_hazard_zone": maximum_hazard_zone,
        "highest_risk_floor": highest_risk_floor,
    }


# =====================================================


def _highest_risk_floor(zones, zone_risk_scores) -> Optional[str]:

    zone_by_id = {zone.id: zone for zone in zones}
    floor_scores: Dict[str, List[float]] = {}

    for entry in zone_risk_scores:

        if entry["risk_score"] is None:
            continue

        zone = zone_by_id.get(entry["zone_id"])

        if zone is None:
            continue

        floor_scores.setdefault(zone.floor_id, []).append(entry["risk_score"])

    if not floor_scores:
        return None

    averages = {
        floor_id: sum(scores) / len(scores) for floor_id, scores in floor_scores.items()
    }

    return max(averages, key=averages.get)
