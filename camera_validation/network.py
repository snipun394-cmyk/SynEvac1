from dataclasses import dataclass
from typing import Tuple


# A camera's own coverage of a zone below this fraction doesn't count
# as "this camera covers that zone" for overlap/isolation purposes --
# a stray sliver of visibility through a doorway shouldn't count as
# real redundant coverage. Illustrative, not validated.
OVERLAP_ZONE_COVERAGE_THRESHOLD = 0.1

# A camera whose covered zones are backed up by at least one other
# camera at or above this fraction is "excessive overlap" -- most of
# what it sees, something else already sees too.
EXCESSIVE_OVERLAP_THRESHOLD = 0.8

# Illustrative per-asset-type penalties applied to the network score
# for each critical, entirely-unmonitored asset -- an unmonitored Exit
# is weighted heaviest (the single most safety-critical asset a
# camera network should never miss), then Stair, then Door.
EXIT_PENALTY = 10.0
STAIR_PENALTY = 7.0
DOOR_PENALTY = 3.0


@dataclass(frozen=True)
class NetworkAnalysis:

    # Phase 3's required whole-network engineering metrics for one
    # floor -- built entirely from FloorCoverage (visibility/coverage.
    # py), never re-deriving wall/visibility geometry itself.

    floor_id: str

    uncovered_zone_ids: Tuple[str, ...]
    uncovered_door_ids: Tuple[str, ...]
    uncovered_exit_ids: Tuple[str, ...]
    uncovered_stair_ids: Tuple[str, ...]

    # A camera counts toward exactly one of these three buckets (or
    # neither, if it covers nothing at all -- see no_coverage_camera_ids).
    excessive_overlap_camera_ids: Tuple[str, ...]
    isolated_camera_ids: Tuple[str, ...]
    redundant_camera_ids: Tuple[str, ...]

    # A camera whose own coverage never reaches
    # OVERLAP_ZONE_COVERAGE_THRESHOLD anywhere -- effectively
    # contributing nothing to this floor's coverage.
    no_coverage_camera_ids: Tuple[str, ...]

    # 0-100 -- combined floor coverage minus a penalty for each
    # entirely-unmonitored critical asset. Illustrative, not a
    # validated life-safety figure (same honesty as
    # CameraPlacementMetrics.placement_score).
    network_score: float


def compute_network_analysis(floor, floor_coverage) -> NetworkAnalysis:

    covered_door_ids = set()
    covered_exit_ids = set()
    covered_stair_ids = set()

    for visibility in floor_coverage.per_camera.values():

        covered_door_ids.update(visibility.visible_door_ids)
        covered_exit_ids.update(visibility.visible_exit_ids)
        covered_stair_ids.update(visibility.visible_stair_ids)

    uncovered_door_ids = tuple(
        door.id for door in floor.doors if door.id not in covered_door_ids
    )
    uncovered_exit_ids = tuple(
        exit_obj.id for exit_obj in floor.exits if exit_obj.id not in covered_exit_ids
    )
    uncovered_stair_ids = tuple(
        stair.id for stair in floor.stairs if stair.id not in covered_stair_ids
    )

    excessive_overlap_ids, isolated_ids, redundant_ids, no_coverage_ids = _classify_cameras(
        floor_coverage,
    )

    combined_coverage_component = floor_coverage.total_floor_coverage_fraction * 100.0

    critical_penalty = (
        EXIT_PENALTY * len(uncovered_exit_ids)
        + STAIR_PENALTY * len(uncovered_stair_ids)
        + DOOR_PENALTY * len(uncovered_door_ids)
    )

    network_score = max(0.0, min(100.0, combined_coverage_component - critical_penalty))

    return NetworkAnalysis(
        floor_id=floor.id,
        uncovered_zone_ids=floor_coverage.uncovered_zone_ids,
        uncovered_door_ids=uncovered_door_ids,
        uncovered_exit_ids=uncovered_exit_ids,
        uncovered_stair_ids=uncovered_stair_ids,
        excessive_overlap_camera_ids=excessive_overlap_ids,
        isolated_camera_ids=isolated_ids,
        redundant_camera_ids=redundant_ids,
        no_coverage_camera_ids=no_coverage_ids,
        network_score=network_score,
    )


# =====================================================


def _classify_cameras(floor_coverage):

    excessive_overlap_ids = []
    isolated_ids = []
    redundant_ids = []
    no_coverage_ids = []

    for camera_id, visibility in floor_coverage.per_camera.items():

        own_zone_ids = {
            zone_id for zone_id, fraction in visibility.zone_coverage.items()
            if fraction > OVERLAP_ZONE_COVERAGE_THRESHOLD
        }

        if not own_zone_ids:

            no_coverage_ids.append(camera_id)
            continue

        backed_up_zone_count = sum(
            1 for zone_id in own_zone_ids
            if _another_camera_covers(floor_coverage, camera_id, zone_id)
        )

        overlap_fraction = backed_up_zone_count / len(own_zone_ids)

        if overlap_fraction >= EXCESSIVE_OVERLAP_THRESHOLD:
            excessive_overlap_ids.append(camera_id)
        elif overlap_fraction == 0.0:
            isolated_ids.append(camera_id)
        else:
            redundant_ids.append(camera_id)

    return (
        tuple(sorted(excessive_overlap_ids)),
        tuple(sorted(isolated_ids)),
        tuple(sorted(redundant_ids)),
        tuple(sorted(no_coverage_ids)),
    )


# =====================================================


def _another_camera_covers(floor_coverage, camera_id, zone_id) -> bool:

    for other_camera_id, other_visibility in floor_coverage.per_camera.items():

        if other_camera_id == camera_id:
            continue

        if other_visibility.zone_coverage.get(zone_id, 0.0) > OVERLAP_ZONE_COVERAGE_THRESHOLD:
            return True

    return False
