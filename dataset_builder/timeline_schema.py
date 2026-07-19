from typing import List

from models.building import Building

from dataset_builder.schema import ordered_zones


# =====================================================
# Timeline-specific column-name generators. Zone occupancy and every
# engineering-state column (Door/Exit/Stair/Detector/Camera) are
# already provided by dataset_builder/schema.py -- V1's own
# zone_occupancy_columns()/door_state_columns()/exit_state_columns()/
# stair_state_columns()/detector_state_columns()/camera_state_columns()
# are reused as-is for Dataset Builder V2's timeline rows, never
# reimplemented here. Only the three hazard-per-zone columns are new,
# since V1 has no notion of a per-tick hazard reading.
# =====================================================


def zone_smoke_columns(building: Building) -> List[str]:

    return [
        f"Zone_{index}_Smoke"
        for index in range(1, len(ordered_zones(building)) + 1)
    ]


def zone_temperature_columns(building: Building) -> List[str]:

    return [
        f"Zone_{index}_Temperature"
        for index in range(1, len(ordered_zones(building)) + 1)
    ]


def zone_visibility_columns(building: Building) -> List[str]:

    return [
        f"Zone_{index}_Visibility"
        for index in range(1, len(ordered_zones(building)) + 1)
    ]


# =====================================================
# Fixed-shape columns that never depend on a particular Building.
# Append-only, same convention as schema.py's SIMULATION_OUTCOME_COLUMNS/
# ZONE_RESULT_COLUMNS.
# =====================================================

TIMELINE_GENERAL_COLUMNS = (
    "scenario_id",
    "simulation_time",
)

TIMELINE_FIRE_COLUMNS = (
    "ignition_zone",
    "current_fire_zones",
    "current_hazard_score",
)

TIMELINE_SIMULATION_COLUMNS = (
    "people_evacuated",
    "people_remaining",
    "people_trapped",
    "current_congestion",
    "maximum_congestion_so_far",
    "current_evacuation_rate",
)

# Optional -- populated only when derivable from data the completed
# run already carries; left as None (blank in CSV) otherwise. See
# timeline.py for exactly what each one is (and isn't) derived from.
TIMELINE_OPTIONAL_COLUMNS = (
    "current_queue_length",
    "current_route_utilization",
    "current_bottleneck",
    "current_hazard_frontier",
)
