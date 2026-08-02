"""
18-Story Complete Merge Validation -- Hybrid Flow Regions (Option D).

Completes the existing NIST 18-story office building recreation
(scripts/run_nist_18story_validation.py, Published Scenario Validation
Campaign V2) with its own remaining, already-published geometry: Stairs
3 and 12, both of which -- per Peacock, Hoskins & Kuligowski (2012),
Section 2.3 -- exit through their own independent 0.91 m doors into the
SAME fifth-floor lobby Stair 7 already discharges into. That original
recreation deliberately modeled only Stair 7 of the three lobby-sharing
stairs ("a disclosed scope reduction to keep the campaign tractable");
this script adds the other two, using only dimensions the paper already
publishes (Table 2, Section 2.3) -- no new geometry is invented.

CORRECTION TO THIS MILESTONE'S OWN STATED BACKGROUND, disclosed here
rather than silently followed: the requesting brief for this milestone
named "Stair 1, Stair 3, Stair 7" as the three stairs sharing the
lobby. The published paper says otherwise, in its own words: "Stairs
3, 7, and 12 exited to the lobby area on the fifth floor ... each
through a 0.91 m (36 in) wide door[]; while Stair 1 continued to the
first floor and exited directly out of the rear of the building."
Stair 1 is a separate, direct-exit topology, already fully and
correctly modeled in run_nist_18story_validation.py -- it is NOT part
of the lobby merge. This script therefore builds Stairs 3 and 12 (not
1 and 3), the two stairs the paper actually documents as sharing the
lobby alongside the already-built Stair 7, per this milestone's own
explicit instruction to use "only the published paper" and "not invent
geometry."

Every construction pattern below is copied verbatim from
run_nist_18story_validation.py's own already-tested
build_nist_18story_building()/build_nist_18story_definition() -- no
new modeling convention is introduced. This is a data-construction
script only: it does not touch FlowRegionCapacityModel,
FlowRegionInferencer, or MultiAgentSimulation, and does not run any
performance/statistics comparison (see the companion investigation
report for what happens after this scenario exists).
"""

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from scenario_definition import FireDefinition, FixedValue, OccupantDefinition, ScenarioDefinition


# =====================================================
# Published parameters (Known) -- identical values to
# run_nist_18story_validation.py, restated here rather than imported so
# this script has no dependency on that module's own internals (same
# convention run_flow_region_nist_validation.py already established
# for reusing sibling scripts' *functions*, but constants are simple
# enough to restate directly and keep this file self-contained).
# =====================================================

FLOOR_COUNT = 18                         # Known -- "18-Story Office Building"
STAIR_WIDTH_M = 1.12                     # Known -- Table 2 (44 in, full width incl. handrails)
LOBBY_DOOR_WIDTH_M = 0.91                # Known -- "each through a 0.91 m (36 in) wide door[]" (Stairs 3/7/12)
FRONT_EXIT_WIDTH_M = 0.83                # Known -- Table 2, "Exit width" for the 18-story building
REAR_EXIT_WIDTH_M = 0.83                 # Estimated -- see run_nist_18story_validation.py's own
                                          # disclosure: one building-level exit width is published,
                                          # reused for both Stair 1's rear exit and the shared lobby exit.
RISER_M = 0.191                          # Known -- Table 2 (191 mm)

STEPS_UPPER = 16                         # Known -- floors 18 to 4 (Stair 1) / floors 18 to 6 (Stairs 3/7/12)
STEPS_STAIR1_LOWER = 18                  # Known -- Stair 1, floors 4-3, 3-2, 2-1 (unchanged, restated
                                          # only so floor heights stay identical to the original recreation)

RISE_UPPER_M = STEPS_UPPER * RISER_M               # 3.056 m -- floors 6..17 (shared by all four stairs)
RISE_STAIR1_LOWER_M = STEPS_STAIR1_LOWER * RISER_M  # 3.438 m -- Stair 1's own floors 1..3

# Known-required approximation, IDENTICAL to and inherited from the
# original recreation (not new to this script): floor_5.height is one
# shared Building-wide value, set to the more common 16-step value
# (matching Stair 1's own geometry) rather than the true 19-step
# (3.629 m) final segment the paper publishes for Stairs 3/7/12's own
# landing into the lobby. In the original recreation this approximation
# affected only Stair 7's one flight; it now identically affects Stairs
# 3 and 12's own final flights too, for the same reason and to the same
# ~16% degree -- disclosed here explicitly rather than silently
# inherited.

STAIR1_OCCUPANTS = 255                   # Known -- Table 2 (unchanged; Stair 1 itself is not touched
                                          # by this script, restated only for the definition builder below)
STAIR3_OCCUPANTS = 292                   # Known -- Table 2 ("292 in Stair 3"), previously unused
STAIR7_OCCUPANTS = 340                   # Known -- Table 2 (unchanged)
STAIR12_OCCUPANTS = 197                  # Known -- Table 2 ("197 in Stair 12"), previously unused

STAIR1_OCCUPIED_FLOORS = list(range(2, FLOOR_COUNT + 1))    # floors 2-18 (17 floors) -- unchanged
STAIR3_OCCUPIED_FLOORS = list(range(6, FLOOR_COUNT + 1))    # floors 6-18 (13 floors) -- Known topology,
                                                              # identical to Stair 7's own range (paper:
                                                              # "In Stairs 3, 7, and 12 ... from floors
                                                              # 18 to 6")
STAIR7_OCCUPIED_FLOORS = list(range(6, FLOOR_COUNT + 1))    # floors 6-18 (13 floors) -- unchanged
STAIR12_OCCUPIED_FLOORS = list(range(6, FLOOR_COUNT + 1))   # floors 6-18 (13 floors) -- same as Stair 3/7


def _split_evenly(total, n):

    base, remainder = divmod(total, n)
    return [base + 1 if i < remainder else base for i in range(n)]


def build_nist_18story_complete_building() -> Building:

    floors = []

    for floor_number in range(1, FLOOR_COUNT + 1):

        floor_id = f"floor-{floor_number}"
        display_order = floor_number - 1

        zones = []
        exits = []
        doors = []
        staircases = []

        # ---- Stair 1 (direct-to-exit) -- present on every floor 1-18 ----
        # Unchanged from the original recreation -- Stair 1 is NOT part
        # of the lobby merge (see this file's own module docstring).
        zone_1 = Zone(
            id=f"zone-1-{floor_number}", name=f"Stair 1 Area F{floor_number}",
            x=0.0, y=0.0, width=5.0, height=5.0,
        )
        zones.append(zone_1)

        if floor_number == 1:
            zones.append(Zone(id="zone-fire-only", name="(Unoccupied, ignition-only zone)", x=-50.0, y=-50.0, width=1.0, height=1.0))

        if floor_number == 1:
            exits.append(Exit(id="exit-1", zone_id="zone-1-1", width=REAR_EXIT_WIDTH_M))
        else:
            staircases.append(
                Staircase(
                    id=f"stair-1-{floor_number}", from_floor_id=floor_id, to_floor_id=f"floor-{floor_number - 1}",
                    from_zone_id=f"zone-1-{floor_number}", to_zone_id=f"zone-1-{floor_number - 1}",
                    width=STAIR_WIDTH_M,
                )
            )

        # ---- Stairs 3, 7, 12 (lobby-merge) -- present on floors 5-18,
        # each an exact structural copy of the pattern already used for
        # Stair 7, differing only in id prefix and (for the lobby
        # itself) never re-created twice. ----
        for stair_prefix in ("3", "7", "12"):

            if floor_number < 5:
                continue

            # x offsets (20 / 50 / 80) are purely for Designer/Replay
            # Studio layout spacing -- chosen to avoid visually
            # overlapping the shared lobby zone at x=40 below -- and
            # have no effect on graph connectivity or simulation.
            stair_x_offset = {"3": 20.0, "7": 50.0, "12": 80.0}[stair_prefix]

            zone_id = f"zone-{stair_prefix}-{floor_number}"
            zones.append(
                Zone(
                    id=zone_id, name=f"Stair {stair_prefix} Area F{floor_number}",
                    x=stair_x_offset, y=0.0, width=5.0, height=5.0,
                )
            )

            if floor_number == 5:

                if stair_prefix == "3":
                    # The shared lobby zone/exit are created exactly
                    # once, the first time this loop reaches floor 5 --
                    # identical object to what Stair 7's own door and
                    # Stair 12's own door both connect to below.
                    lobby_zone = Zone(id="zone-lobby", name="Ground Floor Lobby (Front)", x=40.0, y=0.0, width=8.0, height=8.0)
                    zones.append(lobby_zone)
                    exits.append(Exit(id="exit-lobby", zone_id="zone-lobby", width=FRONT_EXIT_WIDTH_M))

                doors.append(
                    Door(
                        id=f"door-{stair_prefix}-lobby", floor_id=floor_id,
                        zone_a_id=zone_id, zone_b_id="zone-lobby",
                        width=LOBBY_DOOR_WIDTH_M, normally_open=True,
                    )
                )

            else:
                staircases.append(
                    Staircase(
                        id=f"stair-{stair_prefix}-{floor_number}", from_floor_id=floor_id, to_floor_id=f"floor-{floor_number - 1}",
                        from_zone_id=zone_id, to_zone_id=f"zone-{stair_prefix}-{floor_number - 1}",
                        width=STAIR_WIDTH_M,
                    )
                )

        if floor_number == 1:
            height = RISE_STAIR1_LOWER_M
        elif floor_number in (2, 3):
            height = RISE_STAIR1_LOWER_M
        elif floor_number == 5:
            height = RISE_UPPER_M
        elif floor_number < FLOOR_COUNT:
            height = RISE_UPPER_M
        else:
            height = 3.0  # unused -- topmost floor

        floors.append(
            Floor(
                name=f"Floor {floor_number}", id=floor_id, display_order=display_order, height=height,
                zones=zones, exits=exits, doors=doors, stairs=staircases,
            )
        )

    return Building(
        name="NIST 18-Story Office Building, Complete Merge Recreation (Peacock et al. 2012)",
        id="nist-18story-complete-merge-v1", floors=floors,
    )


def build_nist_18story_complete_definition() -> ScenarioDefinition:

    occupancy = {}
    behaviour = {}

    for prefix, total, occupied_floors in (
        ("1", STAIR1_OCCUPANTS, STAIR1_OCCUPIED_FLOORS),
        ("3", STAIR3_OCCUPANTS, STAIR3_OCCUPIED_FLOORS),
        ("7", STAIR7_OCCUPANTS, STAIR7_OCCUPIED_FLOORS),
        ("12", STAIR12_OCCUPANTS, STAIR12_OCCUPIED_FLOORS),
    ):

        per_floor = _split_evenly(total, len(occupied_floors))

        for index, floor_number in enumerate(occupied_floors):

            zone_id = f"zone-{prefix}-{floor_number}"
            occupancy[zone_id] = FixedValue(per_floor[index])
            behaviour[zone_id] = FixedValue("Adult_Default")

    return ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=FixedValue(300.0),
            allowed_ignition_zone_ids=frozenset(["zone-fire-only"]),
        ),
        occupant=OccupantDefinition(occupancy_distribution=occupancy, behaviour_profile_distribution=behaviour),
    )


DEFINITION_ID = "nist-18story-complete-merge-validation-v1"
MASTER_SEED = 20260803  # same building, same seed convention as run_nist_18story_validation.py


if __name__ == "__main__":

    # Structural/mechanical verification only -- no performance or
    # statistics comparison is run here (that is explicitly out of
    # scope for this milestone; see the companion investigation
    # report). This block exists purely to prove the construction is
    # valid and to print FlowRegionInferencer's own verdict on the
    # lobby's region_kind.

    from navigation.flow_region import FlowRegion
    from navigation.graph_builder import NavigationGraphGenerator

    building = build_nist_18story_complete_building()
    graph = NavigationGraphGenerator().build(building)

    report = graph.validate()

    print(f"Zones: {sum(1 for n in graph.nodes.values() if n.node_type == 'Zone')}")
    print(f"Edges: {len(graph.edges)}")
    print(f"Validation errors: {len(report.errors)}, warnings: {len(report.warnings)}")

    for issue in report.errors:
        print("  ERROR:", issue.code, issue.message)

    lobby_region = graph.flow_regions["door-7-lobby"]

    print(f"Lobby region kind: {lobby_region.region_kind}")
    print(f"Lobby region member edges ({len(lobby_region.edge_ids)}): {sorted(lobby_region.edge_ids)}")

    assert graph.flow_regions["door-3-lobby"].id == lobby_region.id
    assert graph.flow_regions["door-12-lobby"].id == lobby_region.id
    assert lobby_region.region_kind == FlowRegion.MERGE

    print("Confirmed: door-3-lobby, door-7-lobby, and door-12-lobby all resolve to the same MERGE FlowRegion.")
