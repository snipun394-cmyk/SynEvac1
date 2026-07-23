from navigation.validation import ValidationReport


# Designer-level authoring completeness -- deliberately separate from
# NavigationGraph.validate() (frozen, unmodified) rather than changing
# that method's own severities. navigation/validation.py's WARNING
# classification for a missing Door/Exit/Stair Zone reference is
# correct and intentional there: "the graph is internally consistent,
# building it succeeds, it's just incomplete" is a true statement
# about the *graph*. But from the Designer's own point of view -- the
# tool a person actually authors a building in -- an unconnected
# Door/Exit/Stair is not a minor completeness gap, it is a connector
# that silently does nothing, and a building can "look" fully wired on
# screen while producing a Navigation Graph with no edge for it at
# all. This module reports exactly that class of problem as an ERROR,
# without touching navigation/validation.py's own, differently-scoped
# severities.
#
# Reuses ValidationReport/ValidationIssue as a plain value type (same
# shape, same by_code()/errors/warnings accessors) -- not a subclass,
# not a modification, just constructing more instances of an
# already-frozen, already-proven data type from a different code path.


def validate_building_authoring(building) -> ValidationReport:

    report = ValidationReport()

    if building is None:
        return report

    for floor in building.ordered_floors():

        for door in floor.doors:
            _check_door(report, floor, door)

        for exit_obj in floor.exits:
            _check_exit(report, floor, exit_obj)

        for stair in floor.stairs:
            _check_stair(report, floor, stair)

        for speaker in floor.speakers:
            _check_zone_assignment(report, floor, speaker, "Speaker", "speaker_missing_zone")

        for smoke_detector in floor.smoke_detectors:
            _check_zone_assignment(report, floor, smoke_detector, "Smoke Detector", "smoke_detector_missing_zone")

        for heat_detector in floor.heat_detectors:
            _check_zone_assignment(report, floor, heat_detector, "Heat Detector", "heat_detector_missing_zone")

        for mcp in floor.manual_call_points:
            _check_zone_assignment(report, floor, mcp, "Manual Call Point", "manual_call_point_missing_zone")

        for emergency_light in floor.emergency_lights:
            _check_zone_assignment(report, floor, emergency_light, "Emergency Light", "emergency_light_missing_zone")

        for sprinkler in floor.sprinklers:
            _check_zone_assignment(report, floor, sprinkler, "Sprinkler", "sprinkler_missing_zone")

        for fire_extinguisher in floor.fire_extinguishers:
            _check_zone_assignment(report, floor, fire_extinguisher, "Fire Extinguisher", "fire_extinguisher_missing_zone")

        for fire_hydrant in floor.fire_hydrants:
            _check_zone_assignment(report, floor, fire_hydrant, "Fire Hydrant", "fire_hydrant_missing_zone")

        for hose_reel in floor.hose_reels:
            _check_zone_assignment(report, floor, hose_reel, "Hose Reel", "hose_reel_missing_zone")

        for tank in floor.fire_water_tanks:
            _check_zone_assignment(report, floor, tank, "Fire Water Tank", "fire_water_tank_missing_zone")

        for pump in floor.fire_pumps:
            _check_zone_assignment(report, floor, pump, "Fire Pump", "fire_pump_missing_zone")

        for jockey_pump in floor.jockey_pumps:
            _check_zone_assignment(report, floor, jockey_pump, "Jockey Pump", "jockey_pump_missing_zone")

        for inlet in floor.fire_service_inlets:
            _check_zone_assignment(report, floor, inlet, "Fire Service Inlet", "fire_service_inlet_missing_zone")

    _detect_duplicate_stairs(report, building)
    _detect_dangling_fire_water_system_references(report, building)

    return report


# =====================================================
# Digital Twin Asset -> Zone Assignment & Live FACP Runtime milestone,
# Phase 5 -- reuses this module's own existing ValidationReport
# mechanism rather than a new framework. Deliberately WARNING, not
# ERROR, unlike Door/Exit/Stair above: an unassigned Speaker/Smoke/Heat
# Detector still exists and still functions as a device (a detector
# still computes its own DetectorState, a speaker is still a real
# asset) -- only its ZONE-SCOPED behavior (voice broadcast routing,
# FACP zone-consistency) is degraded, never a structurally broken
# Navigation Graph the way an unconnected Door/Exit/Stair is.
# =====================================================

def _check_zone_assignment(report, floor, asset, label, code):

    if not asset.zone_ids:

        report.add(
            code,
            f"{label} '{asset.name}' on '{floor.name}' has no zone assigned -- "
            f"it will not participate in zone-scoped voice evacuation/FACP "
            f"consistency checks until one is configured.",
            severity=ValidationReport.WARNING,
            object_id=asset.id,
            floor_id=floor.id,
        )


# =====================================================

def _check_door(report, floor, door):

    if not door.zone_a_id:

        report.add(
            "door_missing_zone_a",
            f"Door '{door.name}' on '{floor.name}' has no Zone A "
            f"assigned -- it will not produce a Navigation Graph edge.",
            severity=ValidationReport.ERROR,
            object_id=door.id,
            floor_id=floor.id,
        )

    if not door.zone_b_id:

        report.add(
            "door_missing_zone_b",
            f"Door '{door.name}' on '{floor.name}' has no Zone B "
            f"assigned -- it will not produce a Navigation Graph edge.",
            severity=ValidationReport.ERROR,
            object_id=door.id,
            floor_id=floor.id,
        )


# =====================================================

def _check_exit(report, floor, exit_obj):

    if not exit_obj.zone_id:

        report.add(
            "exit_missing_zone",
            f"Exit '{exit_obj.name}' on '{floor.name}' has no Zone "
            f"assigned -- it will not produce a Navigation Graph edge.",
            severity=ValidationReport.ERROR,
            object_id=exit_obj.id,
            floor_id=floor.id,
        )


# =====================================================

def _check_stair(report, floor, stair):

    if not stair.from_zone_id:

        report.add(
            "stair_missing_from_zone",
            f"Stair '{stair.name}' on '{floor.name}' has no From Zone "
            f"assigned -- it will not produce a Navigation Graph edge.",
            severity=ValidationReport.ERROR,
            object_id=stair.id,
            floor_id=floor.id,
        )

    if not stair.to_zone_id:

        report.add(
            "stair_missing_to_zone",
            f"Stair '{stair.name}' on '{floor.name}' has no To Zone "
            f"assigned -- it will not produce a Navigation Graph edge.",
            severity=ValidationReport.ERROR,
            object_id=stair.id,
            floor_id=floor.id,
        )


# =====================================================
# Groups every Stair by (floor pair, zone pair) -- two Stairs only
# ever land in the same group when they connect the exact same two
# floors AND the exact same two zones (or are both equally unassigned
# on either end), which is what actually happened in the reported bug:
# two independent Staircase objects standing in for one physical
# connector. A building with several genuinely different physical
# staircases between the same two floors connects different zone
# pairs and is correctly never flagged.
# =====================================================

def _detect_duplicate_stairs(report, building):

    groups = {}

    for floor in building.ordered_floors():

        for stair in floor.stairs:

            key = (
                frozenset({stair.from_floor_id, stair.to_floor_id}),
                frozenset({stair.from_zone_id, stair.to_zone_id}),
            )

            groups.setdefault(key, []).append((floor, stair))

    for (floor_pair, _zone_pair), entries in groups.items():

        if len(entries) < 2:
            continue

        stair_names = ", ".join(
            f"'{stair.name}' ({floor.name})" for floor, stair in entries
        )

        first_floor, first_stair = entries[0]

        report.add(
            "potential_duplicate_stair",
            f"{len(entries)} Stairs appear to represent the same "
            f"physical connection: {stair_names}. If this is really "
            f"one staircase, keep only one of these objects.",
            severity=ValidationReport.WARNING,
            object_id=first_stair.id,
            floor_id=first_floor.id,
        )


# =====================================================
# Fire Water Supply & Suppression Infrastructure milestone -- a
# FireWaterSystem's own *_ids fields are plain, unresolved id
# references (models/fire_water_system.py's own docstring), never
# validated by that class itself. This is the one place a dangling
# reference (an id that no longer names a real asset, e.g. the asset
# was deleted after being added to a system) is reported -- as a
# WARNING, the same severity every other zone/system-scoped
# completeness gap in this module already uses, never a crash.
# =====================================================

def _detect_dangling_fire_water_system_references(report, building):

    if building is None:
        return

    valid_ids_by_field = {
        "tank_ids": set(), "pump_ids": set(), "jockey_pump_ids": set(), "inlet_ids": set(),
        "sprinkler_ids": set(), "hydrant_ids": set(), "hose_reel_ids": set(),
    }

    for floor in building.ordered_floors():

        valid_ids_by_field["tank_ids"].update(tank.id for tank in floor.fire_water_tanks)
        valid_ids_by_field["pump_ids"].update(pump.id for pump in floor.fire_pumps)
        valid_ids_by_field["jockey_pump_ids"].update(pump.id for pump in floor.jockey_pumps)
        valid_ids_by_field["inlet_ids"].update(inlet.id for inlet in floor.fire_service_inlets)
        valid_ids_by_field["sprinkler_ids"].update(sprinkler.id for sprinkler in floor.sprinklers)
        valid_ids_by_field["hydrant_ids"].update(hydrant.id for hydrant in floor.fire_hydrants)
        valid_ids_by_field["hose_reel_ids"].update(hose_reel.id for hose_reel in floor.hose_reels)

    for system in building.fire_water_systems:

        for field_name, valid_ids in valid_ids_by_field.items():

            for asset_id in getattr(system, field_name):

                if asset_id not in valid_ids:

                    report.add(
                        "fire_water_system_dangling_reference",
                        f"Fire Water System '{system.name}' references asset id "
                        f"'{asset_id}' ({field_name}), which no longer exists.",
                        severity=ValidationReport.WARNING,
                        object_id=system.id,
                        floor_id="",
                    )
