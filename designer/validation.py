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

    _detect_duplicate_stairs(report, building)

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
