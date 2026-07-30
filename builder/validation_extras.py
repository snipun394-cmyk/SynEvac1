from navigation.validation import ValidationReport

# ValidationReport.add()'s severity parameter is a free string, not
# restricted to ValidationReport.ERROR/WARNING (see navigation/
# validation.py -- ValidationIssue.severity is just a str field) --
# this module is the one place that also uses a third tier, purely
# advisory notices that are never errors or warnings (e.g. "no floor
# plan imported yet", which is entirely optional). Kept local rather
# than added to ValidationReport itself, which stays exactly the
# frozen two-tier (data-integrity ERROR / completeness WARNING) type
# every other validator in this codebase already relies on.
INFO = "info"


# Builder-only authoring-completeness checks that neither designer.
# validation.validate_building_authoring() nor navigation.graph.
# NavigationGraph.validate() already cover, but that the milestone
# brief explicitly names as examples for Builder's Validation Panel
# ("overlapping geometry", "missing names", "missing scale
# calibration"). Kept as a separate, additive module rather than
# edited into either existing (shared, frozen) validator -- see
# docs/architecture/synevac_builder_feasibility_investigation.md on
# why designer/validation.py and navigation/validation.py are reused
# unchanged. Reuses the SAME ValidationReport/ValidationIssue value
# type those modules already use, just from a third code path.

def validate_builder_extras(building) -> ValidationReport:

    report = ValidationReport()

    if building is None:
        return report

    _check_missing_names(report, building)
    _check_missing_scale_calibration(report, building)
    _check_overlapping_zones(report, building)
    _check_missing_floor_plan(report, building)

    return report


# =====================================================

def _named_collections(floor):

    return (
        (floor.zones, "Zone"),
        (floor.doors, "Door"),
        (floor.exits, "Exit"),
        (floor.stairs, "Stair"),
        (floor.cameras, "Camera"),
        (floor.smoke_detectors, "Smoke Detector"),
        (floor.heat_detectors, "Heat Detector"),
        (floor.speakers, "Speaker"),
        (floor.obstacles, "Obstacle"),
    )


def _check_missing_names(report, building):

    for floor in building.ordered_floors():

        if not floor.name.strip():

            report.add(
                "floor_missing_name",
                "A floor has no name.",
                severity=ValidationReport.WARNING,
                floor_id=floor.id,
            )

        for collection, label in _named_collections(floor):

            for obj in collection:

                if not obj.name.strip():

                    code = label.lower().replace(" ", "_") + "_missing_name"

                    report.add(
                        code,
                        f"A {label} on '{floor.name}' has no name.",
                        severity=ValidationReport.WARNING,
                        object_id=obj.id,
                        floor_id=floor.id,
                    )


# =====================================================

def _check_missing_scale_calibration(report, building):

    for floor in building.ordered_floors():

        if floor.floor_plan and not floor.is_scale_calibrated:

            report.add(
                "floor_missing_scale_calibration",
                (
                    f"Floor '{floor.name}' has an imported floor plan "
                    f"but has not been scale-calibrated -- placed "
                    f"geometry may not reflect real-world dimensions."
                ),
                severity=ValidationReport.WARNING,
                floor_id=floor.id,
            )


# =====================================================
# Overlapping Zones -- a simple axis-aligned bounding-box overlap
# test between every pair of Zones on the same floor. Zones are
# always authored as axis-aligned rectangles (see models/zone.py,
# designer/items/zone_rectangle.py) so this is exact, not an
# approximation.
# =====================================================

def _rects_overlap(a, b):

    return (
        a.x < b.x + b.width
        and a.x + a.width > b.x
        and a.y < b.y + b.height
        and a.y + a.height > b.y
    )


def _check_overlapping_zones(report, building):

    for floor in building.ordered_floors():

        zones = floor.zones

        for i in range(len(zones)):

            for j in range(i + 1, len(zones)):

                zone_a = zones[i]
                zone_b = zones[j]

                if _rects_overlap(zone_a, zone_b):

                    report.add(
                        "overlapping_zones",
                        (
                            f"Zone '{zone_a.name}' and Zone "
                            f"'{zone_b.name}' on '{floor.name}' overlap."
                        ),
                        severity=ValidationReport.WARNING,
                        object_id=zone_a.id,
                        floor_id=floor.id,
                    )


# =====================================================
# Informational only -- a floor legitimately may never need an
# imported floor plan (a simple building can be drawn freehand against
# the grid alone), so this is never a warning, only a notice.
# =====================================================

def _check_missing_floor_plan(report, building):

    for floor in building.ordered_floors():

        if not floor.floor_plan:

            report.add(
                "floor_missing_floor_plan",
                f"Floor '{floor.name}' has no floor plan imported yet.",
                severity=INFO,
                floor_id=floor.id,
            )
