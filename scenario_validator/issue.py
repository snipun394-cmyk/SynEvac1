from dataclasses import dataclass


class FailureCategory:

    # The fixed, closed enum architecture doc
    # docs/architecture/scenario_engine.md §5.5 introduces -- exactly
    # the seven named in this pass's brief. Mapping from the seven
    # validation modules is mostly 1:1, with two deliberate exceptions:
    # Occupant Validation splits across OCCUPANCY (count/proportion)
    # and GEOMETRY (polygon containment), and Door/Exit/Stair checks
    # (engineering_validation.py) land in STRUCTURAL alongside Building
    # Validation's (both are "is the declared/counted state correct,"
    # as opposed to NAVIGATION's "does that state work"). Plain string
    # constants, not a Python Enum -- matches §5.4's own
    # `category: str` typing and the navigation.validation.
    # ValidationReport.ERROR/WARNING convention this package's severity
    # constants (report.py) already follow.

    STRUCTURAL = "STRUCTURAL"
    GEOMETRY = "GEOMETRY"
    OCCUPANCY = "OCCUPANCY"
    FIRE = "FIRE"
    NAVIGATION = "NAVIGATION"
    EVENTS = "EVENTS"
    DATASET = "DATASET"

    ALL = (STRUCTURAL, GEOMETRY, OCCUPANCY, FIRE, NAVIGATION, EVENTS, DATASET)


@dataclass
class ScenarioValidationIssue:

    # §5.4's frozen shape. Deliberately one flat, self-describing issue
    # type rather than four parallel lists (messages/categories/
    # reasons/warnings) -- avoids the indexing bug four separate lists
    # would invite. Not frozen (unlike every model in scenario/ and
    # scenario_definition/): an issue is built once by whichever
    # validation module produced it and never mutated after being
    # appended to a report, but there is no value in paying frozen
    # dataclass overhead for a type that is always constructed
    # complete in one call, mirroring
    # scenario_definition.validation.DefinitionValidationIssue's own
    # choice for the same reason.

    category: str
    severity: str
    code: str
    message: str

    object_id: str = ""
