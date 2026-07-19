from typing import FrozenSet

from scenario import Scenario
from scenario_definition import ScenarioDefinition

from scenario_validator.building_validation import validate_building
from scenario_validator.dataset_validation import validate_dataset
from scenario_validator.engineering_validation import validate_engineering
from scenario_validator.event_validation import validate_events
from scenario_validator.fire_validation import validate_fire
from scenario_validator.navigation_validation import validate_navigation
from scenario_validator.occupant_validation import validate_occupants
from scenario_validator.report import ScenarioValidationReport


# The top-level entry point -- architecture doc §5.2:
# "ScenarioValidator.validate(candidate, definition, accepted_hashes=
# frozenset()) -> ScenarioValidationReport". `building` is added as an
# explicit fourth parameter this phase (this implementation phase's own
# Input section: "Validator receives: Scenario, ScenarioDefinition,
# Building. Nothing else.") -- the frozen §5.2 signature line predates
# that explicit resolution and, taken literally, cannot support
# Building Validation or Navigation Validation at all (both explicitly
# frozen, §5.3, and both require a Building to check against). The same
# kind of gap already resolved the same way for
# scenario_definition.ScenarioDefinition.validate(building=...) and
# scenario_generator.GenerationRequest.building -- filled here, not
# redesigned.
#
# A pure, stateless function of its four inputs (§5.2, unchanged): no
# module below it, and this function itself, ever reads or writes
# anything outside its parameters -- no attempt counter, no seed, no
# memory of a previous call.


def validate(
    candidate: Scenario,
    definition: ScenarioDefinition,
    building,
    accepted_hashes: FrozenSet[str] = frozenset(),
) -> ScenarioValidationReport:

    report = ScenarioValidationReport()

    # Building Validation is a precondition for the rest (§5.3): every
    # other module's checks presuppose the ids they operate on actually
    # resolve against the Building.
    building_report = validate_building(candidate, building)
    report.extend(building_report.issues)

    if not building_report.accepted:

        report.metadata = _base_metadata(candidate)
        report.metadata.update(report.summary())

        return report

    for module_report in (
        validate_occupants(candidate, definition, building),
        validate_fire(candidate, definition),
        validate_engineering(candidate, definition),
        validate_navigation(candidate, definition, building),
        validate_events(candidate, definition),
        validate_dataset(candidate, accepted_hashes=accepted_hashes),
    ):
        report.extend(module_report.issues)

    report.metadata = _base_metadata(candidate)
    report.metadata.update(report.summary())

    return report


def _base_metadata(candidate):

    return {
        "scenario_id": candidate.metadata.scenario_id,
        "definition_id": candidate.metadata.definition_id,
    }


class ScenarioValidator:

    # A thin, stateless wrapper matching the frozen doc's own
    # `ScenarioValidator.validate(...)` naming -- carries no state of
    # its own (§5.2); every call is independent and forwards directly
    # to the module-level validate() function above, which is what
    # actually does the work and is what every test in this package
    # exercises directly.

    def validate(
        self,
        candidate: Scenario,
        definition: ScenarioDefinition,
        building,
        accepted_hashes: FrozenSet[str] = frozenset(),
    ) -> ScenarioValidationReport:

        return validate(candidate, definition, building, accepted_hashes=accepted_hashes)
