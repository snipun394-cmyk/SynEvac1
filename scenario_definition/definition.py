from dataclasses import dataclass, field
from typing import Optional, Tuple

from scenario_definition.engineering_constraints import EngineeringConstraints
from scenario_definition.event_templates import EventTemplate
from scenario_definition.fire_definition import FireDefinition
from scenario_definition.firefighter_definition import FirefighterDeploymentDefinition
from scenario_definition.occupant_definition import OccupantDefinition
from scenario_definition.validation import DefinitionValidationReport, validate_definition


@dataclass(frozen=True)
class ScenarioDefinition:

    # The immutable rulebook describing the legal search space for
    # scenario generation -- architecture doc §3, frozen. Declares
    # what may be sampled; samples nothing itself. Never creates
    # scenarios, never runs simulations, never contains randomness,
    # never contains generated occupants/fire/hazards, never contains
    # simulation, RL, AI, or Navigation state.
    #
    # fire has no default (a Definition must state some fire rule);
    # engineering/occupant/event_templates/seed all default to "no
    # constraint stated" shapes, since a Definition may legitimately
    # leave any of those categories entirely unconstrained.

    fire: FireDefinition

    engineering: EngineeringConstraints = field(default_factory=EngineeringConstraints)
    occupant: OccupantDefinition = field(default_factory=OccupantDefinition)

    # Human Population & Assisted Evacuation Modeling Phase 5 --
    # additive, defaults to "deploy nothing" (see
    # FirefighterDeploymentDefinition's own docstring), so every
    # existing Definition is unaffected.
    firefighter: FirefighterDeploymentDefinition = field(
        default_factory=FirefighterDeploymentDefinition,
    )

    event_templates: Tuple[EventTemplate, ...] = field(default_factory=tuple)

    # Stored so a later Generator run is reproducible -- not used by
    # anything inside this package (§3.2's Reproducibility row).
    seed: Optional[int] = None

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "event_templates", tuple(self.event_templates))

    # =====================================================

    def validate(self, building=None) -> DefinitionValidationReport:

        # See scenario_definition/validation.py for why `building` is
        # an explicit parameter here rather than something this class
        # stores. Checks structural well-formedness only and never
        # samples anything (§3.4) -- validating a generated Scenario is
        # exclusively the Scenario Validator's job, out of scope here.

        return validate_definition(self, building=building)

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "fire": self.fire.to_dict(),
            "engineering": self.engineering.to_dict(),
            "occupant": self.occupant.to_dict(),
            "firefighter": self.firefighter.to_dict(),
            "event_templates": [template.to_dict() for template in self.event_templates],
            "seed": self.seed,
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioDefinition":

        return cls(
            fire=FireDefinition.from_dict(data["fire"]),
            engineering=EngineeringConstraints.from_dict(data.get("engineering", {})),
            occupant=OccupantDefinition.from_dict(data.get("occupant", {})),
            firefighter=FirefighterDeploymentDefinition.from_dict(data.get("firefighter", {})),
            event_templates=tuple(
                EventTemplate.from_dict(item) for item in data.get("event_templates", [])
            ),
            seed=data.get("seed"),
        )
