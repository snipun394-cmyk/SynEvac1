from dataclasses import dataclass, field
from typing import Tuple

from scenario_definition.distributions import Distribution, FixedValue, distribution_from_dict


@dataclass(frozen=True)
class FirefighterDeploymentDefinition:

    # Human Population & Assisted Evacuation Modeling Phase 5 -- pure
    # configuration for how many firefighter teams may be sampled, of
    # what size, arriving when, from which entry points -- the exact
    # same "declares what may be sampled, samples nothing itself" role
    # OccupantDefinition already plays for civilians (scenario_
    # definition/occupant_definition.py), extended to a population this
    # platform generates entirely independently of it (Phase 4's own
    # "kept separate" rule, restated at generation-config time too).
    #
    # Every field defaults to "deploy nothing" (team_count_distribution
    # = FixedValue(0), entry_zone_ids = ()) so a Definition that never
    # mentions firefighters at all -- every existing Definition, before
    # this phase -- generates none, exactly its pre-existing behavior.
    #
    # behaviour_profile_id is the same opaque-identifier-string
    # convention OccupantDefinition.behaviour_profile_distribution's
    # own values already are -- resolved only by
    # behaviour_profile_resolver.firefighter_registrar, at simulation
    # time, never here (§8's "never resolves, interprets, or
    # validates" rule, restated for the firefighter registry).

    team_count_distribution: Distribution = field(default_factory=lambda: FixedValue(0))
    team_size_distribution: Distribution = field(default_factory=lambda: FixedValue(2))
    arrival_time_distribution: Distribution = field(default_factory=lambda: FixedValue(60.0))

    # Candidate entry zones a team may be assigned to (uniform choice
    # among these) -- empty means "no legal entry point stated," the
    # same "nothing to sample from" meaning an empty occupancy_
    # distribution/behaviour_profile_distribution mapping already
    # carries in OccupantDefinition.
    entry_zone_ids: Tuple[str, ...] = field(default_factory=tuple)

    behaviour_profile_id: str = "Firefighter_Default"

    # Probability that a *deployed* team is assigned a specific rescue
    # target (an unreachable-or-stationary civilian, chosen by the
    # Generator among that run's own sampled occupants) rather than
    # searching with no fixed target -- Phase 4's own "searching rooms"
    # vs. "rescuing trapped occupants" split, expressed as one
    # configurable rate rather than a separate distribution shape,
    # since there are only the two outcomes.
    rescue_assignment_probability: float = 0.5

    # =====================================================

    def __post_init__(self):

        if not (0.0 <= self.rescue_assignment_probability <= 1.0):

            raise ValueError(
                f"FirefighterDeploymentDefinition.rescue_assignment_probability must be in "
                f"[0.0, 1.0], got {self.rescue_assignment_probability!r}."
            )

        object.__setattr__(self, "entry_zone_ids", tuple(self.entry_zone_ids))

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "team_count_distribution": self.team_count_distribution.to_dict(),
            "team_size_distribution": self.team_size_distribution.to_dict(),
            "arrival_time_distribution": self.arrival_time_distribution.to_dict(),
            "entry_zone_ids": list(self.entry_zone_ids),
            "behaviour_profile_id": self.behaviour_profile_id,
            "rescue_assignment_probability": self.rescue_assignment_probability,
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "FirefighterDeploymentDefinition":

        return cls(
            team_count_distribution=(
                distribution_from_dict(data["team_count_distribution"])
                if "team_count_distribution" in data else FixedValue(0)
            ),
            team_size_distribution=(
                distribution_from_dict(data["team_size_distribution"])
                if "team_size_distribution" in data else FixedValue(2)
            ),
            arrival_time_distribution=(
                distribution_from_dict(data["arrival_time_distribution"])
                if "arrival_time_distribution" in data else FixedValue(60.0)
            ),
            entry_zone_ids=tuple(data.get("entry_zone_ids", ())),
            behaviour_profile_id=data.get("behaviour_profile_id", "Firefighter_Default"),
            rescue_assignment_probability=data.get("rescue_assignment_probability", 0.5),
        )
