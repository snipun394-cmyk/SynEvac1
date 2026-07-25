from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from ai_registry.training_scenario import make_training_building, make_training_definition

from predictive_dataset.dataset_builder import DEFAULT_HORIZONS
from predictive_dataset.schema import SCHEMA_VERSION


# =====================================================
# Large-Scale Predictive Dataset Campaign & Validation milestone,
# Phase 1 -- a documented, versionable campaign configuration. Building
# and ScenarioDefinition are REUSED from ai_registry.training_scenario
# (the same fixture the prior "Live-Compatible AI Model Training"
# milestone already established and validated, not a new building
# authored for this milestone) -- what is new here is making every
# distribution this campaign actually draws from EXPLICIT and
# SERIALIZABLE, per Phase 1's own "documented campaign configuration"
# requirement, rather than leaving it implicit inside a fixture
# function a reader would have to go read separately.
# =====================================================

CAMPAIGN_VERSION = "predictive_dataset_campaign_v1"


@dataclass(frozen=True)
class CampaignConfig:

    campaign_version: str
    schema_version: str

    scenario_count: int
    master_seed: int

    tick_dt_seconds: float
    horizons_seconds: Tuple[float, ...]

    building_id: str
    definition_id: str

    # A human-readable summary of every distribution this campaign's
    # ScenarioDefinition draws from -- occupant ranges, fire origin/
    # growth, blocked exits/doors, stair availability, behavior
    # profiles, hazard-adjacent (fire) configuration. Not the
    # ScenarioDefinition object itself (that's a rich, code-level
    # object with sampling behavior) -- a serializable, versionable
    # DESCRIPTION of it, suitable for embedding in a campaign report or
    # doc verbatim.
    distributions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "campaign_version": self.campaign_version,
            "schema_version": self.schema_version,
            "scenario_count": self.scenario_count,
            "master_seed": self.master_seed,
            "tick_dt_seconds": self.tick_dt_seconds,
            "horizons_seconds": list(self.horizons_seconds),
            "building_id": self.building_id,
            "definition_id": self.definition_id,
            "distributions": self.distributions,
        }


def build_campaign_config(
    scenario_count: int,
    master_seed: int,
    *,
    tick_dt_seconds: float = 5.0,
    horizons_seconds: Tuple[float, ...] = DEFAULT_HORIZONS,
) -> CampaignConfig:

    building = make_training_building()
    definition = make_training_definition()

    distributions = {
        "occupant_ranges": _describe_mapping(definition.occupant.occupancy_distribution),
        "behaviour_profile_mix": _describe_mapping(definition.occupant.behaviour_profile_distribution),
        "assistance_pairing_probability": definition.occupant.assistance_pairing_probability,
        "fire_ignition_zone_preference": _describe(definition.fire.ignition_zone_preference),
        "fire_growth_parameter_seconds": _describe(definition.fire.growth_parameter_distribution),
        "fire_profiles": sorted(definition.fire.allowed_fire_profiles),
        "blocked_door_states": _describe_mapping(definition.engineering.door_state_distribution),
        "blocked_exit_states": _describe_mapping(definition.engineering.exit_state_distribution),
        "stair_availability_states": _describe_mapping(definition.engineering.stair_state_distribution),
        "camera_availability_states": _describe_mapping(definition.engineering.camera_state_distribution),
        "firefighter_team_count": _describe(definition.firefighter.team_count_distribution),
        "firefighter_arrival_time_seconds": _describe(definition.firefighter.arrival_time_distribution),
    }

    return CampaignConfig(
        campaign_version=CAMPAIGN_VERSION,
        schema_version=SCHEMA_VERSION,
        scenario_count=scenario_count,
        master_seed=master_seed,
        tick_dt_seconds=tick_dt_seconds,
        horizons_seconds=tuple(horizons_seconds),
        building_id=building.id,
        definition_id="def-predictive-dataset-campaign-v1",
        distributions=distributions,
    )


# =====================================================
# Every scenario_definition.Distribution subclass (FixedValue/
# UniformRange/WeightedOptions) already has its own to_dict() -- reused
# directly rather than re-describing the same shape a second way.
# =====================================================


def _describe(distribution) -> Dict[str, Any]:

    if distribution is None:
        return {"kind": "none"}

    return distribution.to_dict()


def _describe_mapping(distribution_map) -> Dict[str, Any]:

    return {key: _describe(distribution) for key, distribution in sorted(distribution_map.items())}
