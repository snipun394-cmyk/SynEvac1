import hashlib
import random

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from behaviour_profile_resolver.category import OccupantCategory, occupant_category


# =====================================================
# Occupant Attributes -- a deterministic, per-occupant realism layer
# ON TOP OF (never replacing) Behaviour Profiles. Behaviour Profiles
# remain the high-level category (Adult_Default, Child_Default, ...);
# this module answers "which specific adult" -- a continuous, unique
# spread of physical/behavioral values within that category's own
# realistic range.
#
# Deliberately NOT stored on Scenario/ScenarioOccupant. scenario/
# occupant.py's own docstring and docs/architecture/scenario_engine.md
# §8 are explicit: a Scenario's occupant record carries exactly one
# behavior-related field (behaviour_profile_id, opaque), and "no
# walking_speed, panic, ... field exists anywhere... and none is added
# by this revision either" -- enforced by tests.test_behaviour_profile_
# resolver.test_scenario_runner_never_imports_this_package. Instead,
# every value here is *derived on demand*, deterministically, from
# (scenario_seed, occupant_id, behaviour_profile_id) -- the same
# sha256(f"{seed}|{namespace}|{occupant_id}") convention already
# established twice in this codebase: behaviour_profile_resolver.
# registrar._derive_occupant_seed() and ground_truth.human_behavior.
# _derive_behavior_seed(). Any caller holding the same three inputs
# (behaviour_profile_resolver at registration time, ground_truth at
# outcome-summary time, dataset_builder at feature-export time) derives
# the identical value independently -- no shared mutable state, no new
# coordination needed.
#
# Ranges are illustrative, documented starting points (same "disclosed
# simplification, not a validated human-factors model" honesty
# DEFAULT_PROFILE_REGISTRY's own walking speeds/compliance levels
# already carry) -- not measured real-world data.
# =====================================================


@dataclass(frozen=True)
class OccupantAttributes:

    # Phase 1 -- Physical. walking_speed_multiplier/stamina/fatigue_
    # resistance/mobility_factor are expressed as a [0, ~1.5] scale
    # where 1.0 is an unremarkable adult baseline; the rest are [0, 1]
    # capability/tolerance scores.
    walking_speed_multiplier: float
    reaction_speed: float
    stamina: float
    smoke_tolerance: float
    visibility_tolerance: float
    fatigue_resistance: float
    mobility_factor: float

    # Phase 2 -- Behavioral, each a [0, 1] level/probability.
    leadership: float
    risk_aversion: float
    route_familiarity: float
    compliance: float
    helping_likelihood: float
    panic_susceptibility: float
    crowd_following_tendency: float

    # =====================================================

    def to_dict(self) -> Dict[str, float]:

        return {
            "walking_speed_multiplier": self.walking_speed_multiplier,
            "reaction_speed": self.reaction_speed,
            "stamina": self.stamina,
            "smoke_tolerance": self.smoke_tolerance,
            "visibility_tolerance": self.visibility_tolerance,
            "fatigue_resistance": self.fatigue_resistance,
            "mobility_factor": self.mobility_factor,
            "leadership": self.leadership,
            "risk_aversion": self.risk_aversion,
            "route_familiarity": self.route_familiarity,
            "compliance": self.compliance,
            "helping_likelihood": self.helping_likelihood,
            "panic_susceptibility": self.panic_susceptibility,
            "crowd_following_tendency": self.crowd_following_tendency,
        }


# =====================================================
# Category-conditioned ranges -- (low, high) per attribute, sampled
# uniformly. Encodes the brief's own worked examples directly (Child:
# faster but less hazard-aware; Elderly: slower with more hesitation;
# Wheelchair: slower, near-zero mobility_factor -- see that field's own
# docstring below for why "cannot use stairs" stops at being disclosed
# data rather than an enforced pathfinding restriction; Firefighter:
# faster, most smoke/visibility tolerant). UNKNOWN (a profile id this
# platform was never told how to classify) falls back to ADULT's
# ranges -- the same honest "closest neutral default, never a guess
# beyond that" convention occupant_category() itself already documents.
# =====================================================

_ADULT_RANGES: Dict[str, Tuple[float, float]] = {
    "walking_speed_multiplier": (0.85, 1.15),
    "reaction_speed": (0.50, 0.90),
    "stamina": (0.50, 0.90),
    "smoke_tolerance": (0.30, 0.60),
    "visibility_tolerance": (0.35, 0.65),
    "fatigue_resistance": (0.45, 0.85),
    "mobility_factor": (0.85, 1.00),
    "leadership": (0.30, 0.60),
    "risk_aversion": (0.40, 0.70),
    "route_familiarity": (0.30, 0.60),
    "compliance": (0.60, 0.90),
    "helping_likelihood": (0.30, 0.60),
    "panic_susceptibility": (0.25, 0.55),
    "crowd_following_tendency": (0.30, 0.60),
}

_RANGES_BY_CATEGORY: Dict[OccupantCategory, Dict[str, Tuple[float, float]]] = {
    OccupantCategory.ADULT: _ADULT_RANGES,
    OccupantCategory.CHILD: {
        "walking_speed_multiplier": (1.05, 1.35),
        "reaction_speed": (0.40, 0.80),
        "stamina": (0.40, 0.75),
        "smoke_tolerance": (0.15, 0.40),
        "visibility_tolerance": (0.20, 0.45),
        "fatigue_resistance": (0.35, 0.70),
        "mobility_factor": (0.85, 1.00),
        "leadership": (0.05, 0.25),
        "risk_aversion": (0.15, 0.40),
        "route_familiarity": (0.10, 0.35),
        "compliance": (0.30, 0.60),
        "helping_likelihood": (0.10, 0.30),
        "panic_susceptibility": (0.45, 0.80),
        "crowd_following_tendency": (0.55, 0.85),
    },
    OccupantCategory.ELDERLY: {
        "walking_speed_multiplier": (0.55, 0.80),
        "reaction_speed": (0.20, 0.50),
        "stamina": (0.20, 0.50),
        "smoke_tolerance": (0.15, 0.40),
        "visibility_tolerance": (0.20, 0.45),
        "fatigue_resistance": (0.15, 0.45),
        "mobility_factor": (0.50, 0.80),
        "leadership": (0.25, 0.55),
        "risk_aversion": (0.55, 0.85),
        "route_familiarity": (0.25, 0.50),
        "compliance": (0.65, 0.90),
        "helping_likelihood": (0.20, 0.45),
        "panic_susceptibility": (0.35, 0.65),
        "crowd_following_tendency": (0.35, 0.65),
    },
    OccupantCategory.WHEELCHAIR_USER: {
        "walking_speed_multiplier": (0.45, 0.70),
        "reaction_speed": (0.30, 0.60),
        "stamina": (0.25, 0.55),
        "smoke_tolerance": (0.20, 0.45),
        "visibility_tolerance": (0.25, 0.50),
        "fatigue_resistance": (0.20, 0.50),
        # Near-zero, deliberately -- the disclosed signal for "cannot
        # use stairs." This module only ever generates and exposes the
        # value; it does not itself restrict routing (pathfinding/
        # navigation are untouched by this feature -- see this
        # package's own plan/report for that explicit boundary).
        "mobility_factor": (0.0, 0.15),
        "leadership": (0.20, 0.50),
        "risk_aversion": (0.50, 0.80),
        "route_familiarity": (0.20, 0.45),
        "compliance": (0.65, 0.90),
        "helping_likelihood": (0.20, 0.45),
        "panic_susceptibility": (0.35, 0.65),
        "crowd_following_tendency": (0.30, 0.60),
    },
    OccupantCategory.VISITOR: {
        "walking_speed_multiplier": (0.85, 1.15),
        "reaction_speed": (0.40, 0.75),
        "stamina": (0.40, 0.80),
        "smoke_tolerance": (0.25, 0.55),
        "visibility_tolerance": (0.30, 0.60),
        "fatigue_resistance": (0.35, 0.75),
        "mobility_factor": (0.80, 1.00),
        "leadership": (0.15, 0.40),
        "risk_aversion": (0.35, 0.65),
        # Visitors are unfamiliar with the building by definition.
        "route_familiarity": (0.05, 0.25),
        "compliance": (0.40, 0.70),
        "helping_likelihood": (0.15, 0.40),
        "panic_susceptibility": (0.35, 0.65),
        "crowd_following_tendency": (0.50, 0.80),
    },
    OccupantCategory.STAFF: {
        "walking_speed_multiplier": (0.95, 1.25),
        "reaction_speed": (0.60, 0.95),
        "stamina": (0.55, 0.90),
        "smoke_tolerance": (0.35, 0.65),
        "visibility_tolerance": (0.40, 0.70),
        "fatigue_resistance": (0.50, 0.85),
        "mobility_factor": (0.85, 1.00),
        "leadership": (0.50, 0.80),
        "risk_aversion": (0.45, 0.75),
        "route_familiarity": (0.70, 0.95),
        "compliance": (0.85, 1.00),
        "helping_likelihood": (0.50, 0.80),
        "panic_susceptibility": (0.20, 0.45),
        "crowd_following_tendency": (0.20, 0.45),
    },
    OccupantCategory.FIRE_WARDEN: {
        "walking_speed_multiplier": (0.95, 1.25),
        "reaction_speed": (0.65, 0.95),
        "stamina": (0.60, 0.90),
        "smoke_tolerance": (0.40, 0.70),
        "visibility_tolerance": (0.45, 0.75),
        "fatigue_resistance": (0.55, 0.85),
        "mobility_factor": (0.85, 1.00),
        "leadership": (0.70, 0.95),
        "risk_aversion": (0.40, 0.70),
        "route_familiarity": (0.75, 0.95),
        "compliance": (0.85, 1.00),
        "helping_likelihood": (0.70, 0.95),
        "panic_susceptibility": (0.10, 0.30),
        "crowd_following_tendency": (0.15, 0.40),
    },
    OccupantCategory.FIREFIGHTER: {
        "walking_speed_multiplier": (1.15, 1.45),
        "reaction_speed": (0.75, 1.00),
        "stamina": (0.75, 1.00),
        "smoke_tolerance": (0.75, 1.00),
        "visibility_tolerance": (0.75, 1.00),
        "fatigue_resistance": (0.70, 1.00),
        "mobility_factor": (0.90, 1.00),
        "leadership": (0.75, 1.00),
        "risk_aversion": (0.20, 0.50),
        "route_familiarity": (0.30, 0.60),
        "compliance": (0.85, 1.00),
        "helping_likelihood": (0.85, 1.00),
        "panic_susceptibility": (0.05, 0.20),
        "crowd_following_tendency": (0.10, 0.30),
    },
}


def _ranges_for(category: OccupantCategory) -> Dict[str, Tuple[float, float]]:

    return _RANGES_BY_CATEGORY.get(category, _ADULT_RANGES)


# =====================================================


def _derive_attribute_seed(scenario_seed: int, occupant_id: str) -> int:

    # Same sha256-of-a-stable-key convention behaviour_profile_resolver.
    # registrar._derive_occupant_seed()/ground_truth.human_behavior.
    # _derive_behavior_seed() already establish, restated here (not
    # imported) for the same "kept independent per package" reason
    # every other seed-derivation site in this codebase already
    # restates rather than imports one shared helper.

    key = f"{scenario_seed}|behaviour_profile_resolver.occupant_attributes|{occupant_id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big")


def derive_occupant_attributes(
    scenario_seed: int, occupant_id: str, behaviour_profile_id: str,
) -> OccupantAttributes:

    # Pure function: the same three inputs always produce the same
    # OccupantAttributes, regardless of call order, which package calls
    # it, or how many other occupants exist in this run -- the same
    # index-keyed-not-order-keyed guarantee scenario_generator.seed_
    # manager.category_rng() documents for its own, independent seed
    # hierarchy (this module's is a separate, parallel one -- it never
    # reads or perturbs scenario_generator's own RNG streams at all).

    rng = random.Random(_derive_attribute_seed(scenario_seed, occupant_id))
    ranges = _ranges_for(occupant_category(behaviour_profile_id))

    return OccupantAttributes(
        walking_speed_multiplier=rng.uniform(*ranges["walking_speed_multiplier"]),
        reaction_speed=rng.uniform(*ranges["reaction_speed"]),
        stamina=rng.uniform(*ranges["stamina"]),
        smoke_tolerance=rng.uniform(*ranges["smoke_tolerance"]),
        visibility_tolerance=rng.uniform(*ranges["visibility_tolerance"]),
        fatigue_resistance=rng.uniform(*ranges["fatigue_resistance"]),
        mobility_factor=rng.uniform(*ranges["mobility_factor"]),
        leadership=rng.uniform(*ranges["leadership"]),
        risk_aversion=rng.uniform(*ranges["risk_aversion"]),
        route_familiarity=rng.uniform(*ranges["route_familiarity"]),
        compliance=rng.uniform(*ranges["compliance"]),
        helping_likelihood=rng.uniform(*ranges["helping_likelihood"]),
        panic_susceptibility=rng.uniform(*ranges["panic_susceptibility"]),
        crowd_following_tendency=rng.uniform(*ranges["crowd_following_tendency"]),
    )


# =====================================================


def attribute_traits(attributes: OccupantAttributes) -> Dict[str, Any]:

    # BehaviorProfile.traits-ready mapping -- every raw attribute value,
    # plus one derived key genuinely read by behavior_library.
    # attribute_aware_strategies.AttributeAwarePreMovementDelayStrategy
    # (see that module's own docstring): faster reaction shortens the
    # sampled pre-movement delay, higher panic susceptibility lengthens
    # it. Clamped to a sane positive floor so a strategy multiplying by
    # this can never zero out or invert a delay.

    traits = dict(attributes.to_dict())

    multiplier = (1.4 - 0.8 * attributes.reaction_speed) * (0.7 + 0.6 * attributes.panic_susceptibility)
    traits["pre_movement_delay_multiplier"] = max(0.2, multiplier)

    return traits
