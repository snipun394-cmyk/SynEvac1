from dataclasses import dataclass, field
from typing import Tuple

from fire_safety_manager.status import FireSafetyAssetStatus


@dataclass(frozen=True)
class SprinklerCounts:

    # Sprinkler's own three-state breakdown (SprinklerActivationState) --
    # kept separate from PassiveAssetCounts below because "activated" is
    # not the same concept as "unavailable" (see models.sprinkler.
    # SprinklerActivationState's own docstring on why Sprinkler never
    # shares vocabulary with the other three).

    total: int = 0
    normal: int = 0
    activated: int = 0
    fault: int = 0


@dataclass(frozen=True)
class PassiveAssetCounts:

    # Shared breakdown shape for FireExtinguisher/FireHydrant/HoseReel
    # (PassiveFireSafetyAvailability's AVAILABLE/UNAVAILABLE/FAULT) --
    # one dataclass, reused three times on FireSafetyStatusSnapshot
    # below, exactly mirroring why models.fire_safety_asset shares one
    # compute_passive_availability() across the same three siblings.

    total: int = 0
    available: int = 0
    unavailable: int = 0
    fault: int = 0


@dataclass(frozen=True)
class FireSafetyStatusSnapshot:

    # BuildingState's additive view over FireSafetyAssetManager (see
    # that class's own snapshot() method, which produces this value --
    # BuildingStateEstimator only ever passes it through, never
    # computes it, the same discipline facp_status/control_status
    # already establish). None on BuildingState when no
    # FireSafetyAssetManager is configured for this cycle -- never
    # fabricated. entries is the full per-asset detail (Command Center's
    # own table reads this); the four count fields are the Phase 11
    # roll-up an operator-facing summary needs without re-deriving it
    # from entries every time.

    entries: Tuple[FireSafetyAssetStatus, ...] = field(default_factory=tuple)

    sprinklers: SprinklerCounts = field(default_factory=SprinklerCounts)
    fire_extinguishers: PassiveAssetCounts = field(default_factory=PassiveAssetCounts)
    fire_hydrants: PassiveAssetCounts = field(default_factory=PassiveAssetCounts)
    hose_reels: PassiveAssetCounts = field(default_factory=PassiveAssetCounts)
