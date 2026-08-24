from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple


# Scenario Campaign Feasibility Preflight -- Phase 1 result model.
# docs/architecture/scenario_campaign_feasibility_preflight_investigation.txt
# Section 5/Part G: a dedicated, campaign-feasibility-scoped model, NOT
# a reuse of designer.campaign.diagnostics.DiagnosticsCollector's
# (category, code)-only shape -- that shape is post-hoc, per-candidate,
# and (per the investigation) loses exactly the zone/fire-zone pairing
# information this analysis exists to preserve. Nothing here is read by
# scenario_generator or scenario_validator, and nothing here is written
# onto ScenarioDefinition/Building -- this is its own, separate,
# campaign-level type.


class ZoneFeasibilityStatus:

    # ERROR: proven zero feasible generation space for this zone (Part
    # H, Case 1 or Case 2) -- the campaign must not start.
    # WARNING: feasible, but not structurally robust (Part H, Case 3) --
    # the campaign may proceed, the user should see why.
    # OK: reachable under the pessimistic bound and every eligible fire
    # zone is SAFE (Part H, Case 4) -- no risk found for this zone.

    ERROR = "ERROR"
    WARNING = "WARNING"
    OK = "OK"

    ALL = (ERROR, WARNING, OK)


@dataclass(frozen=True)
class ZoneFeasibilityResult:

    occupied_zone_id: str

    # Part D's two bounds. optimistic_reachable=False is, on its own,
    # already a proven zero-feasibility finding (Part C/D.1) --
    # independent of the fire dimension entirely.
    optimistic_reachable: bool
    pessimistic_reachable: bool

    # Part E's cut-vertex classification, restricted to this zone's own
    # fire-eligible ignition-zone set (never the whole Building). A
    # zone id equal to `occupied_zone_id` itself is never classified
    # into either set -- see analysis.py's own docstring for why (the
    # same exclusion FIRE_ORIGIN_BLOCKS_EVACUATION already applies).
    safe_fire_zone_ids: FrozenSet[str] = field(default_factory=frozenset)
    lethal_fire_zone_ids: FrozenSet[str] = field(default_factory=frozenset)

    # Part F -- the analytical probability that a fire zone sampled from
    # this Definition's actual sampling population (ignition_zone_
    # preference if stated, else uniform over the eligible set) is
    # LETHAL for this zone. None when it cannot be determined (no
    # eligible/preference population to sample from at all -- itself a
    # separate, pre-existing Definition-authoring problem this analysis
    # does not fabricate an answer for).
    lethal_fire_probability: Optional[float] = None

    status: str = ZoneFeasibilityStatus.OK
    explanation: str = ""

    # =====================================================

    @property
    def is_blocking(self) -> bool:

        return self.status == ZoneFeasibilityStatus.ERROR

    # =====================================================

    @property
    def is_warning(self) -> bool:

        return self.status == ZoneFeasibilityStatus.WARNING


@dataclass(frozen=True)
class CampaignFeasibilityReport:

    zone_results: Tuple[ZoneFeasibilityResult, ...] = field(default_factory=tuple)

    # =====================================================

    @property
    def has_errors(self) -> bool:

        return any(result.is_blocking for result in self.zone_results)

    # =====================================================

    @property
    def has_warnings(self) -> bool:

        return any(result.is_warning for result in self.zone_results)

    # =====================================================

    @property
    def error_results(self) -> Tuple[ZoneFeasibilityResult, ...]:

        return tuple(result for result in self.zone_results if result.is_blocking)

    # =====================================================

    @property
    def warning_results(self) -> Tuple[ZoneFeasibilityResult, ...]:

        return tuple(result for result in self.zone_results if result.is_warning)
