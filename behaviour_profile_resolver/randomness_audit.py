from dataclasses import dataclass
from typing import Iterator, Mapping, Set, Tuple

from behavior_library.decision_strategies import ComplianceDecisionStrategy
from behavior_library.pre_movement_strategies import ProbabilisticPreMovementDelay
from behavior_library.route_choice_strategies import StaticHerdingRouteChoiceStrategy

from behaviour_profile_resolver.template import BehaviorProfileTemplate


# =====================================================
# Calibration Studio Phase 0 -- Deterministic Experiment Infrastructure.
#
# This module does NOT introduce seeded randomness -- that already
# exists, comprehensively, predating this milestone:
# behaviour_profile_resolver.registrar._derive_occupant_seed() derives a
# per-occupant seed from (scenario.metadata.seed, occupant_id), and
# every strategy capable of drawing randomness
# (ComplianceDecisionStrategy, ProbabilisticPreMovementDelay,
# StaticHerdingRouteChoiceStrategy, plus behavior_library.
# attribute_aware_strategies's two inline-rng wrapper strategies, which
# never hold registry-swappable state and are out of this module's
# scope -- see below) already prefers context.rng over its own
# constructor-injected self.rng fallback when one is supplied. This is
# exhaustively covered by tests/test_reproducibility_fix.py.
#
# What was genuinely missing -- and what this module builds -- is
# EVIDENCE: a way to inspect a concrete
# Mapping[str, BehaviorProfileTemplate] (a production registry, or a
# ParameterCandidate's own baseline_registry()/candidate_registry())
# and answer, mechanically rather than by assumption, "does every
# randomness-consuming strategy actually reachable from this registry
# belong to the known, verified-context.rng-preferring set, or does
# something unrecognised -- a future custom ParameterCandidate, a
# hand-authored registry, a regression -- introduce randomness this
# audit cannot vouch for." A future CalibrationSession.reproducible
# field is meant to read exactly this, never to assume it.
#
# Deliberately excludes AttributeAwareComplianceDecisionStrategy and
# CrowdFollowingAwareRouteChoiceStrategy (behavior_library.
# attribute_aware_strategies): both compute `context.rng if context.rng
# is not None else random.Random()` inline, inside decide()/choose(),
# rather than storing an rng on self -- confirmed by direct source
# inspection they are never constructed as part of a
# BehaviorProfileTemplate (registrar.py wraps every template's own
# decision_strategy/route_choice_strategy in these at REGISTRATION
# time, identically regardless of which registry supplied the
# template), so no registry this audit could ever be asked to check
# can vary whether that wrapping happens or how it behaves. Auditing a
# registry's own templates is therefore the right, sufficient scope --
# not a partial one.
# =====================================================


CONTEXT_RNG_SAFE_STRATEGY_CLASSES = frozenset({
    ComplianceDecisionStrategy,
    ProbabilisticPreMovementDelay,
    StaticHerdingRouteChoiceStrategy,
})

_STRATEGY_METHOD_NAMES = ("decide", "choose", "delay")


def _is_strategy_like(value: object) -> bool:

    return any(hasattr(value, name) for name in _STRATEGY_METHOD_NAMES)


def _walk_strategy(root: object, seen: Set[int]) -> Iterator[object]:

    # Generic composition walk (no hardcoded attribute name -- existing
    # wrapper strategies in this codebase use different names for their
    # own inner strategy: `fallback`, `noncompliant_strategy`,
    # `compliant_strategy`) so a future wrapper needs no change here.
    # `seen` guards against a strategy that (unusually) refers back to
    # itself; every existing strategy in this codebase is a simple tree,
    # not a cycle, but nothing enforces that structurally.

    if id(root) in seen or not _is_strategy_like(root):
        return

    seen.add(id(root))
    yield root

    for value in vars(root).values():
        if _is_strategy_like(value):
            yield from _walk_strategy(value, seen)


@dataclass(frozen=True)
class RandomnessControlReport:

    # controlled/uncontrolled entries are "{profile_id}.{template_field}.
    # {ClassName}" labels, not just class names -- so two different
    # profiles' own instances of the same unsafe class are both visible
    # individually, not collapsed into one ambiguous entry.

    controlled: Tuple[str, ...]
    uncontrolled: Tuple[str, ...]

    @property
    def fully_controlled(self) -> bool:

        # Derived, not stored -- never independently settable, so it
        # can never silently drift from what `uncontrolled` actually
        # contains (the exact drift risk already documented elsewhere
        # in this codebase for other hand-maintained derived fields).

        return len(self.uncontrolled) == 0

    def to_dict(self) -> dict:

        return {
            "controlled": list(self.controlled),
            "uncontrolled": list(self.uncontrolled),
            "fully_controlled": self.fully_controlled,
        }


_TEMPLATE_STRATEGY_FIELDS = ("decision_strategy", "route_choice_strategy", "pre_movement_strategy")


def audit_registry(registry: Mapping[str, BehaviorProfileTemplate]) -> RandomnessControlReport:

    # Read-only inspection of already-constructed strategy objects --
    # this function never constructs, seeds, or mutates anything, and
    # therefore cannot itself change any simulation's behaviour. Safe to
    # call unconditionally, on every calibration run, with zero risk to
    # the "do not change default behaviour for production simulations"
    # requirement this milestone was given.

    controlled = []
    uncontrolled = []

    for profile_id, template in registry.items():
        for field_name in _TEMPLATE_STRATEGY_FIELDS:

            root = getattr(template, field_name, None)

            if root is None:
                continue

            for strategy in _walk_strategy(root, set()):

                if not hasattr(strategy, "rng"):
                    # Doesn't hold constructor-level rng state at all --
                    # either draws no randomness (AlwaysEvacuateDecisionStrategy,
                    # ShortestRouteChoiceStrategy, NoPreMovementDelay, ...)
                    # or is one of the two inline-rng wrapper strategies
                    # this module's own docstring excludes by design.
                    continue

                label = f"{profile_id}.{field_name}.{type(strategy).__name__}"

                if type(strategy) in CONTEXT_RNG_SAFE_STRATEGY_CLASSES:
                    controlled.append(label)
                else:
                    uncontrolled.append(label)

    return RandomnessControlReport(controlled=tuple(controlled), uncontrolled=tuple(uncontrolled))
