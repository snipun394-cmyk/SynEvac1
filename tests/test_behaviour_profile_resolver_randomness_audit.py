import random
import unittest
from dataclasses import replace

from behavior_library.decision_strategies import AlwaysWaitDecisionStrategy, ComplianceDecisionStrategy

from behaviour_profile_resolver.randomness_audit import CONTEXT_RNG_SAFE_STRATEGY_CLASSES, audit_registry
from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY


class _UnsafeDecisionStrategy:

    # A deliberately broken strategy: holds constructor-level rng state
    # (same shape as ComplianceDecisionStrategy) but never looks at
    # context.rng in decide() -- exactly the class of regression
    # audit_registry() exists to catch.

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def decide(self, context):
        return self.rng.random()


class AuditRegistryOnDefaultRegistryTests(unittest.TestCase):

    def test_default_registry_is_fully_controlled(self):

        report = audit_registry(DEFAULT_PROFILE_REGISTRY)

        self.assertTrue(report.fully_controlled)
        self.assertEqual(report.uncontrolled, ())

    def test_default_registry_actually_found_real_randomness_sources(self):

        # A vacuous "nothing uncontrolled because nothing was even
        # looked at" pass would be worthless -- confirm the audit really
        # walked the registry and found the known, real
        # ComplianceDecisionStrategy/ProbabilisticPreMovementDelay
        # instances DEFAULT_PROFILE_REGISTRY actually contains.

        report = audit_registry(DEFAULT_PROFILE_REGISTRY)

        self.assertGreater(len(report.controlled), 0)
        self.assertTrue(any("ComplianceDecisionStrategy" in entry for entry in report.controlled))
        self.assertTrue(any("ProbabilisticPreMovementDelay" in entry for entry in report.controlled))

    def test_deterministic_only_profile_contributes_no_entries(self):

        # Staff_Default: AlwaysEvacuateDecisionStrategy + NoPreMovementDelay
        # + ShortestRouteChoiceStrategy -- none hold rng state at all.

        report = audit_registry(DEFAULT_PROFILE_REGISTRY)

        self.assertFalse(any(entry.startswith("Staff_Default.") for entry in report.controlled))
        self.assertFalse(any(entry.startswith("Staff_Default.") for entry in report.uncontrolled))

    def test_nested_noncompliant_strategy_does_not_produce_a_false_uncontrolled_entry(self):

        # Child_Default's decision_strategy is ComplianceDecisionStrategy(
        # noncompliant_strategy=AlwaysWaitDecisionStrategy()) -- the walk
        # must recurse into noncompliant_strategy, find
        # AlwaysWaitDecisionStrategy has no .rng, and correctly emit
        # nothing for it (not a false "uncontrolled").

        report = audit_registry(DEFAULT_PROFILE_REGISTRY)

        child_entries = [entry for entry in report.uncontrolled if entry.startswith("Child_Default.")]
        self.assertEqual(child_entries, [])

        self.assertTrue(any(
            entry == "Child_Default.decision_strategy.ComplianceDecisionStrategy"
            for entry in report.controlled
        ))


class AuditRegistryDetectsUnsafeStrategyTests(unittest.TestCase):

    def test_unrecognised_rng_bearing_strategy_is_flagged_uncontrolled(self):

        broken_registry = dict(DEFAULT_PROFILE_REGISTRY)
        broken_registry["Adult_Default"] = replace(
            broken_registry["Adult_Default"],
            decision_strategy=_UnsafeDecisionStrategy(),
        )

        report = audit_registry(broken_registry)

        self.assertFalse(report.fully_controlled)
        self.assertTrue(any(
            entry == "Adult_Default.decision_strategy._UnsafeDecisionStrategy"
            for entry in report.uncontrolled
        ))

    def test_one_unsafe_profile_does_not_hide_other_profiles_own_controlled_entries(self):

        broken_registry = dict(DEFAULT_PROFILE_REGISTRY)
        broken_registry["Adult_Default"] = replace(
            broken_registry["Adult_Default"],
            decision_strategy=_UnsafeDecisionStrategy(),
        )

        report = audit_registry(broken_registry)

        # Child_Default's own ComplianceDecisionStrategy is untouched by
        # this edit and must still be reported as controlled.
        self.assertTrue(any(
            entry == "Child_Default.decision_strategy.ComplianceDecisionStrategy"
            for entry in report.controlled
        ))


class SafeStrategyClassesInventoryTests(unittest.TestCase):

    def test_safe_set_matches_the_verified_three_class_inventory(self):

        # Locks the allow-list to exactly the three classes this
        # milestone's own investigation verified prefer context.rng --
        # a change to this set should be a deliberate, reviewed decision,
        # never an accidental one.

        self.assertEqual(len(CONTEXT_RNG_SAFE_STRATEGY_CLASSES), 3)
        self.assertIn(ComplianceDecisionStrategy, CONTEXT_RNG_SAFE_STRATEGY_CLASSES)
        self.assertNotIn(AlwaysWaitDecisionStrategy, CONTEXT_RNG_SAFE_STRATEGY_CLASSES)


if __name__ == "__main__":
    unittest.main()
