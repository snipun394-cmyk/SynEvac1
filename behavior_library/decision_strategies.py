import random

from simulator.decision import ActionType

from behavior.intent import ActionIntent, AlwaysEvacuateDecisionStrategy, DecisionStrategy


class AlwaysWaitDecisionStrategy(DecisionStrategy):

    # A trivial, reusable primitive -- the natural "otherwise" branch
    # for composite strategies below (e.g. ComplianceDecisionStrategy's
    # default noncompliant behavior). Mirrors AlwaysEvacuateDecisionStrategy's
    # role in behavior/intent.py, just for the opposite outcome.

    def decide(self, context) -> ActionIntent:

        return ActionIntent(
            occupant_id=context.profile.occupant_id,
            action_type=ActionType.WAIT,
            requires_movement=False,
        )


class AlwaysIgnoreDecisionStrategy(DecisionStrategy):

    # Same role as AlwaysWaitDecisionStrategy, labeled ActionType.IGNORE
    # instead of WAIT -- purely descriptive metadata either way (see
    # ActionType's own docstring: "Simulation never branches on this
    # value"), offered as an alternative "otherwise" branch for
    # composites that want to distinguish "hasn't complied yet" from
    # "will not comply".

    def decide(self, context) -> ActionIntent:

        return ActionIntent(
            occupant_id=context.profile.occupant_id,
            action_type=ActionType.IGNORE,
            requires_movement=False,
        )


class ComplianceDecisionStrategy(DecisionStrategy):

    # Compliance with evacuation instructions -- reads
    # BehaviorProfile.compliance_level (already an existing field, no
    # profile changes needed) as the probability of complying this
    # step, and delegates entirely to one of two constructor-injected
    # sub-strategies. Composable by construction: compliant_strategy/
    # noncompliant_strategy can be any DecisionStrategy, including
    # another strategy from this library (e.g. BasicHelpingDecisionStrategy
    # as the compliant branch), without ComplianceDecisionStrategy
    # itself needing to know anything about what they do.
    #
    # rng is constructor-injected (defaulting to an unseeded
    # random.Random()) so tests can supply a seeded instance for
    # deterministic, reproducible draws -- same reasoning as every
    # other Default*/Strategy default in this codebase being overridable
    # rather than hardcoded.

    def __init__(self, compliant_strategy=None, noncompliant_strategy=None, rng=None):

        self.compliant_strategy = compliant_strategy or AlwaysEvacuateDecisionStrategy()
        self.noncompliant_strategy = noncompliant_strategy or AlwaysWaitDecisionStrategy()
        self.rng = rng or random.Random()

    # =====================================================

    def decide(self, context) -> ActionIntent:

        # Prefers context.rng when supplied -- docs/architecture/
        # reproducibility_review.md §7.2: HumanBehaviorLayer.register()
        # threads a per-occupant, Scenario-seed-derived rng through
        # DecisionContext in the production path; self.rng remains the
        # fallback for direct/standalone construction (e.g. tests that
        # seed this strategy's own rng and call decide() without a full
        # DecisionContext.rng).
        rng = context.rng if context.rng is not None else self.rng

        roll = rng.random()

        if roll <= context.profile.compliance_level:
            return self.compliant_strategy.decide(context)

        return self.noncompliant_strategy.decide(context)


class BasicHelpingDecisionStrategy(DecisionStrategy):

    # "Basic" is a deliberate scope boundary: this strategy does not
    # reason about proximity, capability, capacity limits on how many
    # people one helper can assist, or mutual consent -- it is a
    # simple, explicit opt-in. The occupant's own BehaviorProfile.traits
    # carries "helping_occupant_id" (traits is documented in
    # behavior/profile.py as exactly the extension point for
    # unanticipated, caller-assigned data like this), and that alone
    # decides the outcome. A future, more sophisticated helping model
    # (proximity detection, dynamic reassessment, distress signaling
    # from the person needing help) would be a *different*
    # DecisionStrategy, not an extension of this one.

    def __init__(self, fallback=None):

        self.fallback = fallback or AlwaysEvacuateDecisionStrategy()

    # =====================================================

    def decide(self, context) -> ActionIntent:

        target_id = context.profile.traits.get("helping_occupant_id")

        if target_id is None:
            return self.fallback.decide(context)

        return ActionIntent(
            occupant_id=context.profile.occupant_id,
            action_type=ActionType.HELP,
            requires_movement=True,
            metadata={"helping_occupant_id": target_id},
        )
