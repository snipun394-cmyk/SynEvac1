import random
from typing import Any, FrozenSet

from scenario_definition.distributions import Distribution, FixedValue, UniformRange, WeightedOptions


# The Distribution-sampling interpreter -- architecture doc §4 ("for
# every field the Definition declares, it runs that field's
# Distribution descriptor through its own sampling interpreter... and
# assigns the result -- nothing more") and §4.7 (the one fixed default
# policy: an allow-list/FrozenSet field with no paired preference
# Distribution is sampled uniformly at random over its members).
#
# This is the *only* place in the whole Scenario Engine that knows how
# to turn a Distribution plus an rng into a concrete value --
# scenario_definition/ deliberately has no sample() method anywhere
# (§3.1 finding 1), and this module is exactly the "Generator-side
# concern" that finding deferred to.


def sample(distribution: Distribution, rng: random.Random) -> Any:

    if isinstance(distribution, FixedValue):
        return distribution.value

    if isinstance(distribution, UniformRange):

        if distribution.discrete:
            return rng.randint(int(distribution.low), int(distribution.high))

        return rng.uniform(distribution.low, distribution.high)

    if isinstance(distribution, WeightedOptions):

        # dict iteration order is insertion-order-stable in Python (a
        # language guarantee, unaffected by hash randomization) -- safe
        # to use directly, unlike a set/frozenset (see
        # sample_uniform_choice below).
        options = list(distribution.weights.keys())
        weights = list(distribution.weights.values())

        return rng.choices(options, weights=weights, k=1)[0]

    raise TypeError(f"Unknown Distribution kind: {type(distribution)!r}")


def sample_uniform_choice(allowed: FrozenSet[Any], rng: random.Random) -> Any:

    # §4.7's default policy: a field expressed purely as an allow-list
    # with no paired preference Distribution is sampled uniformly at
    # random over its members. `allowed` is typically a FrozenSet of
    # strings (zone ids, fire profile labels, ...) -- frozenset
    # iteration order is hash-bucket-dependent and, for strings, varies
    # across separate process runs under Python's default hash
    # randomization. sorted() first guarantees a canonical, stable
    # ordering before rng.choice() indexes into it, so the *value*
    # chosen for a given rng state never depends on which process ran
    # it -- the exact same reproducibility property §4.6 requires of
    # the seed hierarchy would otherwise be silently undermined here.

    if not allowed:
        raise ValueError(
            "sample_uniform_choice() received an empty allowed set -- there is "
            "nothing to sample from. This is a Definition-authoring problem the "
            "Generator does not repair; the (future) Scenario Validator is where an "
            "infeasible Definition is expected to surface, not here."
        )

    ordered = sorted(allowed)

    return rng.choice(ordered)
