from behaviour_profile_resolver.template import BehaviorProfileTemplate


class UnknownBehaviourProfileError(LookupError):

    # Raised, never repaired -- architecture doc
    # docs/architecture/scenario_runner.md §7, point 3: "An
    # unrecognized behaviour_profile_id ... is a hard error at this
    # stage -- no repair, no invented default, mirroring the 'no
    # repair, anywhere' principle already frozen throughout the
    # Scenario Engine." A LookupError subclass (not a bare KeyError)
    # so callers can catch specifically "this behaviour profile id
    # isn't registered" without also catching an unrelated KeyError
    # bug elsewhere.

    pass


def resolve_profile(behaviour_profile_id: str, registry) -> BehaviorProfileTemplate:

    # The one place a behaviour_profile_id string is ever interpreted
    # -- a pure lookup, never a decision. `registry` is a plain
    # Mapping[str, BehaviorProfileTemplate]; this function has no
    # opinion about where it came from (registry.py's
    # DEFAULT_PROFILE_REGISTRY, or any caller-supplied alternative --
    # registrar.py's register_occupants() accepts either).

    try:
        return registry[behaviour_profile_id]

    except KeyError:

        raise UnknownBehaviourProfileError(
            f"No behaviour profile is registered for id {behaviour_profile_id!r}."
        ) from None
