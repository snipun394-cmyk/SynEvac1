from enum import Enum, auto


class OccupantCategory(Enum):

    # A coarse, human-readable classification of what a
    # behaviour_profile_id *represents* -- Human Population & Assisted
    # Evacuation Modeling Phase 1's own required minimum set (Adult,
    # Child, Elderly, Wheelchair User, Visitor), plus the two existing
    # non-civilian-count profiles this registry already carried (Staff,
    # Fire Warden) and FIREFIGHTER (a category no default civilian
    # profile ever resolves to -- see behaviour_profile_resolver.
    # firefighter_registry, kept in its own module per this phase's own
    # "keep firefighter logic separate from civilian behavior logic"
    # rule). UNKNOWN is the honest default for any profile id this
    # module was never told how to classify -- never a guessed category.
    #
    # This is the *one* place in the platform allowed to interpret what
    # a behaviour_profile_id means (behaviour_profile_resolver already
    # owns DEFAULT_PROFILE_REGISTRY, the only other place that maps an
    # id to something concrete) -- every downstream consumer that wants
    # a profile-aware count (dataset_builder, ground_truth,
    # decision_policy, ai_training) imports occupant_category() from
    # here rather than re-deriving a classification of its own from the
    # id string. rl_training does NOT import this module (see that
    # package's own tests.test_rl_training.IndependenceTests --
    # behaviour_profile_resolver is a forbidden import there); a
    # SimulationSnapshot-level bridge carries an already-classified
    # plain string instead (see simulation_interactive.state_snapshot).

    ADULT = auto()
    CHILD = auto()
    ELDERLY = auto()
    WHEELCHAIR_USER = auto()
    VISITOR = auto()
    STAFF = auto()
    FIRE_WARDEN = auto()
    FIREFIGHTER = auto()
    UNKNOWN = auto()


# Every default civilian profile id this package's own DEFAULT_PROFILE_
# REGISTRY defines, classified once, here -- restated as a lookup table
# rather than derived from the id string itself (e.g. by stripping a
# "_Default" suffix), so a custom registry's differently-named ids
# (register_occupants() accepts any Mapping[str, BehaviorProfileTemplate]
# in DEFAULT_PROFILE_REGISTRY's place) are never silently misclassified
# by accident -- an id this table has no entry for is honestly UNKNOWN,
# never guessed at from its spelling.
_CATEGORY_BY_PROFILE_ID = {
    "Adult_Default": OccupantCategory.ADULT,
    "Child_Default": OccupantCategory.CHILD,
    "Elderly_Default": OccupantCategory.ELDERLY,
    "Wheelchair_Default": OccupantCategory.WHEELCHAIR_USER,
    "Visitor_Default": OccupantCategory.VISITOR,
    "Staff_Default": OccupantCategory.STAFF,
    "FireWarden_Default": OccupantCategory.FIRE_WARDEN,
    # behaviour_profile_resolver.firefighter_registry.
    # DEFAULT_FIREFIGHTER_PROFILE_REGISTRY's own default id -- included
    # here (the one shared, cross-population classification table) even
    # though civilian and firefighter *behavior* logic stay fully
    # separate (Phase 4's own rule); classifying "what kind of person is
    # this" is a read-only lookup, not decision-making, so one shared
    # table is the correct, minimal seam, not a violation of it.
    "Firefighter_Default": OccupantCategory.FIREFIGHTER,
}


def occupant_category(behaviour_profile_id: str) -> OccupantCategory:

    return _CATEGORY_BY_PROFILE_ID.get(behaviour_profile_id, OccupantCategory.UNKNOWN)


def register_category(behaviour_profile_id: str, category: OccupantCategory) -> None:

    # The extension seam for a caller using a custom profile registry
    # (register_occupants() already accepts any
    # Mapping[str, BehaviorProfileTemplate] in place of
    # DEFAULT_PROFILE_REGISTRY) who wants their own profile ids to
    # classify correctly too -- mutates the same module-level table
    # occupant_category() reads, so it takes effect for every consumer
    # package immediately, with no registry object to thread through
    # every call site.

    _CATEGORY_BY_PROFILE_ID[behaviour_profile_id] = category
