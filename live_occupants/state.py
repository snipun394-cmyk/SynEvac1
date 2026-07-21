from enum import Enum, auto


class OccupantStatus(Enum):

    # Live Occupant Digital Twin milestone, Phase 6 -- an occupant's
    # runtime lifecycle, entirely owned by live_occupants.manager.
    # LiveOccupantManager. Deliberately a separate module from
    # occupant.py, mirroring tracking/track_state.py's own "the state
    # enum lives apart from the model it annotates" convention.

    # First cycle this occupant_id was ever seen -- exactly one update()
    # call reports NEW; every later sighting of the same occupant_id
    # reports ACTIVE instead (see lifecycle.py's own transition rules).
    NEW = auto()

    # Currently being observed (a resolved global identity update
    # arrived for this occupant_id this cycle).
    ACTIVE = auto()

    # Not observed this cycle, but the underlying identity has not been
    # gone long enough to count as expired yet -- still coasting,
    # exactly the same "not deleted immediately" grace period
    # cross_camera_identity.identity_registry.IdentityRegistry.release()
    # already gives a departed local track at the tracking layer,
    # applied one layer up here.
    TEMPORARILY_LOST = auto()

    # A DELIBERATE, DISTINCT alternative to TEMPORARILY_LOST: the
    # occupant's last known world_position was within
    # LiveOccupantManager's own exit_proximity_threshold of a modeled
    # models.exit.Exit segment at the moment they stopped being
    # observed -- an honest, geometry-based guess ("likely left through
    # this exit"), not a certainty. Recoverable: a later update() for
    # the same occupant_id (the guess was wrong -- they did not
    # actually leave) returns it to ACTIVE, exactly like
    # TEMPORARILY_LOST would.
    EXITED = auto()

    # Terminal. The underlying identity has been gone longer than
    # LiveOccupantManager's own expire_after_seconds -- configure this
    # to match (or exceed) the CrossCameraIdentityResolver's own
    # TransitionModel.timeout_seconds (Phase 6's "respect CrossCameraIdentity
    # timeout policy" -- a configuration convention, not a live cross-
    # package query, keeping this package's own dependency surface
    # minimal). Once expired, the occupant is removed from the
    # manager's active store entirely (see LiveOccupantManager.
    # sweep_missing()) -- this status value is only ever observed
    # transiently, in the OccupantExpired event's own payload.
    EXPIRED = auto()
