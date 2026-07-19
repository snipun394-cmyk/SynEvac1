from scenario import (
    DeviceAvailability,
    DoorState,
    PresenceState,
    Scenario,
    ScenarioCameraState,
    ScenarioDetectorState,
    ScenarioDoorState,
    ScenarioEvent,
    ScenarioExitState,
    ScenarioFire,
    ScenarioFirefighter,
    ScenarioObstacleState,
    ScenarioOccupant,
    ScenarioStairState,
    StairAvailability,
)

from behaviour_profile_resolver.category import OccupantCategory, occupant_category

from scenario_generator.metadata_builder import build_metadata
from scenario_generator.request import GenerationRequest
from scenario_generator.sampling import sample, sample_uniform_choice
from scenario_generator.seed_manager import category_rng, derive_attempt_seed


# The construction module's single entry point -- architecture doc §4:
# a fully generic, deterministic function of (ScenarioDefinition,
# Building, seed) that samples every category's fields through
# sampling.sample()/sample_uniform_choice() and assigns the result,
# nothing more. No accept/reject branch exists anywhere in this module
# (§4.1: "not a validator") -- generate_scenario() always returns a
# candidate Scenario; whether that candidate is acceptable is entirely
# the (not-yet-implemented) Scenario Validator's concern.
#
# Reproducibility gap this module resolves, once, at the top level:
# scenario_definition.ScenarioDefinition carries no `definition_id` and
# no Building reference of its own (§3.2's frozen field table has
# neither) -- both are supplied via GenerationRequest instead (see
# request.py's own docstring for why).


def generate_scenario(request: GenerationRequest) -> Scenario:

    definition = request.definition
    building = request.building

    attempt_seed = derive_attempt_seed(request.seed, request.attempt_index)

    fire_rng = category_rng(attempt_seed, "fire")
    occupant_rng = category_rng(attempt_seed, "occupant")
    behaviour_profile_rng = category_rng(attempt_seed, "behaviour_profile")
    door_rng = category_rng(attempt_seed, "door")
    exit_rng = category_rng(attempt_seed, "exit")
    stair_rng = category_rng(attempt_seed, "stair")
    obstacle_rng = category_rng(attempt_seed, "obstacle")
    camera_rng = category_rng(attempt_seed, "camera")
    detector_rng = category_rng(attempt_seed, "detector")
    event_rng = category_rng(attempt_seed, "event")

    # Human Population & Assisted Evacuation Modeling Phase 5 -- two
    # more category streams, same "never draws from another category's
    # stream" convention every existing stream above already follows.
    # assistance_rng affects only ScenarioOccupant.assisting_occupant_id/
    # assistance_type (never occupant count/placement/profile,
    # occupant_rng/behaviour_profile_rng's own exclusive concerns).
    # firefighter_rng is its own, fully independent population, per
    # Phase 5's own "generate firefighter deployments independently
    # from occupant populations" rule.
    assistance_rng = category_rng(attempt_seed, "assistance")
    firefighter_rng = category_rng(attempt_seed, "firefighter")

    metadata = build_metadata(
        definition=definition,
        definition_id=request.definition_id,
        seed=request.seed,
        attempt_index=request.attempt_index,
        # No scenario_validator/ exists yet to reject a candidate, so
        # this phase's own code never discards one -- always 0 here
        # until a future orchestration phase wires in real retries.
        rejected_attempt_count=0,
    )

    occupants = _generate_occupants(
        definition, building, occupant_rng, behaviour_profile_rng, assistance_rng,
    )

    return Scenario(
        metadata=metadata,
        occupants=occupants,
        firefighters=_generate_firefighters(definition, building, occupants, firefighter_rng),
        fire=_generate_fire(definition, building, fire_rng),
        door_states=_generate_door_states(definition, building, door_rng),
        exit_states=_generate_exit_states(definition, building, exit_rng),
        stair_states=_generate_stair_states(definition, building, stair_rng),
        obstacle_states=_generate_obstacle_states(definition, building, obstacle_rng),
        camera_states=_generate_camera_states(definition, building, camera_rng),
        detector_states=_generate_detector_states(definition, building, detector_rng),
        events=_generate_events(definition, event_rng),
        difficulty=None,
    )


# =====================================================
# Building lookups -- read-only resolution of ids to geometry/state,
# never connectivity (§4.2: "never to reason about connectivity, since
# reachability stays exclusively a Validator concern").
# =====================================================


def _zone_lookup(building):

    # zone_id -> (Zone, floor_id)

    lookup = {}

    for floor in building.floors:
        for zone in floor.zones:
            lookup[zone.id] = (zone, floor.id)

    return lookup


def _doors_by_id(building):
    return {door.id: door for floor in building.floors for door in floor.doors}


def _exits_by_id(building):
    return {exit_obj.id: exit_obj for floor in building.floors for exit_obj in floor.exits}


def _stairs_by_id(building):

    # A Staircase spans exactly two floors and is rendered on both
    # (models/staircase.py) -- appears in floor.stairs on each, so the
    # dict naturally dedupes by id regardless.
    stairs = {}

    for floor in building.floors:
        for stair in floor.stairs:
            stairs[stair.id] = stair

    return stairs


def _obstacles_by_id(building):
    return {obs.id: obs for floor in building.floors for obs in floor.obstacles}


def _cameras_by_id(building):
    return {cam.id: cam for floor in building.floors for cam in floor.cameras}


def _detectors_by_id(building):
    return {det.id: det for floor in building.floors for det in floor.detectors}


# =====================================================
# Fire (§4.4/§4.7)
# =====================================================


def _generate_fire(definition, building, fire_rng) -> ScenarioFire:

    fire_def = definition.fire
    zone_floor = {zone_id: floor_id for zone_id, (_, floor_id) in _zone_lookup(building).items()}

    if fire_def.allowed_ignition_zone_ids:
        base = fire_def.allowed_ignition_zone_ids
    else:
        # "empty = no additional restriction beyond the other two
        # fields" (§3.2) -- the eligible universe is every zone on the
        # Building.
        base = frozenset(zone_floor.keys())

    eligible = base - fire_def.forbidden_ignition_zone_ids

    if fire_def.allowed_ignition_floor_ids:
        eligible = eligible & frozenset(
            zone_id for zone_id, floor_id in zone_floor.items()
            if floor_id in fire_def.allowed_ignition_floor_ids
        )

    if fire_def.ignition_zone_preference is not None:
        # §13's open question (whether a *present* preference's support
        # must be a subset of allowed-minus-forbidden) is explicitly
        # not decided by the frozen architecture -- this module trusts
        # the preference distribution as authored, consistent with
        # that question being left open rather than silently resolved
        # here.
        ignition_zone_id = sample(fire_def.ignition_zone_preference, fire_rng)
    else:
        ignition_zone_id = sample_uniform_choice(eligible, fire_rng)

    ignition_floor_id = zone_floor.get(ignition_zone_id, "")

    growth_value = sample(fire_def.growth_parameter_distribution, fire_rng)

    if fire_def.allowed_fire_profiles:
        fire_profile = sample_uniform_choice(fire_def.allowed_fire_profiles, fire_rng)
    else:
        # No allow-list stated and no universe of "every possible fire
        # profile" exists to fall back on (§9: an open, unfinalized
        # namespace, unlike zones) -- resolves to "no profile stated"
        # rather than fabricating one.
        fire_profile = ""

    return ScenarioFire(
        ignition_zone_id=ignition_zone_id,
        ignition_floor_id=ignition_floor_id,
        fire_profile=fire_profile,
        # A single sampled float wrapped under one documented key --
        # scenario.fire.ScenarioFire.growth_parameters is a generic
        # str -> float bag (§4.2's own "plain data" framing); this is
        # the one key this Generator release populates. Interpreting
        # it into an actual FireGrowthCurve is exclusively the
        # Simulator's job.
        growth_parameters={"growth_time": float(growth_value)},
    )


# =====================================================
# Occupants + Behaviour Profile IDs (§4.4: two separate pipeline
# stages, two separate category streams -- "Generate Occupants" never
# draws from behaviour_profile_rng, and vice versa).
# =====================================================


def _generate_occupant_placements(definition, building, occupant_rng):

    zone_lookup = _zone_lookup(building)
    placements = []

    # sorted(): a Mapping's iteration order reflects however it was
    # constructed, which is not itself part of a Definition's *logical*
    # content -- sorting by zone_id guarantees the same sampled
    # scenario regardless of incidental dict-construction order.
    for zone_id, distribution in sorted(definition.occupant.occupancy_distribution.items()):

        zone_info = zone_lookup.get(zone_id)

        if zone_info is None:
            # An id the Definition references that doesn't exist on
            # this Building -- a Definition-authoring problem, not
            # something this module validates or repairs (§4.7); it is
            # simply not resolvable, and is skipped rather than
            # crashing the whole batch.
            continue

        zone, floor_id = zone_info

        raw_count = sample(distribution, occupant_rng)
        count = max(0, int(round(raw_count)))

        for index in range(count):

            x = occupant_rng.uniform(zone.x, zone.x + zone.width)
            y = occupant_rng.uniform(zone.y, zone.y + zone.height)

            placements.append(
                {
                    # Deterministic, not uuid4() -- a pure function of
                    # (zone_id, index) so the same seed always mints
                    # the same occupant ids.
                    "occupant_id": f"occ-{zone_id}-{index}",
                    "zone_id": zone_id,
                    "floor_id": floor_id,
                    "position": (x, y),
                }
            )

    return placements


def _assign_behaviour_profile_ids(placements, occupant_definition, behaviour_profile_rng):

    profile_distribution = occupant_definition.behaviour_profile_distribution
    profile_ids = []

    for placement in placements:

        distribution = profile_distribution.get(placement["zone_id"])

        if distribution is None:
            # No rule stated for this zone -- resolves to "no profile
            # assigned" rather than fabricating one; a (future) Scenario
            # Validator, not this module, is where an incomplete
            # Definition is expected to surface as a problem.
            profile_ids.append("")
            continue

        profile_ids.append(sample(distribution, behaviour_profile_rng))

    return profile_ids


def _generate_occupants(definition, building, occupant_rng, behaviour_profile_rng, assistance_rng):

    placements = _generate_occupant_placements(definition, building, occupant_rng)
    profile_ids = _assign_behaviour_profile_ids(
        placements, definition.occupant, behaviour_profile_rng,
    )

    occupants = tuple(
        ScenarioOccupant(
            occupant_id=placement["occupant_id"],
            zone_id=placement["zone_id"],
            floor_id=placement["floor_id"],
            position=placement["position"],
            behaviour_profile_id=profile_id,
        )
        for placement, profile_id in zip(placements, profile_ids)
    )

    assistance_by_helper_id = _generate_assistance_pairs(
        occupants, definition.occupant, assistance_rng,
    )

    if not assistance_by_helper_id:
        return occupants

    return tuple(
        occupant if occupant.occupant_id not in assistance_by_helper_id
        else _with_assistance(occupant, *assistance_by_helper_id[occupant.occupant_id])
        for occupant in occupants
    )


# =====================================================


def _with_assistance(occupant, target_occupant_id, assistance_type):

    # ScenarioOccupant is frozen -- a new instance, every other field
    # carried through unchanged, is the only way to add the two
    # assistance fields onto an already-built occupant.
    return ScenarioOccupant(
        occupant_id=occupant.occupant_id,
        zone_id=occupant.zone_id,
        floor_id=occupant.floor_id,
        position=occupant.position,
        behaviour_profile_id=occupant.behaviour_profile_id,
        assisting_occupant_id=target_occupant_id,
        assistance_type=assistance_type,
    )


# =====================================================
# Assistance pairing (Phase 3/5) -- a documented, illustrative
# simplification, not a validated mobility-needs model (same honesty
# convention every other Generator default in this module already
# carries): a generated occupant whose OccupantCategory is considered
# potentially in need of assistance (CHILD/ELDERLY/WHEELCHAIR_USER) is,
# with probability occupant_definition.assistance_pairing_probability,
# paired with one other, not-yet-paired, non-vulnerable-category
# occupant in the *same zone* to assist them -- never across zones,
# since an assistance pairing presupposes the two already start
# together. An occupant already paired (either role) is never paired
# again. Iterates occupants in occupant_id order -- a pure function of
# already-deterministic placement/profile-assignment output -- so the
# same seed always produces the same pairings.

_ASSISTANCE_TYPE_BY_CATEGORY = {
    OccupantCategory.WHEELCHAIR_USER: "PUSH_WHEELCHAIR",
    OccupantCategory.ELDERLY: "ESCORT",
    OccupantCategory.CHILD: "HELP",
}


def _generate_assistance_pairs(occupants, occupant_definition, assistance_rng):

    probability = occupant_definition.assistance_pairing_probability

    if probability <= 0.0:
        return {}

    occupants_by_zone = {}
    for occupant in occupants:
        occupants_by_zone.setdefault(occupant.zone_id, []).append(occupant)

    pairs_by_helper_id = {}
    paired_ids = set()

    for occupant in sorted(occupants, key=lambda o: o.occupant_id):

        if occupant.occupant_id in paired_ids:
            continue

        category = occupant_category(occupant.behaviour_profile_id)
        assistance_type = _ASSISTANCE_TYPE_BY_CATEGORY.get(category)

        if assistance_type is None:
            continue

        if assistance_rng.random() >= probability:
            continue

        candidate_ids = sorted(
            candidate.occupant_id
            for candidate in occupants_by_zone.get(occupant.zone_id, [])
            if candidate.occupant_id != occupant.occupant_id
            and candidate.occupant_id not in paired_ids
            and occupant_category(candidate.behaviour_profile_id) not in _ASSISTANCE_TYPE_BY_CATEGORY
        )

        if not candidate_ids:
            continue

        helper_id = assistance_rng.choice(candidate_ids)

        pairs_by_helper_id[helper_id] = (occupant.occupant_id, assistance_type)
        paired_ids.add(helper_id)
        paired_ids.add(occupant.occupant_id)

    return pairs_by_helper_id


# =====================================================
# Firefighter deployment (Phase 4/5) -- generated entirely
# independently of the occupant population above (its own category rng
# stream, never reading occupant_rng/behaviour_profile_rng/
# assistance_rng): the *number* of teams, their size, and their arrival
# time never depend on how many occupants were generated. The one
# place a firefighter *does* read the occupant population is optional
# rescue-target assignment -- picking *which* already-generated
# occupant a team is assigned to rescue, never generating a new one.
# =====================================================


def _generate_firefighters(definition, building, occupants, firefighter_rng):

    firefighter_def = definition.firefighter
    zone_lookup = _zone_lookup(building)

    if not firefighter_def.entry_zone_ids:
        return ()

    raw_team_count = sample(firefighter_def.team_count_distribution, firefighter_rng)
    team_count = max(0, int(round(raw_team_count)))

    if team_count == 0:
        return ()

    entry_zone_ids = frozenset(firefighter_def.entry_zone_ids)
    already_assigned_target_ids = set()
    firefighters = []

    for team_index in range(team_count):

        team_id = f"team-{team_index}"

        raw_team_size = sample(firefighter_def.team_size_distribution, firefighter_rng)
        team_size = max(1, int(round(raw_team_size)))

        arrival_time = float(sample(firefighter_def.arrival_time_distribution, firefighter_rng))

        entry_zone_id = sample_uniform_choice(entry_zone_ids, firefighter_rng)
        zone_info = zone_lookup.get(entry_zone_id)

        if zone_info is None:
            # Same "Definition references an id this Building doesn't
            # have" case _generate_occupant_placements() already treats
            # as skip-not-crash -- not something this module repairs.
            continue

        zone, floor_id = zone_info

        rescue_target = None

        if firefighter_rng.random() < firefighter_def.rescue_assignment_probability:

            remaining = sorted(
                (o for o in occupants if o.occupant_id not in already_assigned_target_ids),
                key=lambda o: o.occupant_id,
            )

            if remaining:
                rescue_target = firefighter_rng.choice(remaining)
                already_assigned_target_ids.add(rescue_target.occupant_id)

        for member_index in range(team_size):

            x = firefighter_rng.uniform(zone.x, zone.x + zone.width)
            y = firefighter_rng.uniform(zone.y, zone.y + zone.height)

            # Only the team's first member carries the explicit rescue
            # assignment -- every other teammate searches instead
            # (Phase 4's other named behavior), avoiding every member
            # of one team independently computing and following the
            # identical rescue route.
            has_assignment = rescue_target is not None and member_index == 0

            firefighters.append(
                ScenarioFirefighter(
                    firefighter_id=f"ff-{team_id}-{member_index}",
                    team_id=team_id,
                    entry_zone_id=entry_zone_id,
                    floor_id=floor_id,
                    position=(x, y),
                    arrival_time=arrival_time,
                    behaviour_profile_id=firefighter_def.behaviour_profile_id,
                    rescue_target_occupant_id=(
                        rescue_target.occupant_id if has_assignment else None
                    ),
                    rescue_target_zone_id=rescue_target.zone_id if has_assignment else None,
                )
            )

    return tuple(firefighters)


# =====================================================
# Engineering object states (§4.4). Door/Stair/Obstacle/Camera/
# Detector share one shape (an enum-valued state, defaulted from the
# Building's own authored state when the Definition states no rule for
# that id -- §3.2's undetermined "default rule applies to any id absent
# from the dict"); Exit is bool-shaped and handled on its own, matching
# scenario_definition's own Distribution[bool] typing for it.
# =====================================================


def _generate_category_states(building_objects, distribution_map, rng, enum_cls, default_for, make_state):

    results = []

    for object_id in sorted(building_objects.keys()):

        distribution = distribution_map.get(object_id)

        if distribution is not None:
            value = enum_cls[sample(distribution, rng)]
        else:
            value = default_for(building_objects[object_id])

        results.append(make_state(object_id, value))

    return tuple(results)


def _generate_door_states(definition, building, door_rng):

    def default_for(door):

        if door.locked:
            return DoorState.LOCKED

        if door.normally_open:
            return DoorState.OPEN

        return DoorState.CLOSED

    return _generate_category_states(
        _doors_by_id(building),
        definition.engineering.door_state_distribution,
        door_rng,
        DoorState,
        default_for,
        lambda door_id, state: ScenarioDoorState(door_id=door_id, state=state),
    )


def _generate_exit_states(definition, building, exit_rng):

    exits_by_id = _exits_by_id(building)
    distribution_map = definition.engineering.exit_state_distribution
    results = []

    for exit_id in sorted(exits_by_id.keys()):

        distribution = distribution_map.get(exit_id)

        if distribution is not None:
            is_open = bool(sample(distribution, exit_rng))
        else:
            is_open = not exits_by_id[exit_id].is_blocked

        results.append(ScenarioExitState(exit_id=exit_id, is_open=is_open))

    return tuple(results)


def _generate_stair_states(definition, building, stair_rng):

    return _generate_category_states(
        _stairs_by_id(building),
        definition.engineering.stair_state_distribution,
        stair_rng,
        StairAvailability,
        # models.Staircase has no active/blocked-style field at all
        # (unlike Door/Exit/Obstacle/Camera/Detector) -- there is no
        # Building-authored state to fall back to, so an unstated rule
        # defaults to the simplest, documented constant: available.
        lambda stair: StairAvailability.AVAILABLE,
        lambda stair_id, availability: ScenarioStairState(
            stair_id=stair_id, availability=availability,
        ),
    )


def _generate_obstacle_states(definition, building, obstacle_rng):

    return _generate_category_states(
        _obstacles_by_id(building),
        definition.engineering.obstacle_state_distribution,
        obstacle_rng,
        PresenceState,
        lambda obstacle: PresenceState.ACTIVE if obstacle.active else PresenceState.INACTIVE,
        lambda obstacle_id, presence: ScenarioObstacleState(
            obstacle_id=obstacle_id, presence=presence,
        ),
    )


def _generate_camera_states(definition, building, camera_rng):

    return _generate_category_states(
        _cameras_by_id(building),
        definition.engineering.camera_state_distribution,
        camera_rng,
        DeviceAvailability,
        lambda camera: DeviceAvailability.AVAILABLE if camera.active else DeviceAvailability.FAILED,
        lambda camera_id, availability: ScenarioCameraState(
            camera_id=camera_id, availability=availability,
        ),
    )


def _generate_detector_states(definition, building, detector_rng):

    return _generate_category_states(
        _detectors_by_id(building),
        definition.engineering.detector_state_distribution,
        detector_rng,
        DeviceAvailability,
        lambda detector: (
            DeviceAvailability.AVAILABLE if detector.active else DeviceAvailability.FAILED
        ),
        lambda detector_id, availability: ScenarioDetectorState(
            detector_id=detector_id, availability=availability,
        ),
    )


# =====================================================
# Events (§4.4/§6). A ScenarioEvent is materialized only for a template
# whose sampled `occurs` is true -- a template that did not occur is
# simply never produced, exactly as scenario.event.ScenarioEvent's own
# docstring already establishes.
# =====================================================


def _generate_events(definition, event_rng):

    events = []

    for index, template in enumerate(definition.event_templates):

        occurs = sample(template.occurs, event_rng)

        if not occurs:
            continue

        time_value = sample(template.time, event_rng)

        events.append(
            ScenarioEvent(
                event_id=f"evt-{index}",
                target_type=template.target_type,
                target_id=template.target_id,
                event_type=template.event_type,
                time=float(time_value),
                parameters=dict(template.parameters),
            )
        )

    return tuple(events)
