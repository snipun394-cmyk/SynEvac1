from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from simulator.occupant import OccupantState


@dataclass(frozen=True)
class RescueOutcome:

    # One firefighter's own rescue *assignment* (Scenario-authored, see
    # scenario.firefighter.ScenarioFirefighter's own docstring) crossed
    # with what the completed simulation actually shows happened --
    # never a claim about intent alone. rescued is True only when the
    # target occupant's own OccupantTimeline shows ARRIVED (the same
    # authoritative "did this occupant actually evacuate" signal
    # ground_truth.evacuation_metrics.compute_evacuation_metrics()
    # already uses) -- a firefighter can be *assigned* to rescue
    # someone and still fail (the target's own OccupantTimelineStep
    # chain never reaches an exit, e.g. the firefighter itself becomes
    # unreachable) without this ever being asserted as a success.

    firefighter_id: str
    target_occupant_id: str
    rescued: bool
    rescue_time_seconds: Optional[float]


def compute_rescue_outcomes(scenario, movement_result) -> Tuple[RescueOutcome, ...]:

    # One entry per firefighter that was ever given an explicit rescue
    # assignment (Phase 4/5) -- a firefighter with no assignment
    # (searching, Phase 4's other named behavior) contributes nothing
    # here, the same "only touched ids get an entry" convention every
    # other ground_truth/*.py module already follows.

    outcomes = []

    for firefighter in scenario.firefighters:

        target_id = firefighter.rescue_target_occupant_id

        if not target_id:
            continue

        target_timeline = movement_result.occupants.get(target_id)
        firefighter_timeline = movement_result.occupants.get(firefighter.firefighter_id)

        if target_timeline is None or firefighter_timeline is None:
            # Either occupant_id never appears in this run's own
            # movement_result at all (a Scenario/movement_result
            # mismatch this module does not repair) -- no honest
            # outcome to report.
            continue

        rescued = target_timeline.state == OccupantState.ARRIVED

        rescue_time_seconds = None

        if rescued and target_timeline.arrival_time is not None:
            # Total time from when the firefighter themselves departed
            # to when the rescued occupant reached safety -- the one
            # honest "how long did this rescue take" figure both
            # occupant records already carry, requiring no new
            # bookkeeping anywhere in the simulator.
            rescue_time_seconds = target_timeline.arrival_time - firefighter_timeline.depart_time

        outcomes.append(
            RescueOutcome(
                firefighter_id=firefighter.firefighter_id,
                target_occupant_id=target_id,
                rescued=rescued,
                rescue_time_seconds=rescue_time_seconds,
            )
        )

    return tuple(outcomes)


# =====================================================


def summarize_rescue_outcomes(outcomes: Tuple[RescueOutcome, ...]) -> Dict[str, Any]:

    rescued = [outcome for outcome in outcomes if outcome.rescued]

    rescue_times = [
        outcome.rescue_time_seconds for outcome in rescued
        if outcome.rescue_time_seconds is not None
    ]

    return {
        # This platform's V1 rescue model assigns at most one civilian
        # target per firefighter *team* (see scenario_generator.
        # generator._generate_firefighters()'s own "only the team's
        # first member carries the explicit rescue assignment" note),
        # so "how many distinct occupants were rescued" and "how many
        # rescue operations succeeded" are numerically identical today
        # -- reported as two separate, independently-named fields
        # (Phase 7's own "Rescued Occupants"/"Firefighter Rescue Count")
        # because a future version that lets one team rescue several
        # people, or several teams collaborate on one rescue, would
        # make them genuinely diverge.
        "rescued_occupant_count": len(rescued),
        "firefighter_rescue_count": len(rescued),
        "average_rescue_time_seconds": (
            sum(rescue_times) / len(rescue_times) if rescue_times else None
        ),
    }
