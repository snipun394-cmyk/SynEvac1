from scenario_definition.distributions import FixedValue

from scenario_validator.issue import FailureCategory, ScenarioValidationIssue
from scenario_validator.report import ScenarioValidationReport


# Event Validation -- architecture doc §5.3, module 5.
#
# `event_type` has no fixed vocabulary anywhere in the frozen schema
# (scenario_definition.event_templates.EventTemplate.event_type is a
# plain str) -- "no sampled event contradicts a FixedValue-pinned
# element" therefore cannot be checked by exact type matching, only by
# a documented, best-effort keyword heuristic over event_type. This is
# the one check in this module built on an inherently fuzzy signal
# rather than an exact structural comparison; it is scoped
# conservatively (case-insensitive substring match against a small,
# named keyword table per engineering category) precisely because the
# schema itself does not commit to more.

_DOOR_KEYWORDS = {"close": {"OPEN"}, "lock": {"OPEN"}, "open": {"CLOSED", "LOCKED"}, "unlock": {"LOCKED"}}
_STAIR_KEYWORDS = {"close": {"AVAILABLE"}, "block": {"AVAILABLE"}, "open": {"CLOSED"}, "clear": {"CLOSED"}}
_OBSTACLE_KEYWORDS = {"remove": {"ACTIVE"}, "clear": {"ACTIVE"}, "place": {"INACTIVE"}, "add": {"INACTIVE"}}
_DEVICE_KEYWORDS = {"fail": {"AVAILABLE"}, "disable": {"AVAILABLE"}, "restore": {"FAILED"}, "repair": {"FAILED"}}
_EXIT_KEYWORDS = {"close": True, "block": True, "open": False, "unblock": False}


def _contradicted_enum_values(event_type, keyword_table):

    lowered = event_type.lower()
    contradicted = set()

    for keyword, values in keyword_table.items():
        if keyword in lowered:
            contradicted |= values

    return contradicted


def _contradicted_exit_open_value(event_type):

    lowered = event_type.lower()

    for keyword, implied_is_open in _EXIT_KEYWORDS.items():
        if keyword in lowered:
            return not implied_is_open

    return None


def _check_pinned_contradiction(report, event, distribution_map):

    distribution = distribution_map.get(event.target_id)

    if not isinstance(distribution, FixedValue):
        return

    if event.target_type == "door":
        contradicted = _contradicted_enum_values(event.event_type, _DOOR_KEYWORDS)
    elif event.target_type == "stair":
        contradicted = _contradicted_enum_values(event.event_type, _STAIR_KEYWORDS)
    elif event.target_type == "obstacle":
        contradicted = _contradicted_enum_values(event.event_type, _OBSTACLE_KEYWORDS)
    elif event.target_type in ("camera", "detector"):
        contradicted = _contradicted_enum_values(event.event_type, _DEVICE_KEYWORDS)
    elif event.target_type == "exit":

        contradicts_open_pin = _contradicted_exit_open_value(event.event_type)

        if contradicts_open_pin is not None and distribution.value == contradicts_open_pin:

            report.add(
                FailureCategory.EVENTS, ScenarioValidationReport.ERROR,
                "EVENT_CONTRADICTS_PINNED_STATE",
                f"Event {event.event_id!r} ({event.event_type!r}) on exit "
                f"{event.target_id!r} contradicts its FixedValue({distribution.value!r}) "
                f"pin.",
                object_id=event.target_id,
            )
        return
    else:
        return

    if distribution.value in contradicted:

        report.add(
            FailureCategory.EVENTS, ScenarioValidationReport.ERROR,
            "EVENT_CONTRADICTS_PINNED_STATE",
            f"Event {event.event_id!r} ({event.event_type!r}) on {event.target_type} "
            f"{event.target_id!r} contradicts its FixedValue({distribution.value!r}) pin.",
            object_id=event.target_id,
        )


_DISTRIBUTION_FIELD_BY_TARGET_TYPE = {
    "door": "door_state_distribution",
    "exit": "exit_state_distribution",
    "stair": "stair_state_distribution",
    "obstacle": "obstacle_state_distribution",
    "camera": "camera_state_distribution",
    "detector": "detector_state_distribution",
}


def validate_events(candidate, definition) -> ScenarioValidationReport:

    report = ScenarioValidationReport()
    engineering = definition.engineering

    events = candidate.events

    for event in events:

        if not event.target_type or not event.target_id or not event.event_type:

            report.add(
                FailureCategory.EVENTS, ScenarioValidationReport.ERROR,
                "EVENT_MISSING_REQUIRED_FIELD",
                f"Event {event.event_id!r} is missing target_type/target_id/event_type.",
                object_id=event.event_id,
            )

        if event.time < 0:

            report.add(
                FailureCategory.EVENTS, ScenarioValidationReport.ERROR,
                "EVENT_INVALID_TIMESTAMP",
                f"Event {event.event_id!r} has a negative time ({event.time}).",
                object_id=event.event_id,
            )

        field_name = _DISTRIBUTION_FIELD_BY_TARGET_TYPE.get(event.target_type)

        if field_name is not None:
            _check_pinned_contradiction(report, event, getattr(engineering, field_name))

    # Ordered events: the candidate's events tuple must already be in
    # non-decreasing time order -- a real, checkable structural
    # property a downstream Simulator consuming this list sequentially
    # would need, and the literal reading of "ordered events" as its
    # own bullet, distinct from "no conflicting events" below.
    for earlier, later in zip(events, events[1:]):

        if earlier.time > later.time:

            report.add(
                FailureCategory.EVENTS, ScenarioValidationReport.ERROR,
                "EVENTS_NOT_ORDERED",
                f"Event {earlier.event_id!r} (t={earlier.time}) is not ordered "
                f"before {later.event_id!r} (t={later.time}).",
            )
            break

    # No conflicting events: more than one event targeting the same
    # (target_type, target_id) at the exact same instant is ambiguous
    # -- which one applies first is undefined. This covers both an
    # accidental exact duplicate and two different event_types racing
    # at the same timestamp, without needing to know event_type's
    # domain-specific meaning.
    seen_at_time = {}

    for event in events:

        key = (event.target_type, event.target_id, event.time)

        if key in seen_at_time:

            report.add(
                FailureCategory.EVENTS, ScenarioValidationReport.ERROR,
                "CONFLICTING_EVENTS",
                f"Events {seen_at_time[key]!r} and {event.event_id!r} both target "
                f"{event.target_type} {event.target_id!r} at t={event.time}.",
                object_id=event.target_id,
            )

        else:
            seen_at_time[key] = event.event_id

    # Every EventTemplate with occurs=FixedValue(True) must have
    # produced its event verbatim.
    for index, template in enumerate(definition.event_templates):

        if not (isinstance(template.occurs, FixedValue) and template.occurs.value is True):
            continue

        matches = [
            event for event in events
            if event.target_type == template.target_type
            and event.target_id == template.target_id
            and event.event_type == template.event_type
        ]

        if not matches:

            report.add(
                FailureCategory.EVENTS, ScenarioValidationReport.ERROR,
                "PINNED_EVENT_MISSING",
                f"event_templates[{index}] is pinned occurs=True but produced no "
                f"matching event in the candidate.",
                object_id=template.target_id,
            )

        elif isinstance(template.time, FixedValue):

            if not any(event.time == template.time.value for event in matches):

                report.add(
                    FailureCategory.EVENTS, ScenarioValidationReport.ERROR,
                    "PINNED_EVENT_TIME_MISMATCH",
                    f"event_templates[{index}] pins time={template.time.value!r} but "
                    f"no matching event landed on it.",
                    object_id=template.target_id,
                )

    return report
