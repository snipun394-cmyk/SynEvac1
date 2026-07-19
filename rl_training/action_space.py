from dataclasses import dataclass
from typing import Optional, Tuple

from gymnasium import spaces

from dataset_builder.schema import ordered_exits, ordered_stairs, ordered_zones

from models.building import Building

from simulation_interactive import Action, InteractiveActionType

NOOP_LABEL = "NOOP"


@dataclass(frozen=True)
class ActionSpaceEntry:

    index: int
    label: str
    action_type: Optional[InteractiveActionType]
    target_id: Optional[str]

    def to_dict(self) -> dict:

        return {
            "index": self.index,
            "label": self.label,
            "action_type": self.action_type.name if self.action_type is not None else None,
            "target_id": self.target_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionSpaceEntry":

        action_type = (
            InteractiveActionType[data["action_type"]] if data["action_type"] is not None else None
        )

        return cls(
            index=data["index"], label=data["label"],
            action_type=action_type, target_id=data["target_id"],
        )


@dataclass(frozen=True)
class ActionSchema:

    entries: Tuple[ActionSpaceEntry, ...]

    def to_dict(self) -> dict:

        return {"entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_dict(cls, data: dict) -> "ActionSchema":

        return cls(entries=tuple(ActionSpaceEntry.from_dict(e) for e in data["entries"]))


class ActionMapper:

    # A fixed, discrete action table built once per building at
    # construction time (never per episode) -- Gymnasium requires a
    # stable action_space for the lifetime of an env instance, and the
    # table only depends on building topology (how many exits/stairs
    # exist), never on a particular generated scenario. Every produced
    # Action/InteractiveActionType is imported straight from
    # simulation_interactive -- no new simulation action is invented
    # here.
    #
    # A "recommend exit/stair" action is applied to every zone in the
    # building (one simulation_interactive.Action per zone) rather than
    # to a single zone, since RouteManager's recommendation bookkeeping
    # is zone-scoped but this environment's action space is building-
    # scoped. RecommendationAwareRouteChoiceStrategy's own graceful
    # fallback (falls back to the base strategy when the recommended
    # edge isn't reachable from a given zone) makes this safe -- zones
    # where the recommendation doesn't make sense simply keep their
    # normal route.

    def __init__(self, building: Building):

        self._zone_ids = tuple(zone.id for zone in ordered_zones(building))

        entries = [
            ActionSpaceEntry(index=0, label=NOOP_LABEL, action_type=None, target_id=None),
        ]

        for exit_obj in ordered_exits(building):
            entries.append(
                ActionSpaceEntry(
                    index=len(entries),
                    label=f"RECOMMEND_EXIT:{exit_obj.id}",
                    action_type=InteractiveActionType.RECOMMEND_EXIT,
                    target_id=exit_obj.id,
                )
            )

        for stair in ordered_stairs(building):
            entries.append(
                ActionSpaceEntry(
                    index=len(entries),
                    label=f"RECOMMEND_STAIR:{stair.id}",
                    action_type=InteractiveActionType.RECOMMEND_STAIR,
                    target_id=stair.id,
                )
            )

        entries.append(
            ActionSpaceEntry(
                index=len(entries), label="BROADCAST_EVACUATE",
                action_type=InteractiveActionType.BROADCAST_EVACUATE, target_id=None,
            )
        )
        entries.append(
            ActionSpaceEntry(
                index=len(entries), label="BROADCAST_SHELTER_IN_PLACE",
                action_type=InteractiveActionType.BROADCAST_SHELTER_IN_PLACE, target_id=None,
            )
        )
        entries.append(
            ActionSpaceEntry(
                index=len(entries), label="DEPLOY_STAFF",
                action_type=InteractiveActionType.DEPLOY_STAFF, target_id=None,
            )
        )

        self._schema = ActionSchema(entries=tuple(entries))

    # =====================================================

    @property
    def schema(self) -> ActionSchema:

        return self._schema

    # =====================================================

    @property
    def space(self) -> spaces.Discrete:

        return spaces.Discrete(len(self._schema.entries))

    # =====================================================

    def describe(self, action_index: int) -> str:

        return self._schema.entries[action_index].label

    # =====================================================

    def decode(self, action_index: int) -> Tuple[Action, ...]:

        entry = self._schema.entries[action_index]

        if entry.action_type is None:
            return ()

        if entry.action_type == InteractiveActionType.RECOMMEND_EXIT:
            return tuple(
                Action(
                    InteractiveActionType.RECOMMEND_EXIT, target_id=zone_id,
                    parameters={"exit_id": entry.target_id},
                )
                for zone_id in self._zone_ids
            )

        if entry.action_type == InteractiveActionType.RECOMMEND_STAIR:
            return tuple(
                Action(
                    InteractiveActionType.RECOMMEND_STAIR, target_id=zone_id,
                    parameters={"stair_id": entry.target_id},
                )
                for zone_id in self._zone_ids
            )

        return (Action(entry.action_type, target_id=entry.target_id),)
