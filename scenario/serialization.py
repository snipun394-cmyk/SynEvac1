from enum import Enum
from typing import Mapping, Sequence, Tuple, Type, TypeVar


# Small, shared, pure-function helpers for the to_dict()/from_dict()
# pairs every model in this package implements. Nothing here performs
# file I/O, randomness, or computation over Scenario content -- these
# only reshape already-in-memory values into/out of plain Python types
# (dict/list/str/float/int/bool/None), the same "serialize only plain
# Python types" contract every model's own to_dict()/from_dict() must
# honor.


E = TypeVar("E", bound=Enum)
T = TypeVar("T")


def position_to_list(position: Tuple[float, float]) -> list:

    x, y = position

    return [x, y]


def position_from_list(data: Sequence[float]) -> Tuple[float, float]:

    x, y = data

    return (float(x), float(y))


def mapping_to_dict(mapping: Mapping[str, float]) -> dict:

    return dict(mapping)


def enum_to_str(value: Enum) -> str:

    return value.name


def enum_from_str(enum_cls: Type[E], name: str) -> E:

    return enum_cls[name]


def sequence_to_list(items: Sequence) -> list:

    return [item.to_dict() for item in items]


def sequence_from_list(item_cls: Type[T], data: Sequence[dict]) -> tuple:

    return tuple(item_cls.from_dict(item_data) for item_data in data)
