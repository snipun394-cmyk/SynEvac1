import hashlib

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# =====================================================
# Dataset Splitting (Phase 4) -- deterministic train/validation/test
# splits over an already-loaded campaign's samples.
#
# Determinism: which split a scenario_id lands in is derived from
# sha256(f"{master_seed}:{scenario_id}") alone -- never from list
# order, dict iteration order, or wall-clock time -- so the same
# master_seed against the same set of scenario_ids always reproduces
# byte-identical splits, on any machine, any run.
#
# No overlap: every scenario_id is assigned to exactly one bucket by
# construction (a single sorted-then-sliced list per stratum), so
# train/validation/test are disjoint by design, not by a post-hoc check.
# =====================================================


@dataclass(frozen=True)
class DatasetSplit:

    train_ids: Tuple[str, ...]
    validation_ids: Tuple[str, ...]
    test_ids: Tuple[str, ...]

    def split_for(self, scenario_id: str) -> Optional[str]:

        if scenario_id in self.train_ids:
            return "train"

        if scenario_id in self.validation_ids:
            return "validation"

        if scenario_id in self.test_ids:
            return "test"

        return None

    def to_dict(self) -> dict:

        return {
            "train": list(self.train_ids),
            "validation": list(self.validation_ids),
            "test": list(self.test_ids),
        }


def stratify_by_fire_profile(sample) -> str:

    # Convenience stratify_by callable -- fire_profile is the single
    # categorical field Phase 3 statistics already treats as the
    # primary bias signal (fire_profile_distribution), so it is the
    # most natural default a caller can opt into for "preserve
    # distributions where practical."

    return sample.scenario_features.get("fire_profile") or "unknown"


def _deterministic_unit_interval(scenario_id: str, master_seed: int) -> float:

    digest = hashlib.sha256(f"{master_seed}:{scenario_id}".encode("utf-8")).hexdigest()

    return int(digest[:16], 16) / float(1 << 64)


def split_dataset(
    dataset: Iterable,
    *,
    master_seed: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    stratify_by: Optional[Callable[[Any], str]] = None,
) -> DatasetSplit:

    if abs((train_fraction + validation_fraction + test_fraction) - 1.0) > 1e-6:
        raise ValueError("train_fraction + validation_fraction + test_fraction must sum to 1.0")

    if min(train_fraction, validation_fraction, test_fraction) < 0:
        raise ValueError("split fractions must be non-negative")

    samples = sorted(dataset, key=lambda sample: sample.scenario_id)

    groups: Dict[str, List] = {}

    for sample in samples:

        key = stratify_by(sample) if stratify_by is not None else "*"
        groups.setdefault(key or "unknown", []).append(sample)

    train_ids: List[str] = []
    validation_ids: List[str] = []
    test_ids: List[str] = []

    for key in sorted(groups):

        ordered = sorted(
            groups[key],
            key=lambda sample: _deterministic_unit_interval(sample.scenario_id, master_seed),
        )

        total = len(ordered)
        n_train = min(round(total * train_fraction), total)
        n_validation = min(round(total * validation_fraction), total - n_train)

        train_ids.extend(sample.scenario_id for sample in ordered[:n_train])
        validation_ids.extend(
            sample.scenario_id for sample in ordered[n_train:n_train + n_validation]
        )
        test_ids.extend(sample.scenario_id for sample in ordered[n_train + n_validation:])

    return DatasetSplit(
        train_ids=tuple(sorted(train_ids)),
        validation_ids=tuple(sorted(validation_ids)),
        test_ids=tuple(sorted(test_ids)),
    )
