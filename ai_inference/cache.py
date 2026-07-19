import hashlib
import json

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


# =====================================================
# Phase 6 -- Prediction Cache. A pure in-memory (never persisted to
# disk, never survives past the owning PredictionCache instance)
# cache keyed by (model_version, feature_hash, prediction_type) -- the
# same feature vector against the SAME model_version for the SAME
# prediction_type always hits; a different model_version (e.g. after
# retraining/redeploying) or a different feature vector always misses,
# forcing fresh inference.
# =====================================================


def compute_feature_hash(X_row: Dict[str, Any]) -> str:

    # Deterministic content hash of a Scenario Feature dict --
    # sort_keys=True means insertion order never affects the hash;
    # default=str absorbs any non-JSON-native value (numpy scalars,
    # etc.) the same way a caller's own feature dict might carry one.

    canonical = json.dumps(X_row, sort_keys=True, default=str)

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# =====================================================


@dataclass(frozen=True)
class CacheKey:

    model_version: str
    feature_hash: str
    prediction_type: str


# =====================================================


class PredictionCache:

    def __init__(self):

        self._store: Dict[CacheKey, Any] = {}
        self.hits = 0
        self.misses = 0

    # =====================================================

    def make_key(self, model_version: str, X_row: Dict[str, Any], prediction_type: str) -> CacheKey:

        return CacheKey(
            model_version=model_version,
            feature_hash=compute_feature_hash(X_row),
            prediction_type=prediction_type,
        )

    # =====================================================

    def contains(self, model_version: str, X_row: Dict[str, Any], prediction_type: str) -> bool:

        return self.make_key(model_version, X_row, prediction_type) in self._store

    # =====================================================

    def get_or_compute(
        self,
        model_version: str,
        X_row: Dict[str, Any],
        prediction_type: str,
        compute_fn: Callable[[], Any],
    ) -> Any:

        key = self.make_key(model_version, X_row, prediction_type)

        if key in self._store:

            self.hits += 1
            return self._store[key]

        self.misses += 1
        value = compute_fn()
        self._store[key] = value

        return value

    # =====================================================

    def invalidate_model_version(self, model_version: str) -> int:

        # Drops every cached entry for a specific model_version (e.g.
        # after that model is retrained/redeployed and a caller no
        # longer wants its stale predictions served) -- returns how
        # many entries were removed.

        stale_keys = [key for key in self._store if key.model_version == model_version]

        for key in stale_keys:
            del self._store[key]

        return len(stale_keys)

    # =====================================================

    def clear(self) -> None:

        self._store.clear()
        self.hits = 0
        self.misses = 0

    # =====================================================

    def __len__(self) -> int:

        return len(self._store)

    # =====================================================

    @property
    def hit_rate(self) -> Optional[float]:

        total = self.hits + self.misses

        return (self.hits / total) if total else None
