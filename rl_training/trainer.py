from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import gymnasium
from stable_baselines3 import A2C, DQN, PPO

# Registry pattern -- Phase 5's own requirement ("structure the code
# so additional algorithms can be added later"). PPO is the only one
# exercised/supported end-to-end by this phase; A2C/DQN are included
# because stable-baselines3 already ships them with the same
# `(policy, env, **kwargs)` construction signature and no SynEvac-side
# code changes are needed to use them -- they are not individually
# tested here. Adding SAC (continuous-only) would additionally require
# a continuous action space this environment doesn't have yet.
ALGORITHM_REGISTRY = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


@dataclass(frozen=True)
class TrainerConfig:

    algorithm: str = "PPO"
    policy: str = "MlpPolicy"
    algorithm_kwargs: Mapping[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    def to_dict(self) -> dict:

        return {
            "algorithm": self.algorithm, "policy": self.policy,
            "algorithm_kwargs": dict(self.algorithm_kwargs), "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainerConfig":

        return cls(
            algorithm=data.get("algorithm", "PPO"), policy=data.get("policy", "MlpPolicy"),
            algorithm_kwargs=data.get("algorithm_kwargs", {}), seed=data.get("seed"),
        )


class RLTrainer:

    # A thin wrapper: all training logic belongs to stable-baselines3
    # itself. This class exists only to (a) select an algorithm by
    # name via ALGORITHM_REGISTRY, and (b) give policy_export.py a
    # single, stable object to save/load against.

    def __init__(self, env: gymnasium.Env, config: Optional[TrainerConfig] = None):

        self.config = config or TrainerConfig()

        try:
            algorithm_cls = ALGORITHM_REGISTRY[self.config.algorithm]
        except KeyError:
            raise ValueError(
                f"Unsupported algorithm {self.config.algorithm!r}. "
                f"Known algorithms: {sorted(ALGORITHM_REGISTRY)!r}."
            ) from None

        self.model = algorithm_cls(
            self.config.policy, env, seed=self.config.seed, **self.config.algorithm_kwargs,
        )

    # =====================================================

    def train(self, total_timesteps: int, callback=None) -> None:

        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    # =====================================================

    def predict(self, observation, deterministic: bool = True):

        return self.model.predict(observation, deterministic=deterministic)
