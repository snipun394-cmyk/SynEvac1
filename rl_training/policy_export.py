import datetime
import json
import os
from typing import Any, Dict, Tuple

from rl_training.action_space import ActionSchema
from rl_training.observation_space import ObservationSchema
from rl_training.reward_function import RewardConfig
from rl_training.trainer import ALGORITHM_REGISTRY, RLTrainer, TrainerConfig


# Mirrors ai_training/experiment.py's own artifact convention
# (ExperimentRunner.save_result/load_result): one directory, one model
# file, one sibling manifest.json holding everything needed to
# reconstruct the surrounding config -- generated_at as an ISO-8601 UTC
# timestamp, the same format ai_training/experiment.py._utc_now_iso()
# produces. SB3's own native `model.zip` format is used for the model
# file itself (not joblib -- SB3 policies aren't joblib-round-trippable
# the way a scikit-learn estimator is).


def _utc_now_iso() -> str:

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_policy(
    trainer: RLTrainer,
    directory: str,
    observation_schema: ObservationSchema,
    action_schema: ActionSchema,
    reward_config: RewardConfig,
    metadata: Dict[str, Any],
) -> Dict[str, str]:

    os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(directory, "model.zip")
    manifest_path = os.path.join(directory, "manifest.json")

    trainer.model.save(model_path)

    manifest = {
        "trainer_config": trainer.config.to_dict(),
        "observation_schema": observation_schema.to_dict(),
        "action_schema": action_schema.to_dict(),
        "reward_config": vars(reward_config),
        "metadata": metadata,
        "generated_at": _utc_now_iso(),
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return {"model": model_path, "manifest": manifest_path}


def load_policy(
    directory: str,
) -> Tuple[Any, ObservationSchema, ActionSchema, RewardConfig, Dict[str, Any]]:

    manifest_path = os.path.join(directory, "manifest.json")
    model_path = os.path.join(directory, "model.zip")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    trainer_config = TrainerConfig.from_dict(manifest["trainer_config"])
    algorithm_cls = ALGORITHM_REGISTRY[trainer_config.algorithm]
    model = algorithm_cls.load(model_path)

    observation_schema = ObservationSchema.from_dict(manifest["observation_schema"])
    action_schema = ActionSchema.from_dict(manifest["action_schema"])
    reward_config = RewardConfig(**manifest["reward_config"])

    return model, observation_schema, action_schema, reward_config, manifest["metadata"]
