#!/usr/bin/env python
"""Export a keyframe-mnemonics run folder as a LeRobot pretrained_model directory.

Reads the selector zip and stage-3 policy .pth from `checkpoints/<run>/`, packs
both into a single KeyframeMnemonicsPolicy, and writes config.json,
model.safetensors and the processor JSONs so `--policy.path=<out>` works.

    python -m lerobot_policy_keyframe_mnemonics.export \
        --run real_robot_RememberColor3_<TAG> \
        --output outputs/km_remember_color_3
"""

import argparse
import glob
import os
from pathlib import Path

import h5py
import torch
import yaml
from stable_baselines3 import PPO

from lerobot.configs import NormalizationMode
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors

from common.helpers import get_problem_type_and_conf_path, parse_experiment_name, stage_dir

from .configuration_keyframe_mnemonics import KeyframeMnemonicsConfig
from .modeling_keyframe_mnemonics import KeyframeMnemonicsPolicy


def _latest_policy_checkpoint(policy_dir, experiment_name, checkpoint_num=None):
    if checkpoint_num is not None:
        path = os.path.join(policy_dir, f"{experiment_name}_policy_{checkpoint_num}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Policy checkpoint not found: {path}")
        return path
    matches = glob.glob(os.path.join(policy_dir, f"{experiment_name}_policy_*.pth"))
    if not matches:
        raise FileNotFoundError(f"No policy checkpoints in {policy_dir}")
    return max(matches, key=lambda p: int(p.split("_")[-1].split(".")[0]))


def _dataset_stats(path_prefix, env_id, split):
    """Recover the normalized space from the proxy H5 for LeRobot's processors."""
    h5_path = Path(path_prefix) / "datasets/real_robot/proxy_dataset" / f"{env_id}_{split}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Proxy H5 not found: {h5_path}. Run make_h5_dataset.py first.")
    with h5py.File(h5_path, "r") as h5f:
        missing = [k for k in ("state_mean", "state_std", "action_min", "action_max")
                   if k not in h5f.attrs]
        if missing:
            raise ValueError(
                f"{h5_path} is missing {missing}. Regenerate it with the current "
                "make_h5_dataset.py, which always normalizes."
            )
        return {
            "observation.state": {
                "mean": [float(v) for v in h5f.attrs["state_mean"]],
                "std": [float(v) for v in h5f.attrs["state_std"]],
            },
            "action": {
                "min": [float(v) for v in h5f.attrs["action_min"]],
                "max": [float(v) for v in h5f.attrs["action_max"]],
            },
        }


# LeRobot silently skips normalization for a feature with no stats, so a missing
# entry would ship raw-vs-normalized units to the robot. Fail at export instead.
_REQUIRED_STATS = {
    NormalizationMode.MEAN_STD: ("mean", "std"),
    NormalizationMode.MIN_MAX: ("min", "max"),
}


def _assert_stats_cover(config, dataset_stats):
    for feature, ftype in (("observation.state", FeatureType.STATE), ("action", FeatureType.ACTION)):
        mode = config.normalization_mapping.get(ftype.value, NormalizationMode.IDENTITY)
        if mode is NormalizationMode.IDENTITY:
            continue
        required = _REQUIRED_STATS.get(mode)
        if required is None:
            raise ValueError(f"Unsupported normalization mode {mode} for {feature}.")
        have = dataset_stats.get(feature, {})
        missing = [k for k in required if k not in have]
        if missing:
            raise ValueError(
                f"{feature} is {mode.value} but dataset_stats is missing {missing}. "
                "The exported policy would emit unnormalized values."
            )


def export(run: str, output: str, checkpoint_num=None, device: str = "cuda"):
    experiment_name = parse_experiment_name(run)
    _, conf_path = get_problem_type_and_conf_path(experiment_name)
    with open(conf_path) as f:
        base_config = yaml.safe_load(f)
    path_prefix = base_config.get("path_prefix", "")

    policy_dir = stage_dir(path_prefix, run, "policy")
    with open(os.path.join(policy_dir, "config.yaml")) as f:
        run_config = yaml.safe_load(f)

    policy_cfg = run_config["policy"]
    model_kwargs = policy_cfg["model_kwargs"]
    problem_cfg = dict(run_config["problem"])
    problem_cfg.update(run_config.get("policy_problem_override", {}))

    selector_cfg = run_config["selector"]
    extractor_kwargs = selector_cfg.get("features_extractor_kwargs", {})

    env_id = run_config["problem"]["env_id"]
    split = run_config["problem"].get("split", "train")
    dataset_stats = _dataset_stats(path_prefix, env_id, split)

    action_dim = run_config["problem"]["action_dim"]
    from problems.real_robot_problem.lerobot_utils import (
        CAMERA_CAPTURE_HW,
        CAMERA_NAME,
        IMAGE_SIZE,
        STATE_DIM,
    )

    queue_kwargs = policy_cfg.get("queue_kwargs", {})

    config = KeyframeMnemonicsConfig(
        buffer_size=problem_cfg["buffer_size"],
        total_slots=model_kwargs["total_slots"],
        rejection_threshold=queue_kwargs.get("rejection_threshold", 0.9),
        no_repeat_threshold=queue_kwargs.get("no_repeat_threshold", 0.1),
        image_size=IMAGE_SIZE,
        state_dim=STATE_DIM,
        camera_name=CAMERA_NAME,
        architecture=model_kwargs["policy_type"],
        chunk_size=model_kwargs.get("horizon", 1),
        n_action_steps=policy_cfg.get("open_loop_steps", 1),
        hidden_dim=model_kwargs["hidden_dim"],
        attn_heads=model_kwargs["attn_heads"],
        num_layers=model_kwargs["num_layers"],
        num_inference_steps=model_kwargs.get("num_inference_steps", 5),
        selector_features_dim=extractor_kwargs.get("features_dim", 128),
        selector_hidden_dim=extractor_kwargs.get("hidden_dim", 256),
        selector_net_arch=list(selector_cfg.get("net_arch", [500, 500])),
        device=device,
    )

    # The live robot observation, not the stage-3 dataset's per-slot expansion.
    # Capture resolution, not IMAGE_SIZE: advertising 128 would let the deployment
    # script resize instead of `resize_camera`, skipping the recorder's bottleneck.
    config.input_features = {
        f"observation.images.{CAMERA_NAME}": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, *CAMERA_CAPTURE_HW)
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
    }

    policy = KeyframeMnemonicsPolicy(config)

    selector_zip = os.path.join(stage_dir(path_prefix, run, "selector"), f"{experiment_name}_selector.zip")
    if not os.path.exists(selector_zip):
        raise FileNotFoundError(f"Selector checkpoint not found: {selector_zip}")
    ppo = PPO.load(selector_zip, device="cpu")
    missing, unexpected = policy.selector.load_state_dict(ppo.policy.state_dict(), strict=False)
    if missing:
        raise RuntimeError(f"Selector weights missing keys: {missing}")
    print(f"Loaded selector: {selector_zip}" + (f" (ignored {len(unexpected)} extra keys)" if unexpected else ""))

    policy_pth = _latest_policy_checkpoint(policy_dir, experiment_name, checkpoint_num)
    ckpt = torch.load(policy_pth, map_location="cpu", weights_only=False)
    policy.policy.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded policy  : {policy_pth}")

    _assert_stats_cover(config, dataset_stats)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset_stats)

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    preprocessor.save_pretrained(out)
    postprocessor.save_pretrained(out)

    print(f"\nExported to {out}")
    print(f"  lerobot-record --policy.path={out} ...")
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a keyframe-mnemonics run as a LeRobot pretrained_model directory."
    )
    parser.add_argument("--run", type=str, required=True,
                        help="Run folder name inside checkpoints/ (e.g. real_robot_RememberColor3_<TAG>)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the LeRobot checkpoint")
    parser.add_argument("--checkpoint_num", type=int, default=None,
                        help="Policy checkpoint number (default: latest)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device recorded in the exported config (default: cuda)")

    args = parser.parse_args()
    export(args.run, args.output, checkpoint_num=args.checkpoint_num, device=args.device)
