#!/usr/bin/env python
"""Export a selector run plus a trained LeRobot policy as a keyframe_buffer checkpoint.

The inner policy stays where `lerobot-train` left it; this only packages the
frozen selector and the config that points at it.

    python -m lerobot_policy_keyframe_buffer.export \
        --run real_robot_RememberColor3_<TAG> \
        --inner_policy_type act \
        --inner_policy_path outputs/train/act_km/checkpoints/last/pretrained_model \
        --output outputs/km_buffer_act
"""

import argparse
import os
from pathlib import Path

import h5py
import yaml
from stable_baselines3 import PPO

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors

from common.helpers import get_problem_type_and_conf_path, parse_experiment_name, stage_dir

from .configuration_keyframe_buffer import KeyframeBufferConfig
from .modeling_keyframe_buffer import KeyframeBufferPolicy


def _state_normalization(path_prefix, env_id, split):
    h5_path = Path(path_prefix) / "datasets/real_robot/proxy_dataset" / f"{env_id}_{split}.h5"
    if not h5_path.exists():
        return None, None
    with h5py.File(h5_path, "r") as h5f:
        if "state_mean" not in h5f.attrs:
            return None, None
        return (
            [float(v) for v in h5f.attrs["state_mean"]],
            [float(v) for v in h5f.attrs["state_std"]],
        )


def export(run, inner_policy_type, inner_policy_path, output,
           chunk_size=1, n_action_steps=1, task=None, device="cuda"):
    experiment_name = parse_experiment_name(run)
    _, conf_path = get_problem_type_and_conf_path(experiment_name)
    with open(conf_path) as f:
        base_config = yaml.safe_load(f)
    path_prefix = base_config.get("path_prefix", "")

    selector_dir = stage_dir(path_prefix, run, "selector")
    with open(os.path.join(selector_dir, "config.yaml")) as f:
        run_config = yaml.safe_load(f)

    problem_cfg = dict(run_config["problem"])
    problem_cfg.update(run_config.get("policy_problem_override", {}))
    selector_cfg = run_config["selector"]
    extractor_kwargs = selector_cfg.get("features_extractor_kwargs", {})

    env_id = run_config["problem"]["env_id"]
    split = run_config["problem"].get("split", "train")
    state_mean, state_std = _state_normalization(path_prefix, env_id, split)

    from problems.real_robot_problem.lerobot_utils import (
        CAMERA_CAPTURE_HW,
        CAMERA_NAME,
        IMAGE_SIZE,
        STATE_DIM,
    )

    queue_kwargs = run_config["policy"].get("queue_kwargs", {})

    buffer_size = problem_cfg["buffer_size"]
    config = KeyframeBufferConfig(
        inner_policy_type=inner_policy_type,
        inner_policy_path=str(Path(inner_policy_path).resolve()),
        buffer_size=buffer_size,
        total_slots=buffer_size + 1,
        rejection_threshold=queue_kwargs.get("rejection_threshold", 0.9),
        no_repeat_threshold=queue_kwargs.get("no_repeat_threshold", 0.1),
        image_size=IMAGE_SIZE,
        state_dim=STATE_DIM,
        camera_name=CAMERA_NAME,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        selector_features_dim=extractor_kwargs.get("features_dim", 128),
        selector_hidden_dim=extractor_kwargs.get("hidden_dim", 256),
        selector_net_arch=list(selector_cfg.get("net_arch", [500, 500])),
        state_mean=state_mean,
        state_std=state_std,
        task=task,
        device=device,
    )

    # Capture resolution, not IMAGE_SIZE — see the note in plugin A's export.
    config.input_features = {
        f"observation.images.{CAMERA_NAME}": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, *CAMERA_CAPTURE_HW)
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(run_config["problem"]["action_dim"],))
    }

    policy = KeyframeBufferPolicy(config)

    selector_zip = os.path.join(selector_dir, f"{experiment_name}_selector.zip")
    if not os.path.exists(selector_zip):
        raise FileNotFoundError(f"Selector checkpoint not found: {selector_zip}")
    ppo = PPO.load(selector_zip, device="cpu")
    missing, _ = policy.selector.load_state_dict(ppo.policy.state_dict(), strict=False)
    if missing:
        raise RuntimeError(f"Selector weights missing keys: {missing}")
    print(f"Loaded selector    : {selector_zip}")
    print(f"Inner policy       : {inner_policy_type} @ {config.inner_policy_path}")

    preprocessor, postprocessor = make_pre_post_processors(config)

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
        description="Export a keyframe_buffer checkpoint wrapping a trained LeRobot policy."
    )
    parser.add_argument("--run", type=str, required=True,
                        help="Selector run folder inside checkpoints/")
    parser.add_argument("--inner_policy_type", type=str, required=True,
                        help="LeRobot policy type of the inner model (e.g. act, smolvla)")
    parser.add_argument("--inner_policy_path", type=str, required=True,
                        help="Path to the trained inner LeRobot checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the wrapper checkpoint")
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--n_action_steps", type=int, default=1)
    parser.add_argument("--task", type=str, default=None,
                        help="Task string passed to the inner policy (needed by VLAs)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    export(args.run, args.inner_policy_type, args.inner_policy_path, args.output,
           chunk_size=args.chunk_size, n_action_steps=args.n_action_steps,
           task=args.task, device=args.device)
