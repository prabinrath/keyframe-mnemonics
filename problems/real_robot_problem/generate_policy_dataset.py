#!/usr/bin/env python
"""Generate the stage-3 policy LeRobot dataset for real_robot using a trained selector.

Reads the collected LeRobot dataset, rebuilding the selector's flat observation
inline so the dataset stays grounded in the original source rather than chained
off the proxy H5.

The output is a plain LeRobot dataset, so it can also be handed to `lerobot-train`
to fit VLA baselines on the same selector-curated buffers. Pass --no_normalization
for that case: it writes raw state and actions, matching the collected dataset, so
LeRobot's own processors handle normalization.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from common.helpers import get_problem_type_and_conf_path, parse_experiment_name, set_seeds, stage_dir
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from stable_baselines3 import PPO
from keyframe_mnemonics.buffers import queue_dict
from problems.real_robot_problem.lerobot_utils import (
    CAMERA_NAME,
    build_policy_observation,
    flatten_observation,
    get_lerobot_features,
)
from problems.real_robot_problem.make_h5_dataset import _normalize_action, _resolve_stats


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_policy_dataset(
    selector_checkpoint: str,
    env_id: str,
    split: str = "train",
    num_episodes: int | None = None,
    queue_strategy: str = "evict_latest_norepeat",
    queue_kwargs: dict | None = None,
    no_normalization: bool = False,
    task: str | None = None,
    fps: int | None = None,
    output_dir: str | Path | None = None,
):
    """Generate the stage-3 policy LeRobot dataset from the collected dataset using a trained selector.

    task and fps default to the collected dataset's values; queue_kwargs to
    `policy.queue_kwargs` in the experiment config. no_normalization writes raw
    state and actions for `lerobot-train`; the default writes the normalized space
    the keyframe_mnemonics policy trains in.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config from checkpoint
    experiment_name = parse_experiment_name(selector_checkpoint)

    _, conf_path = get_problem_type_and_conf_path(experiment_name)
    with open(conf_path) as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {conf_path}")

    path_prefix = config.get("path_prefix", "")

    selector_dir = Path(stage_dir(path_prefix, selector_checkpoint, "selector"))
    ckpt_conf_path = selector_dir / "config.yaml"
    with open(ckpt_conf_path) as f:
        ckpt_config = yaml.safe_load(f)
    ckpt_config["selector"]["logging"] = config["selector"]["logging"]
    assert ckpt_config["selector"] == config["selector"], (
        "Selector config mismatch between current config and checkpoint config."
    )

    if queue_kwargs is None:
        queue_kwargs = config["policy"].get("queue_kwargs", {})

    # Resolve buffer size
    problem_cfg = dict(config["problem"])
    problem_cfg.update(config.get("policy_problem_override", {}))
    problem_cfg["use_current_obs"] = True
    buffer_size = problem_cfg["buffer_size"]
    total_slots = config["policy"]["model_kwargs"]["total_slots"]
    assert total_slots == buffer_size + 1, (
        f"total_slots must be buffer_size + 1 (for current obs). "
        f"Got total_slots={total_slots}, buffer_size={buffer_size}."
    )

    # Open the collected dataset
    raw_root = Path(path_prefix) / "datasets/real_robot/raw_datasets" / env_id
    if not raw_root.exists():
        raise FileNotFoundError(f"Collected dataset not found: {raw_root}")

    src = LeRobotDataset(env_id, root=str(raw_root))
    state_mean, state_std, action_min, action_max = _resolve_stats(src)
    if fps is None:
        fps = int(src.meta.fps)

    episodes = src.meta.episodes.to_dict()
    bounds = list(zip(episodes["dataset_from_index"], episodes["dataset_to_index"]))
    if num_episodes is not None:
        bounds = bounds[:num_episodes]

    print(f"Collected dataset   : {raw_root}")
    print(f"Trajectories to process: {len(bounds)}")
    print(f"Buffer size: {buffer_size}   total_slots: {total_slots}   fps: {fps}")
    print(f"Queue: {queue_strategy}   {queue_kwargs}")
    print(f"Normalization: {'raw (for lerobot-train)' if no_normalization else 'normalized'}")

    # Load selector model
    ckpt_file = selector_dir / f"{experiment_name}_selector.zip"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Selector zip not found: {ckpt_file}")
    ckpt_file = str(ckpt_file)
    selector = PPO.load(ckpt_file, device=device)
    selector.policy.set_training_mode(False)
    print(f"Loaded selector: {ckpt_file}")

    # Create LeRobot dataset
    features = get_lerobot_features(total_slots)
    repo_id = f"{experiment_name}_policy_dataset"
    if output_dir is None:
        output_dir = Path(path_prefix) / "datasets/real_robot/policy_dataset" / repo_id
    output_dir = Path(output_dir)

    print(f"Creating LeRobot dataset at: {output_dir}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=str(output_dir),
        robot_type=src.meta.robot_type,
        features=features,
        use_videos=True,
        image_writer_processes=4,
        image_writer_threads=4,
    )

    # Generate episodes
    with VideoEncodingManager(dataset):
        for start, stop in tqdm(bounds, desc="Generating episodes"):
            buf = queue_dict[queue_strategy](
                maxsize=buffer_size, use_current_obs=problem_cfg["use_current_obs"],
                **queue_kwargs,
            )

            for idx in range(start, stop):
                src_frame = src[idx]
                raw_state = np.asarray(src_frame["observation.state"], dtype=np.float32)
                raw_action = np.asarray(src_frame["action"], dtype=np.float32)

                # The selector always sees the normalized space it was trained on.
                obs_np = flatten_observation(
                    src_frame[f"observation.images.{CAMERA_NAME}"],
                    (raw_state - state_mean) / state_std,
                )
                priority, _ = selector.predict(obs_np, deterministic=True)
                buf.push(torch.as_tensor(obs_np), float(priority.item()))

                frame = build_policy_observation(buf.get().numpy(), total_slots)
                if no_normalization:
                    # Buffer state is normalized; the last slot is this frame, so the
                    # raw value is an exact substitute.
                    frame["observation.state"] = raw_state
                    frame["action"] = raw_action
                else:
                    frame["action"] = _normalize_action(raw_action, action_min, action_max)
                frame["task"] = task if task else str(src_frame["task"])

                dataset.add_frame(frame)

            dataset.save_episode()

    # Finalise dataset
    dataset.finalize()

    print("\nDone.")
    print(f"  Total episodes : {dataset.meta.total_episodes}")
    print(f"  Total frames   : {dataset.meta.total_frames}")
    print(f"  Dataset saved  : {output_dir}")

    return str(output_dir)


def generate(selector_checkpoint, seed=42, **kwargs):
    """Uniform entrypoint used by the train.py orchestrator.

    Derives env_id from the experiment config; forwards remaining kwargs to
    generate_policy_dataset (num_episodes, queue_strategy, ...).
    """
    if seed is not None:
        set_seeds(seed)

    experiment_name = parse_experiment_name(selector_checkpoint)
    _, conf_path = get_problem_type_and_conf_path(experiment_name)
    with open(conf_path) as f:
        config = yaml.safe_load(f)

    return generate_policy_dataset(
        selector_checkpoint=selector_checkpoint,
        env_id=config["problem"]["env_id"],
        split=config["problem"].get("split", "train"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the stage-3 real_robot policy dataset from the collected dataset using a trained selector."
    )
    parser.add_argument(
        "--selector_checkpoint",
        type=str,
        required=True,
        help="Selector checkpoint directory name inside checkpoints/",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="remember_color_2",
        help="Collected dataset directory name under datasets/real_robot/raw_datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split tag (default: train)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of trajectories to process (default: all).",
    )
    parser.add_argument(
        "--queue_strategy",
        type=str,
        default="evict_latest_norepeat",
        choices=list(queue_dict.keys()),
        help="Buffer eviction strategy (default: evict_latest_norepeat)",
    )
    parser.add_argument(
        "--no_normalization",
        action="store_true",
        help="Write raw state and actions, matching the collected dataset, for lerobot-train.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task description string (default: the collected dataset's).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second (default: the collected dataset's).",
    )

    args = parser.parse_args()

    generate_policy_dataset(
        selector_checkpoint=args.selector_checkpoint,
        env_id=args.env_id,
        split=args.split,
        num_episodes=args.num_episodes,
        queue_strategy=args.queue_strategy,
        no_normalization=args.no_normalization,
        task=args.task,
        fps=args.fps,
    )
