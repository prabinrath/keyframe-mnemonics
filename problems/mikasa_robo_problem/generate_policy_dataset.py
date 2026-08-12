#!/usr/bin/env python
"""Generate policy LeRobot dataset for mikasa_robo using a trained selector.

Reads from NPZ files.
State : qpos only (9 dims, joints[7:16]).
Action: raw actions from NPZ (8 dims).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from common.helpers import parse_experiment_name, set_seeds, stage_dir
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from stable_baselines3 import PPO
from keyframe_mnemonics.buffers import queue_dict
from problems.mikasa_robo_problem.lerobot_utils import get_lerobot_features, build_policy_observation
from problems.mikasa_robo_problem.rotate_prompts import inject_rotate_prompt


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_policy_dataset(
    selector_checkpoint: str,
    env_id: str,
    npz_dataset_root: str | Path,
    num_episodes: int | None = None,
    queue_strategy: str = "evict_latest_norepeat",
    task: str = "",
    fps: int = 10,
    filter_failures: bool = True,
    output_dir: str | Path | None = None,
):
    """Generate a policy LeRobot dataset from NPZ files using a trained selector.

    Args:
        selector_checkpoint: Checkpoint directory name (e.g. mikasa_robo_RememberColor3_20260216_102348).
        env_id             : ManiSkill environment ID (e.g. RememberColor3-v0).
        npz_dataset_root   : Path to root directory containing per-env NPZ folders.
        num_episodes       : Number of episodes to process. Defaults to all.
        queue_strategy     : Buffer eviction strategy (e.g. 'evict_latest_norepeat', 'evict_past').
        task               : Task description string embedded in every frame.
        fps                : Frames per second for the LeRobot dataset.
        filter_failures    : Skip trajectories that did not succeed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load config from checkpoint
    # ------------------------------------------------------------------
    # Checkpoint name format: mikasa_robo_{ExperimentName}_{YYYYMMDD}_{HHMMSS}
    experiment_name = parse_experiment_name(selector_checkpoint)

    conf_path = Path("conf/mikasa_robo") / f"{experiment_name}.yaml"
    with open(conf_path) as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {conf_path}")

    path_prefix = config.get("path_prefix", "")

    # Verify selector sub-config matches
    selector_dir = Path(stage_dir(path_prefix, selector_checkpoint, "selector"))
    ckpt_conf_path = selector_dir / "config.yaml"
    with open(ckpt_conf_path) as f:
        ckpt_config = yaml.safe_load(f)
    ckpt_config["selector"]["logging"] = config["selector"]["logging"]
    assert ckpt_config["selector"] == config["selector"], (
        "Selector config mismatch between current config and checkpoint config."
    )

    # ------------------------------------------------------------------
    # Resolve buffer size
    # ------------------------------------------------------------------
    problem_cfg = config["problem"]
    problem_cfg.update(config.get("policy_problem_override", {}))
    problem_cfg["use_current_obs"] = True
    buffer_size = problem_cfg["buffer_size"]
    total_slots = config["policy"]["model_kwargs"]["total_slots"]
    assert total_slots == buffer_size + 1, f"total_slots must be buffer_size + 1 (for current obs). Got total_slots={total_slots}, buffer_size={buffer_size}."

    # ------------------------------------------------------------------
    # Collect NPZ files
    # ------------------------------------------------------------------
    npz_dir = Path(npz_dataset_root) / env_id
    if not npz_dir.exists():
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")

    data_files = sorted(npz_dir.glob("train_data_*.npz"))
    if not data_files:
        raise FileNotFoundError(f"No NPZ files found in {npz_dir}")

    # Hover demos are open-loop replays that drift off the cube after succeeding, so
    # success[-1] holds very few demos.
    is_hover = "Hover" in env_id
    succeeded = ((lambda s: bool(s.any())) if is_hover else (lambda s: bool(s[-1])))

    if filter_failures:
        print("Filtering failed trajectories …"
              + (" (hover: success at any step)" if is_hover else ""))
        data_files = [f for f in data_files if succeeded(np.load(f)["success"])]
        print(f"  {len(data_files)} successful trajectories kept.")

    if num_episodes is not None:
        data_files = data_files[:num_episodes]

    print(f"Episodes to process : {len(data_files)}")
    print(f"Buffer size: {buffer_size}   total_slots: {total_slots}")

    # ------------------------------------------------------------------
    # Load selector model
    # ------------------------------------------------------------------
    ckpt_file = selector_dir / f"{experiment_name}_selector.zip"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Selector zip not found: {ckpt_file}")
    ckpt_file = str(ckpt_file)
    selector = PPO.load(ckpt_file, device=device)
    selector.policy.set_training_mode(False)
    print(f"Loaded selector: {ckpt_file}")

    # ------------------------------------------------------------------
    # Create LeRobot dataset
    # ------------------------------------------------------------------
    features   = get_lerobot_features(total_slots)
    repo_id    = f"{experiment_name}_policy_dataset"
    if output_dir is None:
        output_dir = Path(path_prefix) / "datasets/mikasa_robo/policy_dataset" / repo_id
    output_dir = Path(output_dir)

    print(f"Creating LeRobot dataset at: {output_dir}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=str(output_dir),
        robot_type="mikasa_robo",
        features=features,
        use_videos=True,
        image_writer_processes=4,
        image_writer_threads=4,
    )

    # ------------------------------------------------------------------
    # Generate episodes
    # ------------------------------------------------------------------
    with VideoEncodingManager(dataset):
        for data_file in tqdm(data_files, desc="Generating episodes"):
            data = np.load(data_file)
            rgb     = data["rgb"]      # (T, H, W, 6) uint8
            joints  = data["joints"]   # (T, 25)
            actions = data["action"]   # (T, 8)

            # Original train_data_<i> index — used to look up this trajectory's Rotate
            # target_angle prompt; no-op for non-Rotate tasks.
            orig_idx = int(data_file.stem.split("_")[-1])

            buf = queue_dict[queue_strategy](maxsize=buffer_size, use_current_obs=problem_cfg["use_current_obs"])

            for t in range(len(data["done"])):
                # Build flat obs for selector (images normalised to [0,1])
                overhead = rgb[t, :, :, :3].astype(np.float32) / 255.0
                gripper  = rgb[t, :, :, 3:].astype(np.float32) / 255.0
                obs_np = np.concatenate([
                    overhead.flatten(), gripper.flatten(),
                    joints[t],  # tcp_pose(7) + qpos(9) + qvel(9) = 25 dims
                ]).astype(np.float32)

                # Inject the Rotate prompt into the redundant last-qpos slot before the
                # selector sees it (matches the injected proxy H5 the selector trained on).
                inject_rotate_prompt(obs_np, env_id, orig_idx)

                priority, _ = selector.predict(obs_np, deterministic=True)
                buf.push(torch.as_tensor(obs_np), float(priority.item()))

                frame = build_policy_observation(buf.get().numpy(), total_slots)
                frame["action"] = actions[t].astype(np.float32)
                frame["task"]   = task

                dataset.add_frame(frame)

            dataset.save_episode()

    # ------------------------------------------------------------------
    # Finalise dataset
    # ------------------------------------------------------------------
    dataset.finalize()

    print("\nDone.")
    print(f"  Total episodes : {dataset.meta.total_episodes}")
    print(f"  Total frames   : {dataset.meta.total_frames}")
    print(f"  Dataset saved  : {output_dir}")

    return str(output_dir)


def generate(selector_checkpoint, seed=42, **kwargs):
    """Uniform entrypoint used by the train.py orchestrator.

    Derives env_id from the experiment config; forwards remaining kwargs to
    generate_policy_dataset (npz_dataset_root, num_episodes, queue_strategy, ...).
    """
    if seed is not None:
        set_seeds(seed)

    experiment_name = parse_experiment_name(selector_checkpoint)
    conf_path = Path("conf/mikasa_robo") / f"{experiment_name}.yaml"
    with open(conf_path) as f:
        config = yaml.safe_load(f)

    return generate_policy_dataset(
        selector_checkpoint=selector_checkpoint,
        env_id=config["problem"]["env_id"],
        npz_dataset_root=kwargs.pop("npz_dataset_root", "datasets/mikasa_robo"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate policy mikasa_robo dataset from NPZ files using a trained selector."
    )
    parser.add_argument(
        "--selector_checkpoint",
        type=str,
        default="mikasa_robo_RememberColor3_20260216_102348",
        help="Selector checkpoint directory name inside checkpoints/",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="RememberColor3-v0",
        help="ManiSkill environment ID (e.g. RememberColor3-v0)",
    )
    parser.add_argument(
        "--npz_dataset_root",
        type=str,
        default="datasets/mikasa_robo",
        help="Root directory containing per-env NPZ folders",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to process (default: all).",
    )
    parser.add_argument(
        "--queue_strategy",
        type=str,
        default="evict_latest_norepeat",
        choices=list(queue_dict.keys()),
        help="Buffer eviction strategy (default: evict_latest_norepeat)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Remember the color of the cube and then pick the matching one",
        help="Task description string embedded in every dataset frame.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the LeRobot dataset (default: 10).",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Include failed trajectories (filtered by default).",
    )

    args = parser.parse_args()

    generate_policy_dataset(
        selector_checkpoint=args.selector_checkpoint,
        env_id=args.env_id,
        npz_dataset_root=args.npz_dataset_root,
        num_episodes=args.num_episodes,
        queue_strategy=args.queue_strategy,
        task=args.task,
        fps=args.fps,
        filter_failures=not args.no_filter,
    )
