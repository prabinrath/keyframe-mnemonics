#!/usr/bin/env python
"""Convert a collected LeRobot dataset to the flat proxy H5 format.

Stages 1 and 2 read flat observation vectors from an H5, never from a simulator,
so a real-robot dataset plugs straight into the pipeline once flattened.
Run once per collected dataset.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from problems.real_robot_problem.lerobot_utils import (
    ACTION_DIM,
    CAMERA_NAME,
    OBS_DIM,
    STATE_DIM,
    flatten_observation,
)


def _resolve_stats(dataset):
    """State mean/std and action min/max from the dataset's own statistics.

    Every stage reads these numbers back out of the H5 attrs, so they are the one
    definition of the normalized space.
    """
    state = dataset.meta.stats.get("observation.state")
    action = dataset.meta.stats.get("action")
    for name, stats, keys in (("observation.state", state, ("mean", "std")),
                              ("action", action, ("min", "max"))):
        if not stats or any(k not in stats for k in keys):
            raise ValueError(f"Dataset metadata has no {'/'.join(keys)} for '{name}'.")

    mean = np.asarray(state["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(state["std"], dtype=np.float32).reshape(-1)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    action_min = np.asarray(action["min"], dtype=np.float32).reshape(-1)
    action_max = np.asarray(action["max"], dtype=np.float32).reshape(-1)
    return mean, std, action_min, action_max


def _normalize_action(action, action_min, action_max):
    """Map raw actions to [-1, 1]; matches LeRobot's MIN_MAX so the exported
    postprocessor inverts it exactly."""
    denom = action_max - action_min
    denom = np.where(denom == 0, 1.0, denom)
    return (2.0 * (action - action_min) / denom - 1.0).astype(np.float32)


def create_h5_dataset(dataset_root, repo_id, split="train", output_dir=None,
                      num_episodes=None):
    """Create the flat proxy H5 from a collected LeRobot dataset."""
    dataset = LeRobotDataset(repo_id, root=str(dataset_root))
    print(f"Loaded LeRobot dataset: {dataset_root}")
    print(f"  Episodes : {dataset.num_episodes}")
    print(f"  Frames   : {dataset.num_frames}")

    tasks = [str(t) for t in dataset.meta.tasks.index]
    task = tasks[0] if len(tasks) == 1 else ""
    if len(tasks) == 1:
        print(f"  Task     : {task!r}")
    else:
        print(f"  Task     : {len(tasks)} tasks found; leaving the H5 task attr empty")

    state_mean, state_std, action_min, action_max = _resolve_stats(dataset)
    print(f"  State  mean/std: {state_mean} / {state_std}")
    print(f"  Action min/max : {action_min} / {action_max}")

    output_dir = Path(output_dir or "datasets/real_robot/proxy_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{repo_id}_{split}.h5"

    total_episodes = dataset.num_episodes if num_episodes is None else min(num_episodes, dataset.num_episodes)

    with h5py.File(h5_path, "w") as h5f:
        demo_group = h5f.create_group("demo")

        written = 0
        current_episode = None
        observations, actions = [], []

        # Flush per episode rather than accumulating: the full dataset would be
        # ~6 GB of float32 in memory.
        def flush():
            nonlocal written, observations, actions
            if not observations:
                return
            traj = demo_group.create_group(str(written))
            traj.create_dataset("observations", data=np.stack(observations), dtype=np.float32)
            traj.create_dataset("actions", data=np.stack(actions), dtype=np.float32)
            traj.attrs["episode_length"] = len(observations)
            traj.attrs["source_episode_index"] = int(current_episode)
            written += 1
            observations, actions = [], []

        for idx in tqdm(range(dataset.num_frames), desc="Converting frames"):
            frame = dataset[idx]
            episode_index = int(frame["episode_index"])

            if current_episode is not None and episode_index != current_episode:
                flush()
                if written >= total_episodes:
                    break
            current_episode = episode_index

            state = (np.asarray(frame["observation.state"], dtype=np.float32) - state_mean) / state_std

            observations.append(
                flatten_observation(frame[f"observation.images.{CAMERA_NAME}"], state)
            )
            actions.append(
                _normalize_action(np.asarray(frame["action"], dtype=np.float32),
                                  action_min, action_max)
            )

        if written < total_episodes:
            flush()

        h5f.attrs["num_trajectories"] = written
        h5f.attrs["env_id"] = repo_id
        h5f.attrs["split"] = split
        h5f.attrs["observation_dim"] = OBS_DIM
        h5f.attrs["action_dim"] = ACTION_DIM
        h5f.attrs["state_dim"] = STATE_DIM
        h5f.attrs["fps"] = int(dataset.meta.fps)
        h5f.attrs["task"] = task
        # The exporter carries these into the LeRobot plugin checkpoint so inference
        # reproduces this space exactly.
        h5f.attrs["state_mean"] = state_mean
        h5f.attrs["state_std"] = state_std
        h5f.attrs["action_min"] = action_min
        h5f.attrs["action_max"] = action_max

    print(f"\nDone.\n  Trajectories : {written}\n  Dataset saved: {h5_path}")
    return str(h5_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a collected LeRobot dataset into the flat proxy H5."
    )
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the collected LeRobot dataset directory")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Dataset repo id (also used as env_id in the H5)")
    parser.add_argument("--split", type=str, default="train",
                        help="Split tag for the H5 filename and attrs (default: train)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: datasets/real_robot/proxy_dataset)")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Number of episodes to convert (default: all)")

    args = parser.parse_args()

    create_h5_dataset(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        split=args.split,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
    )
