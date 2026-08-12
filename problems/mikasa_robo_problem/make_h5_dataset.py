#!/usr/bin/env python
"""Convert NPZ dataset to H5 format with preprocessing."""

import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
import argparse

from problems.mikasa_robo_problem.rotate_prompts import inject_rotate_prompt


def process_env_observation(obs_dict):
    """Process environment observation dictionary into flat array.
    
    Extracts and flattens in order:
    1. overhead_camera (rgb[:,:,:3]) - 128x128x3
    2. gripper_camera (rgb[:,:,3:]) - 128x128x3
    3. tcp_pose (joints[0:7]) - 7 dims (xyz + quaternion)
    4. qpos (joints[7:16]) - 9 dims (7 arm + 2 gripper positions)
    5. qvel (joints[16:25]) - 9 dims (7 arm + 2 gripper velocities)
    
    Total: 98329 dimensions
    """
    # Process RGB
    rgb = obs_dict['rgb']
    
    # Split RGB into gripper and overhead cameras
    overhead_camera = rgb[:, :, :3]  # First camera (channels 0-2)
    gripper_camera = rgb[:, :, 3:]  # Second camera (channels 3-5)
    
    # Normalize to [0, 1] range
    overhead_camera = overhead_camera.astype(np.float32) / 255.0
    gripper_camera = gripper_camera.astype(np.float32) / 255.0
    
    # Process joints (state)
    joints = obs_dict['state']
    
    # Extract joint components
    tcp_pose = joints[0:7]  # TCP pose: xyz + quaternion
    qpos = joints[7:16]   # Joint positions: 7 arm + 2 gripper
    qvel = joints[16:25]  # Joint velocities: 7 arm + 2 gripper        
    
    # Flatten and concatenate in the specified order
    return np.concatenate([
        overhead_camera.flatten(),
        gripper_camera.flatten(),
        tcp_pose,
        qpos,
        qvel
    ])


def create_h5_dataset(folder_path, split="train", filter_failures=True):
    """Create H5 dataset from NPZ files."""
    dataset_dir = Path(folder_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Extract env_id from folder path
    env_id = dataset_dir.name
    
    # Get all data files
    file_pattern = f"{split}_data_*.npz"
    data_files = sorted(dataset_dir.glob(file_pattern))
    
    if len(data_files) == 0:
        raise FileNotFoundError(f"No data files found matching pattern: {file_pattern}")
    
    print(f"Found {len(data_files)} {split} trajectories")
    
    # Hover demos are open-loop replays that drift off the cube after succeeding, so
    # success[-1] holds very few demos; the env terminates on first success.
    is_hover = "Hover" in env_id
    succeeded = ((lambda d: bool(d['success'].any())) if is_hover
                 else (lambda d: bool(d['success'][-1])))

    # Filter out failures if requested
    if filter_failures:
        print("Filtering failed trajectories..."
              + (" (hover: success at any step)" if is_hover else ""))
        valid_files = []
        for data_file in data_files:
            if succeeded(np.load(data_file)):
                valid_files.append(data_file)
        print(f"Kept {len(valid_files)} successful trajectories (filtered {len(data_files) - len(valid_files)} failures)")
        data_files = valid_files
    
    if len(data_files) == 0:
        raise ValueError("No valid trajectories after filtering!")
    
    # Create H5 file in datasets/mikasa_robo/proxy_dataset directory
    h5_output_dir = Path("datasets/mikasa_robo/proxy_dataset")
    h5_output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = h5_output_dir / f"{env_id}_{split}.h5"

    print(f"Creating H5 dataset at: {h5_path}")
    total_samples = 0
    
    with h5py.File(h5_path, 'w') as h5f:
        # Create root demo group
        demo_group = h5f.create_group('demo')
        
        # Process each trajectory
        for traj_idx, data_file in enumerate(tqdm(data_files, desc="Processing trajectories")):
            data = np.load(data_file)
            
            # Create trajectory group
            traj_group = demo_group.create_group(str(traj_idx))

            # Get episode length
            episode_length = len(data['done'])

            # Original train_data_<i> index (survives failure filtering) — used to look
            # up this trajectory's Rotate target_angle prompt; no-op for non-Rotate tasks.
            orig_idx = int(data_file.stem.split('_')[-1])

            # Process all timesteps and store as arrays
            observations = []
            actions = []
            for t in range(episode_length):
                obs_dict = {'rgb': data['rgb'][t], 'state': data['joints'][t]}
                observation = process_env_observation(obs_dict)
                inject_rotate_prompt(observation, env_id, orig_idx)
                observations.append(observation)
                actions.append(data['action'][t])
            
            obs_array = np.array(observations, dtype=np.float32)
            action_array = np.array(actions, dtype=np.float32)
            
            # Store as 2D datasets for efficient slicing
            traj_group.create_dataset('observations', data=obs_array)
            traj_group.create_dataset('actions', data=action_array)
            
            # Store trajectory metadata
            traj_group.attrs['episode_length'] = episode_length
            traj_group.attrs['success'] = succeeded(data)
            traj_group.attrs['source_file'] = data_file.name

            total_samples += len(obs_array)
        
        # Store dataset metadata
        h5f.attrs['num_trajectories'] = len(data_files)
        h5f.attrs['env_id'] = env_id
        h5f.attrs['split'] = split
        h5f.attrs['observation_dim'] = 98329
        h5f.attrs['action_dim'] = data['action'].shape[1]
    
    print(f"Successfully created H5 dataset with {len(data_files)} trajectories")
    print(f"Total samples: {total_samples}")
    print(f"Saved to: {h5_path}")

    return h5_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPZ dataset to H5 format")
    parser.add_argument("--folder_path", type=str, default="datasets/mikasa_robo/ShellGameTouch-v0",
                       help="Path to folder containing NPZ files")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                       help="Dataset split")
    parser.add_argument("--no_filter", action="store_true",
                       help="Don't filter out failed trajectories")
    
    args = parser.parse_args()
    
    create_h5_dataset(
        folder_path=args.folder_path,
        split=args.split,
        filter_failures=not args.no_filter
    )
