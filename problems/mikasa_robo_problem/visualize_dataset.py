#!/usr/bin/env python
"""
Real-time visualization of Mikasa Robo demonstration data using rerun.io

Joint structure (25 dims):
  [0:3]   TCP position (x, y, z)
  [3:7]   TCP orientation (quaternion: w, x, y, z)
  [7:14]  Arm joint positions (7 joints)
  [14:16] Gripper finger positions (2)
  [16:23] Arm joint velocities (7 joints)
  [23:25] Gripper finger velocities (2)

Usage:
    python visualize_dataset.py --env-name ShellGameTouch-v0 --episode-index 0
    python visualize_dataset.py --env-name ShellGameTouch-v0 --episode-index 0 5 10
    python visualize_dataset.py --env-name ShellGameTouch-v0 --info
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import rerun as rr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_episode(data_path: Path) -> dict:
    """Load a single episode from NPZ file."""
    data = np.load(data_path)
    return {
        'rgb': data['rgb'],
        'joints': data['joints'],
        'action': data['action'],
        'reward': data['reward'],
        'success': data['success'],
        'done': data['done']
    }


def visualize_episode(
    env_id: str,
    episode_index: int,
    dataset_dir: Path,
) -> None:
    """Visualize a single episode of demonstration data as a separate recording."""
    
    # Find the data file by episode index (matches _xx in filename)
    data_path = dataset_dir / f"train_data_{episode_index}.npz"
    if not data_path.exists():
        raise ValueError(f"Episode {episode_index} not found: {data_path}")
    
    logger.info(f"Loading episode {episode_index} from {data_path.name}")
    
    # Load episode data
    episode_data = load_episode(data_path)
    rgb = episode_data['rgb']
    joints = episode_data['joints']
    actions = episode_data['action']
    rewards = episode_data['reward']
    success = episode_data['success']
    
    num_timesteps = len(rewards)
    logger.info(f"Episode has {num_timesteps} timesteps")
    
    # Split RGB into two cameras (6 channels = 2 cameras × 3 RGB)
    rgb_camera1 = rgb[:, :, :, :3]
    rgb_camera2 = rgb[:, :, :, 3:]
    
    logger.info("Logging data to Rerun")
    cumulative_reward = 0.0
    
    # Main visualization loop
    for t in range(num_timesteps):
        # Set timeline
        rr.set_time("timestep", sequence=t)
        
        # Log RGB images (overhead and gripper views)
        rr.log("camera/overhead", rr.Image(rgb_camera1[t].astype(np.uint8)))
        rr.log("camera/gripper", rr.Image(rgb_camera2[t].astype(np.uint8)))
        
        # tcp_pose: Tool Center Point pose (7 dims) - xyz + quaternion (x,y,z,qx,qy,qz,qw)
        rr.log("tcp_pose", rr.Scalars(joints[t, 0:7]))

        # qpos: Joint positions (9 dims) - 7 arm + 2 gripper
        rr.log("qpos", rr.Scalars(joints[t, 7:16]))
        
        # qvel: Joint velocities (9 dims) - 7 arm + 2 gripper
        rr.log("qvel", rr.Scalars(joints[t, 16:25]))
        
        # Log actions (8 dims)
        rr.log("action", rr.Scalars(actions[t]))
        
        # Log metrics
        rr.log("metrics/reward", rr.Scalars(float(rewards[t])))
        cumulative_reward += float(rewards[t])
        rr.log("metrics/cumulative_reward", rr.Scalars(cumulative_reward))
        rr.log("metrics/success", rr.Scalars(float(success[t])))
        
        if (t + 1) % 10 == 0:
            logger.info(f"  Logged {t + 1}/{num_timesteps} timesteps")
    
    logger.info(f"Episode {episode_index} complete - Final reward: {cumulative_reward:.3f}")


def get_dataset_info(env_id: str, dataset_dir: Path) -> None:
    """Print dataset information."""
    data_files = sorted(dataset_dir.glob("train_data_*.npz"))
    
    if not data_files:
        logger.error(f"No data files found in {dataset_dir}")
        return
    
    sample_data = load_episode(data_files[0])
    
    print(f"\n{'='*60}")
    print("Dataset Information")
    print(f"{'='*60}")
    print(f"Environment: {env_id}")
    print(f"Total Episodes: {len(data_files)}")
    print(f"Episode Length: {len(sample_data['done'])} timesteps")
    print(f"RGB Shape: {sample_data['rgb'].shape}")
    print(f"Joint Dims: {sample_data['joints'].shape[1]}")
    print(f"Action Dims: {sample_data['action'].shape[1]}")
    print(f"Episodes: 0 to {len(data_files) - 1}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize Mikasa Robo data with Rerun")
    parser.add_argument("--env_id", type=str, default="ShellGameTouch-v0",
                       help="Environment name (e.g., ShellGameTouch-v0)")
    parser.add_argument("--episode_index", type=int, nargs="*", default=[0],
                       help="Episode indices to visualize (e.g., 0 1 2)")
    parser.add_argument("--dataset_path", type=str, default="datasets/mikasa_robo",
                       help="Path to dataset directory (default: datasets/mikasa_robo)")
    parser.add_argument("--info", action="store_true",
                       help="Show dataset information and exit")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_path) / args.env_id
    
    if not dataset_dir.exists():
        logger.error(f"Dataset not found: {dataset_dir}")
        return
    
    if args.info:
        get_dataset_info(args.env_id, dataset_dir)
        return
    
    # Visualize all episodes - each in its own viewer/recording
    for i, episode_idx in enumerate(args.episode_index):
        logger.info(f"\n{'='*60}")
        logger.info(f"Visualizing episode {episode_idx} ({i+1}/{len(args.episode_index)})")
        logger.info(f"{'='*60}")
        
        try:
            # Initialize Rerun with unique recording ID and spawn viewer
            rr.init(f"{args.env_id}/episode_{episode_idx}", spawn=True)
            
            # Load and visualize episode
            visualize_episode(
                env_id=args.env_id,
                episode_index=episode_idx,
                dataset_dir=dataset_dir,
            )
            
            # Small delay between episodes
            if i < len(args.episode_index) - 1:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"\nAll {len(args.episode_index)} episodes visualized.")


if __name__ == "__main__":
    main()
