"""Evaluate trained policy on Mikasa Robo problems."""

import torch
import argparse
import yaml
import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from io import BytesIO
from stable_baselines3 import PPO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from keyframe_mnemonics.buffers import Process
from problems import problem_dict
from models import model_dict
from common.helpers import set_seeds, parse_experiment_name, stage_dir
from problems.mikasa_robo_problem.lerobot_utils import build_policy_observation
from problems.mikasa_robo_problem.mikasa_robo_problem import apply_delta_time_override
from problems.mikasa_robo_problem.rotate_prompts import ROTATE_ENV_IDS, PROMPT_SLOT


def eval_policy(
    policy_checkpoint,
    selector_checkpoint=None,
    checkpoint_num=None,
    record_video=False,
    video_dir="rollout/mikasa_robo/videos",
    episode_indices=None,
    seed=42,
    delta_time=None,
):
    """Evaluate a trained policy with a trained selector for buffer management.

    Args:
        policy_checkpoint: Policy checkpoint directory name (also contains selector)
        record_video: Whether to record env + buffer videos
        video_dir: Directory to save videos
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if seed is not None:
        set_seeds(seed)

    # Extract experiment_name from policy checkpoint
    experiment_name = parse_experiment_name(policy_checkpoint)
    
    # Load config from conf/ for path_prefix and problem setup
    with open(f"conf/mikasa_robo/{experiment_name}.yaml") as f:
        config = yaml.safe_load(f)
    path_prefix = config.get("path_prefix", "")

    # Setup test problem from conf/ config
    problem_config = config.get("problem")
    problem_config.update(config.get("policy_problem_override", {}))
    problem_config.update(config.get("test_problem_override", {}))
    problem_config["use_current_obs"] = True
    if delta_time is not None:
        apply_delta_time_override(problem_config, delta_time)
    if record_video:
        problem_config.setdefault("render_mode", "all")

    # Set total_slots
    policy_checkpoint_path = stage_dir(path_prefix, policy_checkpoint, "policy")
    with open(os.path.join(policy_checkpoint_path, "config.yaml")) as f:
        policy_checkpoint_config = yaml.safe_load(f)
    policy_config = policy_checkpoint_config.get("policy", {})
    model_config = policy_config.get("model_kwargs", {})
    total_slots = model_config["total_slots"]

    print(f"num_inference_steps: {model_config.get('num_inference_steps')}")
    print(f"open_loop_steps: {policy_config.get('open_loop_steps')}")

    print("\nTest Configuration:")
    print(f"  Environment: {problem_config['env_id']}")

    # Resolve policy checkpoint file
    policy_checkpoint_dir = policy_checkpoint_path
    if checkpoint_num is not None:
        policy_file = os.path.join(policy_checkpoint_dir, f"{experiment_name}_policy_{checkpoint_num}.pth")
        if not os.path.exists(policy_file):
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_file}")
    else:
        policy_pattern = os.path.join(policy_checkpoint_dir, f"{experiment_name}_policy_*.pth")
        policy_checkpoints = glob.glob(policy_pattern)
        if not policy_checkpoints:
            raise FileNotFoundError(f"No policy checkpoints found matching {policy_pattern}")
        policy_file = max(policy_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_num = policy_file.split('_')[-1].split('.')[0]

    # Create test problem and process buffer
    problem_class = problem_dict.get("mikasa_robo").get("problem")
    problem = problem_class(**problem_config)
    problem.randomize_reset = False
    process = Process(problem, queue_strategy="evict_latest_norepeat")

    # Rotate* tasks carry the goal (target_angle) in the redundant last-qpos slot.
    is_rotate = problem_config["env_id"] in ROTATE_ENV_IDS

    # Setup video recording
    if record_video:
        video_path = Path(video_dir) / policy_checkpoint / checkpoint_num
        video_path.mkdir(parents=True, exist_ok=True)
        print(f"Recording videos to: {video_path}")
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        axes[0].axis('off')
        axes[1].axis('off')
        fig.tight_layout()
    
    # Load selector model — prefer the run's selector subfolder, fall back to selector_checkpoint arg
    selector_dir = stage_dir(path_prefix, policy_checkpoint, "selector")
    selector_file = os.path.join(selector_dir, f"{experiment_name}_selector.zip")
    if not os.path.exists(selector_file):
        if not selector_checkpoint:
            raise FileNotFoundError(f"No selector found in {selector_dir} and --selector-checkpoint not provided")
        selector_file = os.path.join(stage_dir(path_prefix, selector_checkpoint, "selector"), f"{experiment_name}_selector.zip")
        if not os.path.exists(selector_file):
            raise FileNotFoundError(f"Selector checkpoint not found: {selector_file}")
    
    selector_model = PPO.load(selector_file, device=device)
    selector_model.policy.set_training_mode(False)
    print(f"\nLoaded selector from: {selector_file}")
    
    # Load policy model
    # Get model config from policy checkpoint config
    policy_class = model_dict.get("mikasa_robo").get("policy")

    # Create policy model
    policy_model = policy_class(
        output_dim=process.problem.action_dim,
        **model_config
    ).to(device)
    
    # Load policy weights
    checkpoint = torch.load(policy_file, map_location=device, weights_only=False)
    policy_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded policy from: {policy_file}")
    policy_model.eval()
    
    # Setup action charts directory
    charts_path = Path(video_dir) / policy_checkpoint / checkpoint_num
    charts_path.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    episode_list = episode_indices if episode_indices else list(range(process.problem.num_variations))
    print(f"\nRunning evaluation on {len(episode_list)} episodes...")
    episode_successes = []
    episode_rewards = []
    
    for episode in tqdm(episode_list, desc="Evaluating"):
        # Reset process and problem
        process.reset(episode)
        buffer_frames = [] if record_video else None
        env_frames = [] if record_video else None
        episode_actions = []
        action_horizon = 0
        
        # Run episode
        for idx in range(process.problem.seq_len):
            # Get observation
            obs, t = process.get_obs(idx)
            
            # Convert to tensor if needed
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32)

            # Rotate*: overwrite the redundant last-qpos slot with the live target_angle,
            # matching the training-time injection (no-op for other tasks).
            if is_rotate:
                obs[PROMPT_SLOT] = float(process.problem.env.unwrapped.target_angle.reshape(-1)[0])

            # Use selector to get priority (deterministic)
            priority, _ = selector_model.predict(obs, deterministic=True)
            
            # Add observation to buffer with priority
            process.set_action(obs, priority)
            
            # Get buffer content
            buffer = process.get_buffer()
            
            # Convert to numpy for parsing
            if isinstance(buffer, torch.Tensor):
                buffer = buffer.cpu().numpy()
            
            # Parse buffer into policy observation format
            policy_obs = build_policy_observation(buffer, total_slots)
            
            # Convert to tensors and add batch dimension
            for key in policy_obs:
                if 'image' in key:
                    # Images: (H, W, C) uint8 -> (1, C, H, W) float32 [0, 1]
                    policy_obs[key] = torch.from_numpy(policy_obs[key]).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                else:
                    # State: (D,) -> (1, D)
                    policy_obs[key] = torch.from_numpy(policy_obs[key]).unsqueeze(0).to(device)
            
            # Get action from policy (already has inference_mode inside)
            if action_horizon == 0:
                pred_target = policy_model.get_action(policy_obs, deterministic=True).cpu().squeeze(0)
                action_horizon = policy_config.get("open_loop_steps")

            action_vec = pred_target[policy_config.get("open_loop_steps")-action_horizon]
            process.problem.optimal_action = action_vec.unsqueeze(0)
            action_horizon -= 1
            episode_actions.append(action_vec.cpu().numpy())

            if record_video:
                env_frame = process.problem.env.render()
                if isinstance(env_frame, torch.Tensor):
                    env_frame = env_frame.detach().cpu().numpy()
                env_frame = np.asarray(env_frame)
                if env_frame.ndim == 4:
                    env_frame = env_frame[0]
                env_frames.append(env_frame.astype(np.uint8))

                buf_np = buffer.reshape(total_slots, -1)
                overhead_imgs = [np.clip(buf_np[i, :49152].reshape(128, 128, 3), 0, 1) for i in range(total_slots)]
                gripper_imgs  = [np.clip(buf_np[i, 49152:98304].reshape(128, 128, 3), 0, 1) for i in range(total_slots)]
                axes[0].clear(); axes[0].imshow(np.concatenate(overhead_imgs, axis=1)); axes[0].set_title(f"Overhead Buffer - Step {idx+1}"); axes[0].axis('off')
                axes[1].clear(); axes[1].imshow(np.concatenate(gripper_imgs,  axis=1)); axes[1].set_title(f"Gripper Buffer  - Step {idx+1}"); axes[1].axis('off')
                bio = BytesIO(); fig.savefig(bio, format='png', dpi=100, bbox_inches='tight'); bio.seek(0)
                buffer_frames.append(imageio.imread(bio)); bio.close()
            
            # Break early if episode is successful
            if process.problem.episode_success:
                break
        
        # Save action line chart
        if episode_actions:
            actions_arr = np.array(episode_actions)  # (T, action_dim)
            fig_ac, ax_ac = plt.subplots(figsize=(10, 4))
            for dim in range(actions_arr.shape[1]):
                ax_ac.plot(actions_arr[:, dim], label=f"dim {dim}")
            ax_ac.set_xlabel("Step"); ax_ac.set_ylabel("Action")
            ax_ac.set_title(f"Episode {episode} Actions")
            ax_ac.legend(loc="upper right", fontsize=6, ncol=4)
            fig_ac.tight_layout()
            fig_ac.savefig(charts_path / f"{episode}_action.png", dpi=100)
            plt.close(fig_ac)

        # Collect episode results
        episode_successes.append(process.problem.episode_success)
        episode_rewards.append(process.problem.episode_reward)

        if record_video and env_frames:
            imageio.mimwrite(str(video_path / f"{episode}_env.mp4"), env_frames, fps=30, codec='h264', quality=8)

        if record_video and buffer_frames:
            out = video_path / f"{episode}_buffer.mp4"
            frames_u8 = [(f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8) for f in buffer_frames]
            imageio.mimwrite(str(out), frames_u8, fps=5, codec='h264', quality=8)
    
    # Compute metrics
    process.reset()  # flush last video
    if record_video:
        plt.close('all')

    success_rate = sum(episode_successes) / len(episode_successes)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    
    print(f"\n{'='*60}")
    print("Evaluation Results:")
    print(f"  Success Rate: {success_rate:.2%} ({sum(episode_successes)}/{len(episode_successes)})")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"{'='*60}")
    
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'episode_successes': episode_successes,
        'episode_rewards': episode_rewards
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained policy on Mikasa Robo problems")
    parser.add_argument("--policy_checkpoint", type=str,
                       default="",
                       help="Policy checkpoint directory name")
    parser.add_argument("--selector_checkpoint", type=str, default="",
                       help="Fallback selector checkpoint directory if not found in policy checkpoint")
    parser.add_argument("--checkpoint_num", type=int, default=None,
                       help="Policy checkpoint number to load (default: max checkpoint)")
    parser.add_argument("--record_video", action="store_true", default=False,
                       help="Record env and buffer videos")
    parser.add_argument("--video_dir", type=str, default="rollout/mikasa_robo/videos",
                       help="Directory to save videos")
    parser.add_argument("--episode_indices", type=int, nargs="*", default=list(range(10)),
                       help="Episode indices to evaluate (default: first 10)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--delta_time", type=int, default=None,
                       help="Override the env cue-to-action delay (default: the config's value)")
    args = parser.parse_args()

    eval_policy(
        policy_checkpoint=args.policy_checkpoint,
        selector_checkpoint=args.selector_checkpoint,
        checkpoint_num=str(args.checkpoint_num) if args.checkpoint_num is not None else None,
        record_video=args.record_video,
        video_dir=args.video_dir,
        episode_indices=args.episode_indices,
        seed=args.seed,
        delta_time=args.delta_time,
    )
