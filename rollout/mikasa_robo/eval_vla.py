"""Evaluate trained VLA on Mikasa Robo problems."""

import torch
import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from io import BytesIO
from stable_baselines3 import PPO
from mani_skill.utils.wrappers import RecordEpisode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import re

from keyframe_mnemonics.buffers import Process
from problems import problem_dict
from common.helpers import set_seeds, parse_experiment_name, stage_dir
from problems.mikasa_robo_problem.lerobot_utils import get_lerobot_features, build_vla_observation

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import build_inference_frame


def eval_vla(
    selector_checkpoint,
    vla_checkpoint,
    policy_type="smolvla",
    task=None,
    record_video=False,
    video_dir="rollout/mikasa_robo/videos",
    episode_indices=None,
    seed=42,
    open_loop_steps=1,
):
    """Evaluate a trained VLA with a trained selector for buffer management.
    
    Args:
        selector_checkpoint: Selector checkpoint directory name
        vla_checkpoint: VLA checkpoint directory name
        policy_type: Policy type string for get_policy_class (e.g. 'smolvla')
        record_video: Whether to record env + buffer videos
        video_dir: Directory to save videos
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if seed is not None:
        set_seeds(seed)
    
    # Extract experiment_name from selector checkpoint
    experiment_name = parse_experiment_name(selector_checkpoint)
    
    # Load config from conf/ for path_prefix and problem setup
    with open(f"conf/mikasa_robo/{experiment_name}.yaml") as f:
        config = yaml.safe_load(f)
    path_prefix = config.get("path_prefix", "")

    # Load VLA model first to determine buffer_size from camera keys
    checkpoint_name = "last"
    vla_path = os.path.join(path_prefix, "checkpoints", vla_checkpoint, "checkpoints", checkpoint_name, "pretrained_model")

    vla_model = get_policy_class(policy_type).from_pretrained(str(vla_path))
    vla_model.to(device)
    vla_model.config.n_action_steps = open_loop_steps
    vla_model.eval()
    print(f"Loaded VLA from: {vla_path}")

    # Derive buffer_size from VLA config camera keys
    camera_keys = [k for k in vla_model.config.input_features if 'camera' in k]
    buffer_size = 0
    for key in camera_keys:
        # Extract number from keys like 'observation.images.gripper_camera1'
        match = re.search(r'camera(\d+)', key)
        if match:
            num = int(match.group(1))
            buffer_size = max(buffer_size, num)
    print(f"Resolved buffer_size from VLA config: {buffer_size}")

    # Setup test problem from conf/ config
    problem_config = config.get("problem")
    problem_config.update(config.get("policy_problem_override", {}))
    problem_config.update(config.get("test_problem_override", {}))
    problem_config["use_current_obs"] = True
    if record_video:
        problem_config.setdefault("render_mode", "all")

    print("\nTest Configuration:")
    print(f"  Environment: {problem_config['env_id']}")
    
    # Create test problem and process buffer
    problem_class = problem_dict.get("mikasa_robo").get("problem")
    problem = problem_class(**problem_config)
    problem.randomize_reset = False
    process = Process(problem, queue_strategy="evict_latest_norepeat")

    # Setup video recording
    if record_video:
        video_path = Path(video_dir) / vla_checkpoint / checkpoint_name
        video_path.mkdir(parents=True, exist_ok=True)
        print(f"Recording videos to: {video_path}")
        process.problem.env = RecordEpisode(
            process.problem.env,
            output_dir=str(video_path),
            save_trajectory=False,
            info_on_video=True,
            max_steps_per_video=process.problem.seq_len,
            video_fps=30
        )
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        axes[0].axis('off')
        axes[1].axis('off')
        fig.tight_layout()
    
    # Load selector model
    selector_file = os.path.join(stage_dir(path_prefix, selector_checkpoint, "selector"),
                                 f"{experiment_name}_selector.zip")
    if not os.path.exists(selector_file):
        raise FileNotFoundError(f"Selector checkpoint not found: {selector_file}")
    
    selector_model = PPO.load(selector_file, device=device)
    selector_model.policy.set_training_mode(False)
    print(f"\nLoaded selector from: {selector_file}")

    # Setup pre/post processors for VLA
    device_override = {"device": device}
    preprocess, postprocess = make_pre_post_processors(
        vla_model.config,
        pretrained_path=str(vla_path),
        preprocessor_overrides={
            "device_processor": device_override,
        },
        postprocessor_overrides={"device_processor": device_override},
    )
    dataset_features = get_lerobot_features(buffer_size=buffer_size)
    
    # Setup action charts directory
    charts_path = Path(video_dir) / vla_checkpoint / checkpoint_name
    charts_path.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    episode_list = episode_indices if episode_indices else list(range(process.problem.num_variations))
    print(f"\nRunning evaluation on {len(episode_list)} episodes...")
    episode_successes = []
    episode_rewards = []
    
    for episode_idx, episode in enumerate(tqdm(episode_list, desc="Evaluating")):
        # Reset process and problem
        process.reset(episode)
        buffer_frames = [] if record_video else None
        episode_actions = []
        
        # Run episode
        for idx in range(process.problem.seq_len):
            # Get observation
            obs, t = process.get_obs(idx)
            
            # Convert to tensor if needed
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32)
            
            # Use selector to get priority (deterministic)
            priority, _ = selector_model.predict(obs, deterministic=True)
            
            # Add observation to buffer with priority
            process.set_action(obs, priority)
            
            # Get buffer content
            buffer = process.get_buffer()
            
            # Convert to numpy for parsing
            if isinstance(buffer, torch.Tensor):
                buffer = buffer.cpu().numpy()
            
            # Parse buffer into VLA observation format
            policy_obs = build_vla_observation(buffer, buffer_size)
            
            # Convert to tensors and add batch dimension
            obs_frame = build_inference_frame(
                observation=policy_obs, ds_features=dataset_features, device=device, task=task
            )
            
            obs_frame = preprocess(obs_frame)
            pred_target = vla_model.select_action(obs_frame)
            pred_target = postprocess(pred_target).cpu()
            process.problem.optimal_action = pred_target
            episode_actions.append(pred_target.squeeze().cpu().numpy())

            if record_video:
                buf_np = buffer.reshape(buffer_size, -1)
                overhead_imgs = [np.clip(buf_np[i, :49152].reshape(128, 128, 3), 0, 1) for i in range(buffer_size)]
                gripper_imgs  = [np.clip(buf_np[i, 49152:98304].reshape(128, 128, 3), 0, 1) for i in range(buffer_size)]
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
    parser = argparse.ArgumentParser(description="Evaluate trained VLA on Mikasa Robo problems")
    parser.add_argument("--selector_checkpoint", type=str,
                       default="",
                       help="Selector checkpoint directory name")
    parser.add_argument("--vla_checkpoint", type=str,
                       default="smolvla_RememberColor3_ce_policy",
                       help="VLA checkpoint directory name")
    parser.add_argument("--policy_type", type=str, default="smolvla",
                       help="Policy type string for get_policy_class (e.g. 'smolvla')")
    parser.add_argument("--task", type=str, default="Remember the color of the cube and then pick the matching one",
                       help="Task description for the VLA model")
    parser.add_argument("--record_video", action="store_true", default=False,
                       help="Record env and buffer videos")
    parser.add_argument("--video_dir", type=str, default="rollout/mikasa_robo/videos",
                       help="Directory to save videos")
    parser.add_argument("--episode_indices", type=int, nargs="*", default=list(range(10)),
                       help="Episode indices to evaluate (default: first 10)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (default: 42)")
    parser.add_argument("--open_loop_steps", type=int, default=1,
                       help="Override open_loop_steps (default: 1)")
    
    args = parser.parse_args()
    
    eval_vla(
        selector_checkpoint=args.selector_checkpoint,
        vla_checkpoint=args.vla_checkpoint,
        policy_type=args.policy_type,
        task=args.task,
        record_video=args.record_video,
        video_dir=args.video_dir,
        episode_indices=args.episode_indices,
        seed=args.seed,
        open_loop_steps=args.open_loop_steps,
    )
