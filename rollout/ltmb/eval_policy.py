"""Evaluate trained LTMB policy on LTMB environments."""

import torch
import argparse
import yaml
import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from io import BytesIO
import gymnasium as gym
import ltmb
import minigrid
import imageio
import matplotlib
from stable_baselines3 import PPO
from minigrid.core.grid import Grid

from keyframe_mnemonics.buffers import Process
from problems import problem_dict
from models import model_dict
from common.helpers import set_seeds, parse_experiment_name, stage_dir


def eval_policy(
    policy_checkpoint,
    selector_checkpoint=None,
    checkpoint_num=None,
    deterministic=True,
    render=False,
    record_video=False,
    video_dir="rollout/ltmb/videos",
    episode_indices=None,
    seed=42,
    override_env_options=True,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if seed is not None:
        set_seeds(seed)

    experiment_name = parse_experiment_name(policy_checkpoint)

    with open(f"conf/ltmb/{experiment_name}.yaml") as f:
        config = yaml.safe_load(f)
    path_prefix = config.get("path_prefix", "")

    problem_config = config.get("problem")
    problem_config.update(config.get("policy_problem_override", {}))
    test_override = dict(config.get("test_problem_override", {}))
    if not override_env_options:
        # Validation: keep split=test but drop the length override so the env is built
        # at the demo's natural (ID) length, not the OOD report length.
        test_override.pop("env_options", None)
    problem_config.update(test_override)

    # Load policy checkpoint config for model_kwargs
    policy_checkpoint_dir = stage_dir(path_prefix, policy_checkpoint, "policy")
    with open(os.path.join(policy_checkpoint_dir, "config.yaml")) as f:
        saved_config = yaml.safe_load(f)
    policy_config = saved_config.get("policy", {})
    model_config = policy_config.get("model_kwargs", {})
    total_slots = model_config["total_slots"]

    # Resolve policy checkpoint file
    if checkpoint_num is not None:
        policy_file = os.path.join(policy_checkpoint_dir, f"{experiment_name}_policy_{checkpoint_num}.pth")
        if not os.path.exists(policy_file):
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_file}")
    else:
        policy_pattern = os.path.join(policy_checkpoint_dir, f"{experiment_name}_policy_*.pth")
        policy_files = glob.glob(policy_pattern)
        if not policy_files:
            raise FileNotFoundError(f"No policy checkpoints found matching {policy_pattern}")
        policy_file = max(policy_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_num = policy_file.split('_')[-1].split('.')[0]

    # Create problem and process
    problem_config["use_current_obs"] = True
    problem_class = problem_dict.get("ltmb").get("problem")
    problem = problem_class(**problem_config)
    problem.randomize_reset = False
    process = Process(problem, queue_strategy="evict_latest_norepeat",
                           queue_kwargs={"rejection_threshold": 0.9})

    # Load selector — prefer the run's selector subfolder, fall back to selector_checkpoint arg
    selector_dir = stage_dir(path_prefix, policy_checkpoint, "selector")
    selector_file = os.path.join(selector_dir, f"{experiment_name}_selector.zip")
    if not os.path.exists(selector_file):
        if not selector_checkpoint:
            raise FileNotFoundError(
                f"No selector found in {selector_dir} and --selector-checkpoint not provided"
            )
        selector_file = os.path.join(
            stage_dir(path_prefix, selector_checkpoint, "selector"), f"{experiment_name}_selector.zip"
        )
        if not os.path.exists(selector_file):
            raise FileNotFoundError(f"Selector checkpoint not found: {selector_file}")
    selector_model = PPO.load(selector_file, device=device)
    selector_model.policy.set_training_mode(False)
    print(f"Loaded selector from: {selector_file}")

    # Load policy model
    policy_class = model_dict.get("ltmb").get("policy")
    policy_model = policy_class(output_dim=process.problem.action_dim, device=device, **model_config).to(device)
    checkpoint = torch.load(policy_file, map_location=device, weights_only=False)
    policy_model.load_state_dict(checkpoint['model_state_dict'])
    policy_model.eval()
    print(f"Loaded policy from: {policy_file}")

    # Setup rendering
    if record_video:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if render and not record_video:
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        ax.axis('off')
    elif record_video:
        video_path = Path(video_dir) / policy_checkpoint / str(checkpoint_num)
        video_path.mkdir(parents=True, exist_ok=True)
        print(f"Recording videos to: {video_path}")
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        ax.axis('off')
        fig.tight_layout()

    # Evaluation loop
    episode_list = episode_indices if episode_indices else list(range(process.problem.num_variations))
    episode_successes = []
    episode_rewards = []

    for episode in tqdm(episode_list, desc="Evaluating"):
        process.reset()
        env_options = process.problem.get_env_options()

        if record_video:
            render_mode = "rgb_array"
        elif render:
            render_mode = "human"
        else:
            render_mode = None

        env = gym.make(process.problem.env_id, **env_options, render_mode=render_mode)
        obs, _ = env.reset(seed=int(process.problem.sample.get("seed")))

        buffer_frames = [] if record_video else None
        env_frames = [] if record_video else None

        # Capture the initial env frame (post-reset state) before any actions
        if record_video:
            env_frame = env.render()
            if env_frame is not None:
                env_frames.append(env_frame)

        reward = 0.0
        info = {}
        max_steps = process.problem.get_rollout_step_limit(default_limit=500)

        for step in range(max_steps):
            smp = torch.as_tensor(np.concatenate((
                obs['image'].flatten(),
                np.array([obs['direction']], dtype=np.int32),
            )))
            priority, _ = selector_model.predict(smp, deterministic=True)
            process.set_action(smp, priority)

            buffer = process.get_buffer()
            pred_target = policy_model.get_action(
                buffer.unsqueeze(0).to(device), deterministic=deterministic
            ).cpu()
            env_action = int(pred_target[0, 0])

            obs, reward, terminated, truncated, info = env.step(env_action)

            if render or record_video:
                buf_np = buffer.numpy().reshape(total_slots, 148)[:, :147].reshape(
                    total_slots, 7, 7, 3
                )
                imgs = []
                for i in range(total_slots):
                    decoded_grid, _ = Grid.decode(buf_np[i])
                    imgs.append(decoded_grid.render(tile_size=32, agent_pos=None, agent_dir=None))
                ax.clear()
                ax.imshow(np.concatenate(imgs, axis=1))
                ax.set_title(f"Buffer Images - Step {step + 1} ({len(imgs)} images)")
                ax.axis('off')

                if record_video:
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    buffer_frames.append(imageio.imread(buf))
                    buf.close()

                    env_frame = env.render()
                    if env_frame is not None:
                        env_frames.append(env_frame)
                else:
                    plt.pause(0.1)

            if terminated or truncated:
                break

        success = bool(info.get('success', reward > 0))
        episode_successes.append(success)
        episode_rewards.append(reward)
        env.close()
        print(f"  Episode {episode}: {'Success' if success else 'Failed'}, reward={reward:.3f}")

        if record_video:
            if buffer_frames:
                imageio.mimwrite(str(video_path / f"{episode}_buffer.mp4"), buffer_frames, fps=5, codec='h264', quality=8)
            if env_frames:
                imageio.mimwrite(str(video_path / f"{episode}_env.mp4"), env_frames, fps=10, codec='h264', quality=8)

    if render and not record_video:
        plt.ioff()
    if render or record_video:
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
        'episode_rewards': episode_rewards,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained LTMB policy")
    parser.add_argument("--policy_checkpoint", type=str, default="ltmb_Hallway_20260719_192255",
                        help="Policy checkpoint directory name")
    parser.add_argument("--selector_checkpoint", type=str, default="",
                        help="Fallback selector checkpoint directory if not found in policy checkpoint")
    parser.add_argument("--checkpoint_num", type=int, default=None,
                        help="Policy checkpoint number to load (default: max checkpoint)")
    parser.add_argument("--deterministic", default=True, action=argparse.BooleanOptionalAction,
                        help="Use deterministic policy actions (default: True)")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Render buffer interactively")
    parser.add_argument("--record_video", action="store_true", default=False,
                        help="Record env and buffer videos")
    parser.add_argument("--video_dir", type=str, default="rollout/ltmb/videos",
                        help="Directory to save videos")
    parser.add_argument("--episode_indices", type=int, nargs="*", default=list(range(10)),
                        help="Episode indices to evaluate (default: first 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--override_env_options", default=True, action=argparse.BooleanOptionalAction,
                        help="Apply test_problem_override.env_options (OOD length). "
                             "Use --no-override_env_options for ID validation at natural lengths.")
    args = parser.parse_args()

    eval_policy(
        policy_checkpoint=args.policy_checkpoint,
        selector_checkpoint=args.selector_checkpoint,
        checkpoint_num=str(args.checkpoint_num) if args.checkpoint_num is not None else None,
        deterministic=args.deterministic,
        render=args.render,
        record_video=args.record_video,
        video_dir=args.video_dir,
        episode_indices=args.episode_indices,
        seed=args.seed,
        override_env_options=args.override_env_options,
    )
