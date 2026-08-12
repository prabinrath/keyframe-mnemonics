"""
Inspect what a trained LTMB selector keeps in the memory buffer.

Drives each episode along the expert (ground-truth) trajectory and traces the
selector's priority per step plus the buffer state (rendered grid images).
No proxy/policy involved.

Usage:
    python rollout/ltmb/test_selector.py --checkpoint_path ltmb_Hallway_20260101_120000 --render
    python rollout/ltmb/test_selector.py --checkpoint_path ltmb_Hallway_20260101_120000 --record_video
"""
import argparse
import numpy as np
import torch
import yaml
import os
import gymnasium as gym
import ltmb  # register LTMB envs
import minigrid  # register MiniGrid envs
import matplotlib
import imageio
from pathlib import Path
from io import BytesIO

from minigrid.core.grid import Grid
from stable_baselines3 import PPO
from problems import problem_dict
from keyframe_mnemonics.buffers import Process
from common.helpers import set_seeds, parse_experiment_name, stage_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LTMB-SelectorTrace')
    parser.add_argument('--render', default=False, action=argparse.BooleanOptionalAction, help='Render buffer interactively')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint_path', type=str, default="ltmb_Hallway_20260719_120655", help='Run folder name (under checkpoints/)')
    parser.add_argument('--record_video', default=False, action=argparse.BooleanOptionalAction, help='Record buffer/env videos')
    parser.add_argument('--video_dir', type=str, default="rollout/ltmb/videos", help='Directory to save videos')
    parser.add_argument('--total_episodes', type=int, default=5, help='Total episodes to trace')
    args = parser.parse_args()

    if args.record_video:
        matplotlib.use('Agg')  # non-interactive backend for headless video capture
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.seed is not None:
        set_seeds(args.seed)

    if args.render and not args.record_video:
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        ax.set_title("Buffer Images"); ax.axis('off')
        fig.canvas.manager.window.wm_geometry("+0+600")
    elif args.record_video:
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        ax.set_title("Buffer Images"); ax.axis('off')
        fig.tight_layout()

    checkpoint_folder = args.checkpoint_path
    experiment_name = parse_experiment_name(checkpoint_folder)

    with open(f"conf/ltmb/{experiment_name}.yaml") as f:
        problem_config = yaml.safe_load(f)
    path_prefix = problem_config.get("evaluator", {}).get("path_prefix", "")
    checkpoint_path = stage_dir(path_prefix, checkpoint_folder, "selector")
    test_problem_config = problem_config.get("problem")
    override = dict(problem_config.get("test_problem_override", {}))
    override.pop("env_options", None)
    test_problem_config.update(override)

    problem_class = problem_dict.get("ltmb").get("problem")
    test_problem = problem_class(**test_problem_config)
    test_problem.randomize_reset = False
    test_process = Process(test_problem, queue_strategy="evict_latest_norepeat",
                           queue_kwargs={"rejection_threshold": 0.9})

    selector_model = PPO.load(
        os.path.join(checkpoint_path, f"{experiment_name}_selector.zip"),
        device=device
    )
    selector_model.policy.set_training_mode(False)

    if args.record_video:
        video_path = Path(args.video_dir) / f"{checkpoint_folder}"
        video_path.mkdir(parents=True, exist_ok=True)
        print(f"Recording videos to: {video_path}")

    for episode in range(args.total_episodes):
        print(f"\nEpisode {episode}")
        test_process.reset()
        env_options = test_process.problem.get_env_options()
        render_mode = "rgb_array" if args.record_video else ("human" if args.render else None)

        env = gym.make(test_process.problem.env_id, **env_options, render_mode=render_mode)
        obs, _ = env.reset(seed=int(test_process.problem.sample.get("seed")))

        buffer_frames = [] if args.record_video else None
        env_frames = [] if args.record_video else None
        if args.record_video:
            env_frame = env.render()
            if env_frame is not None:
                env_frames.append(env_frame)

        max_steps = test_process.problem.get_rollout_step_limit(default_limit=200)
        for step in range(max_steps):
            smp = torch.as_tensor(np.concatenate((
                obs['image'].flatten(),
                np.array([obs['direction']], dtype=np.int32),
            )))
            p, _ = selector_model.predict(smp, deterministic=True)
            test_process.set_action(smp, p)
            priority = float(np.asarray(p).reshape(-1)[0])

            # advance the env along the expert (ground-truth) trajectory
            test_process.get_obs(step)
            env_action = int(test_process.get_target(step).cpu().numpy()[0])
            obs, reward, terminated, truncated, info = env.step(env_action)

            print(f"  t={step:03d} priority={priority:.4f}")

            if args.render or args.record_video:
                buffer = test_process.get_buffer().numpy()
                buffer_imgs = buffer.reshape(test_problem.buffer_size, 148)[:, :147].reshape(
                    test_problem.buffer_size, 7, 7, 3
                )
                imgs_in_buffer = []
                for i in range(buffer_imgs.shape[0]):
                    decoded_grid, _ = Grid.decode(buffer_imgs[i])
                    imgs_in_buffer.append(decoded_grid.render(tile_size=32, agent_pos=None, agent_dir=None))
                ax.clear()
                ax.imshow(np.concatenate(imgs_in_buffer, axis=1))
                ax.set_title(f"Buffer Images - Step {step + 1} ({len(imgs_in_buffer)} images)")
                ax.axis('off')

                if args.record_video:
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    buffer_frames.append(imageio.imread(buf)); buf.close()
                    env_frame = env.render()
                    if env_frame is not None:
                        env_frames.append(env_frame)
                else:
                    plt.pause(0.1)

            if terminated or truncated:
                break

        env.close()
        if args.record_video:
            if buffer_frames:
                frames_uint8 = [(f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8) for f in buffer_frames]
                imageio.mimwrite(str(video_path / f"{episode}_buffer.mp4"), frames_uint8, fps=5, codec='h264', quality=8)
            if env_frames:
                frames_uint8 = [(f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8) for f in env_frames]
                imageio.mimwrite(str(video_path / f"{episode}_env.mp4"), frames_uint8, fps=10, codec='h264', quality=8)

    if args.render and not args.record_video:
        plt.ioff()
    plt.close('all')
