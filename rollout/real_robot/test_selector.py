"""
Inspect what a trained real_robot selector keeps in the memory buffer.

Replays each demonstration from the proxy H5 and traces the selector's priority
per step plus the buffer state (rendered camera images). No proxy/policy involved,
and no simulator -- the episodes are recorded teleop, so there is nothing to step.

Usage:
    python rollout/real_robot/test_selector.py --checkpoint_path real_robot_RememberColor3_20260101_120000 --record_video
"""
import argparse
from stable_baselines3 import PPO
from problems import problem_dict
from keyframe_mnemonics.buffers import Process
from problems.real_robot_problem.lerobot_utils import IMAGE_SIZE, PIXELS_PER_CAMERA
from common.helpers import set_seeds, parse_experiment_name, stage_dir
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import os
from pathlib import Path
import imageio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealRobot-SelectorTrace')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint_path', type=str, default="", help='Run folder name (under checkpoints/)')
    parser.add_argument('--record_video', default=False, action=argparse.BooleanOptionalAction, help='Record buffer video')
    parser.add_argument('--video_dir', type=str, default="rollout/real_robot/videos", help='Directory to save videos')
    parser.add_argument('--episode_indices', type=int, nargs='*', default=list(range(5)), help='Episode indices to trace')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.seed is not None:
        set_seeds(args.seed)

    if args.record_video:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Wrist Camera Buffer"); ax.axis('off')
        fig.tight_layout()

    checkpoint_folder = args.checkpoint_path
    experiment_name = parse_experiment_name(checkpoint_folder)

    with open(f"conf/real_robot/{experiment_name}.yaml") as f:
        problem_config = yaml.safe_load(f)
    path_prefix = problem_config.get("path_prefix", "")
    checkpoint_path = stage_dir(path_prefix, checkpoint_folder, "selector")

    with open(os.path.join(checkpoint_path, "config.yaml")) as f:
        problem_config = yaml.safe_load(f)
        test_problem_config = problem_config.get("problem")

    queue_kwargs = problem_config.get("policy", {}).get("queue_kwargs", {})
    print(f"Queue kwargs: {queue_kwargs}")

    problem_class = problem_dict.get("real_robot").get("problem")
    test_problem = problem_class(**test_problem_config)
    test_problem.randomize_reset = False
    test_process = Process(test_problem, queue_strategy="evict_latest_norepeat",
                           queue_kwargs=queue_kwargs)

    selector_model = PPO.load(
        os.path.join(checkpoint_path, f"{experiment_name}_selector.zip"), device=device)
    selector_model.policy.set_training_mode(False)

    if args.record_video:
        video_path = Path(args.video_dir) / f"{checkpoint_folder}"
        video_path.mkdir(parents=True, exist_ok=True)
        print(f"Recording videos to: {video_path}")

    episode_list = args.episode_indices or list(range(test_problem.num_variations))

    for episode in episode_list:
        print(f"\nEpisode {episode}")
        test_process.reset(episode)
        buffer_frames = [] if args.record_video else None

        for step in range(test_process.problem.seq_len):
            smp, t = test_process.get_obs(step)
            p, _ = selector_model.predict(smp.numpy(), deterministic=True)
            test_process.set_action(smp, p)
            priority = float(np.asarray(p).reshape(-1)[0])

            print(f"  t={step:03d} priority={priority:.4f}")

            if args.record_video:
                buffer = test_process.get_buffer().numpy().reshape(test_problem.buffer_size, -1)
                wrist_imgs = [np.clip(buffer[i, :PIXELS_PER_CAMERA].reshape(IMAGE_SIZE, IMAGE_SIZE, 3), 0, 1)
                              for i in range(buffer.shape[0])]
                ax.clear(); ax.imshow(np.concatenate(wrist_imgs, axis=1))
                ax.set_title(f"Wrist Camera Buffer - Step {step + 1}"); ax.axis('off')
                fig.canvas.draw()
                buffer_frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

        if args.record_video and buffer_frames:
            frames_uint8 = [(f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8) for f in buffer_frames]
            imageio.mimwrite(str(video_path / f"{episode}_buffer.mp4"), frames_uint8, fps=10, codec='h264', quality=8)

    test_process.problem.close()
    if args.record_video:
        plt.close('all')
