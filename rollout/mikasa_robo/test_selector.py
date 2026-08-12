"""
Inspect what a trained mikasa_robo selector keeps in the memory buffer.

Drives each episode along the expert (ground-truth) trajectory and traces the
selector's priority per step plus the buffer state (rendered camera images).
No proxy/policy involved.

Usage:
    python rollout/mikasa_robo/test_selector.py --checkpoint_path mikasa_robo_RememberColor3_20260101_120000 --record_video
"""
import argparse
from stable_baselines3 import PPO
from problems import problem_dict
from keyframe_mnemonics.buffers import Process
from common.helpers import set_seeds, parse_experiment_name, stage_dir
import yaml
import numpy as np
import torch
from mani_skill.utils.wrappers import RecordEpisode
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import os
from pathlib import Path
import imageio
from io import BytesIO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MikasaRobo-SelectorTrace')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint_path', type=str, default="", help='Run folder name (under checkpoints/)')
    parser.add_argument('--record_video', default=False, action=argparse.BooleanOptionalAction, help='Record buffer video')
    parser.add_argument('--video_dir', type=str, default="rollout/mikasa_robo/videos", help='Directory to save videos')
    parser.add_argument('--episode_indices', type=int, nargs='*', default=list(range(5)), help='Episode indices to trace')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.seed is not None:
        set_seeds(args.seed)

    if args.record_video:
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        axes[0].set_title("Overhead Camera Buffer"); axes[0].axis('off')
        axes[1].set_title("Gripper Camera Buffer"); axes[1].axis('off')
        fig.tight_layout()

    checkpoint_folder = args.checkpoint_path
    experiment_name = parse_experiment_name(checkpoint_folder)

    with open(f"conf/mikasa_robo/{experiment_name}.yaml") as f:
        problem_config = yaml.safe_load(f)
    path_prefix = problem_config.get("evaluator", {}).get("path_prefix", "")
    checkpoint_path = stage_dir(path_prefix, checkpoint_folder, "selector")

    with open(os.path.join(checkpoint_path, "config.yaml")) as f:
        problem_config = yaml.safe_load(f)
        test_problem_config = problem_config.get("problem")
        test_problem_config.update(problem_config.get("test_problem_override", {}))
        if args.record_video:
            test_problem_config.setdefault("render_mode", "all")

    problem_class = problem_dict.get("mikasa_robo").get("problem")
    test_problem = problem_class(**test_problem_config)
    test_problem.randomize_reset = False
    test_process = Process(test_problem, queue_strategy="evict_latest_norepeat")

    selector_model = PPO.load(
        os.path.join(checkpoint_path, f"{experiment_name}_selector.zip"), device=device)
    selector_model.policy.set_training_mode(False)

    if args.record_video:
        video_path = Path(args.video_dir) / f"{checkpoint_folder}"
        video_path.mkdir(parents=True, exist_ok=True)
        print(f"Recording videos to: {video_path}")
        test_process.problem.env = RecordEpisode(
            test_process.problem.env,
            output_dir=str(video_path),
            save_trajectory=False,
            info_on_video=True,
            max_steps_per_video=test_process.problem.seq_len,
            video_fps=30,
        )

    episode_list = args.episode_indices or list(range(test_problem.num_variations))

    for episode in episode_list:
        print(f"\nEpisode {episode}")
        test_process.reset(episode)
        buffer_frames = [] if args.record_video else None

        for step in range(test_process.problem.seq_len):
            smp, t = test_process.get_obs(step)
            p, _ = selector_model.predict(smp, deterministic=True)
            test_process.set_action(smp, p)
            priority = float(np.asarray(p).reshape(-1)[0])

            # advance the env along the expert (ground-truth) trajectory
            test_process.problem.optimal_action = test_process.get_target(t)

            print(f"  t={step:03d} priority={priority:.4f}")

            if args.record_video:
                buffer = test_process.get_buffer().numpy().reshape(test_problem.buffer_size, -1)
                overhead_imgs = [np.clip(buffer[i, :49152].reshape(128, 128, 3), 0, 1) for i in range(buffer.shape[0])]
                gripper_imgs = [np.clip(buffer[i, 49152:98304].reshape(128, 128, 3), 0, 1) for i in range(buffer.shape[0])]
                axes[0].clear(); axes[0].imshow(np.concatenate(overhead_imgs, axis=1))
                axes[0].set_title(f"Overhead Camera Buffer - Step {step + 1}"); axes[0].axis('off')
                axes[1].clear(); axes[1].imshow(np.concatenate(gripper_imgs, axis=1))
                axes[1].set_title(f"Gripper Camera Buffer - Step {step + 1}"); axes[1].axis('off')
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                buffer_frames.append(imageio.imread(buf)); buf.close()

            if test_process.problem.episode_success:
                break

        if args.record_video and buffer_frames:
            frames_uint8 = [(f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8) for f in buffer_frames]
            imageio.mimwrite(str(video_path / f"{episode}_buffer.mp4"), frames_uint8, fps=5, codec='h264', quality=8)

    test_process.problem.reset()  # flush last video
    if args.record_video:
        plt.close('all')
