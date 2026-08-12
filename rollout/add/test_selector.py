"""
Inspect what a trained add selector keeps in the memory buffer.

Traces each step: the observation, the selector's priority (its action), and the
resulting buffer state. No proxy/policy involved.

Usage:
    python rollout/add/test_selector.py --checkpoint_path add_20260101_120000
"""
import argparse
import os

import torch
import yaml
from stable_baselines3 import PPO

from common.helpers import parse_experiment_name, set_seeds, stage_dir
from problems import problem_dict
from keyframe_mnemonics.buffers import Process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add-SelectorTrace")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_path", type=str, default="add_20260718_181839",
                        help="Run folder name (under checkpoints/)")
    parser.add_argument("--episode_indices", type=int, nargs="*", default=list(range(5)),
                        help="Episode indices to trace")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        set_seeds(args.seed)

    checkpoint_folder = args.checkpoint_path
    experiment_name = parse_experiment_name(checkpoint_folder)

    with open("conf/add.yaml") as f:
        config = yaml.safe_load(f)
    path_prefix = config.get("evaluator", {}).get("path_prefix", "")
    checkpoint_path = stage_dir(path_prefix, checkpoint_folder, "selector")

    test_problem_config = dict(config.get("problem", {}))
    test_problem_config.update(config.get("test_problem_override", {}))

    problem_class = problem_dict.get("add").get("problem")
    test_problem = problem_class(**test_problem_config)
    test_problem.randomize_reset = False
    test_process = Process(test_problem)

    selector_model = PPO.load(
        os.path.join(checkpoint_path, f"{experiment_name}_selector.zip"),
        device=device,
    )
    selector_model.policy.set_training_mode(False)

    episode_list = args.episode_indices or list(range(test_process.problem.num_variations))
    for episode in episode_list:
        test_process.reset(episode)
        print(f"\nEpisode {episode}")
        for step in range(test_process.problem.seq_len):
            obs, t = test_process.get_obs(step)
            priority, _ = selector_model.predict(obs.unsqueeze(0).numpy(), deterministic=True)
            priority = float(priority[0][0])
            test_process.set_action(obs, priority)
            buffer_state = test_process.get_buffer().tolist()
            print(f"  t={step:03d} obs={obs.tolist()} priority={priority:.4f} buffer={buffer_state}")
