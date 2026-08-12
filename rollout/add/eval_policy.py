"""
Evaluate a trained add policy using the saved selector + policy checkpoints.

Usage:
    python rollout/add/eval_policy.py --policy_checkpoint add_20260412_120000
"""
import argparse
import glob
import os

import torch
import yaml
from stable_baselines3 import PPO
from tqdm import tqdm

from common.helpers import parse_experiment_name, set_seeds, stage_dir
from models import model_dict
from problems import problem_dict
from keyframe_mnemonics.buffers import Process


def eval_policy(policy_checkpoint, checkpoint_num=None, episode_indices=None, seed=42, show_trace=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        set_seeds(seed)

    experiment_name = parse_experiment_name(policy_checkpoint)
    with open(f"conf/{experiment_name}.yaml") as f:
        config = yaml.safe_load(f)
    path_prefix = config.get("path_prefix", "")

    policy_checkpoint_dir = stage_dir(path_prefix, policy_checkpoint, "policy")
    selector_checkpoint_dir = stage_dir(path_prefix, policy_checkpoint, "selector")
    with open(os.path.join(policy_checkpoint_dir, "config.yaml")) as f:
        saved_config = yaml.safe_load(f)
    policy_config = saved_config.get("policy", {})
    model_config = policy_config.get("model_kwargs", {})

    if checkpoint_num is not None:
        policy_file = os.path.join(policy_checkpoint_dir, f"{experiment_name}_policy_{checkpoint_num}.pth")
        if not os.path.exists(policy_file):
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_file}")
    else:
        policy_pattern = os.path.join(policy_checkpoint_dir, f"{experiment_name}_policy_*.pth")
        policy_files = glob.glob(policy_pattern)
        if not policy_files:
            raise FileNotFoundError(f"No policy checkpoints found matching {policy_pattern}")
        policy_file = max(policy_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    selector_file = os.path.join(selector_checkpoint_dir, f"{experiment_name}_selector.zip")
    if not os.path.exists(selector_file):
        raise FileNotFoundError(f"Selector checkpoint not found: {selector_file}")

    problem_config = config.get("problem")
    problem_config.update(config.get("policy_problem_override", {}))
    problem_config.update(config.get("test_problem_override", {}))
    problem_config["use_current_obs"] = True
    problem_class = problem_dict.get("add").get("problem")
    problem = problem_class(**problem_config)
    problem.randomize_reset = False
    process = Process(problem)

    selector_model = PPO.load(selector_file, device=device)
    selector_model.policy.set_training_mode(False)
    print(f"Loaded selector from: {selector_file}")

    policy_class = model_dict.get("add").get("policy")
    policy_model = policy_class(output_dim=process.problem.action_dim, **model_config).to(device)
    checkpoint = torch.load(policy_file, map_location=device, weights_only=False)
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    policy_model.eval()
    print(f"Loaded policy from: {policy_file}")

    episode_list = episode_indices if episode_indices else list(range(process.problem.num_variations))
    final_mse_episode = []

    for episode in tqdm(episode_list, desc="Evaluating"):
        process.reset(episode)
        if show_trace:
            print(f"\nEpisode {episode}")
        for idx in range(process.problem.seq_len):
            obs, t = process.get_obs(idx)
            target = process.get_target(t)
            priority, _ = selector_model.predict(obs.unsqueeze(0).numpy(), deterministic=True)
            priority = float(priority[0][0])
            process.set_action(obs, priority)

            pred_target = policy_model.get_action(
                process.get_buffer().unsqueeze(0).to(device)
            ).cpu().squeeze()

            if show_trace:
                buffer_state = process.get_buffer().tolist()
                print(
                    f"  t={idx:03d} obs={obs.tolist()} priority={priority:.4f} "
                    f"buffer={buffer_state} pred={pred_target.tolist()} target={target.tolist()}"
                )

            if idx == process.problem.seq_len - 1:
                episode_final_mse = float(((target - pred_target) ** 2).mean().item())
                final_mse_episode.append(episode_final_mse)
                if show_trace:
                    print(
                        f"Episode {episode} final: target={float(target.item()):.6f} "
                        f"pred={float(pred_target.item()):.6f}"
                    )

    mean_final_mse = sum(final_mse_episode) / len(final_mse_episode)
    print(f"\nMean final MSE: {mean_final_mse:.6f}")
    return mean_final_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained add policy")
    parser.add_argument("--policy_checkpoint", type=str, default="add_20260718_181839")
    parser.add_argument("--checkpoint_num", type=int, default=None,
                        help="Policy checkpoint number to load (default: max checkpoint)")
    parser.add_argument("--episode_indices", type=int, nargs="*", default=list(range(100)),
                        help="Episode indices to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_trace", default=False, action=argparse.BooleanOptionalAction,
                        help="Print obs, priorities, buffer, policy prediction, and target during rollout")
    args = parser.parse_args()

    eval_policy(
        policy_checkpoint=args.policy_checkpoint,
        checkpoint_num=str(args.checkpoint_num) if args.checkpoint_num is not None else None,
        episode_indices=args.episode_indices,
        seed=args.seed,
        show_trace=args.show_trace,
    )
