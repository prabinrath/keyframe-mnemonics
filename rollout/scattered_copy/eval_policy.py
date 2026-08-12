"""
Evaluate a trained scattered-copy policy using the saved selector + policy checkpoints.

Usage:
    python rollout/scattered_copy/eval_policy.py --policy_checkpoint scattered_copy_<TAG>
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
    problem_class = problem_dict.get("scattered_copy").get("problem")
    problem = problem_class(**problem_config)
    problem.randomize_reset = False
    process = Process(problem)

    selector_model = PPO.load(selector_file, device=device)
    selector_model.policy.set_training_mode(False)
    print(f"Loaded selector from: {selector_file}")

    policy_class = model_dict.get("scattered_copy").get("policy")
    policy_model = policy_class(output_dim=process.problem.action_dim, **model_config).to(device)
    checkpoint = torch.load(policy_file, map_location=device, weights_only=False)
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    policy_model.eval()
    print(f"Loaded policy from: {policy_file}")

    episode_list = episode_indices if episode_indices else list(range(process.problem.num_variations))
    step_match = []
    emit_match = []
    copy_success = []

    for episode in tqdm(episode_list, desc="Evaluating"):
        process.reset(episode)
        matched = True
        if show_trace:
            print(f"\nEpisode {episode}")
        for idx in range(process.problem.seq_len):
            obs, t = process.get_obs(idx)
            target = process.get_target(t)
            priority, _ = selector_model.predict(obs.unsqueeze(0).numpy(), deterministic=True)
            process.set_action(obs, float(priority[0][0]))

            token = int(obs.item())
            pred_target = policy_model.get_action(
                process.get_buffer().unsqueeze(0).to(device), deterministic=True
            ).cpu().squeeze()
            is_match = int(pred_target.item()) == int(target.item())
            step_match.append(float(is_match))
            if int(target.item()) != 0:
                emit_match.append(float(is_match))
            matched = matched and is_match

            if show_trace:
                buffer_state = [int(x) for x in process.get_buffer().tolist()]
                target_val = int(target.item())
                pred_val = int(pred_target.item())
                print(
                    f"  t={idx:03d} obs={token:2d} buffer={buffer_state} "
                    f"pred={pred_val:2d} target={target_val:2d} is_match={is_match}"
                )

        copy_success.append(float(matched))

    step_accuracy = sum(step_match) / len(step_match)
    emit_accuracy = (sum(emit_match) / len(emit_match)) if emit_match else 0.0
    copy_success_rate = sum(copy_success) / len(copy_success)
    print(f"\nStep accuracy: {step_accuracy}")
    print(f"Emit accuracy: {emit_accuracy}")
    print(f"Copy success rate: {copy_success_rate}")
    return step_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained scattered-copy policy")
    parser.add_argument("--policy_checkpoint", type=str, default="scattered_copy_20260718_183248")
    parser.add_argument("--checkpoint_num", type=int, default=None,
                        help="Policy checkpoint number to load (default: max checkpoint)")
    parser.add_argument("--episode_indices", type=int, nargs="*", default=list(range(100)),
                        help="Episode indices to evaluate (default: first 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_trace", default=False, action=argparse.BooleanOptionalAction,
                        help="Print obs, buffer, policy action, and match info during rollout")
    args = parser.parse_args()

    eval_policy(
        policy_checkpoint=args.policy_checkpoint,
        checkpoint_num=str(args.checkpoint_num) if args.checkpoint_num is not None else None,
        episode_indices=args.episode_indices,
        seed=args.seed,
        show_trace=args.show_trace,
    )
