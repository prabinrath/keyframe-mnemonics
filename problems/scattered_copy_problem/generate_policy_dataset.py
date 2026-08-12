"""
Generate policy dataset for scattered copy by rolling out a trained selector.

Saves (buffer, target_token) pairs to an HDF5 file using MemoryManager.

Usage:
    python problems/scattered_copy_problem/generate_policy_dataset.py --selector_checkpoint scattered_copy_20260101_120000
"""
import argparse
import os
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from tqdm import tqdm

from common.helpers import parse_experiment_name, set_seeds, stage_dir
from keyframe_mnemonics.memory_manager import MemoryManager
from problems import problem_dict
from keyframe_mnemonics.buffers import Process


def generate(selector_checkpoint, cache_path=None,
             batch_size=1024, seed=42):
    """Roll out the frozen selector; return a MemoryManager of (buffer, target).

    cache_path="" -> collected in RAM; a directory -> written to an H5 there.
    cache_path=None -> read from the config's policy.cache_path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        set_seeds(seed)

    experiment_name = parse_experiment_name(selector_checkpoint)

    conf_path = Path("conf") / f"{experiment_name}.yaml"
    with open(conf_path) as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {conf_path}")

    path_prefix = config.get("path_prefix", "")
    checkpoint_path = stage_dir(path_prefix, selector_checkpoint, "selector")

    ckpt_conf_path = Path(checkpoint_path) / "config.yaml"
    with open(ckpt_conf_path) as f:
        ckpt_config = yaml.safe_load(f)
    assert ckpt_config["selector"] == config["selector"], (
        "Selector config mismatch between current config and checkpoint config."
    )

    problem_cfg = deepcopy(config["problem"])
    problem_cfg.update(config.get("policy_problem_override", {}))
    problem_cfg["use_current_obs"] = True
    buffer_size = problem_cfg["buffer_size"]
    total_slots = config["policy"]["model_kwargs"]["total_slots"]
    assert total_slots == buffer_size + 1, (
        f"total_slots must be buffer_size + 1 (for current obs). "
        f"Got total_slots={total_slots}, buffer_size={buffer_size}."
    )
    print(f"Buffer size (buffer_size): {buffer_size}   total_slots: {total_slots}")

    problem_class = problem_dict.get("scattered_copy").get("problem")
    problem = problem_class(**problem_cfg)
    problem.randomize_reset = False
    process = Process(problem)
    processes = [deepcopy(process) for _ in range(batch_size)]

    # Resolve the selector zip (fixed name -- one per run)
    selector_file = os.path.join(checkpoint_path, f"{experiment_name}_selector.zip")
    if not os.path.exists(selector_file):
        raise FileNotFoundError(f"Selector zip not found: {selector_file}")
    selector = PPO.load(selector_file, device=device)
    selector.policy.set_training_mode(False)
    print(f"Loaded selector: {selector_file}")

    if cache_path is None:
        # A config-relative path needs the prefix; an explicit one is used as given
        cache_path = config.get("policy", {}).get("cache_path", "")
        cache_path = str(Path(path_prefix) / cache_path) if cache_path else ""
    mm_dir = cache_path
    h5_filename = f"{experiment_name}_policy_dataset.h5"
    memory = MemoryManager(cache_path=mm_dir, batch_size=128, h5_filename=h5_filename)
    dest = os.path.join(mm_dir, h5_filename) if mm_dir else "in-memory"

    total_episodes = problem.num_variations
    print(f"Generating policy dataset: {total_episodes} episodes -> {dest}")

    remaining = total_episodes
    sidx = 0
    with tqdm(total=total_episodes, desc="Episodes") as pbar:
        while remaining:
            used = 0
            for proc in processes:
                proc.reset(sidx)
                sidx = (sidx + 1) % problem.num_variations
                used += 1
                remaining -= 1
                pbar.update(1)
                if not remaining:
                    break

            episode_data = [[] for _ in range(used)]
            max_seq_len = max(proc.problem.seq_len for proc in processes[:used])
            for idx in range(max_seq_len):
                obs_batch, t_batch, active = [], [], []
                for j, proc in enumerate(processes[:used]):
                    if idx < proc.problem.seq_len:
                        obs, t = proc.get_obs(idx)
                        obs_batch.append(obs)
                        t_batch.append(t)
                        active.append(j)

                priorities, _ = selector.predict(torch.stack(obs_batch).numpy(), deterministic=True)

                for k, j in enumerate(active):
                    proc = processes[j]
                    proc.set_action(obs_batch[k], float(priorities[k][0]))
                    episode_data[j].append((proc.get_buffer().clone(), proc.get_target(t_batch[k]).clone()))

            for j in range(used):
                for buf, tgt in episode_data[j]:
                    memory.write(buf, tgt)

    memory.finalize()
    print(f"Done. {len(memory)} samples -> {dest}")
    return memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScatteredCopy-GeneratePolicyDataset")
    parser.add_argument("--selector_checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1024, help="Number of processes to run in parallel.")
    parser.add_argument("--cache_path", type=str, default="datasets/scattered_copy/policy_dataset",
                        help="Directory for the H5 (empty string keeps it in memory)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(args.selector_checkpoint, cache_path=args.cache_path,
             batch_size=args.batch_size, seed=args.seed)
