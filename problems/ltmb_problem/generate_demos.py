"""
Generate imitation learning dataset from expert policies and save to HDF5.

The flat structure of datasets:
    observations   (N_total, H*W*3 + 1) int32 -- flattened obs['image'] + direction
    actions        (N_total,)         int32
    rewards        (N_total,)         float32
    episode_starts  (num_episodes,)    int64   -- index into flat arrays where each episode begins
    episode_seeds   (num_episodes,)    int64
    episode_lengths (num_episodes,)    int32   -- env length param used for each episode
    attrs: env_id, num_episodes

Uses streaming HDF5 writes (resizable datasets) to avoid RAM accumulation.
Automatically writes train and test splits to:
    <output_dir>/<env_id>_train.h5
    <output_dir>/<env_id>_test.h5

Supports variable-length episodes by sampling length uniformly from [min_length, max_length].

Usage (fixed length):
    python problems/ltmb_problem/generate_demos.py --env_id LTMB-Hallway-v0 --length 10 --num_demos 100000

Usage (variable length):
    python problems/ltmb_problem/generate_demos.py --env_id LTMB-Hallway-v0 --min_length 3 --max_length 10 --num_demos 100000
"""

import argparse
import random
import numpy as np
import gymnasium as gym
import h5py
import ltmb
from tqdm import tqdm
from pathlib import Path
from ltmb.policies import ExpertHallwayPolicy, ExpertOrderingPolicy, ExpertCountingPolicy

EXPERTS = {
    'LTMB-Hallway-v0':  ExpertHallwayPolicy,
    'LTMB-Ordering-v0': ExpertOrderingPolicy,
    'LTMB-Counting-v0': ExpertCountingPolicy,
}

OBS_DIM = 7 * 7 * 3 + 1  # flattened image + direction
CHUNK_SIZE = 1000     # rows per H5 chunk


def _init_h5(path):
    """Open an H5 file and create resizable datasets for streaming writes."""
    f = h5py.File(path, 'w')
    f.create_dataset('observations', shape=(0, OBS_DIM), maxshape=(None, OBS_DIM),
                     dtype=np.int32, chunks=(CHUNK_SIZE, OBS_DIM))
    f.create_dataset('actions', shape=(0,), maxshape=(None,),
                     dtype=np.int32, chunks=(CHUNK_SIZE,))
    f.create_dataset('rewards', shape=(0,), maxshape=(None,),
                     dtype=np.float32, chunks=(CHUNK_SIZE,))
    return f


def _write_episode(f, step_cursor, ep_obs, ep_actions, ep_rewards):
    """Append one episode to an open H5 file. Returns new step_cursor."""
    n = len(ep_obs)
    f['observations'].resize((step_cursor + n, OBS_DIM))
    f['actions'].resize((step_cursor + n,))
    f['rewards'].resize((step_cursor + n,))
    f['observations'][step_cursor:step_cursor + n] = np.array(ep_obs,     dtype=np.int32)
    f['actions']     [step_cursor:step_cursor + n] = np.array(ep_actions, dtype=np.int32)
    f['rewards']     [step_cursor:step_cursor + n] = np.array(ep_rewards, dtype=np.float32)
    return step_cursor + n


def _finalize_h5(f, env_name, episode_starts, episode_seeds, episode_lengths):
    """Write metadata and close an H5 file."""
    f.create_dataset('episode_starts',  data=np.array(episode_starts,  dtype=np.int64))
    f.create_dataset('episode_seeds',   data=np.array(episode_seeds,   dtype=np.int64))
    f.create_dataset('episode_lengths', data=np.array(episode_lengths, dtype=np.int32))
    f.attrs['env_id'] = env_name
    f.attrs['num_episodes'] = len(episode_starts)
    f.close()


def collect_and_save(env_name, expert_cls, num_trajectories, lengths, extra_options,
                     output_dir, train_split, seed):
    num_train = int(num_trajectories * train_split)
    num_test  = num_trajectories - num_train

    train_path = Path(output_dir) / f"{env_name}_train.h5"
    test_path  = Path(output_dir) / f"{env_name}_test.h5"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Creating {len(lengths)} env(s) for lengths {lengths} ...")
    print(f"Split: {num_train} train / {num_test} test  →  {train_path.name} / {test_path.name}")

    envs = {l: gym.make(env_name, length=l, **extra_options) for l in lengths}
    rng  = random.Random(seed)

    train_f = _init_h5(train_path)
    test_f  = _init_h5(test_path)

    train_starts, train_seeds, train_lengths, train_cursor = [], [], [], 0
    test_starts,  test_seeds,  test_lengths,  test_cursor  = [], [], [], 0
    failed = 0
    written = 0

    pbar = tqdm(total=num_trajectories, desc=env_name, unit="ep")
    while written < num_trajectories:
        length = rng.choice(lengths)
        env    = envs[length]
        ep_seed = rng.randint(0, 10**9)
        policy  = expert_cls()  # non-Markovian: reinit each episode

        try:
            obs, _ = env.reset(seed=ep_seed)
            done = False
            ep_obs, ep_actions, ep_rewards = [], [], []

            while not done:
                action = policy.select_action(obs)
                ep_obs.append(np.concatenate((
                    obs['image'].flatten(),
                    np.array([obs['direction']], dtype=np.int32),
                )))
                ep_actions.append(int(action))
                obs, reward, terminated, truncated, info = env.step(action)
                ep_rewards.append(float(reward))
                done = terminated or truncated

            if not info.get('success', False):
                failed += 1
                del ep_obs, ep_actions, ep_rewards
                continue

        except Exception:
            failed += 1
            del ep_obs, ep_actions, ep_rewards
            continue

        # Route to train or test
        if written < num_train:
            train_starts.append(train_cursor)
            train_seeds.append(ep_seed)
            train_lengths.append(length)
            train_cursor = _write_episode(train_f, train_cursor, ep_obs, ep_actions, ep_rewards)
        else:
            test_starts.append(test_cursor)
            test_seeds.append(ep_seed)
            test_lengths.append(length)
            test_cursor = _write_episode(test_f, test_cursor, ep_obs, ep_actions, ep_rewards)

        del ep_obs, ep_actions, ep_rewards
        written += 1
        pbar.update(1)

    pbar.close()

    _finalize_h5(train_f, env_name, train_starts, train_seeds, train_lengths)
    _finalize_h5(test_f,  env_name, test_starts,  test_seeds,  test_lengths)

    for env in envs.values():
        env.close()

    print(f"Saved {len(train_starts)} train + {len(test_starts)} test episodes to {output_dir}/")
    if failed:
        print(f"  Skipped {failed} failed episodes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', required=True, choices=list(EXPERTS.keys()))
    parser.add_argument('--num_demos', type=int, default=1000, help='Total number of trajectories')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Fraction of episodes for train split (default: 0.9)')
    parser.add_argument('--output_dir', type=str, default='datasets/ltmb/proxy_dataset',
                        help='Output directory (default: datasets/ltmb/proxy_dataset)')
    parser.add_argument('--seed', type=int, default=0)

    # Length: fixed or range
    length_group = parser.add_mutually_exclusive_group(required=True)
    length_group.add_argument('--length', type=int, help='Fixed task length')
    length_group.add_argument('--min_length', type=int, help='Minimum task length (use with --max_length)')
    parser.add_argument('--max_length', type=int, help='Maximum task length (use with --min_length)')

    # Counting-specific
    parser.add_argument('--test_freq', type=float, default=0.3,
                        help='Counting task test room frequency (default: 0.3)')

    args = parser.parse_args()

    if args.min_length is not None:
        if args.max_length is None:
            parser.error('--max_length is required when --min_length is specified')
        if args.max_length < args.min_length:
            parser.error('--max_length must be >= --min_length')
        lengths = list(range(args.min_length, args.max_length + 1))
    else:
        lengths = [args.length]

    extra_options = {}
    if args.env_id == 'LTMB-Counting-v0':
        extra_options['test_freq'] = args.test_freq

    collect_and_save(args.env_id, EXPERTS[args.env_id], args.num_demos, lengths, extra_options,
                     args.output_dir, args.train_split, args.seed)


if __name__ == '__main__':
    main()
