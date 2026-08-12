"""
Replay recorded minigrid/LTMB demonstrations in the environment.

Recreates the gym env using the saved seed and steps through saved actions.
For LTMB envs (env_id starts with 'LTMB-') the length param is passed to
gym.make. For other minigrid envs (e.g. MiniGrid-MemoryS17Random-v0) gym.make
is called without extra params.

Usage:
    python problems/ltmb_problem/validate_demos.py --file datasets/ltmb/proxy_dataset/LTMB-Hallway-v0_train.h5
    python problems/ltmb_problem/validate_demos.py --file datasets/ltmb/proxy_dataset/MiniGrid-MemoryS17Random-v0_train.h5
    python problems/ltmb_problem/validate_demos.py --file datasets/ltmb/proxy_dataset/LTMB-Hallway-v0_train.h5 --episodes 10 --episode_idx 5 --fps 4
"""

import argparse
import time
import h5py
import gymnasium as gym
import minigrid
import ltmb      # noqa: F401 — registers LTMB envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to .h5 demo file')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to replay')
    parser.add_argument('--episode_idx', type=int, default=0, help='Starting episode index')
    parser.add_argument('--fps', type=float, default=5, help='Steps per second')
    args = parser.parse_args()

    with h5py.File(args.file, 'r') as f:
        env_id          = f.attrs['env_id']
        num_episodes    = f.attrs['num_episodes']
        actions         = f['actions'][:]
        ep_starts       = f['episode_starts'][:]
        ep_seeds        = f['episode_seeds'][:]
        ep_lengths      = f['episode_lengths'][:]

    print(f"{env_id}  |  {num_episodes} total episodes")

    delay = 1.0 / args.fps

    for i in range(args.episodes):
        ep_idx = args.episode_idx + i
        if ep_idx >= num_episodes:
            break

        start  = ep_starts[ep_idx]
        end    = ep_starts[ep_idx + 1] if ep_idx + 1 < num_episodes else len(actions)
        ep_act = actions[start:end]
        seed   = int(ep_seeds[ep_idx])
        length = int(ep_lengths[ep_idx])

        if env_id.startswith('LTMB-'):
            env = gym.make(env_id, length=length, render_mode='human')
        else:
            env = gym.make(env_id, render_mode='human')
        env.reset(seed=seed)

        for action in ep_act:
            env.render()
            time.sleep(delay)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        success = info.get('success', terminated and not truncated)
        status = 'SUCCESS' if success else 'FAIL'
        print(f"Episode {ep_idx}  length={length}  steps={len(ep_act)}  {status}")
        env.close()
        time.sleep(0.5)


if __name__ == '__main__':
    main()
