import argparse
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import minigrid
from minigrid.core.constants import OBJECT_TO_IDX


LEFT = 0
RIGHT = 1
FORWARD = 2
NO_OP = 5

KEY_IDX = OBJECT_TO_IDX["key"]
BALL_IDX = OBJECT_TO_IDX["ball"]


def cue_visible(env, cue_xy, obs):
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "agent_sees"):
        return bool(unwrapped.agent_sees(int(cue_xy[0]), int(cue_xy[1])))
    obj_layer = obs["image"][..., 0]
    return np.any((obj_layer == KEY_IDX) | (obj_layer == BALL_IDX))


def find_memory_layout(unwrapped):
    obj_positions = []
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            obj = unwrapped.grid.get(x, y)
            if obj is not None and obj.type in {"key", "ball"}:
                obj_positions.append((x, y, obj.type))

    if len(obj_positions) != 3:
        raise RuntimeError("Expected exactly 3 key/ball objects (1 cue + 2 branch objects)")

    cue_obj = min(obj_positions, key=lambda p: (p[0], p[1]))
    branch_objs = [p for p in obj_positions if p != cue_obj]

    split_x = branch_objs[0][0]
    if branch_objs[1][0] != split_x:
        raise RuntimeError("Branch objects are not aligned on one split column")

    top_obj = min(branch_objs, key=lambda p: p[1])
    bot_obj = max(branch_objs, key=lambda p: p[1])
    return cue_obj, split_x, top_obj, bot_obj


def turn_to_dir_actions_ccw(curr_dir, target_dir):
    left_steps = (curr_dir - target_dir) % 4
    return [LEFT] * left_steps


def turn_to_dir_actions_cw(curr_dir, target_dir):
    right_steps = (target_dir - curr_dir) % 4
    return [RIGHT] * right_steps


def step_actions(env, obs, actions, traj, visualize=False, render_delay=0.03):
    """Execute actions, append transitions to traj, and return latest step outputs."""
    step_reward = 0.0
    terminated = False
    truncated = False
    info = {}

    for a in actions:
        traj.append({
            "obs_image": np.array(obs["image"], copy=True),
            "obs_direction": int(obs["direction"]),
            "action": a,
        })
        obs, step_reward, terminated, truncated, info = env.step(a)
        traj[-1]["reward"] = float(step_reward)
        if visualize:
            env.render()
            time.sleep(render_delay)
        if terminated or truncated:
            break

    return obs, step_reward, terminated, truncated, info


def step_one(env, obs, action, traj, visualize=False, render_delay=0.03):
    return step_actions(
        env,
        obs,
        [action],
        traj,
        visualize=visualize,
        render_delay=render_delay,
    )


def target_xy_from_cue(cue_type, split_x, top_obj, bot_obj):
    if top_obj[2] == cue_type:
        return (split_x, top_obj[1] + 1)
    if bot_obj[2] == cue_type:
        return (split_x, bot_obj[1] - 1)
    raise RuntimeError("No branch object matches cue")


def rollout_expert_episode(
    env,
    seed=None,
    visualize=False,
    render_delay=0.03,
    cue_noops=0,
):
    obs, _ = env.reset(seed=seed)

    if visualize:
        env.render()
        time.sleep(render_delay)

    unwrapped = env.unwrapped
    height = int(unwrapped.height)
    mid_y = height // 2

    cue_obj, split_x, top_obj, bot_obj = find_memory_layout(unwrapped)
    cue_xy = (int(cue_obj[0]), int(cue_obj[1]))
    cue_type = cue_obj[2]
    cue_visible_at_reset = cue_visible(env, cue_xy, obs)

    traj = []
    if not cue_visible_at_reset:
        # Phase 1: get into hallway row, face left, then move until cue enters view.
        curr_x, curr_y = int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1])
        curr_dir = int(unwrapped.agent_dir)
        phase1_actions = []

        if curr_y != mid_y:
            target_dir = 1 if mid_y > curr_y else 3
            phase1_actions.extend(turn_to_dir_actions_ccw(curr_dir, target_dir))
            phase1_actions.extend([FORWARD] * abs(mid_y - curr_y))
            curr_dir = target_dir

        phase1_actions.extend(turn_to_dir_actions_ccw(curr_dir, 2))

        for a in phase1_actions:
            obs, step_reward, terminated, truncated, _ = step_one(
                env,
                obs,
                a,
                traj,
                visualize=visualize,
                render_delay=render_delay,
            )
            if terminated or truncated:
                return traj, step_reward, terminated, truncated
            if cue_visible(env, cue_xy, obs):
                break

        while not cue_visible(env, cue_xy, obs):
            obs, step_reward, terminated, truncated, _ = step_one(
                env,
                obs,
                FORWARD,
                traj,
                visualize=visualize,
                render_delay=render_delay,
            )
            if terminated or truncated:
                return traj, step_reward, terminated, truncated

    target_xy = target_xy_from_cue(cue_type, split_x, top_obj, bot_obj)

    # Phase 2: pause on the cue, then U-turn, go right to split, then go to matching branch.
    curr_x, curr_y = int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1])
    curr_dir = int(unwrapped.agent_dir)

    actions = []
    actions.extend([NO_OP] * max(0, int(cue_noops)))
    actions.extend(turn_to_dir_actions_cw(curr_dir, 0))  # face right using CW turns
    actions.extend([FORWARD] * max(0, split_x - curr_x))

    target_y = int(target_xy[1])
    if target_y < curr_y:
        actions.append(LEFT)  # junction rule: upper branch => ACW turn
    elif target_y > curr_y:
        actions.append(RIGHT)  # junction rule: lower branch => CW turn

    actions.extend([FORWARD] * abs(target_y - curr_y))

    _, step_reward, terminated, truncated, _ = step_actions(
        env, obs, actions, traj, visualize=visualize, render_delay=render_delay
    )

    return traj, step_reward, terminated, truncated


def collect_demos(
    env_id="MiniGrid-MemoryS11-v0",
    n_episodes=100,
    seed=0,
    visualize=False,
    render_delay=0.03,
    max_attempts=0,
    cue_noops=0,
):
    render_mode = "human" if visualize else None
    env = gym.make(env_id, render_mode=render_mode)
    # env = FullyObsWrapper(env)  # uncomment if you want fully observable obs

    demos = []
    episode_seeds = []
    episode_lengths = []
    attempts = 0
    attempts_limit = None if max_attempts <= 0 else int(max_attempts)

    while len(demos) < n_episodes and (attempts_limit is None or attempts < attempts_limit):
        traj, reward, terminated, truncated = rollout_expert_episode(
            env,
            seed=seed + attempts,
            visualize=visualize,
            render_delay=render_delay,
            cue_noops=cue_noops,
        )

        if terminated and not truncated:
            demos.append(traj)
            episode_seeds.append(seed + attempts)
            episode_lengths.append(int(env.unwrapped.width))
            print(f"[{len(demos)}/{n_episodes}] collected (seed={seed + attempts}, return={reward:.3f})")

        attempts += 1

    env.close()
    return demos, attempts, episode_seeds, episode_lengths


def save_demos_h5(output_path, demos, env_id, episode_seeds, episode_lengths):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    obs_dim = 7 * 7 * 3 + 1
    total_steps = int(sum(len(traj) for traj in demos))
    observations = np.zeros((total_steps, obs_dim), dtype=np.int32)
    actions = np.zeros((total_steps,), dtype=np.int32)
    rewards = np.zeros((total_steps,), dtype=np.float32)
    episode_starts = []

    cursor = 0
    for traj in demos:
        episode_starts.append(cursor)
        n = len(traj)
        for i, step in enumerate(traj):
            observations[cursor + i] = np.concatenate((
                step["obs_image"].reshape(-1).astype(np.int32),
                np.array([step["obs_direction"]], dtype=np.int32),
            ))
            actions[cursor + i] = int(step["action"])
            rewards[cursor + i] = float(step["reward"])
        cursor += n

    with h5py.File(output_path, "w") as f:
        f.attrs["env_id"] = env_id
        f.attrs["num_episodes"] = int(len(demos))
        f.create_dataset("observations", data=observations, dtype=np.int32)
        f.create_dataset("actions", data=actions, dtype=np.int32)
        f.create_dataset("rewards", data=rewards, dtype=np.float32)
        f.create_dataset("episode_starts", data=np.asarray(episode_starts, dtype=np.int64))
        f.create_dataset("episode_seeds", data=np.asarray(episode_seeds, dtype=np.int64))
        f.create_dataset("episode_lengths", data=np.asarray(episode_lengths, dtype=np.int32))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate expert demos for MiniGrid Memory environments")
    parser.add_argument("--env_id", type=str, default="MiniGrid-MemoryS17Random-v0")
    parser.add_argument("--num_demos", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default="datasets/ltmb/proxy_dataset")
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=0,
        help="Maximum rollout attempts. Use 0 for unlimited.",
    )
    parser.add_argument("--visualize", action="store_true", help="Render while collecting demos")
    parser.add_argument("--render_delay", type=float, default=0.03)
    parser.add_argument(
        "--cue_noops",
        type=int,
        default=0,
        help="Number of no-op actions to take at the start of phase 2 after the cue is visible.",
    )
    args = parser.parse_args()

    valid_memory_envs = sorted(env_id for env_id in gym.registry.keys() if "Memory" in env_id)
    assert args.env_id in valid_memory_envs, (
        f"Invalid --env-id '{args.env_id}'. Must be one of: {valid_memory_envs}"
    )
    assert 0.0 < args.train_split < 1.0, "--train-split must be in (0, 1)"
    assert args.cue_noops >= 0, "--cue_noops must be >= 0"

    return args


if __name__ == "__main__":
    args = parse_args()

    demos, attempts, episode_seeds, episode_lengths = collect_demos(
        env_id=args.env_id,
        n_episodes=args.num_demos,
        seed=args.seed,
        visualize=args.visualize,
        render_delay=args.render_delay,
        max_attempts=args.max_attempts,
        cue_noops=args.cue_noops,
    )
    num_train = int(len(demos) * args.train_split)
    train_path = Path(args.output_dir) / f"{args.env_id}_train.h5"
    test_path = Path(args.output_dir) / f"{args.env_id}_test.h5"

    save_demos_h5(
        train_path,
        demos[:num_train],
        args.env_id,
        episode_seeds[:num_train],
        episode_lengths[:num_train],
    )
    save_demos_h5(
        test_path,
        demos[num_train:],
        args.env_id,
        episode_seeds[num_train:],
        episode_lengths[num_train:],
    )

    print(f"Collected {len(demos)} successful demos")
    print(f"Split: {num_train} train / {len(demos) - num_train} test")
    print(f"Attempts: {attempts}")
    if demos:
        mean_return = np.mean([traj[-1]["reward"] for traj in demos])
        print(f"Mean return: {mean_return:.4f}")
    print(f"Saved train: {train_path}")
    print(f"Saved test: {test_path}")
