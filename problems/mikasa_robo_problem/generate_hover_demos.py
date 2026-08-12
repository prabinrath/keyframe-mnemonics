"""Build the hover-augmented RememberColor3 demos (RememberColor3Hover-v0).

Replays each successful RememberColor3-v0 demo with a hover spliced in after the cue
disappears and ``delta_time`` extended to match, so the cubes reveal as the hover ends.
Forces the source's cube layout before stepping and reads success from the env. Writes
released-MIKASA-format NPZs, keeping only source-successful demos, renumbered from 0.

Usage:
    MUJOCO_GL=egl python problems/mikasa_robo_problem/generate_hover_demos.py \
        --hover_steps 10 \
        --src_dir datasets/mikasa_robo/RememberColor3-v0 \
        --output_dir datasets/mikasa_robo/RememberColor3Hover-v0
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import gymnasium as gym
import mani_skill.envs  # noqa: F401  registers builtin envs
import mikasa_robo_suite.memory_envs  # noqa: F401  registers RememberColor3-v0
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mikasa_robo_suite.memory_envs.remember_color import RememberColorBaseEnv
from mikasa_robo_suite.utils.wrappers import InitialZeroActionWrapper

from problems.mikasa_robo_problem.mikasa_robo_problem import env_steps


ENV_ID = "RememberColor3-v0"
BATCH_SIZE = 250                 # source demos were collected 250 envs at a time, seed=batch_idx
ACTION_DIM = 8
TIME_OFFSET = RememberColorBaseEnv.TIME_OFFSET                # memorize phase length
DEFAULT_DELTA_TIME = RememberColorBaseEnv.DEFAULT_DELTA_TIME  # default delay phase length


def build_env(env_id, delta_time, max_episode_steps):
    env = gym.make(
        env_id,
        num_envs=BATCH_SIZE,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense",
        max_episode_steps=max_episode_steps,
        delta_time=delta_time,
    )
    env = InitialZeroActionWrapper(env, n_initial_steps=0)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=False)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, BATCH_SIZE, ignore_terminations=True, record_metrics=False)
    return env


def to_np(x):
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def get_joints_np(env):
    """joints = tcp_pose(7) + qpos(9) + qvel(9) -> (B, 25), read straight off the env."""
    agent = env.unwrapped.agent
    return np.concatenate([to_np(agent.tcp.pose.raw_pose),
                           to_np(agent.robot.get_qpos()),
                           to_np(agent.robot.get_qvel())], axis=-1).astype(np.float32)


def sample_hover_arm_deltas(hover_steps, scale, rng):
    """Return-trip arm deltas, (hover_steps, BATCH_SIZE, 7), summing to 0 over time."""
    if hover_steps == 0:
        return np.zeros((0, BATCH_SIZE, 7), dtype=np.float32)
    if hover_steps % 2 != 0:
        raise ValueError(f"hover_steps must be even (got {hover_steps})")
    half = rng.uniform(-scale, scale, size=(hover_steps // 2, BATCH_SIZE, 7)).astype(np.float32)
    return np.concatenate([half, -half[::-1]], axis=0)


def episode_len(t_orig, args):
    """Episode length; defaults to the env horizon so the hover shifts the demo later
    inside a fixed-length episode (60 = 5 memorize + 15 delay + 40 action)."""
    return t_orig if args.max_steps is None else args.max_steps


def cube_order(rgb_overhead, reveal):
    """Colour -> slot ordering (left to right) at the reveal frame. Cubes are isolated by
    differencing against the prior frame; the saturation test needs mid channel < 70."""
    a = rgb_overhead[reveal].astype(int)
    b = rgb_overhead[reveal - 1].astype(int)
    changed = np.abs(a - b).sum(-1) > 90
    srt = np.sort(a, axis=-1)
    sat = (srt[..., 2] > 110) & (srt[..., 1] < 70)
    xs = {}
    for k in range(3):
        m = changed & sat & (a.argmax(-1) == k)
        if m.sum() < 4:
            return None
        xs[k] = float(np.nonzero(m)[1].mean())
    return tuple(sorted(xs, key=xs.get))


def force_source_layout(env, src_order):
    """Permute cubes so each colour takes the slot it held in the source. Positions already
    match (<1mm); only the permutation differs. Returns the number of envs remapped."""
    u = env.unwrapped
    keys = sorted(u.initial_raw_poses)
    raw = {k: u.initial_raw_poses[k].clone() for k in keys}
    xyz = np.stack([raw[k][:, :3].cpu().numpy() for k in keys], axis=1)      # (B, 3, 3)
    new = xyz.copy()
    n = 0
    for i in range(xyz.shape[0]):
        if src_order[i] is None:
            continue
        slots = xyz[i][np.argsort(xyz[i][:, 1])]        # our slots, screen left to right
        for k in range(3):
            new[i, k] = slots[src_order[i].index(k)]
        n += 1
    for j, k in enumerate(keys):
        t = raw[k]
        t[:, :3] = torch.as_tensor(new[:, j], device=t.device, dtype=t.dtype)
        u.initial_raw_poses[k] = t
        cp = u.cubes[k].pose.raw_pose.clone()
        cp[:, :3] = t[:, :3]
        u.cubes[k].pose = cp
    u._zero_cube_velocities(sleep=True)
    u._sync_cube_gpu_state()
    return n


def replay_batch(env, batch_idx, batch_actions, src_done, args, rng, device, t_orig,
                 src_order):
    """Step one 250-env batch through the hover-stitched action tape, with the cube layout
    forced to the source's and success read from the env each step."""
    T_new = episode_len(t_orig, args)
    hover_at = args.hover_at

    # The hover shifts the demo later by hover_steps, so the tail of the source tape
    # falls outside the episode; keep only as much as fits.
    full_actions = np.zeros((T_new, BATCH_SIZE, ACTION_DIM), dtype=np.float32)
    full_actions[:hover_at] = batch_actions[:hover_at]
    n_post = T_new - (hover_at + args.hover_steps)
    full_actions[hover_at + args.hover_steps:] = batch_actions[hover_at:hover_at + n_post]

    hover_arm = sample_hover_arm_deltas(args.hover_steps, args.hover_scale, rng)
    last_gripper = (batch_actions[hover_at - 1, :, 7]
                    if hover_at > 0 else np.zeros(BATCH_SIZE, np.float32))
    for k in range(args.hover_steps):
        full_actions[hover_at + k, :, :7] = hover_arm[k]
        full_actions[hover_at + k, :, 7] = last_gripper

    done_arr = np.zeros((T_new, BATCH_SIZE), dtype=np.int32)
    done_arr[:hover_at] = src_done[:hover_at]
    done_arr[hover_at + args.hover_steps:] = src_done[hover_at:hover_at + n_post]

    obs, _ = env.reset(seed=batch_idx)
    n_forced = force_source_layout(env, src_order)
    print(f"[batch {batch_idx}] forced source cube layout on {n_forced}/{BATCH_SIZE} envs")

    rgb_arr = np.zeros((T_new, BATCH_SIZE, 128, 128, 6), dtype=np.uint8)
    joints_arr = np.zeros((T_new, BATCH_SIZE, 25), dtype=np.float32)
    reward_arr = np.zeros((T_new, BATCH_SIZE), dtype=np.float32)
    success_arr = np.zeros((T_new, BATCH_SIZE), dtype=np.int32)
    rgb_arr[0] = to_np(obs["rgb"]).astype(np.uint8)
    joints_arr[0] = get_joints_np(env)

    u = env.unwrapped
    for t in range(T_new):
        obs, reward, _, _, _ = env.step(torch.from_numpy(full_actions[t]).to(device))
        reward_arr[t] = to_np(reward).astype(np.float32)
        success_arr[t] = to_np(u.evaluate()["success"]).astype(np.int32)
        if t + 1 < T_new:
            rgb_arr[t + 1] = to_np(obs["rgb"]).astype(np.uint8)
            joints_arr[t + 1] = get_joints_np(env)

    return {"rgb": rgb_arr, "joints": joints_arr, "action": full_actions,
            "reward": reward_arr, "success": success_arr, "done": done_arr}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hover_steps", type=int, default=10,
                   help="Hover length (must be even). delta_time = %d + hover_steps."
                        % DEFAULT_DELTA_TIME)
    p.add_argument("--hover_at", type=int, default=None,
                   help="Step index to insert the hover at. "
                        f"Default = TIME_OFFSET + DEFAULT_DELTA_TIME = {TIME_OFFSET + DEFAULT_DELTA_TIME}.")
    p.add_argument("--hover_scale", type=float, default=0.02,
                   help="Per-step arm-delta amplitude in normalized [-1,1] action space "
                        "(0.02 ~ 0.1 deg per joint per step).")
    p.add_argument("--src_dir", type=str, default=f"datasets/mikasa_robo/{ENV_ID}",
                   help="Directory of released train_data_*.npz demos to replay.")
    p.add_argument("--output_dir", type=str,
                   default="datasets/mikasa_robo/RememberColor3Hover-v0")
    p.add_argument("--n_batches", type=int, default=None,
                   help="Batches of 250 to process (default: all source demos).")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Episode length (default: the env's own horizon, 60). The hover "
                        "shifts the demo later within it rather than extending the episode.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0, help="Seed for the hover-delta sampler.")
    args = p.parse_args()

    if args.hover_at is None:
        args.hover_at = TIME_OFFSET + DEFAULT_DELTA_TIME
    if args.hover_steps % 2 != 0:
        raise SystemExit(f"--hover_steps must be even (got {args.hover_steps})")

    src_dir = Path(args.src_dir)
    n_src = len(sorted(src_dir.glob("train_data_*.npz")))
    if n_src == 0:
        raise SystemExit(f"No train_data_*.npz found in {src_dir}")
    n_batches = args.n_batches if args.n_batches is not None else n_src // BATCH_SIZE
    if n_batches == 0:
        raise SystemExit(f"{n_src} source demos in {src_dir}; need at least {BATCH_SIZE} "
                         "(replay must reproduce the 250-env collection batches)")

    t_orig = env_steps(ENV_ID)
    delta_time = DEFAULT_DELTA_TIME + args.hover_steps       # sync invariant
    T_new = episode_len(t_orig, args)

    print(f"[config] hover_steps={args.hover_steps}  hover_at={args.hover_at}  "
          f"hover_scale={args.hover_scale}")
    print(f"[config] env delta_time={delta_time}  cubes reveal at step {TIME_OFFSET + delta_time}")
    print(f"[config] T_orig={t_orig}  T_new={T_new}  "
          f"(memorize {TIME_OFFSET} + delay {delta_time} + action {T_new - TIME_OFFSET - delta_time})")
    print(f"[config] batches={n_batches} x {BATCH_SIZE}")
    print(f"[config] {src_dir} -> {args.output_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Headroom only: the vector env still truncates on the time limit and would
    # auto-reset mid-replay. Demos are still T_new long.
    print(f"[env] building (max_episode_steps={T_new + 5}, recording {T_new} steps) ...")
    env = build_env(ENV_ID, delta_time=delta_time, max_episode_steps=T_new + 5)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    saved = n_hit = 0
    for batch_idx in range(n_batches):
        print(f"\n[batch {batch_idx}/{n_batches - 1}] loading {BATCH_SIZE} source npzs ...")
        batch_actions = np.zeros((t_orig, BATCH_SIZE, ACTION_DIM), dtype=np.float32)
        batch_done = np.zeros((t_orig, BATCH_SIZE), dtype=np.int32)
        succeeded = np.zeros(BATCH_SIZE, dtype=bool)
        src_order = []
        for env_idx in range(BATCH_SIZE):
            d = np.load(src_dir / f"train_data_{batch_idx * BATCH_SIZE + env_idx}.npz")
            batch_actions[:, env_idx] = d["action"].astype(np.float32)
            batch_done[:, env_idx] = d["done"].astype(np.int32)
            succeeded[env_idx] = bool(d["success"][-1])
            src_order.append(cube_order(d["rgb"][:, :, :, :3], TIME_OFFSET + DEFAULT_DELTA_TIME))
        n_read = sum(o is not None for o in src_order)
        print(f"[batch {batch_idx}] {int(succeeded.sum())}/{BATCH_SIZE} originally successful | "
              f"source layout read on {n_read}/{BATCH_SIZE}")
        if n_read < BATCH_SIZE:
            raise SystemExit(f"could not read the source cube layout for "
                             f"{BATCH_SIZE - n_read} episodes of batch {batch_idx}")

        print(f"[batch {batch_idx}] replaying with hover ...")
        out = replay_batch(env, batch_idx, batch_actions, batch_done,
                           args, rng, device, t_orig, src_order)

        # Guarantee: the layout we rendered must equal the source's, per episode.
        reveal = TIME_OFFSET + delta_time
        bad = [i for i in range(BATCH_SIZE)
               if succeeded[i] and cube_order(out["rgb"][:, i, :, :, :3], reveal) != src_order[i]]
        if bad:
            raise SystemExit(f"batch {batch_idx}: cube layout mismatch on {len(bad)} episodes "
                             f"(first: env {bad[0]})")
        print(f"[batch {batch_idx}] layout verified against source on all "
              f"{int(succeeded.sum())} saved episodes")

        for env_idx in range(BATCH_SIZE):
            if not succeeded[env_idx]:
                continue
            np.savez(out_dir / f"train_data_{saved}.npz",
                     **{k: v[:, env_idx] for k, v in out.items()})
            saved += 1
        # Replay success under the env's own semantics: the episode terminates at the
        # first success, so "succeeded at any step" is the metric eval_policy.py uses.
        hit = out["success"][:, succeeded].any(axis=0)
        n_hit += int(hit.sum())
        print(f"[batch {batch_idx}] replay SR {hit.mean():.1%} "
              f"({int(hit.sum())}/{int(succeeded.sum())}) | cumulative saved: {saved}")

    env.close()
    print(f"\n[done] wrote {saved} trajectories to {out_dir}")
    print(f"[done] hover replay SR (success at any step): {n_hit}/{saved} = {n_hit / saved:.1%}")
    print(f"[done] cube layout + colour sequence verified against source on all {saved} demos")


if __name__ == "__main__":
    main()
