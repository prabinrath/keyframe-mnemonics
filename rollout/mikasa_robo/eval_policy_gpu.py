"""Vectorized GPU-sim eval for Mikasa Robo policies.

Runs `--num_envs` episodes in parallel through ManiSkill 3 GPU sim
(`sim_backend="gpu"`), batching the selector/policy forward passes and the
keyframe-buffer ops. Reports aggregate Success Rate / Average Reward — the fast
quantitative counterpart to ``eval_policy.py`` (which runs one CPU-sim env with
per-step buffer video and action charts).

The keyframe buffer is vectorized across all N envs as a single (N, maxsize, D)
CUDA tensor (``BatchedEvictingBuffer``), which reproduces the per-env
``InferencePriorityQueue`` ("evict_latest_norepeat") semantics exactly.

Usage:
    python rollout/mikasa_robo/eval_policy_gpu.py \
        --policy_checkpoint <run_folder> --n_episodes 100 --num_envs 25
"""
import argparse
import glob
import os
import time
import yaml
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

from models import model_dict
from common.helpers import set_seeds, parse_experiment_name, stage_dir
from problems.mikasa_robo_problem.mikasa_robo_problem import (
    apply_delta_time_override, base_env_id, env_steps)
from problems.mikasa_robo_problem.rotate_prompts import ROTATE_ENV_IDS, PROMPT_SLOT
from mikasa_robo_suite.utils.wrappers import StateOnlyTensorToDictWrapper


OBS_DIM = 98329          # overhead(49152) + gripper(49152) + tcp_pose(7) + qpos(9) + qvel(9)
IMG_DIM = 49152          # one 128x128x3 camera, flattened
QPOS_SLICE = slice(98311, 98320)   # qpos (9) within a flat obs — the policy's observation.state


class BatchedEvictingBuffer:
    """Vectorized ``InferencePriorityQueue`` ("evict_latest_norepeat") across N envs.

    All N per-env buffers live in one set of device tensors:
      values        (N, maxsize, D)  filled-slot observations
      priorities    (N, maxsize)
      keys          (N, maxsize)     insertion id per slot; -1 marks an empty slot
      sizes         (N,)             filled-slot count
      last_priority (N,)             last accepted priority (for the no-repeat check)
      latest_value  (N, D)           most-recent obs (the current-obs slot)

    ``get_batched()`` returns (N, n_slots, D) with filled slots first in insertion
    order (empties zeroed) and, when ``use_current_obs``, ``latest_value`` in the
    last slot — the batched equivalent of stacking each env's
    ``InferencePriorityQueue.get()``.
    """

    def __init__(self, num_envs, maxsize, data_dim, use_current_obs=True,
                 rejection_threshold=0.5, no_repeat_threshold=0.05,
                 device=torch.device("cpu")):
        self.num_envs = num_envs
        self.maxsize = maxsize
        self.D = int(data_dim)
        self.use_current_obs = use_current_obs
        self.n_slots = maxsize + (1 if use_current_obs else 0)
        self.rejection_threshold = float(rejection_threshold)
        self.no_repeat_threshold = float(no_repeat_threshold)
        self.device = device

        self.values = torch.zeros(num_envs, maxsize, self.D, device=device)
        self.priorities = torch.zeros(num_envs, maxsize, device=device)
        self.keys = torch.full((num_envs, maxsize), -1, dtype=torch.int64, device=device)
        self.sizes = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.next_key = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.last_priority = torch.zeros(num_envs, device=device)
        self.latest_value = torch.zeros(num_envs, self.D, device=device) if use_current_obs else None
        self._out = torch.zeros(num_envs, self.n_slots, self.D, device=device)

    def reset(self, env_idx=None):
        """Clear all envs, or the subset selected by a bool/long ``env_idx``."""
        if env_idx is None:
            sel = slice(None)
        else:
            if env_idx.dtype == torch.bool:
                env_idx = torch.nonzero(env_idx, as_tuple=False).flatten()
            if env_idx.numel() == 0:
                return
            sel = env_idx
        self.values[sel] = 0
        self.priorities[sel] = 0
        self.keys[sel] = -1
        self.sizes[sel] = 0
        self.next_key[sel] = 0
        self.last_priority[sel] = 0
        if self.use_current_obs:
            self.latest_value[sel] = 0

    def push(self, values, priorities):
        """Push one obs per env. ``values`` (N, D), ``priorities`` (N,), on device."""
        if self.use_current_obs:
            self.latest_value.copy_(values)

        # Accept rule identical to InferencePriorityQueue.push: above threshold AND not a
        # near-repeat of the previous accepted priority.
        accept = (priorities >= self.rejection_threshold) & \
                 ((priorities - self.last_priority).abs() >= self.no_repeat_threshold)

        # Only accepted envs touch slots; each per-env body is a few (maxsize,) ops.
        for env_i in torch.nonzero(accept, as_tuple=False).flatten().tolist():
            p = priorities[env_i]
            self.last_priority[env_i] = p          # updated on accept, even if eviction is declined
            sz = int(self.sizes[env_i])
            if sz < self.maxsize:
                slot = sz
                self.sizes[env_i] = sz + 1
            else:
                # Evict the min-priority slot, tie-broken by largest key (most recent),
                # matching InferencePriorityQueue's min(key=(priority, -key)).
                ps = self.priorities[env_i]
                tie = ps == ps.min()
                slot = int(torch.where(tie, self.keys[env_i], -1).argmax())
                if p < ps[slot]:
                    continue                        # incoming priority loses; keep the buffer
            self.values[env_i, slot] = values[env_i]
            self.priorities[env_i, slot] = p
            self.keys[env_i, slot] = self.next_key[env_i]
            self.next_key[env_i] += 1

    def get_batched(self):
        """Assemble the (N, n_slots, D) buffer view on device."""
        # Sort slots by key ascending (== insertion order); route empty slots (-1) last.
        sentinel = int(self.next_key.max()) + 1
        keys_sort = torch.where(self.keys == -1, sentinel, self.keys)
        order = torch.argsort(keys_sort, dim=1)                      # (N, maxsize)
        vals = torch.gather(self.values, 1, order.unsqueeze(-1).expand(-1, -1, self.D))
        filled = (torch.gather(self.keys, 1, order) != -1).unsqueeze(-1).to(vals.dtype)

        out = self._out
        out[:, :self.maxsize] = vals * filled
        if self.use_current_obs:
            out[:, self.maxsize] = self.latest_value
        return out


def flat_obs_batched(obs_dict, device):
    """Batched flat obs (N, 98329) on ``device``: [overhead/255 | gripper/255 | tcp_pose | qpos | qvel].

    Mirrors ``MikasaRoboProblem._process_env_observation`` (base_camera -> overhead,
    hand_camera -> gripper).
    """
    def to_dev(x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        return x.to(device, non_blocking=True)

    base = to_dev(obs_dict["sensor_data"]["base_camera"]["rgb"]).float() / 255.0
    hand = to_dev(obs_dict["sensor_data"]["hand_camera"]["rgb"]).float() / 255.0
    qpos = to_dev(obs_dict["agent"]["qpos"]).float()
    qvel = to_dev(obs_dict["agent"]["qvel"]).float()
    tcp = to_dev(obs_dict["extra"]["tcp_pose"]).float()

    n = base.shape[0]
    return torch.cat([base.reshape(n, -1), hand.reshape(n, -1), tcp, qpos, qvel], dim=-1)


def build_policy_obs_batched(buffer_NSD, total_slots):
    """Vectorized ``build_policy_observation`` across N envs.

    ``buffer_NSD`` is (N, total_slots, 98329) float in [0, 1]. Returns the policy
    obs dict (both cameras per slot + ``observation.state`` from the last slot's
    qpos), tensors on the same device. Images pass through the float->uint8->float
    round-trip ``build_policy_observation`` applies, matching the uint8 lerobot
    frames the policy was trained on.
    """
    N, n_slots, _ = buffer_NSD.shape
    assert n_slots == total_slots, f"buffer slots {n_slots} != total_slots {total_slots}"

    def to_chw(flat):
        img = (flat * 255.0).round().clamp_(0, 255).div_(255.0)
        return img.reshape(N, n_slots, 128, 128, 3).permute(0, 1, 4, 2, 3).contiguous()

    overhead = to_chw(buffer_NSD[:, :, :IMG_DIM])
    gripper = to_chw(buffer_NSD[:, :, IMG_DIM:2 * IMG_DIM])

    out = {"observation.state": buffer_NSD[:, -1, QPOS_SLICE]}     # (N, 9) current-obs qpos
    for k in range(n_slots):
        out[f"observation.images.overhead_camera{k + 1}"] = overhead[:, k]
        out[f"observation.images.gripper_camera{k + 1}"] = gripper[:, k]
    return out


def _to_cpu(t, dtype, num_envs):
    if t is None:
        return torch.zeros(num_envs, dtype=dtype)
    if isinstance(t, torch.Tensor):
        return t.detach().to(dtype=dtype, device="cpu")
    return torch.as_tensor(np.asarray(t), dtype=dtype)


class _EpisodeRecorder:
    """Per-episode visual outputs for the first ``n_rec`` envs of a batch, in the same
    format as eval_policy.py's --record_video (written to <video_dir>/<ckpt>/<num>/):
      {i}_env.mp4     full-scene render (render_mode="all"), fps=30
      {i}_buffer.mp4  overhead/gripper buffer slots, 2 rows, per step, fps=5
      {i}_action.png  per-dim executed-action line chart

    Frames are captured only while an env is still live (each env stops at its own
    success/termination step, matching the CPU eval's break-on-success).
    """

    def __init__(self, n_rec, total_slots, out_dir, base=0):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.n_rec = n_rec
        self.total_slots = total_slots
        self.out_dir = Path(out_dir)
        self.base = base                                   # global episode index of env 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env_frames = [[] for _ in range(n_rec)]
        self.buf_frames = [[] for _ in range(n_rec)]
        self.actions = [[] for _ in range(n_rec)]
        self.done = [False] * n_rec
        self.fig, self.axes = plt.subplots(2, 1, figsize=(8, 10))
        for ax in self.axes:
            ax.axis("off")
        self.fig.tight_layout()

    def capture_step(self, buf_NSD, action_vec, step):
        """Record the buffer viz + executed action for still-live envs (call pre-step)."""
        for i in range(self.n_rec):
            if self.done[i]:
                continue
            self.actions[i].append(action_vec[i].detach().cpu().numpy())
            buf = buf_NSD[i].detach().cpu().numpy()          # (n_slots, D)
            overhead = [np.clip(buf[s, :IMG_DIM].reshape(128, 128, 3), 0, 1) for s in range(self.total_slots)]
            gripper = [np.clip(buf[s, IMG_DIM:2 * IMG_DIM].reshape(128, 128, 3), 0, 1) for s in range(self.total_slots)]
            self.axes[0].clear(); self.axes[0].imshow(np.concatenate(overhead, axis=1))
            self.axes[0].set_title(f"Overhead Buffer - Step {step}"); self.axes[0].axis("off")
            self.axes[1].clear(); self.axes[1].imshow(np.concatenate(gripper, axis=1))
            self.axes[1].set_title(f"Gripper Buffer  - Step {step}"); self.axes[1].axis("off")
            from io import BytesIO
            import imageio.v2 as imageio
            bio = BytesIO(); self.fig.savefig(bio, format="png", dpi=100, bbox_inches="tight"); bio.seek(0)
            self.buf_frames[i].append(imageio.imread(bio)); bio.close()

    def capture_render(self, render, dones):
        """Record the env frame for still-live envs, then latch newly-done envs (call post-step)."""
        render = render.detach().cpu().numpy() if isinstance(render, torch.Tensor) else np.asarray(render)
        for i in range(self.n_rec):
            if not self.done[i]:
                self.env_frames[i].append(render[i].astype(np.uint8))
        for i in range(self.n_rec):
            if bool(dones[i]):
                self.done[i] = True

    def write(self):
        import imageio.v2 as imageio
        for i in range(self.n_rec):
            ep = self.base + i
            if self.env_frames[i]:
                imageio.mimwrite(str(self.out_dir / f"{ep}_env.mp4"), self.env_frames[i],
                                 fps=30, codec="h264", quality=8)
            if self.buf_frames[i]:
                frames = [(f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
                          for f in self.buf_frames[i]]
                imageio.mimwrite(str(self.out_dir / f"{ep}_buffer.mp4"), frames,
                                 fps=5, codec="h264", quality=8)
            if self.actions[i]:
                arr = np.array(self.actions[i])                # (T, action_dim)
                fig, ax = self.plt.subplots(figsize=(10, 4))
                for dim in range(arr.shape[1]):
                    ax.plot(arr[:, dim], label=f"dim {dim}")
                ax.set_xlabel("Step"); ax.set_ylabel("Action"); ax.set_title(f"Episode {ep} Actions")
                ax.legend(loc="upper right", fontsize=6, ncol=4); fig.tight_layout()
                fig.savefig(self.out_dir / f"{ep}_action.png", dpi=100); self.plt.close(fig)
        self.plt.close(self.fig)


def eval_policy_gpu(policy_checkpoint, selector_checkpoint="", checkpoint_num=None,
                    seed=42, n_episodes=100, num_envs=25, output_log=None,
                    record_video=False, video_dir="rollout/mikasa_robo/videos",
                    delta_time=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(seed)

    experiment_name = parse_experiment_name(policy_checkpoint)
    with open(f"conf/mikasa_robo/{experiment_name}.yaml") as f:
        config = yaml.safe_load(f)
    path_prefix = config.get("path_prefix", "")

    # Compose eval problem config (policy + test overrides), matching eval_policy.py.
    problem_config = dict(config.get("problem", {}))
    problem_config.update(config.get("policy_problem_override", {}))
    problem_config.update(config.get("test_problem_override", {}))
    problem_config["use_current_obs"] = True
    if delta_time is not None:
        apply_delta_time_override(problem_config, delta_time)
    env_id = problem_config["env_id"]
    buffer_size = problem_config["buffer_size"]

    # Policy config comes from the checkpoint's frozen config.yaml.
    policy_dir = stage_dir(path_prefix, policy_checkpoint, "policy")
    with open(os.path.join(policy_dir, "config.yaml")) as f:
        policy_config = yaml.safe_load(f)["policy"]
    model_config = policy_config["model_kwargs"]
    total_slots = model_config["total_slots"]
    open_loop_steps = policy_config.get("open_loop_steps", 1)
    assert total_slots == buffer_size + 1, \
        f"total_slots ({total_slots}) must equal buffer_size + 1 ({buffer_size + 1})"

    # Resolve checkpoint files.
    if checkpoint_num is not None:
        policy_file = os.path.join(policy_dir, f"{experiment_name}_policy_{checkpoint_num}.pth")
        if not os.path.exists(policy_file):
            raise FileNotFoundError(policy_file)
    else:
        policy_files = glob.glob(os.path.join(policy_dir, f"{experiment_name}_policy_*.pth"))
        if not policy_files:
            raise FileNotFoundError(os.path.join(policy_dir, f"{experiment_name}_policy_*.pth"))
        policy_file = max(policy_files, key=lambda x: int(x.rsplit("_", 1)[-1].split(".")[0]))
    ckpt_num = os.path.basename(policy_file).rsplit("_", 1)[-1].split(".")[0]   # for video dir

    selector_file = os.path.join(stage_dir(path_prefix, policy_checkpoint, "selector"),
                                 f"{experiment_name}_selector.zip")
    if not os.path.exists(selector_file):
        if not selector_checkpoint:
            raise FileNotFoundError(f"No selector in {os.path.dirname(selector_file)} "
                                    f"and --selector_checkpoint not given")
        selector_file = os.path.join(stage_dir(path_prefix, selector_checkpoint, "selector"),
                                     f"{experiment_name}_selector.zip")
        if not os.path.exists(selector_file):
            raise FileNotFoundError(selector_file)

    print("Test Configuration (GPU sim):")
    print(f"  Environment: {env_id}")
    print(f"  num_envs: {num_envs}   n_episodes: {n_episodes}")
    print(f"  buffer_size: {buffer_size}   total_slots: {total_slots}   open_loop_steps: {open_loop_steps}")
    print(f"  variation_seeds: [{seed}, {seed + n_episodes - 1}]")

    # Selector (SB3 PPO) — deterministic priorities from a direct CUDA forward.
    selector = PPO.load(selector_file, device=device)
    selector.policy.set_training_mode(False)
    sel_lo = float(selector.action_space.low.min())
    sel_hi = float(selector.action_space.high.max())
    print(f"Loaded selector from: {selector_file}")

    # Vectorized GPU-sim env.
    env_kwargs = dict(num_envs=num_envs, obs_mode="rgb", sim_backend="gpu")
    if record_video:
        env_kwargs["render_mode"] = "all"
    if problem_config.get("delta_time") is not None:
        env_kwargs["delta_time"] = int(problem_config["delta_time"])
    if problem_config.get("max_episode_steps") is not None:
        env_kwargs["max_episode_steps"] = int(problem_config["max_episode_steps"])
    print(f"Building env: gym.make({base_env_id(env_id)!r}, **{env_kwargs}) (loads sapien, ~20s)")
    env = gym.make(base_env_id(env_id), **env_kwargs)
    env = StateOnlyTensorToDictWrapper(env)
    action_dim = int(env.action_space.shape[-1])
    seq_len = problem_config.get("seq_len", env_steps(env_id))
    # Rotate* tasks carry the goal (target_angle) in the redundant last-qpos slot.
    is_rotate = env_id in ROTATE_ENV_IDS

    # Policy.
    policy_class = model_dict.get("mikasa_robo").get("policy")
    policy = policy_class(output_dim=action_dim, **model_config).to(device)
    policy.load_state_dict(torch.load(policy_file, map_location=device, weights_only=False)["model_state_dict"])
    policy.eval()
    print(f"Loaded policy from: {policy_file}")

    buffer = BatchedEvictingBuffer(num_envs, buffer_size, OBS_DIM, use_current_obs=True, device=device)

    n_batches = (n_episodes + num_envs - 1) // num_envs
    all_successes, all_rewards = [], []
    t_start = time.time()
    print(f"\nRunning {n_episodes} episodes in {n_batches} batch(es) of <= {num_envs}...")

    for batch_i in range(n_batches):
        start = batch_i * num_envs
        active_n = min(num_envs, n_episodes - start)
        seeds = [seed + start + i for i in range(active_n)]
        seeds += [seeds[-1]] * (num_envs - active_n)               # pad; padded results discarded

        buffer.reset()
        try:
            obs_dict, _ = env.reset(seed=seeds)
        except (TypeError, ValueError):
            obs_dict, _ = env.reset(seed=seeds[0])

        ep_success = torch.zeros(num_envs, dtype=torch.bool)
        ep_reward = torch.zeros(num_envs, dtype=torch.float32)
        done_for_metric = torch.zeros(num_envs, dtype=torch.bool)

        action_horizon = 0
        pred_target = None
        recorder = _EpisodeRecorder(active_n, total_slots,
                                    Path(video_dir) / policy_checkpoint / ckpt_num, base=start) \
            if record_video else None
        t_batch = time.time()

        for idx in range(seq_len):
            flat = flat_obs_batched(obs_dict, device)             # (N, 98329)

            # Rotate*: overwrite the redundant last-qpos slot with the live per-env
            # target_angle, matching the training-time injection (no-op otherwise).
            if is_rotate:
                flat[:, PROMPT_SLOT] = env.unwrapped.target_angle.to(device=device, dtype=flat.dtype).reshape(-1)

            with torch.inference_mode():
                actions, _, _ = selector.policy(flat, deterministic=True)
            # Match SB3 predict()'s post-forward action-space clip; unclipped priorities
            # would perturb the buffer's no-repeat check and bias keyframe selection.
            priorities = actions.reshape(-1).clamp(sel_lo, sel_hi).float()

            buffer.push(flat, priorities)
            buf_NSD = buffer.get_batched()
            policy_obs = build_policy_obs_batched(buf_NSD, total_slots)

            if action_horizon == 0:
                with torch.inference_mode():
                    pred_target = policy.get_action(policy_obs, deterministic=True)   # (N, horizon, act)
                action_horizon = open_loop_steps
            action_vec = pred_target[:, open_loop_steps - action_horizon, :]
            action_horizon -= 1

            if recorder is not None:
                recorder.capture_step(buf_NSD, action_vec, idx + 1)

            obs_dict, reward, terminated, truncated, info = env.step(action_vec)

            r = _to_cpu(reward, torch.float32, num_envs)
            term = _to_cpu(terminated, torch.bool, num_envs)
            trunc = _to_cpu(truncated, torch.bool, num_envs)
            succ = _to_cpu(info.get("success"), torch.bool, num_envs)

            if recorder is not None:
                recorder.capture_render(env.render(), succ | term | trunc)

            # Metric semantics match eval_policy.py, which breaks on first success and records
            # that step's reward (else the last step's). Per env, until it is "done":
            #   ep_reward -> latest live-step reward   ep_success -> latched OR of success
            still_live = ~done_for_metric
            ep_reward[still_live] = r[still_live]
            ep_success |= succ & still_live
            done_for_metric |= succ | term | trunc

            # GPU sim auto-resets terminated/truncated envs; wipe their buffers so the next
            # push starts from the fresh episode's observations.
            reset_mask = (term | trunc).to(device)
            if reset_mask.any():
                buffer.reset(env_idx=reset_mask)

        sr_b = ep_success[:active_n].float().mean().item()
        rw_b = ep_reward[:active_n].float().mean().item()
        dt = time.time() - t_batch
        print(f"  Batch {batch_i + 1}/{n_batches}: {active_n} eps in {dt:.1f}s "
              f"({dt / seq_len:.2f}s/step)  SR={sr_b:.2%}  r={rw_b:.3f}")

        all_successes.extend(ep_success[:active_n].tolist())
        all_rewards.extend(ep_reward[:active_n].tolist())

        if recorder is not None:
            recorder.write()
            print(f"  wrote {active_n} episode video(s) -> {recorder.out_dir}")

    env.close()

    n_succ = sum(all_successes)
    n_tot = len(all_successes)
    sr = n_succ / n_tot
    avg_rw = sum(all_rewards) / n_tot
    wall = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"Evaluation Results (GPU sim, num_envs={num_envs}):")
    print(f"  Success Rate: {sr:.2%} ({n_succ}/{n_tot})")
    print(f"  Average Reward: {avg_rw:.3f}")
    print(f"  Wall time: {wall:.1f}s ({wall / n_tot:.2f}s/episode amortized)")
    print(f"{'=' * 60}")

    if output_log:
        Path(output_log).parent.mkdir(parents=True, exist_ok=True)
        with open(output_log, "w") as f:
            f.write(f"Environment: {env_id}\n")
            f.write(f"Success Rate: {sr:.2%} ({n_succ}/{n_tot})\n")
            f.write(f"Average Reward: {avg_rw:.3f}\n")
            f.write(f"Wall time: {wall:.1f}s\n")
            f.write(f"per_episode_success: {all_successes}\n")
            f.write(f"per_episode_reward: {all_rewards}\n")

    return {"success_rate": sr, "avg_reward": avg_rw,
            "episode_successes": all_successes, "episode_rewards": all_rewards}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorized GPU-sim eval for Mikasa Robo policies")
    parser.add_argument("--policy_checkpoint", type=str, required=True,
                        help="Policy checkpoint run-folder name (contains policy/ and selector/)")
    parser.add_argument("--selector_checkpoint", type=str, default="",
                        help="Fallback selector run-folder if not found in the policy checkpoint")
    parser.add_argument("--checkpoint_num", type=int, default=None,
                        help="Policy checkpoint step to load (default: max)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base env seed; variation_seeds = [seed, seed + n_episodes)")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=25, help="parallel GPU-sim envs per batch")
    parser.add_argument("--output_log", type=str, default=None,
                        help="Optional path to write summary + per-episode metrics")
    parser.add_argument("--record_video", action="store_true",
                        help="Write env/buffer/action visualizations (use a low --num_envs to bound RAM)")
    parser.add_argument("--video_dir", type=str, default="rollout/mikasa_robo/videos",
                        help="Directory for recorded videos (<video_dir>/<ckpt>/<num>/)")
    parser.add_argument("--delta_time", type=int, default=None,
                        help="Override the env cue-to-action delay (default: the config's value)")
    args = parser.parse_args()

    eval_policy_gpu(
        policy_checkpoint=args.policy_checkpoint,
        selector_checkpoint=args.selector_checkpoint,
        checkpoint_num=args.checkpoint_num,
        seed=args.seed,
        n_episodes=args.n_episodes,
        num_envs=args.num_envs,
        output_log=args.output_log,
        record_video=args.record_video,
        video_dir=args.video_dir,
        delta_time=args.delta_time,
    )
