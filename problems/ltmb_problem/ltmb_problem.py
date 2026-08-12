from problems.problem import Problem, Evaluator
from common.helpers import parse_experiment_name
import gymnasium as gym
from minigrid.core.actions import Actions
import ltmb  # register LTMB envs
import minigrid # register MiniGrid envs
from itertools import cycle
from pathlib import Path
import numpy as np
import wandb
import random
import h5py
import torch
import os


class LTMBProblem(Problem):
    def __init__(self,
                 env_list,
                 split,
                 horizon,
                 env_options=None,
                 path_prefix="",
                 **kwargs):
        self.env_list = env_list
        self.split = split
        self.env_id = None
        self.horizon = int(horizon)
        self.env_options = env_options or {}
        self.min_val, self.max_val = 0, 10
        self.path_prefix = path_prefix

        kwargs.setdefault("seq_len", 0)  # overridden after reset
        super().__init__(
            observation_shape=(7 * 7 * 3 + 1,),  # flattened image + direction
            observation_type=int,
            **kwargs
        )

        for env_id in env_list:
            demo_file_path = Path(os.path.join(self.path_prefix, "datasets/ltmb/proxy_dataset")) / f"{env_id}_{split}.h5"
            if not demo_file_path.exists():
                raise FileNotFoundError(f"Demo file not found: {demo_file_path}")
            with h5py.File(demo_file_path, 'r') as f:
                assert env_id == f.attrs.get("env_id"), \
                    f"env_id mismatch: expected {env_id}, got {f.attrs.get('env_id')}"
                assert self.num_variations <= int(f.attrs['num_episodes']), \
                    f"num_variations ({self.num_variations}) > total_episodes ({f.attrs['num_episodes']})"

        self.sample = None
        self.optimal_action = None
        self.current_reward = 0.0
        self.env_gen = cycle(self.env_list)
        self.reset()

    def _load_current_episode(self):
        demo_file_path = Path(os.path.join(self.path_prefix, "datasets/ltmb/proxy_dataset")) / f"{self.env_id}_{self.split}.h5"
        with h5py.File(demo_file_path, 'r') as f:
            total_episodes = int(f.attrs['num_episodes'])
            total_steps = f['actions'].shape[0]
            start_idx, end_idx = f['episode_starts'][self.sidx:self.sidx + 2] \
                if self.sidx < total_episodes - 1 \
                else (f['episode_starts'][self.sidx], total_steps)
            return {
                'observations': f['observations'][start_idx:end_idx],
                'actions': f['actions'][start_idx:end_idx],
                'rewards': f['rewards'][start_idx:end_idx],
                'seed': f['episode_seeds'][self.sidx],
                'length': f['episode_lengths'][self.sidx]
            }

    def get_sample(self, idx):
        obs = self.sample['observations'][idx]
        self.optimal_action = self.sample['actions'][idx:idx + self.horizon]
        self.current_reward = float(self.sample['rewards'][idx])
        return obs

    def get_reward(self):
        return self.current_reward

    def get_target(self, t):
        actions = self.optimal_action
        if len(actions) < self.horizon:
            padding = np.full((self.horizon - len(actions),), Actions.done)
            actions = np.concatenate([actions, padding])
        return actions

    def reset(self):
        if self.sample is None:
            self.sidx = 0
            self.env_id = next(self.env_gen)
        else:
            if self.randomize_reset:
                self.sidx = random.randint(0, self.num_variations - 1)
                self.env_id = next(self.env_gen)
            else:
                if self.sidx == self.num_variations - 1:
                    self.env_id = next(self.env_gen)
                self.sidx = (self.sidx + 1) % self.num_variations

        self.sample = self._load_current_episode()
        self.seq_len = len(self.sample['observations'])

    def reset_sidx(self, sidx):
        self.sidx = sidx
        self.env_id = next(self.env_gen)
        self.sample = self._load_current_episode()
        self.seq_len = len(self.sample['observations'])

    def get_env_options(self):
        env_options = dict(self.env_options)

        if self.env_id and self.sample is not None:
            if self.env_id.startswith("LTMB-") and "length" not in env_options:
                env_options["length"] = int(self.sample.get("length"))
            elif self.env_id.startswith("MiniGrid-Memory") and "size" not in env_options:
                env_options["size"] = int(self.sample.get("length"))

        return env_options

    def get_rollout_step_limit(self, default_limit):
        step_limit = int(default_limit)

        if self.env_id:
            if self.env_id.startswith("MiniGrid-Memory"):
                env_options = self.get_env_options()
                size = int(env_options.get("size", (self.sample.get("length") if self.sample is not None else 0)))
                step_limit = max(step_limit, 2 * size + 10)
            elif self.env_id.startswith("LTMB-"):
                env_options = self.get_env_options()
                length = int(env_options.get("length", (self.sample.get("length") if self.sample is not None else 0)))
                step_limit = max(step_limit, 4 * length + 10)

        return step_limit


class LTMBEvaluator(Evaluator):
    def __init__(self, sr_threshold, rw_threshold, logging, evaluation_rollouts,
                 path_prefix="", **kwargs):
        super().__init__(**kwargs)
        self.logging = logging
        self.sr_threshold = sr_threshold
        self.rw_threshold = rw_threshold
        self.evaluation_rollouts = evaluation_rollouts

        self.checkpoint_path = os.path.join(path_prefix, f"checkpoints/{self.checkpoint_folder}")
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        self.experiment_name = parse_experiment_name(self.checkpoint_folder)

        if self.logging:
            wandb.define_metric("eval_step")
            wandb.define_metric("eval/*", step_metric="eval_step")
            self.step = 1

    def evaluate(self, selector, proxy, train_process, test_process):
        print("\n------------ID Evaluation------------")
        train_avg_sr, train_avg_rw = self.evaluate_end_to_end(selector, proxy, train_process)
        print(f"success rate: {train_avg_sr:.3f}")
        print(f"reward: {train_avg_rw:.3f}")
        print("------------OOD Evaluation------------")
        test_avg_sr, test_avg_rw = self.evaluate_end_to_end(selector, proxy, test_process)
        print(f"success rate: {test_avg_sr:.3f}")
        print(f"reward: {test_avg_rw:.3f}")
        print("--------------------------------------\n")

        if self.logging:
            wandb.log({
                "eval/train_avg_sr": train_avg_sr,
                "eval/train_avg_rw": train_avg_rw,
                "eval/test_avg_sr": test_avg_sr,
                "eval/test_avg_rw": test_avg_rw,
                "eval_step": self.step
            })
            self.step += 1

        return test_avg_sr > self.sr_threshold and test_avg_rw > self.rw_threshold

    def evaluate_end_to_end(self, selector, proxy, process):
        process.problem.randomize_reset = True
        success_episode = []
        reward_episode = []

        for _ in range(len(process.problem.env_list) * self.evaluation_rollouts):
            process.reset()
            env_options = process.problem.get_env_options()
            env = gym.make(process.problem.env_id, **env_options)
            obs, _ = env.reset(seed=int(process.problem.sample.get("seed")))

            info = {}
            reward = 0.0
            max_steps = process.problem.get_rollout_step_limit(default_limit=200)
            for step in range(max_steps):
                smp = torch.as_tensor(np.concatenate((
                    obs['image'].flatten(),
                    np.array([obs['direction']], dtype=np.int32),
                )))
                p, _ = selector.get_priority(smp)
                process.set_action(smp, p)
                pred_target = proxy.get_action(
                    process.get_buffer().unsqueeze(0).to(proxy.device)
                ).cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(pred_target[:, 0])
                if terminated or truncated:
                    break

            success_episode.append(bool(info.get('success', reward > 0)))
            reward_episode.append(reward)
            env.close()

        avg_sr = sum(success_episode) / len(success_episode)
        avg_rw = sum(reward_episode) / len(reward_episode)
        return (avg_sr, avg_rw)


class LTMBPolicyEvaluator(Evaluator):
    """Policy-stage evaluator: runs selector+policy eval via the rollout registry
    in an isolated subprocess, logs the success rate to wandb, and signals early
    stopping. Validation is in-distribution (test split at natural lengths); the OOD
    length-100 result is reserved for final reporting."""

    def __init__(self, sr_threshold, logging, evaluation_rollouts, **kwargs):
        super().__init__(**kwargs)
        self.sr_threshold = sr_threshold
        self.logging = logging
        self.evaluation_rollouts = evaluation_rollouts
        if self.logging:
            wandb.define_metric("eval_step")
            wandb.define_metric("eval/*", step_metric="eval_step")
            self.step = 1

    def evaluate(self, eval_fn, policy_checkpoint, checkpoint_num):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
            result = ex.submit(
                eval_fn,
                policy_checkpoint=policy_checkpoint,
                checkpoint_num=str(checkpoint_num),
                episode_indices=list(range(self.evaluation_rollouts)),
                record_video=False,
                override_env_options=False,  # validate ID (test split, natural lengths), not the OOD report length
            ).result()
        sr = result['success_rate']
        rw = result['avg_reward']
        print(f"Policy Eval — ID SR: {sr:.3f}, Reward: {rw:.3f}")
        if self.logging:
            wandb.log({
                "eval/policy_sr": sr,
                "eval/policy_rw": rw,
                "eval_step": self.step,
            })
            self.step += 1
        return sr > self.sr_threshold
