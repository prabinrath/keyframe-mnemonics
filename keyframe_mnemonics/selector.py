from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv
from models import model_dict
from common.callbacks import WandbCallback
from common.helpers import parse_experiment_name
import gymnasium as gym
import numpy as np
from copy import deepcopy
import torch
import os


class SelectorActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Selector():
    def __init__(self, vec_env, problem_type, device="cuda", logging=False, **kwargs):
        self.device = device
        self.problem_type = problem_type
        self.total_timesteps = kwargs.get("total_timesteps")
        self.logging = logging
        self.ppo_callback = WandbCallback() if self.logging else None
        
        selector_class = model_dict.get(problem_type).get("selector")
        self.model = PPO(
            SelectorActorCriticPolicy,
            vec_env,
            verbose=1,
            policy_kwargs=dict(
                features_extractor_class=selector_class,
                features_extractor_kwargs=kwargs.get("features_extractor_kwargs"),
                net_arch=kwargs.get("net_arch")
            ),
            device=device,
            ent_coef=kwargs.get("ent_coef", 0),
            n_steps=kwargs.get("n_steps", 2048),
            batch_size=kwargs.get("batch_size", 64),
            n_epochs=kwargs.get("n_epochs", 10),
            learning_rate=kwargs.get("learning_rate", 0.0003),
        )

    def train(self):
        if self.model.env.reward_scaling_tau is not None:
            print("Collecting proxy surprise ...")
            self.model.env.collect_surprise_coef(self.model)

        print("Training selector model ...")
        self.model.learn(self.total_timesteps, callback=self.ppo_callback)
    
    def load_checkpoint(self, checkpoint_path):
        """Load selector PPO model from checkpoint directory."""
        # Extract experiment_name from checkpoint_path (tolerates a trailing stage subfolder)
        experiment_name = parse_experiment_name(checkpoint_path)
        
        checkpoint_file = os.path.join(checkpoint_path, f"{experiment_name}_selector.zip")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Selector checkpoint not found: {checkpoint_file}")
        self.model = PPO.load(checkpoint_file, device=self.device)
        self.model.policy.set_training_mode(False)  # Set to evaluation mode
        print(f"Loaded selector model from {checkpoint_file}")
    
    def get_priority(self, observation, deterministic=True):
        return self.model.predict(observation, deterministic=deterministic)


class VectorizedSelectorEnv(VecEnv):
    """Batched vectorized environment for efficient parallel stepping."""
    
    def __init__(self, proxy, process, observation_shape, observation_type, n_envs, action_penalty, min_loss, reward_scaling_tau=None):
        self.proxy = proxy
        self.n_envs = n_envs
        self.processes = [deepcopy(process) for _ in range(n_envs)]
        self.sidx = 0
        self.num_variations = self.processes[0].problem.num_variations
        self.action_penalty = action_penalty
        self.min_loss = min_loss
        if reward_scaling_tau is not None and reward_scaling_tau <= 0:
            raise ValueError(f"reward_scaling_tau must be > 0 or None, got {reward_scaling_tau}")
        self.reward_scaling_tau = reward_scaling_tau
        self.render_mode = None
        
        action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        observation_space = gym.spaces.Box(
            low=self.processes[0].problem.min_val,
            high=self.processes[0].problem.max_val,
            shape=observation_shape,
            dtype=observation_type
        )
        super().__init__(n_envs, observation_space, action_space)
        
        self.counters = np.zeros(n_envs, dtype=int)
        self.trajectory_indices = np.zeros(n_envs, dtype=int)
        self.observations = [None] * n_envs
        self.ts = [None] * n_envs
        self._actions = None
        self.surprise_coef = None
        
        # Episode tracking for PPO logging
        self.episode_rewards = np.zeros(n_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(n_envs, dtype=int)

    def collect_surprise_coef(self, model):
        loss_surprise = [[] for _ in range(self.num_variations)]

        remaining = self.num_variations
        sidx = 0
        while remaining:
            used_pidx = 0
            trajectory_indices = []
            for i in range(self.n_envs):
                self.processes[i].reset(sidx)
                trajectory_indices.append(sidx)
                sidx += 1
                used_pidx += 1
                remaining -= 1
                if not remaining:
                    break

            batch_max_seq_len = max(
                process.problem.seq_len for process in self.processes[:used_pidx]
            )
            for step_idx in range(batch_max_seq_len):
                obs_batch = []
                target_batch = []
                active_processes = []
                for i, process in enumerate(self.processes[:used_pidx]):
                    if step_idx < process.problem.seq_len:
                        obs, t = process.get_obs(step_idx)
                        obs_batch.append(obs)
                        target_batch.append(process.get_target(t))
                        active_processes.append(i)

                batch_obs = torch.stack(obs_batch)
                priorities, _ = model.predict(batch_obs, deterministic=True)

                buffers = []
                for j, process_idx in enumerate(active_processes):
                    process = self.processes[process_idx]
                    process.set_action(obs_batch[j], priorities[j])
                    buffers.append(process.get_buffer())

                losses = self.proxy.compute_loss_for_reward(
                    torch.stack(buffers).to(self.proxy.device),
                    torch.stack(target_batch),
                )

                for j, process_idx in enumerate(active_processes):
                    traj_idx = trajectory_indices[process_idx]
                    loss_surprise[traj_idx].append(losses[j].item())

        max_seq_len = max(len(traj_losses) for traj_losses in loss_surprise)
        surprise_coef = np.full((self.num_variations, max_seq_len), 0.0, dtype=np.float32)
        for traj_idx, traj_losses in enumerate(loss_surprise):
            traj_len = len(traj_losses)
            traj_losses = np.asarray(traj_losses, dtype=np.float32)
            # calculate reward scaling coefficient using proxy surprise
            # softmax proxy loss over the trajectory and scale with traj_len
            shifted_losses = traj_losses - np.max(traj_losses)
            exp_scores = np.exp(shifted_losses / self.reward_scaling_tau)
            traj_scores = (exp_scores / exp_scores.sum()) * traj_len
            surprise_coef[traj_idx, :traj_len] = traj_scores

        self.surprise_coef = surprise_coef
    
    def reset(self):
        for i in range(self.n_envs):
            self.trajectory_indices[i] = self.sidx
            self.processes[i].reset(self.sidx)
            self.sidx = (self.sidx + 1) % self.num_variations
            self.counters[i] = 0
            self.observations[i], self.ts[i] = self.processes[i].get_obs(0)
        self.episode_rewards[:] = 0
        self.episode_lengths[:] = 0
        return np.stack([obs.numpy() for obs in self.observations])
    
    def step_async(self, actions):
        self._actions = actions
    
    def step_wait(self):
        buffers, targets = [], []
        for i in range(self.n_envs):
            targets.append(self.processes[i].get_target(self.ts[i]))  # get target for proxy
            self.processes[i].set_action(self.observations[i], self._actions[i])  # update the buffer based on selector action
            buffers.append(self.processes[i].get_buffer())
        
        # use proxy to predict target and compute loss
        batch_buffers = torch.stack(buffers).to(self.proxy.device)
        batch_targets = torch.stack(targets)
        losses = self.proxy.compute_loss_for_reward(batch_buffers, batch_targets)
        
        # get reward based on proxy performance
        # if selector is accurate then the buffer should help the proxy in reducing loss
        log_min_loss = np.log(self.min_loss)
        rewards = (torch.log(losses + self.min_loss) / log_min_loss).numpy() - \
                  self.action_penalty * self._actions.squeeze(-1)
        if self.reward_scaling_tau is not None and self.surprise_coef is not None:
            # scale rewards with proxy surprise to focus on confusing timesteps
            rewards = rewards * self.surprise_coef[self.trajectory_indices, self.counters]
        
        # update episode stats
        self.episode_rewards += rewards
        self.episode_lengths += 1
        
        # advance counters and check termination
        self.counters += 1
        dones = np.array([self.counters[i] >= self.processes[i].problem.seq_len 
                         for i in range(self.n_envs)])
        
        # get next obs or auto-reset if done
        infos = [{} for _ in range(self.n_envs)]
        for i in range(self.n_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self.observations[i].numpy()
                # Add episode info for PPO logging
                infos[i]["episode"] = {
                    "r": self.episode_rewards[i],
                    "l": self.episode_lengths[i],
                }
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0
                self.trajectory_indices[i] = self.sidx
                self.processes[i].reset(self.sidx)
                self.sidx = (self.sidx + 1) % self.num_variations
                self.counters[i] = 0
            self.observations[i], self.ts[i] = self.processes[i].get_obs(self.counters[i])
        
        obs = np.stack([obs.numpy() for obs in self.observations])
        return obs, rewards, dones, infos
    
    def close(self):
        pass
    
    def env_method(self, method_name, *args, indices=None, **kwargs):
        indices = self._get_indices(indices)
        return [getattr(self.processes[i], method_name)(*args, **kwargs) for i in indices]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(list(self._get_indices(indices)))

    def get_attr(self, attr_name, indices=None):
        indices = list(self._get_indices(indices))
        # Handle render_mode
        if attr_name == "render_mode":
            return [None] * len(indices)
        return [getattr(self.processes[i], attr_name) for i in indices]

    def set_attr(self, attr_name, value, indices=None):
        for i in self._get_indices(indices):
            setattr(self.processes[i], attr_name, value)
    
    def seed(self, seed=None):
        return [None] * self.n_envs
