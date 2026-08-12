from problems.problem import Problem, Evaluator
from common.helpers import parse_experiment_name
from pathlib import Path
import numpy as np
import wandb
import random
import torch
import h5py
import gymnasium as gym
from mikasa_robo_suite.utils.wrappers import StateOnlyTensorToDictWrapper
from problems.mikasa_robo_problem.rotate_prompts import ROTATE_ENV_IDS, PROMPT_SLOT
import os


# A variant's env_id names its dataset (NPZ folder, proxy H5, LeRobot repo); the live
# env is the base task built with the variant's env kwargs.
ENV_ID_ALIASES = {"RememberColor3Hover-v0": "RememberColor3-v0"}


def base_env_id(env_id):
    """Resolve a dataset-variant env_id to the gym-registered task it instantiates."""
    return ENV_ID_ALIASES.get(env_id, env_id)


def env_steps(env_id):
    env_id = base_env_id(env_id)
    if env_id in ['ShellGamePush-v0', 'ShellGamePick-v0', 'ShellGameTouch-v0']:
        EPISODE_TIMEOUT = 90
    elif env_id in ['InterceptSlow-v0', 'InterceptMedium-v0', 'InterceptFast-v0', 
                    'InterceptGrabSlow-v0', 'InterceptGrabMedium-v0', 'InterceptGrabFast-v0']:
        EPISODE_TIMEOUT = 90
    elif env_id in ['RotateLenientPos-v0', 'RotateLenientPosNeg-v0',
                    'RotateStrictPos-v0', 'RotateStrictPosNeg-v0']:
        EPISODE_TIMEOUT = 90
    elif env_id in ['CameraShutdownPush-v0', 'CameraShutdownPick-v0']:
        EPISODE_TIMEOUT = 90
    elif env_id in ['TakeItBack-v0']:
        EPISODE_TIMEOUT = 180
    elif env_id in ['RememberColor3-v0', 'RememberColor5-v0', 'RememberColor9-v0']:
        EPISODE_TIMEOUT = 60
    elif env_id in ['RememberShape3-v0', 'RememberShape5-v0', 'RememberShape9-v0']:
        EPISODE_TIMEOUT = 60
    elif env_id in ['RememberShapeAndColor3x2-v0', 'RememberShapeAndColor3x3-v0', 'RememberShapeAndColor5x3-v0']:
        EPISODE_TIMEOUT = 60
    elif env_id in ['BunchOfColors3-v0', 'BunchOfColors5-v0', 'BunchOfColors7-v0']:
        EPISODE_TIMEOUT = 120
    elif env_id in ['SeqOfColors3-v0', 'SeqOfColors5-v0', 'SeqOfColors7-v0']:
        EPISODE_TIMEOUT = 120
    elif env_id in ['ChainOfColors3-v0', 'ChainOfColors5-v0', 'ChainOfColors7-v0']:
        EPISODE_TIMEOUT = 120
    else:
        raise ValueError(f"Unknown environment: {env_id}")
    
    return EPISODE_TIMEOUT


def apply_delta_time_override(problem_config, delta_time):
    """Retarget an eval problem config to a new cue-to-action delay.

    Holds the non-delay part of the episode fixed (memorize phase + action phase),
    so the timeout grows one-for-one with the delay. For RememberColor3Hover,
    trained at delta_time=15 in a 60-step episode, that is 45 steps.
    """
    train_delta_time = problem_config.get("delta_time")
    problem_config["delta_time"] = int(delta_time)
    if train_delta_time is None:
        print(f"[eval] delta_time={int(delta_time)}; timeout left at the env registry default")
        return
    action_window = env_steps(problem_config["env_id"]) - int(train_delta_time)
    problem_config["max_episode_steps"] = problem_config["seq_len"] = int(delta_time) + action_window
    print(f"[eval] delta_time={int(delta_time)} (trained at {train_delta_time}); "
          f"max_episode_steps -> {problem_config['seq_len']} (action_window={action_window})")


class MikasaRoboProblem(Problem):
    def __init__(self, 
                 env_id,
                 split,
                 horizon,
                 seed=None,  # seed for test split environment resets
                 render_mode=None,  # render mode for environment in maniskill
                 path_prefix="",
                 delta_time=None,  # env delay-phase length; None -> env default
                 max_episode_steps=None,  # gym.make timeout; None -> env registry default
                 **kwargs):
        self.env_id = env_id
        self.split = split
        self.horizon = int(horizon)
        self.seed = seed
        self.render_mode = render_mode
        self.min_val, self.max_val = 0, 2*np.pi
        self.path_prefix = path_prefix
        self.delta_time = delta_time
        self.max_episode_steps = max_episode_steps
        if max_episode_steps is not None:
            kwargs["seq_len"] = int(max_episode_steps)
        kwargs.setdefault("seq_len", env_steps(env_id))
        super().__init__(
            observation_shape=(128*128*3 + 128*128*3 + 7 + 9 + 9,),  # overhead_camera + gripper_camera + tcp_pose + qpos + qvel = 98329
            observation_type=float,
            **kwargs
        )
        
        # Get all data files based on split
        if split == "train":
            # Store H5 path for lazy loading
            self.dataset_dir = Path(os.path.join(self.path_prefix, "datasets/mikasa_robo/proxy_dataset"))
            if not self.dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
            
            self.h5_path = self.dataset_dir / f"{env_id}_train.h5"
            if not self.h5_path.exists():
                raise FileNotFoundError(f"H5 dataset not found: {self.h5_path}. Run make_h5_dataset.py first.")
            
            # Initialize lazy-loaded attributes
            self._h5_file = None
            self._demo_group = None
            
            # Open file temporarily to get num_trajectories, then close
            with h5py.File(self.h5_path, 'r') as h5f:
                num_trajectories = h5f.attrs['num_trajectories']
                if self.num_variations is None:
                    self.num_variations = num_trajectories
            
            # For train split, verify num_variations doesn't exceed available data
            if self.num_variations > num_trajectories:
                raise ValueError(f"num_variations ({self.num_variations}) exceeds available trajectories ({num_trajectories})")
            
        else:  # split == "test"
            # Environment will be created in reset()
            self.env = None
            self.current_obs = None
            
            # Generate seeds for test split now that we have num_variations
            if self.seed is not None:
                self.variation_seeds = [self.seed + i for i in range(self.num_variations)]
            else:
                self.variation_seeds = [None] * self.num_variations
        
        self.optimal_action = None
        self.reset()
    
    @property
    def h5_file(self):
        """Lazy load H5 file handle."""
        if self.split == "train" and self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file
    
    @property
    def demo_group(self):
        """Lazy load demo group from H5 file."""
        if self.split == "train" and self._demo_group is None:
            self._demo_group = self.h5_file['demo']
        return self._demo_group
    
    def _process_env_observation(self, obs_dict):
        """Process environment observation dictionary into flat array.
        
        Extracts and flattens in order:
        1. overhead_camera (rgb[:,:,:3]) - 128x128x3
        2. gripper_camera (rgb[:,:,3:]) - 128x128x3
        3. tcp_pose (joints[0:7]) - 7 dims (xyz + quaternion (x,y,z,qx,qy,qz,qw))
        4. qpos (joints[7:16]) - 9 dims (7 arm + 2 gripper positions)
        5. qvel (joints[16:25]) - 9 dims (7 arm + 2 gripper velocities)
        """
        # Process RGB
        rgb = obs_dict['rgb']
        
        # Split RGB into gripper and overhead cameras
        overhead_camera = rgb[:, :, :3]  # First camera (channels 0-2)
        gripper_camera = rgb[:, :, 3:]  # Second camera (channels 3-5)
        
        # Normalize to [0, 1] range
        overhead_camera = overhead_camera.astype(np.float32) / 255.0
        gripper_camera = gripper_camera.astype(np.float32) / 255.0
        
        # Process joints (state)
        joints = obs_dict['state']
        
        # Extract joint components
        tcp_pose = joints[0:7]  # TCP pose: xyz + quaternion
        qpos = joints[7:16]   # Joint positions: 7 arm + 2 gripper
        qvel = joints[16:25]  # Joint velocities: 7 arm + 2 gripper        
        
        # Flatten and concatenate in the specified order
        return np.concatenate([
            overhead_camera.flatten(),
            gripper_camera.flatten(),
            tcp_pose,
            qpos,
            qvel
        ])
    
    def step_env(self, idx):
        assert self.split=="test"
        if idx == 0:
            # Reset environment with current seed
            seed = self.variation_seeds[self.sidx]
            if seed is not None:
                current_obs, _ = self.env.reset(seed=seed)
            else:
                current_obs, _ = self.env.reset()
        else:
            # Step environment with the action from previous timestep
            assert self.optimal_action is not None, "optimal_action must be set by model before stepping"
            action_tensor = self.optimal_action[0].unsqueeze(0).clone()
            
            current_obs, reward, _, _, info = self.env.step(action_tensor)
            
            # Track rewards and success
            if torch.is_tensor(reward):
                reward = reward.cpu().numpy()
            self.episode_reward = float(reward[0]) if hasattr(reward, '__len__') else float(reward)
            
            # Check for success in info
            if 'success' in info:
                success = info['success']
                if torch.is_tensor(success):
                    success = success.cpu().numpy()
                self.episode_success = bool(success[0]) if hasattr(success, '__len__') else bool(success)
        
        # Extract observation components from nested structure
        base_camera_rgb = current_obs["sensor_data"]["base_camera"]["rgb"]  # overhead camera
        hand_camera_rgb = current_obs["sensor_data"]["hand_camera"]["rgb"]  # gripper camera
        qpos = current_obs["agent"]["qpos"]
        qvel = current_obs["agent"]["qvel"]
        tcp_pose = current_obs["extra"]["tcp_pose"]
        
        # Convert torch tensors to numpy if needed
        if torch.is_tensor(base_camera_rgb):
            base_camera_rgb = base_camera_rgb.cpu().numpy().squeeze()
        if torch.is_tensor(hand_camera_rgb):
            hand_camera_rgb = hand_camera_rgb.cpu().numpy().squeeze()
        if torch.is_tensor(qpos):
            qpos = qpos.cpu().numpy().squeeze()
        if torch.is_tensor(qvel):
            qvel = qvel.cpu().numpy().squeeze()
        if torch.is_tensor(tcp_pose):
            tcp_pose = tcp_pose.cpu().numpy().squeeze()
        
        # Concatenate RGB cameras along channel axis (base + hand)
        rgb = np.concatenate([base_camera_rgb, hand_camera_rgb], axis=-1)
        
        # Concatenate state components (tcp_pose + qpos + qvel)
        state = np.concatenate([tcp_pose, qpos, qvel], axis=-1)
        
        # Create obs_dict for processing
        obs_dict = {
            'rgb': rgb,
            'state': state
        }
        
        # Process observation
        self.current_obs = self._process_env_observation(obs_dict)
        self.optimal_action = None

    def get_sample(self, idx):
        """Get observation at timestep idx and update optimal action."""
        if self.split == "train":
            # Read directly from H5 file using efficient slicing with string path
            obs = self.demo_group[f'{self.sidx}/observations'][idx]
            
            # Load action horizon with efficient slicing
            end_idx = min(idx + self.horizon, self.seq_len)
            actions = self.demo_group[f'{self.sidx}/actions'][idx:end_idx]
            
            self.optimal_action = actions
            return obs
        else:  # test split
            self.step_env(idx)
            return self.current_obs
    
    def get_target(self, t):
        """Get the target action horizon."""
        if self.optimal_action is None:
            return None
        actions = self.optimal_action
        # Pad with last action if needed
        if len(actions) < self.horizon:
            padding = np.tile(actions[-1:], (self.horizon - len(actions), 1))
            actions = np.concatenate([actions, padding])
        return actions.astype(np.float32)
    
    def reset(self):
        """Reset to a new episode."""
        if self.split == "train":
            if not hasattr(self, 'sidx'):
                self.sidx = 0
            else:
                if self.randomize_reset:
                    self.sidx = random.randint(0, self.num_variations - 1)
                else:
                    self.sidx = (self.sidx + 1) % self.num_variations
        else:  # test split
            # Create environment on first reset
            if self.env is None:
                env_kwargs = dict(num_envs=1, obs_mode="rgb",
                                  render_mode=self.render_mode, sim_backend="cpu")
                if self.delta_time is not None:
                    env_kwargs["delta_time"] = int(self.delta_time)
                if self.max_episode_steps is not None:
                    env_kwargs["max_episode_steps"] = int(self.max_episode_steps)
                self.env = gym.make(base_env_id(self.env_id), **env_kwargs)
                self.env = StateOnlyTensorToDictWrapper(self.env)
                self.sidx = 0
            else:
                if self.randomize_reset:
                    self.sidx = random.randint(0, self.num_variations - 1)
                else:
                    self.sidx = (self.sidx + 1) % self.num_variations
            
            self.episode_reward = 0
            self.episode_success = False
            self.step_env(0)

    def reset_sidx(self, sidx):
        """Reset to a specific episode index."""
        if sidx >= self.num_variations:
            raise IndexError(f"Episode index {sidx} out of range (max: {self.num_variations - 1})")
        
        if self.split == "train":
            self.sidx = sidx
        else:  # test split
            self.sidx = sidx
            self.episode_reward = 0
            self.episode_success = False
            self.step_env(0)
    
    def close(self):
        """Close environment and H5 file."""
        if self.split == "train" and hasattr(self, '_h5_file') and self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._demo_group = None
            
        if self.split == "test" and self.env is not None:
            self.env.close()
            self.env = None


class MikasaEvaluator(Evaluator):
    def __init__(self, sr_threshold, rw_threshold, logging, evaluation_rollouts, path_prefix="", **kwargs):
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
        """Evaluate the selector and proxy on train and test processes."""
        print("\n------------ID Evaluation------------")
        train_res = self.evaluate_end_to_end(selector, proxy, train_process)
        train_max_err = train_res.get("max_err")
        print(f"Maximum action error: {train_max_err}")

        print("------------OOD Evaluation------------")
        test_res = self.evaluate_end_to_end(selector, proxy, test_process)
        test_avg_sr = test_res.get("avg_sr")
        test_avg_rw = test_res.get("avg_rw")
        print(f"success rate: {test_avg_sr}")
        print(f"reward: {test_avg_rw}")
        print("--------------------------------------\n")
        
        if self.logging:
            wandb.log({
                "eval/train_max_err": train_max_err,
                "eval/test_avg_sr": test_avg_sr,
                "eval/test_avg_rw": test_avg_rw,
                "eval_step": self.step
            })
            self.step += 1
        
        return test_avg_sr > self.sr_threshold

    def evaluate_end_to_end(self, selector, proxy, process):
        """
        Evaluate the policy by running it on the problem.
        For train split: uses offline dataset with expert actions.
        For test split: runs the policy on live gym environments.
        """
        process.problem.randomize_reset = True
        success_episode = []
        reward_episode = []
        action_errors = []

        # Rotate*: the live test env carries the real gripper in the last-qpos slot; overwrite
        # it with target_angle to match training. Train split reads the already-injected H5.
        inject_prompt = process.problem.env_id in ROTATE_ENV_IDS and process.problem.split == "test"

        for _ in range(self.evaluation_rollouts):
            process.reset()
            episode_action_error = []

            for step in range(process.problem.seq_len):
                smp, t = process.get_obs(step)
                if inject_prompt:
                    smp[PROMPT_SLOT] = float(process.problem.env.unwrapped.target_angle.reshape(-1)[0])
                expert_action = process.get_target(t)
                p, _ = selector.get_priority(smp)
                process.set_action(smp, p)
                pred_action = proxy.get_action(
                    process.get_buffer().unsqueeze(0).to(proxy.device)
                ).cpu().squeeze()
                
                if process.problem.split == "train":
                    # For train split, compare with expert actions
                    action_error = ((pred_action - expert_action) ** 2).mean()
                    episode_action_error.append(action_error)
                else:  # test split
                    # Update the optimal_action so next step uses it
                    process.problem.optimal_action = pred_action
                
                    if process.problem.episode_success:
                        break

            # Collect episode statistics
            if process.problem.split == "train":
                action_errors.append(sum(episode_action_error)/len(episode_action_error))
            else:  # test split
                # Get statistics from environment tracking
                success_episode.append(process.problem.episode_success)
                reward_episode.append(process.problem.episode_reward)
        
        if process.problem.split == "train":
            max_action_error = max(action_errors)
            return {
                "avg_sr": 0.0,
                "avg_rw": 0.0,
                "max_err": max_action_error
            }
        else:
            avg_sr = sum(success_episode) / len(success_episode)
            avg_rw = sum(reward_episode) / len(reward_episode)
            return {
                "avg_sr": avg_sr,
                "avg_rw": avg_rw,
                "max_err": 0.0
            }
    
class MikasaPolicyEvaluator(Evaluator):
    """Evaluator for policy training: wraps eval_policy rollout, logs to wandb, supports early stopping."""

    def __init__(self, sr_threshold, rw_threshold, logging, evaluation_rollouts, **kwargs):
        super().__init__(**kwargs)
        self.sr_threshold = sr_threshold
        self.rw_threshold = rw_threshold
        self.logging = logging
        self.evaluation_rollouts = evaluation_rollouts

        if self.logging:
            wandb.define_metric("eval_step")
            wandb.define_metric("eval/*", step_metric="eval_step")
            self.step = 1

    def evaluate(self, eval_fn, policy_checkpoint, checkpoint_num):
        """Run eval_fn and log SR/reward. Returns True if success criteria is met."""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
            results = ex.submit(eval_fn,
                policy_checkpoint=policy_checkpoint,
                checkpoint_num=str(checkpoint_num),
                episode_indices=list(range(self.evaluation_rollouts)),
                record_video=False,
            ).result()
        avg_sr = results['success_rate']
        avg_rw = results['avg_reward']

        print(f"Policy Eval — SR: {avg_sr:.3f}, Reward: {avg_rw:.3f}")

        if self.logging:
            wandb.log({
                "eval/policy_sr": avg_sr,
                "eval/policy_rw": avg_rw,
                "eval_step": self.step,
            })
            self.step += 1

        return avg_sr > self.sr_threshold
