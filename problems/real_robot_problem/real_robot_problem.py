from problems.problem import Problem
from pathlib import Path
import numpy as np
import random
import h5py
import os


class RealRobotProblem(Problem):
    """Replays teleoperated demonstrations from the flat proxy H5.

    There is no test split: a real robot cannot be stepped programmatically, so
    the configs omit the evaluator blocks and evaluation happens on hardware
    through the LeRobot plugin.

    Teleop episodes are variable length, so seq_len is a per-episode value
    refreshed on every reset rather than a constant.
    """

    def __init__(self,
                 env_id,
                 horizon,
                 split="train",
                 path_prefix="",
                 **kwargs):
        self.env_id = env_id
        self.split = split
        self.horizon = int(horizon)
        self.path_prefix = path_prefix
        # Bounds of the selector's observation space: images are [0, 1] and joint
        # values are radians, so +-2pi covers both (and standardized state too).
        self.min_val, self.max_val = -2 * np.pi, 2 * np.pi

        self.dataset_dir = Path(os.path.join(self.path_prefix, "datasets/real_robot/proxy_dataset"))
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.h5_path = self.dataset_dir / f"{env_id}_{split}.h5"
        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"H5 dataset not found: {self.h5_path}. Run make_h5_dataset.py first."
            )

        self._h5_file = None
        self._demo_group = None

        with h5py.File(self.h5_path, 'r') as h5f:
            num_trajectories = int(h5f.attrs['num_trajectories'])
            observation_dim = int(h5f.attrs['observation_dim'])
            action_dim = int(h5f.attrs['action_dim'])
            self.episode_lengths = [
                int(h5f['demo'][str(i)]['observations'].shape[0]) for i in range(num_trajectories)
            ]
            self.state_mean = h5f.attrs['state_mean'] if 'state_mean' in h5f.attrs else None
            self.state_std = h5f.attrs['state_std'] if 'state_std' in h5f.attrs else None

        if kwargs.get("num_variations") is None:
            kwargs["num_variations"] = num_trajectories
        if kwargs["num_variations"] > num_trajectories:
            raise ValueError(
                f"num_variations ({kwargs['num_variations']}) exceeds available "
                f"trajectories ({num_trajectories})"
            )

        # Seeded with the longest episode; reset() narrows it to the selected one.
        kwargs.setdefault("seq_len", max(self.episode_lengths))
        kwargs.setdefault("action_dim", action_dim)
        super().__init__(
            observation_shape=(observation_dim,),
            observation_type=float,
            **kwargs
        )

        self.optimal_action = None
        self.reset()

    @property
    def h5_file(self):
        """Lazy load H5 file handle."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    @property
    def demo_group(self):
        """Lazy load demo group from H5 file."""
        if self._demo_group is None:
            self._demo_group = self.h5_file['demo']
        return self._demo_group

    def _select(self, sidx):
        self.sidx = sidx
        self.seq_len = self.episode_lengths[sidx]

    def get_sample(self, idx):
        """Get observation at timestep idx and update optimal action."""
        obs = self.demo_group[f'{self.sidx}/observations'][idx]

        end_idx = min(idx + self.horizon, self.seq_len)
        actions = self.demo_group[f'{self.sidx}/actions'][idx:end_idx]

        self.optimal_action = actions
        return obs

    def get_target(self, t):
        """Get the target action horizon, padded with the last action if the episode ends early."""
        if self.optimal_action is None:
            return None
        actions = self.optimal_action
        if len(actions) < self.horizon:
            padding = np.tile(actions[-1:], (self.horizon - len(actions), 1))
            actions = np.concatenate([actions, padding])
        return actions.astype(np.float32)

    def reset(self):
        """Reset to a new episode."""
        if not hasattr(self, 'sidx') or self.sidx is None:
            self._select(0)
        elif self.randomize_reset:
            self._select(random.randint(0, self.num_variations - 1))
        else:
            self._select((self.sidx + 1) % self.num_variations)

    def reset_sidx(self, sidx):
        """Reset to a specific episode index."""
        if sidx >= self.num_variations:
            raise IndexError(f"Episode index {sidx} out of range (max: {self.num_variations - 1})")
        self._select(sidx)

    def close(self):
        """Close the H5 file."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._demo_group = None
