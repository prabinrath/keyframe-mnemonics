"""
Proxy training dataset (stage 1). Two sampling modes:
  rollout (synth/grid: collect a fixed dataset by replaying Process episodes,
  held in RAM or on an H5 cache via MemoryManager) and demos (robot: sample a
  random buffer per __getitem__ from an on-disk demonstration H5).
"""
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import h5py

from keyframe_mnemonics.memory_manager import MemoryManager


class ProxyDataset(Dataset):
    """
        Stage-1 dataset simulating an agent with a fixed insertion budget.

        Models a buffer where an agent has a strictly limited budget of buffer
        insertions across an episode. At each timestep t, the agent has
        chosen to either insert an observation into the buffer or reject it
        (each past observation is accepted independently with probability 0.5,
        capped at buffer_size). The proxy model learns to predict actions
        based on the current buffer contents. Random sampling covers all
        reachable buffer states given the insertion budget constraint.

        Two sampling modes:
        - ProxyDataset.from_rollouts: collect a fixed dataset of
          (buffer, action) pairs upfront by replaying Process episodes into a
          MemoryManager (synthetic/grid domains).
        - ProxyDataset.from_demos: sample a random buffer on the fly per
          __getitem__ from an on-disk demonstration H5 (robot domains).
    """

    def __init__(self, mode):
        assert mode in ("rollout", "demos")
        self.mode = mode

    # ------------------------------------------------------------------
    # rollout mode (fixed dataset collected via Process + MemoryManager)
    # ------------------------------------------------------------------

    @classmethod
    def from_rollouts(cls, process, rollout_multiplier=1, cache_path="",
                      h5_filename="proxy_data.h5"):
        """Collect (buffer, action) pairs by replaying Process episodes.

        cache_path="" keeps the collected dataset in RAM; a directory writes it to
        an H5 cache there and streams it during training (deleted afterwards).
        """
        dataset = cls(mode="rollout")
        dataset.memory_manager = MemoryManager(cache_path, batch_size=128,
                                               h5_filename=h5_filename)

        print("Collecting rollouts ...")
        num_variations = process.problem.num_variations
        sidx = 0
        dataset_len = num_variations * rollout_multiplier
        pbar = tqdm(total=dataset_len, desc="Rollouts")
        while dataset_len:
            process.reset(sidx)
            dataset_len -= 1
            pbar.update(1)
            sidx = (sidx + 1) % num_variations

            allobs_array, allaction_array = [], []
            for timestep in range(process.problem.seq_len):
                allobs_array.append(process.get_obs(timestep)[0].numpy())
                allaction_array.append(process.get_target(0).numpy())
            allobs_array, allaction_array = np.asarray(allobs_array), np.asarray(allaction_array)

            for timestep in range(process.problem.seq_len):
                obs_array = np.zeros((process.problem.buffer_size,) + process.problem.observation_shape,
                                     dtype=process.problem.observation_type)
                # Each past observation is accepted independently with probability 0.5
                # Then the queue keeps at most buffer_size accepted items
                num_positive = min(np.random.binomial(timestep + 1, 0.5), process.problem.buffer_size)
                if num_positive > 0:
                    obs_steps = np.random.choice(timestep + 1, size=num_positive, replace=False)
                    obs_steps.sort()
                    obs_array[:num_positive] = allobs_array[obs_steps]

                # Fetch actions starting from timestep
                output_data = allaction_array[timestep]

                # Flatten observations
                input_data = obs_array.reshape(-1)

                dataset.memory_manager.write(input_data.astype(np.float32),
                                             output_data.astype(np.float32))

        pbar.close()
        dataset.memory_manager.finalize()
        return dataset

    @property
    def use_h5_cache(self):
        """Whether the rollout-mode cache is HDF5-backed (worker-safe)."""
        return self.mode == "rollout" and self.memory_manager.use_h5

    def cleanup(self):
        """Release resources: deletes the rollout-mode HDF5 cache, closes the demos H5."""
        if self.mode == "rollout":
            self.memory_manager.cleanup()
        elif self.mode == "demos" and self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
            self.demo_group = None

    # ------------------------------------------------------------------
    # demos mode (random per-batch sampling from demonstration H5)
    # ------------------------------------------------------------------

    @classmethod
    def from_demos(cls, h5_path, buffer_size, action_horizon, padding="repeat"):
        """Sample random buffers on the fly from an on-disk demo H5."""
        dataset = cls(mode="demos")
        dataset.h5_path = h5_path
        dataset.buffer_size = buffer_size
        dataset.action_horizon = action_horizon
        dataset.padding = padding

        # Open h5 file with larger chunk cache for better performance
        dataset.h5_file = h5py.File(h5_path, 'r', rdcc_nbytes=1024**3, rdcc_nslots=10007)
        dataset.demo_group = dataset.h5_file["demo"]
        dataset.num_episodes = dataset.h5_file.attrs['num_trajectories']

        # Get observation shape and dtype for zero padding
        dataset.obs_shape = dataset.demo_group['0/observations'].shape[1:]
        dataset.obs_dtype = dataset.demo_group['0/observations'].dtype
        dataset.action_dtype = dataset.demo_group['0/actions'].dtype

        # Create list of (episode, timestep, seq_len) for all timesteps
        dataset.index_list = []
        for ep in range(dataset.num_episodes):
            seq_len = dataset.demo_group[f'{ep}/observations'].shape[0]
            max_step = seq_len if dataset.padding == "repeat" else seq_len - dataset.action_horizon + 1
            for step in range(max_step):
                dataset.index_list.append((ep, step, seq_len))
        return dataset

    def _get_demo_item(self, idx):
        episode, timestep, seq_len = self.index_list[idx]

        # Each past observation is accepted independently with probability 0.5
        # Then the queue keeps at most buffer_size accepted items
        obs_array = np.zeros((self.buffer_size,) + self.obs_shape, dtype=self.obs_dtype)
        num_positive = min(np.random.binomial(timestep + 1, 0.5), self.buffer_size)
        if num_positive > 0:
            obs_steps = np.random.choice(timestep + 1, size=num_positive, replace=False)
            obs_steps.sort()
            obs_array[:num_positive] = self.demo_group[f'{episode}/observations'][obs_steps]

        # Fetch actions starting from timestep
        end_idx = min(timestep + self.action_horizon, seq_len)
        actions = self.demo_group[f'{episode}/actions'][timestep:end_idx]

        # Pad actions by repeating last action if needed
        num_actions = len(actions)
        if num_actions < self.action_horizon:
            padded_actions = np.empty((self.action_horizon,) + actions.shape[1:], dtype=self.action_dtype)
            padded_actions[:num_actions] = actions
            padded_actions[num_actions:] = actions[-1]
            actions = padded_actions

        # Flatten observations
        input_data = obs_array.reshape(-1)

        return torch.from_numpy(input_data), torch.from_numpy(actions)

    # ------------------------------------------------------------------

    def __len__(self):
        if self.mode == "rollout":
            return len(self.memory_manager)
        return len(self.index_list)

    def __getitem__(self, idx):
        if self.mode == "rollout":
            return self.memory_manager.read(idx)
        return self._get_demo_item(idx)
