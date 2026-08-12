"""
Shared (input, output) sample store for the proxy and policy stages.

`cache_path` is the single switch:
  - "" (empty)  -> samples held in RAM; nothing is written to disk.
  - a directory -> samples written to an HDF5 file there and streamed back.

Writing (proxy rollout collection / policy dataset generation): construct with a
`cache_path`, `write()` the pairs, then `finalize()`. The same object then serves
reads (`read(idx)`), streaming per-row for the H5 backend. To read a pre-existing
H5 without re-collecting, use `MemoryManager.open(h5_path)`.

The proxy stage calls `cleanup()` (deletes its transient H5); the policy stage
calls `close()` (keeps its H5 as a reusable artifact).
"""
import os
import h5py
import numpy as np
import torch


class MemoryManager:
    def __init__(self, cache_path="", batch_size=64, h5_filename="proxy_data.h5"):
        self.batch_size = batch_size
        self.use_h5 = cache_path != ""
        self.finalized = False
        self.h5_file = None

        if self.use_h5:
            self.h5_path = os.path.join(cache_path, h5_filename)
            os.makedirs(cache_path, exist_ok=True)
            self.count = 0
            self._in_buffer = []
            self._out_buffer = []
        else:
            self.inputs = []
            self.outputs = []

    # ------------------------------------------------------------------
    # Read an existing finalized H5 (streaming) without re-collecting
    # ------------------------------------------------------------------
    @classmethod
    def open(cls, h5_path):
        mm = cls.__new__(cls)
        mm.batch_size = None
        mm.use_h5 = True
        mm.finalized = True
        mm.h5_path = h5_path
        mm.h5_file = None
        with h5py.File(h5_path, 'r') as f:
            mm.count = int(f['inputs'].shape[0])
        return mm

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def _flush(self):
        if len(self._in_buffer) == 0:
            return
        new_size = self.count + len(self._in_buffer)
        self.h5_file['inputs'].resize((new_size,) + self.h5_file['inputs'].shape[1:])
        self.h5_file['outputs'].resize((new_size,) + self.h5_file['outputs'].shape[1:])
        self.h5_file['inputs'][self.count:new_size] = np.stack(self._in_buffer)
        self.h5_file['outputs'][self.count:new_size] = np.stack(self._out_buffer)
        self.count = new_size
        self._in_buffer.clear()
        self._out_buffer.clear()

    def write(self, input_data, output_data):
        if self.use_h5:
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, 'w')
                input_shape = tuple(input_data.shape)
                output_shape = tuple(output_data.shape)
                self.h5_file.create_dataset(
                    'inputs', shape=(0,) + input_shape, maxshape=(None,) + input_shape,
                    dtype='float32', chunks=(self.batch_size,) + input_shape, compression=None)
                self.h5_file.create_dataset(
                    'outputs', shape=(0,) + output_shape, maxshape=(None,) + output_shape,
                    dtype='float32', chunks=(self.batch_size,) + output_shape, compression=None)

            if isinstance(input_data, torch.Tensor):
                input_data = input_data.cpu().numpy()
            if isinstance(output_data, torch.Tensor):
                output_data = output_data.cpu().numpy()

            self._in_buffer.append(input_data)
            self._out_buffer.append(output_data)
            if len(self._in_buffer) >= self.batch_size:
                self._flush()
        else:
            self.inputs.append(input_data)
            self.outputs.append(output_data)

    def finalize(self):
        if self.use_h5 and self.h5_file is not None:
            self._flush()
            self.h5_file.close()
            self.h5_file = None
        self.finalized = True

    def cleanup(self):
        """Close and delete the H5 file (proxy: its dataset is transient)."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
        if self.use_h5 and os.path.exists(self.h5_path):
            os.remove(self.h5_path)

    def close(self):
        """Release the streaming H5 handle, keeping the file (policy: it's an artifact)."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def __len__(self):
        return self.count if self.use_h5 else len(self.inputs)

    def read(self, idx):
        if self.use_h5:
            assert self.finalized, "Must call finalize() before reading from h5 file"
            # Lazy open per-worker - h5py handles are not fork-safe
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, 'r')
            return (torch.from_numpy(self.h5_file['inputs'][idx]),
                    torch.from_numpy(self.h5_file['outputs'][idx]))
        return torch.as_tensor(self.inputs[idx]), torch.as_tensor(self.outputs[idx])

    def outputs_array(self):
        """All outputs as a numpy array (used for class-balanced sampling)."""
        if self.use_h5:
            with h5py.File(self.h5_path, 'r') as f:
                return f['outputs'][:]
        if self.outputs and isinstance(self.outputs[0], torch.Tensor):
            return torch.stack(self.outputs).numpy()
        return np.asarray(self.outputs)
