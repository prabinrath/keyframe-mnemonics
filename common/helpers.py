import torch
import random
import numpy as np
import os
import sys


class _Tee:
    """A minimal stream that forwards writes to several underlying streams."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def tee_stdout(log_path):
    """Mirror stdout and stderr to a (line-buffered) log file for the rest of the process.

    Called once per training stage after its checkpoint folder exists, so each
    stage's console output — including tracebacks written to stderr — is also
    captured in `<stage folder>/train.log`.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_file


def get_problem_type_and_conf_path(experiment_name, conf_dir="conf"):
    """Return problem_type and config path from either a top-level config or a conf subdirectory prefix."""
    top_level_config = os.path.join(conf_dir, f"{experiment_name}.yaml")
    if os.path.isfile(top_level_config):
        return experiment_name, top_level_config

    for d in os.listdir(conf_dir):
        if os.path.isdir(os.path.join(conf_dir, d)) and experiment_name.startswith(d + "_"):
            return d, os.path.join(conf_dir, d, f"{experiment_name}.yaml")
    raise ValueError(f"Cannot infer problem_type from experiment_name '{experiment_name}'")


# Per-stage subfolders inside a single training-run checkpoint folder.
STAGE_SUBDIRS = ("proxy", "selector", "policy")


def parse_experiment_name(checkpoint_folder):
    """Extract experiment name from a run folder by stripping trailing YYYYMMDD_HHMMSS.

    Accepts either a bare run folder ("tmaze_20260101_120000") or one with a
    trailing stage subfolder ("tmaze_20260101_120000/proxy"); the stage
    component is ignored.
    """
    folder = checkpoint_folder.rstrip("/")
    if os.path.basename(folder) in STAGE_SUBDIRS:
        folder = os.path.dirname(folder)
    parts = os.path.basename(folder).split('_')
    return '_'.join(parts[:-2])


def stage_dir(path_prefix, run_folder, stage):
    """Absolute path to a per-stage subfolder inside a training-run checkpoint folder.

    A single run produces one folder ``checkpoints/{experiment}_{tag}/`` with the
    ``proxy``, ``selector`` and ``policy`` stages each written to their own
    subfolder inside it.
    """
    return os.path.join(path_prefix, "checkpoints", run_folder, stage)


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
