"""
Stage 2: train the PPO selector against a frozen proxy model.

For rollout-mode domains (see config `pipeline.proxy_dataset`) an
end-to-end selector+proxy evaluation (ID/OOD) runs after training.

Usage:
    python -m keyframe_mnemonics.train_selector --experiment_name tmaze --proxy_checkpoint tmaze_<TAG>
"""

import os
_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))
os.environ["OMP_NUM_THREADS"] = str(_NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(_NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(_NUM_THREADS)

import torch
torch.set_num_threads(_NUM_THREADS)
torch.set_num_interop_threads(_NUM_THREADS)

import argparse
from pathlib import Path
import yaml
import wandb
import glob
from copy import deepcopy

from keyframe_mnemonics.buffers import Process
from problems import problem_dict
from models import model_dict
from keyframe_mnemonics.selector import Selector, VectorizedSelectorEnv
from common.helpers import set_seeds, parse_experiment_name, get_problem_type_and_conf_path, stage_dir, tee_stdout


def train(experiment_name, proxy_checkpoint, seed=None, resume_checkpoint=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if seed is not None:
        set_seeds(seed=seed)

    # Load config
    problem_type, conf_path = get_problem_type_and_conf_path(experiment_name)
    with open(conf_path) as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from: {conf_path}")
    path_prefix = config.get("path_prefix", "")

    # Verify proxy config matches the checkpoint config (excluding path_prefix)
    proxy_checkpoint_path = stage_dir(path_prefix, proxy_checkpoint, "proxy")
    checkpoint_config_path = os.path.join(proxy_checkpoint_path, "config.yaml")
    with open(checkpoint_config_path) as f:
        checkpoint_config = yaml.safe_load(f)
    current_proxy_config = deepcopy(config["proxy"])
    checkpoint_proxy_config = deepcopy(checkpoint_config["proxy"])
    current_proxy_config.pop("path_prefix", None)
    checkpoint_proxy_config.pop("path_prefix", None)
    assert checkpoint_proxy_config == current_proxy_config, \
        "proxy config mismatch between current config and checkpoint config (excluding path_prefix)"

    problem_config = config.get("problem")
    selector_config = config.get("selector")
    selector_env_config = config.get("selector_env")
    proxy_config = config.get("proxy")

    # The selector writes into the selector subfolder of the same run folder
    # that already holds the frozen proxy stage.
    run_folder = proxy_checkpoint
    checkpoint_folder = run_folder
    checkpoint_path = Path(stage_dir(path_prefix, run_folder, "selector"))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    tee_stdout(checkpoint_path / "train.log")
    print(f"Checkpoint directory: {checkpoint_path}")

    # Save config to checkpoint folder
    config_save_path = checkpoint_path / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}")

    # Initialize wandb if logging is enabled
    if config.get("logging", False):
        wandb.init(
            project="keyframe_mnemonics",
            name=f"{run_folder}_selector",
            group=run_folder,
            job_type="selector",
            config=config,
            tags=[experiment_name]
        )

    # Setup problem
    problem_class = problem_dict.get(problem_type).get("problem")
    train_problem = problem_class(**problem_config)
    train_process = Process(train_problem)

    # Get proxy model class from model dict
    proxy_class = model_dict.get(problem_type).get("proxy")
    if proxy_class is None:
        raise ValueError(f"No proxy model found for problem_type: {problem_type}")

    # Create proxy model
    model_config = proxy_config.get("model_kwargs")
    print("Initializing proxy model...")
    proxy_model = proxy_class(
        output_dim=train_problem.action_dim,
        device=device,
        **model_config
    ).to(device)

    # Load pretrained proxy checkpoint (latest by number)
    print(f"Loading proxy from: {proxy_checkpoint_path}")
    checkpoint_experiment_name = parse_experiment_name(proxy_checkpoint)

    pattern = os.path.join(proxy_checkpoint_path, f"{checkpoint_experiment_name}_proxy_*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found matching {pattern}")
    checkpoint_file = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    proxy_model.load_state_dict(checkpoint['model_state_dict'])
    proxy_model.eval()  # Set to evaluation mode
    print(f"Proxy loaded from {checkpoint_file}")

    # Initialize Selector Environment
    vec_env = VectorizedSelectorEnv(
        proxy_model,
        deepcopy(train_process),
        observation_shape=train_problem.observation_shape,
        observation_type=train_problem.observation_type,
        **selector_env_config
    )

    # Initialize Selector
    selector = Selector(
        vec_env,
        problem_type=problem_type,
        device=device,
        **selector_config
    )

    # Resume selector training from checkpoint if provided
    if resume_checkpoint:
        resume_path = stage_dir(path_prefix, resume_checkpoint, "selector")
        print(f"Resuming selector training from: {resume_path}")
        selector.load_checkpoint(resume_path)
        selector.model.set_env(vec_env)

    # Train selector
    selector.train()

    # Save final checkpoints (both selector and proxy for a self-contained folder).
    # The selector uses a fixed name (there is only ever one); the frozen proxy copy
    # keeps the epoch number of the source proxy it was loaded from.
    proxy_epoch = os.path.basename(checkpoint_file).split('_')[-1].split('.')[0]
    selector_checkpoint_file = checkpoint_path / f"{experiment_name}_selector.zip"
    proxy_checkpoint_file = checkpoint_path / f"{experiment_name}_proxy_{proxy_epoch}.pth"

    selector.model.save(selector_checkpoint_file)
    torch.save({
        'model_state_dict': proxy_model.state_dict()
    }, proxy_checkpoint_file)

    print("Selector training completed.")
    print(f"Selector checkpoint saved: {selector_checkpoint_file}")
    print(f"Proxy checkpoint saved: {proxy_checkpoint_file}")

    # End-to-end selector+proxy evaluation (ID/OOD) for rollout-mode domains
    if config.get("evaluator"):
        test_problem_config = deepcopy(problem_config)
        test_problem_config.update(config.get("test_problem_override", {}))
        test_problem = problem_class(**test_problem_config)
        test_process = Process(test_problem)

        evaluator = problem_dict.get(problem_type).get("evaluator")(
            checkpoint_folder=checkpoint_folder,
            **config.get("evaluator")
        )
        evaluator.evaluate(selector, proxy_model, train_process, test_process)

    # Close environment
    vec_env.close()

    if wandb.run is not None:
        wandb.finish()

    return checkpoint_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Selector with Pretrained Proxy")

    parser.add_argument("--experiment_name", type=str, default="tmaze",
                        help="Experiment name (yaml config filename, encodes problem type as prefix)")
    parser.add_argument("--proxy_checkpoint", type=str, required=True,
                        help="Pretrained proxy checkpoint folder name")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Resume selector training from checkpoint folder name (optional)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")

    args = parser.parse_args()
    train(args.experiment_name, args.proxy_checkpoint,
          seed=args.seed, resume_checkpoint=args.resume_checkpoint)
