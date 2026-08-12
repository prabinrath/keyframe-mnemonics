"""
Stage 1: train the proxy model on randomly sampled memory buffers.

The dataset sourcing is dispatched on config `pipeline.proxy_dataset`:
  - "rollout": collect a fixed dataset upfront by replaying Process episodes
    (synthetic and grid domains).
  - "demos": sample random buffers per batch from an on-disk demonstration H5
    (robot domains).

Usage:
    python -m keyframe_mnemonics.train_proxy --experiment_name tmaze
"""
import torch
from torch.utils.data import DataLoader
from models import model_dict
from problems import problem_dict
from keyframe_mnemonics.buffers import Process
from keyframe_mnemonics.proxy import ProxyDataset
from common.helpers import set_seeds, parse_experiment_name, get_problem_type_and_conf_path, stage_dir, tee_stdout
from pathlib import Path
import argparse
from datetime import datetime
import yaml
import wandb
import signal
import glob
import os


def train(experiment_name, seed=None, resume_checkpoint=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if seed is not None:
        set_seeds(seed=seed)

    # Load config
    problem_type, conf_path = get_problem_type_and_conf_path(experiment_name)
    print(f"Loading config: {conf_path}")

    with open(conf_path) as f:
        config = yaml.safe_load(f)
    if resume_checkpoint:
        path_prefix = config.get("path_prefix", "")
        resume_path = stage_dir(path_prefix, resume_checkpoint, "proxy")
        conf_path = os.path.join(resume_path, "config.yaml")
        with open(conf_path) as cf:
            config = yaml.safe_load(cf)
        print(f"Reloaded config: {conf_path}")

    problem_config = config.get("problem")
    proxy_config = config.get("proxy")
    model_config = proxy_config.get("model_kwargs")
    dataset_mode = config.get("pipeline").get("proxy_dataset")
    path_prefix = config.get("path_prefix", "")

    # Create the run folder (one per training run) and the proxy stage subfolder
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"{experiment_name}_{tag}"
    checkpoint_path = Path(stage_dir(path_prefix, run_folder, "proxy"))
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
            name=f"{run_folder}_proxy",
            group=run_folder,
            job_type="proxy",
            config=config,
            tags=[experiment_name]
        )

    # Setup problem. The proxy never sees the current observation, 
    # so force use_current_obs=False regardless of the config value.
    problem_class = problem_dict.get(problem_type).get("problem")
    train_problem = problem_class(**{**problem_config, "use_current_obs": False})

    # Create dataset based on the pipeline dispatch
    if dataset_mode == "rollout":
        # cache_path "" -> collect in RAM; anything else -> H5 cache in the run
        # folder (deleted after training)
        cache_path = proxy_config.get("cache_path", "")
        if cache_path:
            cache_path = stage_dir(path_prefix, run_folder, "proxy_cache")
        dataset = ProxyDataset.from_rollouts(
            Process(train_problem),
            rollout_multiplier=proxy_config.get("rollout_multiplier", 1),
            cache_path=cache_path,
        )

        # Cleanup cache on SIGTERM/SIGINT
        def cleanup_handler(signum, frame):
            dataset.cleanup()
            raise SystemExit(1)

        signal.signal(signal.SIGTERM, cleanup_handler)
        signal.signal(signal.SIGINT, cleanup_handler)

        if dataset.use_h5_cache:
            dataloader = DataLoader(dataset, batch_size=proxy_config.get("batch_size"),
                                    shuffle=True, num_workers=8, pin_memory=False,
                                    persistent_workers=True)
        else:
            dataloader = DataLoader(dataset, batch_size=proxy_config.get("batch_size"),
                                    shuffle=True)
    elif dataset_mode == "demos":
        env_id = problem_config.get("env_id")
        h5_path = os.path.join(path_prefix, f"datasets/{problem_type}/proxy_dataset/{env_id}_train.h5")
        print(f"Loading dataset: {h5_path}")
        dataset = ProxyDataset.from_demos(
            h5_path,
            buffer_size=problem_config.get("buffer_size"),
            action_horizon=problem_config.get("horizon"),
            padding=proxy_config.get("padding", "repeat"),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=proxy_config.get("batch_size"),
            shuffle=True,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=True
        )
    else:
        raise ValueError(f"Unknown pipeline.proxy_dataset mode: {dataset_mode}")
    print(f"Dataset length: {len(dataset)}")

    try:
        # Get proxy model class from model dict
        proxy_class = model_dict.get(problem_type).get("proxy")
        if proxy_class is None:
            raise ValueError(f"No proxy model found for problem_type: {problem_type}")

        # Create model
        model = proxy_class(
            output_dim=train_problem.action_dim,
            device=device,
            **model_config
        ).to(device)

        # Optimizer
        lr = proxy_config.get("learning_rate")
        weight_decay = proxy_config.get("weight_decay", 0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Resume from checkpoint if provided
        if resume_checkpoint:
            # Extract experiment_name from the run folder
            resume_experiment_name = parse_experiment_name(resume_checkpoint)

            # Find the last checkpoint automatically
            pattern = os.path.join(resume_path, f"{resume_experiment_name}_proxy_*.pth")
            checkpoints = glob.glob(pattern)
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found matching {pattern}")
            checkpoint_file = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))

            checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded proxy model and optimizer from {checkpoint_file}")

        # Training loop
        model.train()
        print("Starting training...")

        epochs = proxy_config.get("training_epochs")
        save_interval = proxy_config.get("save_interval", 10)
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_inputs, batch_outputs in dataloader:
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)

                optimizer.zero_grad()
                loss = model.compute_loss(batch_inputs, batch_outputs)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")

            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch + 1,
                    'proxy_loss': avg_loss
                })

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_file = checkpoint_path / f"{experiment_name}_proxy_{epoch+1}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_file)
                print(f"Saved checkpoint: {checkpoint_file}")

        # Save final model only if not already saved at last interval
        if epochs % save_interval != 0:
            final_file = checkpoint_path / f"{experiment_name}_proxy_{epochs}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, final_file)
            print(f"Final model saved: {final_file}")
        print("Proxy training completed.")
    finally:
        dataset.cleanup()

    if wandb.run is not None:
        wandb.finish()

    return run_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Proxy Model")

    parser.add_argument("--experiment_name", type=str, default="tmaze",
                        help="Experiment name (yaml config filename, encodes problem type as prefix)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Resume training from checkpoint directory")

    args = parser.parse_args()
    train(args.experiment_name, seed=args.seed, resume_checkpoint=args.resume_checkpoint)
