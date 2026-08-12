"""
Orchestrator for the sequential training pipeline:

    1. proxy  -- train the proxy model on randomly sampled buffers
    2. selector  -- freeze the proxy, train the PPO selector against it
    3. policy    -- train the policy on the selector-curated buffers

The policy stage generates its own dataset by rolling out the frozen selector
(in RAM or an H5 for the flat backend, a LeRobot dataset on disk for the lerobot
backend).

All stages of a run share a single checkpoint folder
`checkpoints/{experiment}_{tag}/`, with the proxy, selector and policy stages
each writing to their own `proxy/`, `selector/` and `policy/` subfolder inside it.

Each stage is also runnable standalone (keyframe_mnemonics/train_proxy.py,
keyframe_mnemonics/train_selector.py, keyframe_mnemonics/train_policy.py); this
script chains them, passing the shared run folder between stages.

Usage:
    python -m keyframe_mnemonics.train --experiment_name tmaze
    python -m keyframe_mnemonics.train --experiment_name tmaze --stages selector,policy --proxy_checkpoint tmaze_<TAG>
    python -m keyframe_mnemonics.train --experiment_name tmaze --stages policy --selector_checkpoint tmaze_<TAG>
"""
import argparse
import sys

from common.helpers import get_problem_type_and_conf_path

STAGES = ["proxy", "selector", "policy"]


def main():
    parser = argparse.ArgumentParser(description='Keyframe-Mnemonics Training Pipeline')
    parser.add_argument('--experiment_name', type=str, default="ltmb_Hallway",
                        help='Name of the experiment (yaml config filename, encodes problem type as prefix)')
    parser.add_argument('--stages', type=str, default=",".join(STAGES),
                        help=f'Comma-separated subset of stages to run, in order: {",".join(STAGES)}')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for deterministic training')
    parser.add_argument('--proxy_checkpoint', type=str, default="",
                        help='Proxy checkpoint folder (required when starting at the selector stage)')
    parser.add_argument('--selector_checkpoint', type=str, default="",
                        help='Selector checkpoint folder (required when starting at the policy stage)')
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        parser.error(f"Unknown stage(s): {unknown}. Valid stages: {STAGES}")
    stages = [s for s in STAGES if s in stages]  # canonical order

    experiment_name = args.experiment_name
    problem_type, _ = get_problem_type_and_conf_path(experiment_name)
    print(f"Experiment: {experiment_name} (problem type: {problem_type})")
    print(f"Stages: {' -> '.join(stages)}")

    proxy_checkpoint = args.proxy_checkpoint
    selector_checkpoint = args.selector_checkpoint

    # Validate upstream checkpoint requirements upfront (before any stage runs)
    if "selector" in stages and "proxy" not in stages and not proxy_checkpoint:
        parser.error("The selector stage requires --proxy_checkpoint "
                     "(or include the proxy stage)")
    if "policy" in stages and "selector" not in stages and not selector_checkpoint:
        parser.error("The policy stage requires --selector_checkpoint "
                     "(or include the selector stage)")

    for stage in stages:
        # Each stage tees stdout to its own train.log; reset so orchestrator
        # output isn't captured into the previous stage's log.
        sys.stdout = sys.__stdout__
        print(f"\n{'='*30} Stage: {stage} {'='*30}\n")

        if stage == "proxy":
            from keyframe_mnemonics import train_proxy
            proxy_checkpoint = train_proxy.train(experiment_name, seed=args.seed)

        elif stage == "selector":
            if not proxy_checkpoint:
                parser.error("The selector stage requires --proxy_checkpoint "
                             "(or run the proxy stage first)")
            from keyframe_mnemonics import train_selector
            selector_checkpoint = train_selector.train(
                experiment_name, proxy_checkpoint, seed=args.seed)

        elif stage == "policy":
            if not selector_checkpoint:
                parser.error("The policy stage requires --selector_checkpoint "
                             "(or run the selector stage first)")
            from keyframe_mnemonics import train_policy
            train_policy.train(experiment_name, seed=args.seed,
                               selector_checkpoint=selector_checkpoint)

    print("\nPipeline finished.")
    run_folder = selector_checkpoint or proxy_checkpoint
    if run_folder:
        print(f"Run folder: checkpoints/{run_folder}")


if __name__ == '__main__':
    main()
