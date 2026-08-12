# Real-robot data preparation

Converts a collected LeRobot dataset into the flat proxy H5 that stages 1 and 2
read. Run once per dataset, from the repo root.

```bash
python problems/real_robot_problem/make_h5_dataset.py \
    --dataset_root ../lerobot/datasets/remember_color_3 \
    --repo_id remember_color_3
```

Writes `{repo_id}_train.h5` to `datasets/real_robot/proxy_dataset/`. Only the
wrist camera is extracted, downsampled to 128x128 — the collected dataset keeps
its front camera, but nothing in this pipeline reads it. `--normalize mean_std`
standardizes the state block using the dataset's own statistics (default is raw).

Then run the pipeline as usual:

```bash
python -m keyframe_mnemonics.train --experiment_name real_robot_RememberColor3
```

Stage 3 writes a LeRobot dataset to `checkpoints/<run>/policy_dataset/`, which
doubles as a standalone dataset for fitting VLA baselines with `lerobot-train`.
Running `generate_policy_dataset.py` standalone still writes to
`datasets/real_robot/policy_dataset/` instead.

There is no evaluator: a real robot cannot be stepped programmatically, so the
configs omit the `evaluator` / `policy_evaluator` blocks and evaluation happens on
hardware through the LeRobot plugin.

## H5 schema

Grouped per trajectory under `demo/{idx}/`:

| dataset | shape | dtype | notes |
|---|---|---|---|
| `demo/{idx}/observations` | `(T, 49160)` | float32 | wrist_img (128×128×3) + joint state (8), flattened; image HWC in [0, 1] |
| `demo/{idx}/actions` | `(T, 8)` | float32 | joint1..joint7 + gripper |

attrs: `num_trajectories`, `env_id`, `split`, `observation_dim`, `action_dim`,
`state_dim`, `normalize`, `fps`, `task`, and `state_mean` / `state_std` when
normalized; per-trajectory `episode_length`, `source_episode_index`.

Episodes keep their true length — `RealRobotProblem` adopts each trajectory's
length as `seq_len` on reset.
