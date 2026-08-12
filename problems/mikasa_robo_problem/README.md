# MIKASA-Robo data preparation

Expert demos for the `mikasa_robo` proxy stage, from the
[MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo) project — one zip per
task (1000 episodes each) on
[avanturist/mikasa-robo](https://huggingface.co/datasets/avanturist/mikasa-robo).
Run from the repo root.

```bash
# Download a task's NPZ dataset (RememberColor3-v0 shown; see the repo/HF for all 32 tasks).
HF_HUB_ENABLE_HF_TRANSFER=1 hf download avanturist/mikasa-robo RememberColor3-v0.zip \
    --repo-type dataset --local-dir datasets/mikasa_robo
unzip -q datasets/mikasa_robo/RememberColor3-v0.zip -d datasets/mikasa_robo/

# Convert the NPZ trajectories into the proxy H5.
python problems/mikasa_robo_problem/make_h5_dataset.py \
    --folder_path datasets/mikasa_robo/RememberColor3-v0 --split train
```

Writes `{env_id}_train.h5` to `datasets/mikasa_robo/proxy_dataset/` (failed
trajectories filtered out by default; `--no_filter` keeps them). The extracted NPZ
files are also used by `generate_policy_dataset.py` for the stage-3 policy dataset.

## RememberColor3Hover

A dataset variant, not a registered env: `RememberColor3-v0` demos re-replayed with the
cue-to-action delay stretched from 5 to 15 steps and a 10-step hover spliced in after the
cue disappears. Episodes stay at the env's 60-step horizon. Requires the full 1000-demo
`RememberColor3-v0` download.

```bash
# Replay the source demos with the hover spliced in.
MUJOCO_GL=egl python problems/mikasa_robo_problem/generate_hover_demos.py \
    --hover_steps 10 \
    --src_dir datasets/mikasa_robo/RememberColor3-v0 \
    --output_dir datasets/mikasa_robo/RememberColor3Hover-v0

# Check the cube layout matches the source and the replay succeeds.
python problems/mikasa_robo_problem/validate_hover_demos.py \
    --hover_dir datasets/mikasa_robo/RememberColor3Hover-v0 \
    --src_dir datasets/mikasa_robo/RememberColor3-v0

# Convert to the proxy H5.
python problems/mikasa_robo_problem/make_h5_dataset.py \
    --folder_path datasets/mikasa_robo/RememberColor3Hover-v0 --split train
```

## H5 schema

Grouped per trajectory under `demo/{idx}/`:

| dataset | shape | dtype | notes |
|---|---|---|---|
| `demo/{idx}/observations` | `(T, 98329)` | float32 | overhead_rgb (128×128×3) + gripper_rgb (128×128×3) + tcp_pose (7) + qpos (9) + qvel (9), flattened; RGB in [0, 1] |
| `demo/{idx}/actions` | `(T, action_dim)` | float32 | |

attrs: `num_trajectories`, `env_id`, `split`, `observation_dim`, `action_dim`; per-trajectory `episode_length`, `success`.
