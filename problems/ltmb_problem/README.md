# LTMB data generation

Expert demos for the `ltmb` proxy stage, written to
`datasets/ltmb/proxy_dataset/{env_id}_{train,test}.h5`. Run from the repo root.

```bash
# Hallway
python problems/ltmb_problem/generate_demos.py --env_id LTMB-Hallway-v0 \
    --num_demos 120000 --min_length 5 --max_length 10

# MiniGrid Memory
python problems/ltmb_problem/generate_minigrid_memory_demos.py \
    --env_id MiniGrid-MemoryS17Random-v0 --num_demos 6000
```

Validate: `python problems/ltmb_problem/validate_demos.py --file <path-to-h5>`

## H5 schema

| dataset | shape | dtype | notes |
|---|---|---|---|
| `observations` | `(N, 148)` | int32 | flattened 7×7×3 image + direction |
| `actions` | `(N,)` | int32 | |
| `rewards` | `(N,)` | float32 | |
| `episode_starts` | `(num_episodes,)` | int64 | start index into the flat arrays |
| `episode_seeds` | `(num_episodes,)` | int64 | seed to recreate the episode |
| `episode_lengths` | `(num_episodes,)` | int32 | env length/size per episode |

attrs: `env_id`, `num_episodes`.
