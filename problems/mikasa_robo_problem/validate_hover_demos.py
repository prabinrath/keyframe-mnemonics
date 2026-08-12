"""Validate a hover dataset against the source demos it was replayed from.

Reads the generated artifact and checks: (1) cube colour->slot ordering matches the
source, a hard precondition since otherwise the actions reach for a cube that is no
longer there; (2) replay success, from the env's own per-step label, with "succeeded at
any step" semantics; (3) how closely the replay tracks the source up to first success.

Usage:
    python problems/mikasa_robo_problem/validate_hover_demos.py \
        --hover_dir datasets/mikasa_robo/RememberColor3Hover-v0 \
        --src_dir   datasets/mikasa_robo/RememberColor3-v0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from mikasa_robo_suite.memory_envs.remember_color import RememberColorBaseEnv
from problems.mikasa_robo_problem.mikasa_robo_problem import env_steps


ENV_ID = "RememberColor3-v0"
BATCH_SIZE = 250
TIME_OFFSET = RememberColorBaseEnv.TIME_OFFSET
DEFAULT_DELTA_TIME = RememberColorBaseEnv.DEFAULT_DELTA_TIME


def cube_order(rgb_overhead, reveal):
    """Colour -> slot ordering (left to right) at the reveal frame. Cubes are isolated by
    differencing against the prior frame; the saturation test needs mid channel < 70."""
    a = rgb_overhead[reveal].astype(int)
    b = rgb_overhead[reveal - 1].astype(int)
    changed = np.abs(a - b).sum(-1) > 90
    srt = np.sort(a, axis=-1)
    sat = (srt[..., 2] > 110) & (srt[..., 1] < 70)
    xs = {}
    for k in range(3):
        m = changed & sat & (a.argmax(-1) == k)
        if m.sum() < 4:
            return None
        xs[k] = float(np.nonzero(m)[1].mean())
    return tuple(sorted(xs, key=xs.get))


def source_files(src_dir, n_batches):
    """Successful source demos in generator order, so index i pairs with hover demo i."""
    out = []
    for i in range(n_batches * BATCH_SIZE):
        f = src_dir / f"train_data_{i}.npz"
        if int(np.load(f)["success"][-1]):
            out.append(f)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hover_dir", default="datasets/mikasa_robo/RememberColor3Hover-v0")
    p.add_argument("--src_dir", default="datasets/mikasa_robo/RememberColor3-v0")
    p.add_argument("--hover_steps", type=int, default=10)
    p.add_argument("--n_batches", type=int, default=4)
    p.add_argument("--track_atol", type=float, default=0.05,
                   help="max |qpos - source qpos| (rad) for a demo to count as tracked")
    p.add_argument("--min_success", type=float, default=0.90,
                   help="fail the run if replay success falls below this fraction")
    args = p.parse_args()

    src_dir, hov_dir = Path(args.src_dir), Path(args.hover_dir)
    t_orig = env_steps(ENV_ID)
    hover_at = TIME_OFFSET + DEFAULT_DELTA_TIME
    src_reveal = TIME_OFFSET + DEFAULT_DELTA_TIME
    hov_reveal = TIME_OFFSET + DEFAULT_DELTA_TIME + args.hover_steps

    hov_files = sorted(hov_dir.glob("train_data_*.npz"),
                       key=lambda f: int(f.stem.split("_")[-1]))
    print(f"[config] hover_steps={args.hover_steps} reveal: source@{src_reveal} hover@{hov_reveal}")
    print(f"[config] {hov_dir} ({len(hov_files)} demos)  vs  {src_dir}")

    print("[map] pairing hover demos with their source demos ...")
    src_files = source_files(src_dir, args.n_batches)
    if len(src_files) != len(hov_files):
        print(f"[err] {len(hov_files)} hover demos but {len(src_files)} successful sources")
        return 1

    n = len(hov_files)
    scene_bad, unread, n_hit, tracked = [], 0, 0, 0
    end_dev = []

    for i, (hf, sf) in enumerate(zip(hov_files, src_files)):
        h, s = np.load(hf), np.load(sf)

        # (1) scene: colour->slot ordering must match the source exactly
        ho = cube_order(h["rgb"][:, :, :, :3], hov_reveal)
        so = cube_order(s["rgb"][:, :, :, :3], src_reveal)
        if ho is None or so is None:
            unread += 1
        elif ho != so:
            scene_bad.append((hf.name, sf.name, so, ho))

        # (2) success: the env's own label, recorded during generation
        n_hit += int(h["success"].any())

        # (3) tracking: qpos vs source up to first success (the env terminates there);
        # comparing past it just measures drift after the episode would have ended.
        hq, sq = h["joints"][:, 7:16], s["joints"][:, 7:16]
        T = hq.shape[0]                       # read the horizon from the data, not assumed
        stop = int(np.argmax(h["success"])) if h["success"].any() else T - 1
        f = np.arange(stop + 1)
        f = f[~((f >= hover_at) & (f < hover_at + args.hover_steps))]   # hover has no source counterpart
        d = float(np.abs(hq[f] - sq[np.where(f < hover_at, f, f - args.hover_steps)]).max()) if len(f) else 0.0
        end_dev.append(d)
        tracked += int(d <= args.track_atol)

    end_dev = np.array(end_dev)
    print(f"\n[result] demos validated: {n}")
    print(f"[1 scene   ] colour/slot ordering matches source : {n - len(scene_bad) - unread:>5}"
          f"  ({(n - len(scene_bad) - unread) / n:.1%})")
    print(f"[1 scene   ] mismatches                          : {len(scene_bad):>5}")
    print(f"[1 scene   ] unreadable                          : {unread:>5}")
    print(f"[2 success ] replay SR (success at any step)     : {n_hit:>5}  ({n_hit / n:.1%})")
    print(f"[3 tracking] within {args.track_atol} rad of source up to success : {tracked:>5}  ({tracked / n:.1%})")
    print(f"[3 tracking] max |qpos-src| rad before success   : median {np.median(end_dev):.3e}"
          f"  max {end_dev.max():.3e}")
    if scene_bad:
        print("\n[scene mismatches] first up to 10 (hover, source, source_order, hover_order):")
        for row in scene_bad[:10]:
            print(f"   {row}")

    ok = not scene_bad and unread == 0 and (n_hit / n) >= args.min_success
    print(f"\n[{'ok' if ok else 'FAIL'}] "
          + (f"all {n} demos share the source cube layout and colour sequence; "
             f"replay SR {n_hit / n:.1%}."
             if ok else
             f"scene mismatches {len(scene_bad)}, unreadable {unread}, "
             f"replay SR {n_hit / n:.1%} (min {args.min_success:.0%})."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
