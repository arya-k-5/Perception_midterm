"""
Is the metric map right? (local mazes only; needs data_gt/local-<seed>/ from record_gt.py)

    uv run --no-sync source/check_poses.py --seed 8 2>&1 | tee logs/check_poses_8.txt

Builds the map exactly as path_agent.py does, then compares every frame's estimated pose with the true pose
(after the best single rigid transform, since the map has its own origin). Prints position and heading error for
plain dead reckoning and for the optimised map, and writes results/poses_<seed>.json for plotting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def align_error(est: np.ndarray, truth: np.ndarray):
    mu_e, mu_t = est[:, :2].mean(0), truth[:, :2].mean(0)
    a, b = est[:, :2] - mu_e, truth[:, :2] - mu_t
    u, _, vt = np.linalg.svd(a.T @ b)
    r = u @ np.diag([1, np.sign(np.linalg.det(u @ vt))]) @ vt
    aligned = a @ r + mu_t
    err = np.hypot(*(aligned - truth[:, :2]).T)
    hd = (est[:, 2] + np.arctan2(r[0, 1], r[0, 0]) - truth[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return err, np.degrees(np.abs(hd)), aligned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--focal", type=float, default=92.0)
    ap.add_argument("--data", type=Path, default=None)
    args = ap.parse_args()
    data = args.data or Path("data_gt") / f"local-{args.seed}"

    from classical_scorer import ClassicalScorer
    from path_agent import PathAgent

    agent = PathAgent(data, ClassicalScorer(Path("cache") / f"sptm-{data.name}"), geo_min=22, seq_min=20)
    agent.focal = args.focal
    agent.build_map()
    truth = []
    for f in agent.frames:
        poses = json.loads((f.path.parent / "poses.json").read_text())
        truth.append(poses[int(f.path.stem)])
    truth = np.array(truth)
    drive = np.concatenate([[t] * (e - s) for t, (s, e) in enumerate(agent.bounds)])
    out = {}
    for name, P in (("dead reckoning", agent.odo), ("optimised map", agent.P)):
        err, hd, aligned = align_error(P, truth)
        print(f"{name:15s} position error: median {np.median(err):.2f} m, 90% {np.percentile(err, 90):.2f} m, "
              f"max {err.max():.2f} m | heading error: median {np.median(hd):.0f} deg, 90% {np.percentile(hd, 90):.0f} deg")
        for d in range(len(agent.bounds)):
            print(f"    drive {d}: median {np.median(err[drive == d]):.2f} m, max {err[drive == d].max():.2f} m")
        out[name] = dict(aligned=aligned.tolist(), err=err.tolist(), heading_err=hd.tolist())
    # does the planner's route stay near the true route? goal node truth vs map
    print(f"goal node {agent.goal if hasattr(agent, 'goal') else '(set at setup)'}")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/poses_{args.seed}.json").write_text(json.dumps(dict(truth=truth.tolist(), drive=drive.tolist(), **out)))
    print(f"wrote results/poses_{args.seed}.json")


if __name__ == "__main__":
    main()
