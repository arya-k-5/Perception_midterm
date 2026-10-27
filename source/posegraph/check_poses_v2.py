"""
Pose-graph check with a calibrated forward distance (local mazes only; needs data_gt/).

Same procedure as source/check_poses.py, but the forward distance per tick is a parameter.
The existing modules are imported, never edited. The constant is set at runtime on the
imported modules before the map is built.

    uv run --no-sync source/posegraph/check_poses_v2.py --seed 110 --forward 0.037
    uv run --no-sync source/posegraph/check_poses_v2.py --seed 110 --forward 0.030   # the old constant

Writes source/posegraph/out/poses_<seed>_<forward>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))

import metric_map  # noqa: E402
import path_agent  # noqa: E402
from check_poses import align_error  # noqa: E402
from classical_scorer import ClassicalScorer  # noqa: E402
from path_agent import PathAgent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=110)
    ap.add_argument("--forward", type=float, default=0.037, help="metres per tick, FORWARD")
    ap.add_argument("--focal", type=float, default=92.0)
    args = ap.parse_args()

    metric_map.TICK_M = args.forward  # MetricMap reads this when it is built
    path_agent.TICK_M = args.forward
    print(f"forward distance per tick: {args.forward} m (turns: {metric_map.TICK_DEG} deg, unchanged)")

    data = Path("data_gt") / f"local-{args.seed}"
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

    dest = HERE / "out"
    dest.mkdir(exist_ok=True)
    path = dest / f"poses_{args.seed}_{args.forward}.json"
    path.write_text(json.dumps(dict(truth=truth.tolist(), drive=drive.tolist(), forward=args.forward, **out)))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
