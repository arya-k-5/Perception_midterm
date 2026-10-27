"""
Measure the odometry constants from ground truth (local mazes only; needs data_gt/).

    uv run --no-sync source/posegraph/calibrate_odometry.py --seeds 7,8,9,10,11,110

For every recorded tick, compare the TRUE motion (data_gt poses.json: x, y, heading) with what the
map assumes: FORWARD = 0.03 m along the heading, LEFT/RIGHT = 3.44 deg. A systematic bias shows up
as a shifted median; random per-tick noise shows up as a wide spread (the simulator adds motion
noise, so the spread is what dead reckoning can never remove).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ASSUMED_TICK_M = 0.03
ASSUMED_TICK_DEG = 3.44


def summarise(name: str, values: np.ndarray, assumed: float) -> None:
    if len(values) == 0:
        print(f"{name}: no samples")
        return
    p10, med, p90 = np.percentile(values, [10, 50, 90])
    print(f"{name:<10} n={len(values):5d}  median {med:+.4f}  p10 {p10:+.4f}  p90 {p90:+.4f}  "
          f"(assumed {assumed:+.4f}, ratio {med / assumed:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,8,9,10,11,110")
    ap.add_argument("--root", type=Path, default=Path("data_gt"))
    args = ap.parse_args()

    fwd, left, right, back = [], [], [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for tdir in sorted((args.root / f"local-{seed}").glob("traj_*")):
            poses = np.array(json.loads((tdir / "poses.json").read_text()), float)
            recs = json.loads((tdir / "data_info.json").read_text())
            if np.abs(poses[:, 2]).max() > 7.0:  # heading stored in degrees, not radians
                poses[:, 2] = np.radians(poses[:, 2])
            for i in range(min(len(recs), len(poses) - 1)):
                acts = recs[i]["action"]
                if len(acts) != 1:
                    continue
                a = acts[0]
                h = poses[i, 2]
                dx, dy = poses[i + 1, 0] - poses[i, 0], poses[i + 1, 1] - poses[i, 1]
                along = dx * math.cos(h) + dy * math.sin(h)  # displacement along the heading
                dh = (poses[i + 1, 2] - poses[i, 2] + math.pi) % (2 * math.pi) - math.pi
                if a == "FORWARD":
                    fwd.append(along)
                elif a == "BACKWARD":
                    back.append(along)
                elif a == "LEFT":
                    left.append(math.degrees(dh))
                elif a == "RIGHT":
                    right.append(math.degrees(dh))

    print(f"seeds {args.seeds}: {len(fwd)} FORWARD, {len(back)} BACKWARD, {len(left)} LEFT, {len(right)} RIGHT ticks")
    summarise("FORWARD m", np.array(fwd), ASSUMED_TICK_M)
    summarise("BACKWARD m", np.array(back), -ASSUMED_TICK_M)
    summarise("LEFT deg", np.array(left), ASSUMED_TICK_DEG)
    summarise("RIGHT deg", np.array(right), -ASSUMED_TICK_DEG)


if __name__ == "__main__":
    main()
