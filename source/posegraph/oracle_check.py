"""
Which part of the pose graph is wrong? (local mazes only; needs data_gt/)

Four maps on the same maze, each optimised with the same least-squares back end:
  0  baseline        image odometry + image loop headings      (what the agent builds today)
  A  true odometry   ground-truth motion between frames         + image loop headings
  B  image odometry  + ground-truth loop headings
  C  true odometry   + ground-truth loop headings                (ceiling for this optimiser)
  D  true turns, image forward distance + image loops
  E  image turns, true forward distance + image loops

Loops in metric_map.py constrain HEADING only (i, j, yaw). So B and C show whether the loop
headings are the limit; A shows whether the odometry is the limit. Errors are measured after one
rigid alignment to ground truth, as in check_poses.py.

    uv run --no-sync source/posegraph/oracle_check.py --seed 110 --forward 0.03
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import metric_map  # noqa: E402
import path_agent  # noqa: E402
from check_poses import align_error  # noqa: E402
from classical_scorer import ClassicalScorer  # noqa: E402
from geometry import yaw_deg  # noqa: E402
from metric_map import MetricMap  # noqa: E402
from path_agent import PathAgent  # noqa: E402

D2R = math.pi / 180


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def report(name: str, P: np.ndarray, truth: np.ndarray) -> None:
    err, hd, _ = align_error(P, truth)
    print(f"{name:<34} position median {np.median(err):.2f} m, 90% {np.percentile(err, 90):.2f} m | "
          f"heading median {np.median(hd):.0f} deg, 90% {np.percentile(hd, 90):.0f} deg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=110)
    ap.add_argument("--forward", type=float, default=0.03)
    ap.add_argument("--focal", type=float, default=92.0)
    args = ap.parse_args()
    metric_map.TICK_M = args.forward
    path_agent.TICK_M = args.forward

    data = Path("data_gt") / f"local-{args.seed}"
    agent = PathAgent(data, ClassicalScorer(Path("cache") / f"sptm-{data.name}"), geo_min=22, seq_min=20)
    agent.focal = args.focal
    agent.build_map()
    mm0 = agent.map

    # ground truth per frame (the frame's image number is its record number)
    truth = []
    for f in agent.frames:
        poses = json.loads((f.path.parent / "poses.json").read_text())
        truth.append(poses[int(f.path.stem)])
    truth = np.array(truth, float)
    if np.abs(truth[:, 2]).max() > 7.0:
        truth[:, 2] = np.radians(truth[:, 2])

    # ground-truth odometry for each odometry step k: i = mm0.oi[k] -> i + 1
    gt_lx = np.zeros(len(mm0.oi))
    gt_dth = np.zeros(len(mm0.oi))
    for k, i in enumerate(mm0.oi):
        h = truth[i, 2]
        dx, dy = truth[i + 1, 0] - truth[i, 0], truth[i + 1, 1] - truth[i, 1]
        gt_lx[k] = dx * math.cos(h) + dy * math.sin(h)
        gt_dth[k] = wrap(truth[i + 1, 2] - h)
    img_lx, img_dth = mm0.o_lx.copy(), mm0.o_dth.copy()

    # loop headings: image (as build_map does) and ground truth
    img_loops, gt_loops = [], []
    for i, j in agent.shortcuts:
        m = agent.matcher.match(agent._feat(i), agent._feat(j))
        if m.n >= 15:
            img_loops.append((i, j, yaw_deg(m, agent.focal) * D2R))
        gt_loops.append((i, j, wrap(truth[j, 2] - truth[i, 2])))

    def solve(lx, dth, loops):
        tpf = agent._ticks_per_frame()
        mm = MetricMap([f.action for f in agent.frames], agent.bounds, tpf)
        mm.o_lx, mm.o_dth = lx.copy(), dth.copy()
        mm.integrate()
        mm.init_align(loops)
        mm.optimise(loops, anchor_pos=agent.anchor_pos, odo_yaw_deg=1.0)
        return mm.P

    print(f"frames {len(agent.frames)}, loops: {len(img_loops)} image, {len(gt_loops)} ground-truth")
    report("0 baseline (image odo + image loops)", agent.P, truth)
    report("A true odometry + image loops", solve(gt_lx, gt_dth, img_loops), truth)
    report("B image odometry + true loops", solve(img_lx, img_dth, gt_loops), truth)
    report("C true odometry + true loops", solve(gt_lx, gt_dth, gt_loops), truth)
    report("D true turns + image forward + image loops", solve(img_lx, gt_dth, img_loops), truth)
    report("E image turns + true forward + image loops", solve(gt_lx, img_dth, img_loops), truth)


if __name__ == "__main__":
    main()
