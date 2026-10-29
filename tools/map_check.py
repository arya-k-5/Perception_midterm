"""
Stitch the exploration frames into a metric map and check it against the truth.   NEW FILE.

    /Users/arya/vis_nav_player_work/.venv/bin/python tools/map_check.py --seed 8
    ... --truth-data /Users/arya/vis_nav_player_work/data_gt/local-8      (true poses, if recorded there)

1. every exploration frame gets an (x, y, heading): drives dead-reckoned from the recorded actions, with each
   turn REPLACED by the rotation SuperGlue + the essential matrix measure between consecutive frames
   (the heading drift of pure action counting is what ruined the earlier map);
2. the agent's verified shortcuts (SuperGlue + sequence check, 0 false in the team's checks) and the spawn joins
   are loop closures: "these two frames are the same place, rotated by yaw";
3. one least-squares pose graph over all frames (lecture 8, SLAM back end) puts every drive in one frame;
4. with true poses available it prints the map error and writes logs/map_<seed>.png + .json.
The point: if the map error is well below 0.2 m, navigating the map metrically is worth building. If it is
metres, it is not, and the photo graph stays the planner.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))  # the agent's modules live in source/


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--truth-data", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--r", type=Path, default=Path("models/sptm_r.pt"))
    ap.add_argument("--l", type=Path, default=Path("models/sptm_l.pt"))
    ap.add_argument("--min-inliers", type=int, default=25)
    args = ap.parse_args()
    data = args.data or Path("data") / f"local-{args.seed}"

    from anyloc import AnyLocScorer
    from metric_map_claude import TICK_DEG, MetricMap
    from my_agent import MyAgent
    from sg_matcher import SGMatcher
    from sptm_models import TorchScorer
    from geometry import yaw_from_e

    gm = SGMatcher(device=args.device)
    scorer = AnyLocScorer(TorchScorer(args.r, args.l, args.device), device=args.device)
    scorer.fit(data)
    agent = MyAgent(data, scorer, goal_matcher=gm, goal_min=20, geo_min=20, join_starts=True, min_run=8)
    focal = 92.0
    frames, bounds = agent.frames, agent.bounds
    n = len(frames)
    feats: dict[int, object] = {}

    def feat(i):
        if i not in feats:
            feats[i] = gm.features(cv2.imread(str(frames[i].path)))
        return feats[i]

    def rel_yaw(a: int, b: int):
        m = gm.match(feat(a), feat(b), focal)
        if m.n < args.min_inliers:
            return None
        y = yaw_from_e(m, focal)
        return None if y is None else (float(y), m.n)

    actions = [f.action for f in frames]
    mm = MetricMap(actions, bounds, ticks_per_frame=3.0)

    # --- 1. measured turn per consecutive pair
    meas = {}
    for s, e in bounds:
        for i in range(s, e - 1):
            r = rel_yaw(i, i + 1)
            if r is not None:
                meas[i] = r[0]
    # sign: a LEFT frame must give a positive (counter-clockwise) turn; flip if it does not
    left = [meas[i] for i in meas if actions[i] == "LEFT"]
    right = [meas[i] for i in meas if actions[i] == "RIGHT"]
    sign = 1.0 if (np.median(left) - np.median(right)) > 0 else -1.0
    print(f"turn measurements: {len(meas)}/{sum(e - s - 1 for s, e in bounds)} pairs; "
          f"median LEFT {np.median(left):+.1f} deg, RIGHT {np.median(right):+.1f} deg -> sign {sign:+.0f}")
    mm.set_turns({i: math.radians(sign * y) for i, y in meas.items()})
    mm.integrate()

    # --- 2. loop closures
    pairs = list(agent.shortcuts) + [e for e in agent.trusted if abs(e[0] - e[1]) != 1]
    loops = []
    for i, j in pairs:
        r = rel_yaw(i, j)
        if r is not None:
            loops.append((i, j, math.radians(sign * r[0])))
    print(f"loop closures: {len(loops)} of {len(pairs)} candidate pairs")

    # --- 3. pose graph
    print(f"drives placed by the loops: {mm.init_align(loops)} of {len(bounds)}")
    P = mm.optimise(loops, odo_yaw_deg=1.0)
    print(f"pose graph residual (rms, in sigmas): {mm.cost0:.2f}")
    out = Path("logs")
    out.mkdir(exist_ok=True)
    (out / f"map_{args.seed}.json").write_text(json.dumps(dict(poses=P.tolist(), bounds=bounds, goal=None)))

    # --- 4. truth
    truth = None
    if args.truth_data is not None:
        from vlad import load_frames
        tf, _ = load_frames(args.truth_data)
        if len(tf) == n:
            truth = np.array([json.loads((f.path.parent / "poses.json").read_text())[int(f.path.stem)] for f in tf])
        else:
            print(f"truth data has {len(tf)} frames, not {n}: skipped")
    if truth is not None:
        mu_e, mu_t = P[:, :2].mean(0), truth[:, :2].mean(0)
        a, b = P[:, :2] - mu_e, truth[:, :2] - mu_t
        u, _, vt = np.linalg.svd(a.T @ b)
        R = u @ np.diag([1, np.sign(np.linalg.det(u @ vt))]) @ vt
        al = a @ R + mu_t
        err = np.hypot(*(al - truth[:, :2]).T)
        print(f"MAP ERROR vs truth: median {np.median(err):.2f} m, 90% {np.percentile(err, 90):.2f} m, max {err.max():.2f} m")
        for d, (s, e) in enumerate(bounds):
            print(f"   drive {d}: median {np.median(err[s:e]):.2f} m, max {err[s:e].max():.2f} m")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            for d, (s, e) in enumerate(bounds):
                ax[0].plot(truth[s:e, 0], truth[s:e, 1], ".", ms=2, label=f"drive {d}")
                ax[1].plot(al[s:e, 0], al[s:e, 1], ".", ms=2, label=f"drive {d}")
            ax[0].set_title("true positions"); ax[1].set_title(f"stitched map (median error {np.median(err):.2f} m)")
            for x in ax: x.set_aspect("equal"); x.legend()
            fig.savefig(out / f"map_{args.seed}.png", dpi=110)
            print(f"wrote logs/map_{args.seed}.png")
        except Exception as ex:  # noqa: BLE001
            print(f"no plot: {ex!r}")


if __name__ == "__main__":
    main()
