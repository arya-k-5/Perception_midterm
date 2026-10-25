"""
Offline map audit with ground truth. Builds the SAME graph the agent builds, then checks it against the
true poses recorded by record_gt.py (data_gt/local-<seed>/traj_*/poses.json).

    python source/map_truth.py --seeds 7-11 --out results/map_truth

Prints per maze: goal node error, false-shortcut rate, and whether the planned route start->goal
contains false jumps. Writes a PNG per maze. Local only.
"""
import argparse, json, math, os, sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import networkx as nx

from classical_scorer import ClassicalScorer
from sptm_agent import SptmAgent


def seeds(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7-11")
    ap.add_argument("--out", type=Path, default=Path("results/map_truth"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for seed in seeds(args.seeds):
        data = Path("data_gt") / f"local-{seed}"
        scorer = ClassicalScorer(Path("cache") / f"sptm-gt-{data.name}")
        ag = SptmAgent(data, scorer, geo_min=22, budget=10000)
        # true pose of every node
        P = np.zeros((ag.n, 3))
        for i, f in enumerate(ag.frames):
            traj = json.loads((Path(f.path).parent / "poses.json").read_text())
            k = int(Path(f.path).stem)
            P[i] = traj[k]
        tgt = cv2.imread(str(data / "target.jpg"))
        w = tgt.shape[1] // 4
        views = [tgt[:, k * w:(k + 1) * w] for k in range(4)]
        class Info: pass
        info = Info(); info.targets = views
        info.camera = None
        # focal: use the sim's value if present, else 92 px for 320 wide
        ag.focal = 92.0
        ag.scorer.set_focal(92.0) if hasattr(ag.scorer, "set_focal") else None
        ag._reset()
        ag.tfeat = [ag.matcher.features(t) for t in views]
        tembs = np.asarray(ag.scorer.embed(list(views)), np.float32)
        best = (-1, -1, 0); fb = (-1.0, -1)
        for k in range(4):
            pk = ag._prob_to_all(tembs[k])
            if pk.max() > fb[0]:
                fb = (float(pk.max()), int(np.argmax(pk)))
            for i in np.argsort(-pk)[:6]:
                n = ag.matcher.match(ag.tfeat[k], ag._feat(int(i))).n
                if n > best[0]:
                    best = (n, int(i), k)
        goal = best[1] if best[0] >= 12 else fb[1]
        ag.mem.set_goal(goal)
        spawn = np.array([P[s] for s, _ in ag.bounds])[0]
        # true goal pose = pose of the last frame? use sim target from the groundtruth json if present
        gt_json = Path("logs/gt") / f"local{seed}.json"
        target = None
        if gt_json.exists():
            target = json.loads(gt_json.read_text())["maze"].get("target")
        gdist = float(np.hypot(*(P[goal, :2] - np.array(target[:2])))) if target else float("nan")
        # shortcut audit
        sc = ag.shortcuts
        d = np.array([np.hypot(*(P[i, :2] - P[j, :2])) for i, j in sc])
        yaw = np.array([abs(wrap(P[i, 2] - P[j, 2])) for i, j in sc])
        good = (d < 0.5) & (yaw < math.radians(60))
        print(f"\n=== maze {seed}: {ag.n} nodes, {len(sc)} verified shortcuts")
        print(f"goal node {goal} ({best[0]} inliers, view {best[2]}): true distance to the real goal = {gdist:.2f} m")
        print(f"shortcuts that join places < 0.5 m and < 60 deg apart: {good.sum()} / {len(sc)} ({100*good.mean():.0f}%); "
              f"median error of the false ones {np.median(d[~good]) if (~good).any() else 0:.1f} m")
        # route start -> goal on the graph
        starts = [s for s, _ in ag.bounds]
        for s0 in starts[:1]:
            try:
                path = nx.shortest_path(ag.G, s0, goal)
            except Exception:
                print("no path from spawn node to goal"); continue
            jumps = [(a, b, float(np.hypot(*(P[a, :2] - P[b, :2])))) for a, b in zip(path[:-1], path[1:]) if abs(a - b) > 1]
            bad = [(a, b, round(x, 2)) for a, b, x in jumps if x >= 0.5]
            print(f"route spawn->goal: {len(path)-1} hops, {len(jumps)} shortcut jumps, {len(bad)} of them false: {bad[:6]}")
        # drives: how close does any exploration frame get to the real goal?
        if target:
            dd = np.hypot(P[:, 0] - target[0], P[:, 1] - target[1])
            print(f"closest exploration frame to the real goal: {dd.min():.2f} m (node {int(dd.argmin())}); frames within 0.5 m: {(dd<0.5).sum()}")
        # plot
        if gt_json.exists():
            m = json.loads(gt_json.read_text())["maze"]; G = np.array([[1 if c == '#' else 0 for c in r] for r in m["grid"]]); cs = m["cell_size"]
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(G, cmap="Greys", origin="lower", extent=[0, 31 * cs, 0, 31 * cs], alpha=.75)
            for (s, e), c in zip(ag.bounds, ["tab:blue", "tab:green", "tab:orange"]):
                ax.plot(P[s:e, 0], P[s:e, 1], ".", ms=1.5, color=c)
            for (i, j), g in zip(sc, good):
                if not g:
                    ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], "r-", lw=.3, alpha=.4)
            ax.plot(*m["spawn"][:2], "g^", ms=14); ax.plot(*m["target"][:2], "r*", ms=18)
            ax.plot(*P[goal, :2], "mo", ms=10, mfc="none")
            ax.set_title(f"maze {seed}: drives (blue/green/orange), false shortcuts (red), chosen goal node (magenta)")
            fig.savefig(args.out / f"maze{seed}.png", dpi=80); plt.close(fig)


if __name__ == "__main__":
    main()
