"""
Which shortcut test separates real shortcuts from look-alikes? Measured against true poses.

    uv run --no-sync source/shortcut_lab.py --seeds 8,9,11 --out results/shortcut_lab.csv

For every appearance-candidate pair (i, j) the agent would consider, writes: the true distance and heading
difference, and several verification scores (fundamental-matrix inliers = what the agent uses now,
essential-matrix inliers with the known focal length, and the same on neighbouring frames i+d, j+d).
Local only (needs data_gt/ from record_gt.py).
"""
import argparse, csv, json, math, os, sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from classical_scorer import ClassicalScorer
from sptm_agent import SptmAgent

FOCAL = 92.0


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
    ap.add_argument("--seeds", default="8,9,11")
    ap.add_argument("--out", type=Path, default=Path("results/shortcut_lab.csv"))
    ap.add_argument("--max-pairs", type=int, default=2500)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "i", "j", "dist_m", "yaw_deg", "true", "nF", "nE", "nE_nb", "nE_nb_med", "nF_nb_med"]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fields); w.writeheader()
        for seed in seeds(args.seeds):
            data = Path("data_gt") / f"local-{seed}"
            ag = SptmAgent(data, ClassicalScorer(Path("cache") / f"sptm-gt-{data.name}"), geo_min=0, budget=10000)
            P = np.zeros((ag.n, 3))
            for i, f in enumerate(ag.frames):
                P[i] = json.loads((Path(f.path).parent / "poses.json").read_text())[int(Path(f.path).stem)]
            pairs = list(ag.shortcuts)
            rng = np.random.default_rng(0)
            if len(pairs) > args.max_pairs:
                pairs = [pairs[k] for k in rng.choice(len(pairs), args.max_pairs, replace=False)]
            print(f"maze {seed}: {len(pairs)} candidate pairs", flush=True)
            lo = {t: s for t, (s, _) in enumerate(ag.bounds)}
            hi = {t: e for t, (_, e) in enumerate(ag.bounds)}
            for k, (i, j) in enumerate(pairs):
                fi, fj = ag._feat(i), ag._feat(j)
                nF = ag.matcher.match(fi, fj).n
                nE = ag.matcher.match(fi, fj, focal=FOCAL).n
                nbE, nbF = [], []
                for d in (-6, -3, 3, 6):
                    ii, jj = i + d, j + d
                    if lo[ag.traj[i]] <= ii < hi[ag.traj[i]] and lo[ag.traj[j]] <= jj < hi[ag.traj[j]]:
                        nbE.append(ag.matcher.match(ag._feat(ii), ag._feat(jj), focal=FOCAL).n)
                        nbF.append(ag.matcher.match(ag._feat(ii), ag._feat(jj)).n)
                dist = float(np.hypot(*(P[i, :2] - P[j, :2])))
                yaw = abs(math.degrees(wrap(P[i, 2] - P[j, 2])))
                w.writerow(dict(seed=seed, i=i, j=j, dist_m=round(dist, 3), yaw_deg=round(yaw, 1), true=int(dist < 0.5 and yaw < 60),
                                nF=nF, nE=nE, nE_nb=sum(1 for x in nbE if x >= 10),
                                nE_nb_med=float(np.median(nbE)) if nbE else -1, nF_nb_med=float(np.median(nbF)) if nbF else -1))
                if k % 200 == 0:
                    print(f"  {k}/{len(pairs)}", flush=True)
                    fh.flush()
    print("done ->", args.out)


if __name__ == "__main__":
    main()
