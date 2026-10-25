"""
Re-record the local exploration drives WITH ground-truth poses (poses.json), same seeds as cli.py
so the frames are identical to data/local-<seed>.

    uv run --no-sync source/record_gt.py --seeds 7-11

Writes data_gt/local-<seed>/ (traj_*/..., poses.json) and checks that data_info.json matches the
normal data/local-<seed>/ one. Local only; nothing here is available on the real server.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import vis_nav_sim as sim


def seeds(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="7-11")
    args = p.parse_args()
    textures = sim.Textures.find()
    for seed in seeds(args.seeds):
        dest = Path("data_gt") / f"local-{seed}"
        if (dest / "target.jpg").exists():
            print(f"maze {seed}: already there")
            continue
        world = sim.Simulator(textures, seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed)
        s = world.record(dest, routes=3, seed=seed, capture_every=5, student_files=False)
        same = "n/a"
        a, b = Path("data") / f"local-{seed}" / "traj_0" / "data_info.json", dest / "traj_0" / "data_info.json"
        if a.exists() and b.exists():
            same = json.loads(a.read_text()) == json.loads(b.read_text())
        print(f"maze {seed}: {s['frames']} frames on {s['routes']} routes, same actions as data/: {same}")


if __name__ == "__main__":
    main()
