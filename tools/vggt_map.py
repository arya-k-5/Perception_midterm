"""
Stitch the exploration frames with VGGT (Wang et al., CVPR 2025 -- course reading list, "3D from images in
one pass") and check the result against the true poses.   NEW FILE. Local mazes with recorded truth only.

    one-time install (does NOT touch torch):
      uv pip install --python /Users/arya/vis_nav_player_work/.venv/bin/python --no-deps \
          git+https://github.com/facebookresearch/vggt.git
      uv pip install --python /Users/arya/vis_nav_player_work/.venv/bin/python einops safetensors huggingface_hub
    run (first run downloads the VGGT-1B weights, ~5 GB):
      /Users/arya/vis_nav_player_work/.venv/bin/python tools/vggt_map.py \
          --data /Users/arya/vis_nav_player_work/data_gt/local-8 --drive 0
    quick test first: add --max-frames 120

Step 1 (this file): per drive, VGGT gives camera poses for overlapping windows of frames; windows are chained
by a similarity transform fitted on the frames they share. The drive's 2D track (x, y, heading) is compared with
the truth after one similarity alignment (VGGT has no metric scale). If this per-drive error is far below the
1.0 m of the action-based map, step 2 is to join the drives with the verified shortcuts (pose graph), as
map_check.py does, and then navigate on it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))


def umeyama(src: np.ndarray, dst: np.ndarray, scale: bool = True):
    """s, R, t with dst ~ s R src + t (least squares; Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    sgn = np.ones(len(d))
    sgn[-1] = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag(sgn) @ vt
    s = (d * sgn).sum() / (a ** 2).sum(1).mean() if scale else 1.0
    return s, r, mu_d - s * r @ mu_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="data_gt/local-<seed> (frames + poses.json)")
    ap.add_argument("--drive", type=int, default=0)
    ap.add_argument("--window", type=int, default=24, help="frames per VGGT pass")
    ap.add_argument("--overlap", type=int, default=8, help="frames shared by consecutive windows")
    ap.add_argument("--every", type=int, default=2, help="use every n-th frame")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vlad import load_frames

    dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    frames, bounds = load_frames(args.data)
    s0, e0 = bounds[args.drive]
    idx = list(range(s0, e0, args.every))
    if args.max_frames:
        idx = idx[: args.max_frames]
    truth = np.array([json.loads((frames[i].path.parent / "poses.json").read_text())[int(frames[i].path.stem)]
                      for i in idx])
    print(f"drive {args.drive}: {len(idx)} frames (every {args.every}), windows of {args.window}, device {dev}")

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    step = args.window - args.overlap
    centres: dict[int, np.ndarray] = {}  # k (position in idx) -> camera centre in the chained frame
    fwd: dict[int, np.ndarray] = {}  # k -> camera viewing direction in the chained frame
    t0 = time.time()
    win_err = []
    for w0 in range(0, len(idx), step):
        ks = list(range(w0, min(w0 + args.window, len(idx))))
        if len(ks) < 3:
            break
        imgs = load_and_preprocess_images([str(frames[idx[k]].path) for k in ks]).to(dev)
        with torch.no_grad():
            pred = model(imgs)
        ext, _ = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
        ext = ext[0].float().cpu().numpy()  # (S, 3, 4) camera-from-world (OpenCV: x right, y down, z ahead)
        r, t = ext[:, :, :3], ext[:, :, 3]
        c = -np.einsum("sji,sj->si", r, t)  # camera centres C = -R^T t
        z = r[:, 2, :]  # viewing direction in world = third row of R
        # diagnostic: this window alone, its own similarity fit to the truth (VGGT quality, no chaining)
        wxy = np.stack([c[:, 0], c[:, 2]], 1)
        wtr = truth[ks][:, :2]
        if np.ptp(wtr, axis=0).max() > 0.2:
            fits = []
            for flip in (1.0, -1.0):
                q = wxy * np.array([1.0, flip])
                s_, r_, t_ = umeyama(q, wtr)
                fits.append(np.median(np.hypot(*((s_ * (r_ @ q.T)).T + t_ - wtr).T)))
            spread = np.median(np.hypot(*(wtr - wtr.mean(0)).T))
            win_err.append((min(fits), spread))
        shared = [k for k in ks if k in centres]
        if shared:
            src = np.array([c[ks.index(k)] for k in shared])
            dst = np.array([centres[k] for k in shared])
            if len(shared) >= 3 and np.ptp(src, axis=0).max() > 1e-6:
                s, rr, tt = umeyama(src, dst)
            else:
                s, rr, tt = 1.0, np.eye(3), dst.mean(0) - src.mean(0)
            c = (s * (rr @ c.T)).T + tt
            z = (rr @ z.T).T
        for j, k in enumerate(ks):
            if k not in centres:
                centres[k], fwd[k] = c[j], z[j]
        print(f"  window {w0 // step + 1}: frames {ks[0]}..{ks[-1]}  ({time.time() - t0:.0f} s)", flush=True)

    ks = sorted(centres)
    C = np.array([centres[k] for k in ks])
    Z = np.array([fwd[k] for k in ks])
    # the robot drives on a plane and the camera is upright, so the floor is the x-z plane of the first
    # camera (the chain's reference frame); heading = direction of the viewing axis in that plane
    xy = np.stack([C[:, 0], C[:, 2]], 1)
    head = np.arctan2(Z[:, 2], Z[:, 0])
    tr = truth[ks]
    best = None
    for flip in (1.0, -1.0):  # the plane's second axis may come out mirrored: keep the better fit
        q = xy * np.array([1.0, flip])
        s, r2, t2 = umeyama(q, tr[:, :2])
        al = (s * (r2 @ q.T)).T + t2
        err = np.hypot(*(al - tr[:, :2]).T)
        if best is None or np.median(err) < np.median(best[1]):
            best = (al, err, flip * head + math.atan2(r2[1, 0], r2[0, 0]))
    al, err, h = best
    d = (h - tr[:, 2] + np.pi) % (2 * np.pi) - np.pi
    off = math.atan2(np.median(np.sin(d)), np.median(np.cos(d)))  # camera vs robot heading convention
    hd = np.degrees(np.abs((d - off + np.pi) % (2 * np.pi) - np.pi))
    print(f"VGGT drive {args.drive}: POSITION error median {np.median(err):.2f} m, 90% {np.percentile(err, 90):.2f} m, "
          f"max {err.max():.2f} m | HEADING error median {np.median(hd):.0f} deg, 90% {np.percentile(hd, 90):.0f} deg")
    if win_err:
        we = np.array(win_err)
        print(f"single windows (no chaining): median error {np.median(we[:, 0]):.2f} m where the frames spread "
              f"{np.median(we[:, 1]):.2f} m around their centre (error ~ spread = VGGT finds nothing)")
    print(f"trivial map (every frame at the centre of the truth): median {np.median(np.hypot(*(tr[:, :2] - tr[:, :2].mean(0)).T)):.2f} m")
    print("(action-based map on maze 8 for comparison: drive 0 median 1.20 m, max 3.43 m, heading ~27 deg)")
    out = Path("logs")
    out.mkdir(exist_ok=True)
    (out / f"vggt_{args.data.name}_drive{args.drive}.json").write_text(json.dumps(dict(
        frames=[idx[k] for k in ks], xy=al.tolist(), truth=tr.tolist(), err=err.tolist())))
    print(f"wrote logs/vggt_{args.data.name}_drive{args.drive}.json")


if __name__ == "__main__":
    main()
