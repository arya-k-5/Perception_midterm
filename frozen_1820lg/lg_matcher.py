"""SuperPoint + LightGlue (Lindenberger, Lindner, Pollefeys, ICCV 2023) behind the same interface as sg_matcher.SGMatcher:
features(image) -> features, match(a, b, focal=None) -> geometry.Match (RANSAC inliers, same check as SGMatcher).
NEW FILE. Official code: pip install git+https://github.com/cvg/LightGlue.git (weights download on first use)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from geometry import Match


@dataclass
class LGFeat:
    pts: np.ndarray  # (n, 2) float32, x y
    feats: dict  # LightGlue feature dict (batched, on device)


class LGMatcher:
    def __init__(self, max_keypoints: int = 512, ransac_px: float = 1.5, device: str | None = None, **_):
        from lightglue import LightGlue, SuperPoint

        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                           else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.sp = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.dev)
        self.lg = LightGlue(features="superpoint").eval().to(self.dev)
        self.ransac_px = ransac_px
        print(f"Loaded SuperPoint + LightGlue on {self.dev}")

    @torch.no_grad()
    def features(self, image: np.ndarray) -> LGFeat:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x = torch.from_numpy(gray).float().div(255.0)[None].to(self.dev)  # (1, H, W)
        f = self.sp.extract(x)
        return LGFeat(f["keypoints"][0].cpu().numpy().astype(np.float32), f)

    @torch.no_grad()
    def match(self, a: LGFeat, b: LGFeat, focal: float | None = None) -> Match:
        empty = Match(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))
        if len(a.pts) < 8 or len(b.pts) < 8:
            return empty
        out = self.lg({"image0": a.feats, "image1": b.feats})
        m = out["matches"][0].cpu().numpy()  # (k, 2) indices into a / b
        if len(m) < 8:
            return empty
        pa, pb = a.pts[m[:, 0]], b.pts[m[:, 1]]
        if focal is not None:
            k = np.array([[focal, 0, 160.0], [0, focal, 120.0], [0, 0, 1.0]])
            _, mask = cv2.findEssentialMat(pa, pb, k, cv2.RANSAC, 0.999, 1.0)
        else:
            _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, self.ransac_px, 0.99)
        if mask is None:
            return empty
        keep = mask.ravel().astype(bool)
        return Match(pa[keep], pb[keep])
