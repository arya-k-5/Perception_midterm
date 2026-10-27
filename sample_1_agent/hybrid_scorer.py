"""
Hybrid scorer: VLAD AND the trained network R must agree; L (or feature matching) steers.

    embed   : [VLAD | R-embedding] stacked per frame
    prob    : sqrt(p_vlad * p_R)  -- a disagreement pulls the score down, so a look-alike corridor
              that fools one of them rarely clears the agent's thresholds
    waypoint: RANSAC visibility (ClassicalScorer.pick_waypoint)
    locomote: trained L if ``use_l`` else feature matching

Same interface as the other scorers, so ``SptmAgent`` does not change.
"""

from __future__ import annotations

import numpy as np


class HybridScorer:
    def __init__(self, classical, trained, use_l: bool = True):
        self.c, self.t, self.use_l = classical, trained, use_l
        self.dv = classical.ex.dim
        self.ticks = 1
        self.dbg = ""

    def prepare(self, frames) -> None:
        self.c.prepare(frames)

    def set_focal(self, focal: float) -> None:
        self.c.set_focal(focal)

    def calibrate(self, emb: np.ndarray, bounds) -> None:
        self.c.calibrate(emb[:, : self.dv], bounds)

    def embed(self, images) -> np.ndarray:
        v = np.asarray(self.c.embed(images), np.float32)
        r = np.asarray(self.t.embed(images), np.float32)
        return np.concatenate([v, r], axis=1)

    def prob(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        pv = self.c.prob(a[:, : self.dv], b[:, : self.dv])
        pr = self.t.prob(a[:, self.dv :], b[:, self.dv :])
        return np.sqrt(np.clip(pv, 0, 1) * np.clip(pr, 0, 1)).astype(np.float32)

    def pick_waypoint(self, cur, cands, get_img, min_inliers: int = 20):
        return self.c.pick_waypoint(cur, cands, get_img, min_inliers)

    def locomote(self, cur, wp, node: int = -1) -> str:
        if self.use_l:
            name = self.t.locomote(cur, wp, node)
            self.ticks = 1 if name == "FORWARD" else 2
            self.dbg = f"L={name}"
            return name
        name = self.c.locomote(cur, wp, node)
        self.ticks, self.dbg = self.c.ticks, self.c.dbg
        return name
