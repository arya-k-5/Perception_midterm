"""
A scorer for ``SptmAgent`` that needs NO training: VLAD stands in for R, feature matching for L.

    R  (same place?)  : cosine of VLAD descriptors, squashed to a probability with a threshold
                        calibrated on THIS maze (adjacent frames vs. far-apart frames).
    L  (which move?)  : match the waypoint frame to the live frame; the sideways shift of the
                        matched features is the heading error. Turn until it is small, else go
                        forward. No matches at all (waypoint behind us): keep turning one way.

It has the same interface as ``sptm_models.TorchScorer`` (embed / prob / locomote), so the
trained networks can replace it later without touching the agent.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from geometry import Matcher, shift_deg
from vlad import Frame, VLADExtractor

TURN_DEG = 3.44


class ClassicalScorer:
    def __init__(self, cache_dir: Path, sign: int = +1, turn_tol: float = 9.0, n_clusters: int = 128):
        self.ex = VLADExtractor(n_clusters, cache_dir)
        self.matcher = Matcher()
        self.sign, self.turn_tol = sign, turn_tol
        self.focal = 160.0
        self.mu, self.scale = 0.0, 0.05
        self.ticks = 1
        self._db: np.ndarray | None = None
        self._wp_feat: dict[int, object] = {}
        self._last_turn = "LEFT"

    # ---- setup (called by the agent before and after embedding the database)

    def prepare(self, frames: list[Frame]) -> None:
        self._db = self.ex.fit(frames)

    def set_focal(self, focal: float) -> None:
        self.focal = float(focal)

    def calibrate(self, emb: np.ndarray, bounds: list[tuple[int, int]], seed: int = 0) -> None:
        """Map cosine similarity to P(same place) so the agent's thresholds mean something."""
        rng = np.random.default_rng(seed)
        e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        pos, neg = [], []
        for s, t in bounds:
            for i in range(s, t - 3, 2):
                pos.append(float(e[i] @ e[i + 3]))
        n = len(e)
        for _ in range(20000):
            i, j = rng.integers(n, size=2)
            if abs(i - j) > 150:
                neg.append(float(e[i] @ e[j]))
        p10, n99 = float(np.percentile(pos, 10)), float(np.percentile(neg, 99))
        if p10 > n99 + 1e-3:
            self.mu, self.scale = (p10 + n99) / 2, (p10 - n99) / 6
        else:
            self.mu, self.scale = n99, 0.01
        print(f"classical R: adjacent-frame cos p10 {p10:.3f}, far-pair cos p99 {n99:.3f} -> mu {self.mu:.3f}")

    # ---- the scorer interface

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        if self._db is not None and len(images) == len(self._db):
            db, self._db = self._db, None  # the agent embeds the whole database exactly once
            return db
        return np.stack([self.ex.extract(im) for im in images])

    def prob(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        z = ((a * b).sum(1) - self.mu) / self.scale
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def locomote(self, cur: np.ndarray, wp: np.ndarray, node: int = -1) -> str:
        if node not in self._wp_feat:
            self._wp_feat[node] = self.matcher.features(wp)
        m = self.matcher.match(self._wp_feat[node], self.matcher.features(cur))
        if m.n < 8:  # cannot see the waypoint: look around in a fixed direction
            self.ticks = 3
            return self._last_turn
        left = self.sign * shift_deg(m, self.focal)  # how far left of the waypoint we point
        if abs(left) > self.turn_tol:
            self._last_turn = "RIGHT" if left > 0 else "LEFT"
            self.ticks = int(np.clip(round(abs(left) / TURN_DEG) - 1, 1, 4))
            return self._last_turn
        self.ticks = 1
        return "FORWARD"
