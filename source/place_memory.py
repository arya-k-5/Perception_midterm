"""
Memory + planner for the map-following agent.

The map is the exploration-frame graph. On top of it the agent keeps a per-node PENALTY that
grows where the robot has been stuck, looped, or turned around, and plans with Dijkstra on
edge weights 1 + mean(penalty of the two ends). After a failure the cheapest route changes, so
the robot does not walk into the same dead end again.

Pure numpy/scipy, no simulator: tests/test_place_memory.py covers it.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

BUCKET = 5  # nodes this close count as "the same spot" for loop detection


class PlaceMemory:
    def __init__(self, n: int, edges: np.ndarray, goal: int = -1, window: int = 150, max_spots: int = 8):
        self.n = n
        self.u, self.v = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
        self.pen = np.zeros(n)
        self.visits = np.zeros(n, np.int32)
        self.hist: deque[int] = deque(maxlen=window)
        self.window, self.max_spots = window, max_spots
        self.last = -1
        self.goal = goal
        self.dirty = True
        self.pred = np.full(n, -1)
        self.dist = np.full(n, np.inf)
        self.hops = np.full(n, -1)
        self.cap = 12.0  # one node never costs more than this extra

    # ---- planning

    def _matrix(self, weighted: bool) -> csr_matrix:
        if weighted:
            w = 1.0 + 0.5 * (self.pen[self.u] + self.pen[self.v])
        else:
            w = np.ones(len(self.u))
        rows = np.concatenate([self.u, self.v])
        cols = np.concatenate([self.v, self.u])
        return csr_matrix((np.concatenate([w, w]), (rows, cols)), shape=(self.n, self.n))

    def set_goal(self, goal: int) -> None:
        self.goal = goal
        d = dijkstra(self._matrix(False), directed=False, indices=goal)
        self.hops = np.where(np.isfinite(d), d, -1).astype(np.int64)
        self.dirty = True

    def replan(self) -> None:
        if not self.dirty or self.goal < 0:
            return
        d, p = dijkstra(self._matrix(True), directed=False, indices=self.goal, return_predecessors=True)
        self.dist, self.pred = d, np.where(p < 0, -1, p)
        self.dirty = False

    def reachable(self) -> np.ndarray:
        return self.hops >= 0

    def path(self, node: int, limit: int = 40) -> list[int]:
        """node, then the next frames toward the goal (at most ``limit`` after it)."""
        self.replan()
        out = [int(node)]
        while out[-1] != self.goal and len(out) <= limit:
            nxt = int(self.pred[out[-1]])
            if nxt < 0:
                break
            out.append(nxt)
        return out

    # ---- remembering

    def punish(self, nodes, amount: float, radius: int = 3) -> None:
        for c in set(int(x) for x in nodes if x >= 0):
            lo, hi = max(0, c - radius), min(self.n, c + radius + 1)
            self.pen[lo:hi] = np.minimum(self.pen[lo:hi] + amount, self.cap)
        self.dirty = True

    def arrive(self, node: int) -> None:
        """Call once per step with the current estimate."""
        if node >= 0 and (self.last < 0 or abs(node - self.last) > 2):
            self.visits[node] += 1
            self.punish([node], 0.05, 2)  # a place we keep coming back to gets slowly dearer
            self.last = node
        self.hist.append(int(node))

    def stagnating(self) -> bool:
        if len(self.hist) < self.window:
            return False
        spots = {n // BUCKET for n in self.hist if n >= 0}
        return 0 < len(spots) <= self.max_spots

    def escape_zone(self) -> set[int]:
        z = set(n for n in self.hist if n >= 0)
        self.hist.clear()
        self.last = -1
        return z

    def mark_turnarounds(self, actions: list[str], bounds, min_ticks: int = 30, amount: float = 4.0) -> int:
        """Prior from the exploration data itself: where the recorded robot turned (nearly)
        around it met a dead end or a wall. Make those places dear before the run starts."""
        marked = 0
        for s, e in bounds:
            i = s
            while i < e:
                if actions[i] in ("LEFT", "RIGHT"):
                    j = i
                    while j < e and actions[j] == actions[i]:
                        j += 1
                    if j - i >= min_ticks:
                        self.punish(range(i, j), amount, 2)
                        marked += 1
                    i = j
                else:
                    i += 1
        return marked
