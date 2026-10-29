"""
A metric map of the exploration drives: an (x, y, heading) for every recorded frame.

Step 1, odometry : each recorded frame says which single action produced the next frame, so a drive
                   can be dead-reckoned (FORWARD moves a fixed distance, LEFT/RIGHT turn a fixed angle).
                   It drifts, mainly in heading.
Step 2, loops    : the verified "same place" pairs between far-apart frames (and their relative yaw,
                   measured from feature matches) pin the drifting drives together.
Step 3, optimise : one least-squares problem over all poses (a pose graph, the back end of a SLAM system).

Units: metres and radians. Heading 0 = +x, positive = counter-clockwise (LEFT).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

TICK_M = 0.03  # FORWARD distance per tick (course simulator)
TICK_DEG = 3.44  # LEFT / RIGHT angle per tick


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class MetricMap:
    def __init__(self, actions: list[str], bounds: list[tuple[int, int]], ticks_per_frame: float = 3.0):
        self.actions = list(actions)
        self.bounds = list(bounds)
        self.n = len(self.actions)
        self.tpf = float(ticks_per_frame)
        self.fwd = self.tpf * TICK_M
        self.rot = math.radians(self.tpf * TICK_DEG)
        oi, lx, dth = [], [], []
        for s, e in self.bounds:
            for i in range(s, e - 1):
                a = self.actions[i]
                oi.append(i)
                lx.append(self.fwd if a == "FORWARD" else -self.fwd if a == "BACKWARD" else 0.0)
                dth.append(self.rot if a == "LEFT" else -self.rot if a == "RIGHT" else 0.0)
        self.oi = np.asarray(oi, np.int64)
        self.o_lx = np.asarray(lx, float)
        self.o_dth = np.asarray(dth, float)
        self.P = np.zeros((self.n, 3))

    def set_turns(self, measured: dict[int, float]) -> int:
        """Replace the nominal turn of frame i -> i+1 by a measured yaw (radians). Between two recorded frames
        the robot can turn more than one frame's worth (IDLE frames are dropped but the robot keeps turning)."""
        pos = {int(i): k for k, i in enumerate(self.oi)}
        n = 0
        for i, yaw in measured.items():
            if i in pos:
                self.o_dth[pos[i]] = yaw
                n += 1
        return n

    # ---- step 1

    def integrate(self) -> np.ndarray:
        """Dead reckoning from the origin, one drive at a time."""
        P = np.zeros((self.n, 3))
        for i, lx, dth in zip(self.oi, self.o_lx, self.o_dth, strict=True):
            x, y, t = P[i]
            P[i + 1] = (x + lx * math.cos(t), y + lx * math.sin(t), t + dth)
        self.P = P
        return P

    # ---- step 2a: put the drives in one frame before optimising (the problem is not convex)

    def init_align(self, loops: list[tuple[int, int, float]]) -> int:
        """Rotate and shift each drive onto the ones already placed, using the loop pairs between them.
        Returns how many drives could be placed. ``loops``: (i, j, yaw) with heading_j - heading_i = yaw."""
        drive = np.zeros(self.n, np.int64)
        for d, (s, e) in enumerate(self.bounds):
            drive[s:e] = d
        placed = {0}
        changed = True
        while changed:
            changed = False
            for d in range(len(self.bounds)):
                if d in placed:
                    continue
                rot, pairs = [], []
                for i, j, s in loops:
                    di, dj = drive[i], drive[j]
                    if di in placed and dj == d:
                        rot.append(self.P[i, 2] + s - self.P[j, 2])
                        pairs.append((i, j))
                    elif dj in placed and di == d:
                        rot.append(self.P[j, 2] - s - self.P[i, 2])
                        pairs.append((j, i))  # (placed, moving)
                if len(pairs) < 3:
                    continue
                dr = math.atan2(float(np.mean(np.sin(rot))), float(np.mean(np.cos(rot))))
                c, sn = math.cos(dr), math.sin(dr)
                s0, e0 = self.bounds[d]
                xy = self.P[s0:e0, :2] @ np.array([[c, sn], [-sn, c]])  # rotate by dr (row vectors)
                self.P[s0:e0, :2] = xy
                self.P[s0:e0, 2] += dr
                moved = np.array([self.P[b] for _, b in pairs])[:, :2]
                fixed = np.array([self.P[a] for a, _ in pairs])[:, :2]
                self.P[s0:e0, :2] += np.median(fixed - moved, axis=0)
                placed.add(d)
                changed = True
        return len(placed)

    # ---- step 3

    def optimise(
        self,
        loops: list[tuple[int, int, float]],
        *,
        anchor_pos: float | None = 0.3,
        anchor_yaw: float = 0.35,
        loop_pos: float = 0.3,
        loop_yaw_deg: float = 8.0,
        odo_pos: float = 0.03,
        odo_yaw_deg: float = 1.5,
        max_nfev: int = 100,
    ) -> np.ndarray:
        """Least squares over all poses. ``anchor_pos=None``: do not tie the drive starts to the origin."""
        n = self.n
        oi, oj = self.oi, self.oi + 1
        li = np.array([l[0] for l in loops], np.int64)
        lj = np.array([l[1] for l in loops], np.int64)
        ls = np.array([l[2] for l in loops], float)
        starts = np.array([s for s, e in self.bounds if e > s], np.int64)
        s_op, s_oy = odo_pos, math.radians(odo_yaw_deg)
        s_lp, s_ly = loop_pos, math.radians(loop_yaw_deg)
        use_anchor = anchor_pos is not None

        def resid(v):
            P = v.reshape(n, 3)
            ti, tj = P[oi, 2], P[oj, 2]
            c, s = np.cos(ti), np.sin(ti)
            dx, dy = P[oj, 0] - P[oi, 0], P[oj, 1] - P[oi, 1]
            parts = [
                (c * dx + s * dy - self.o_lx) / s_op,
                (-s * dx + c * dy) / s_op,
                wrap(tj - ti - self.o_dth) / s_oy,
                (P[li, 0] - P[lj, 0]) / s_lp,
                (P[li, 1] - P[lj, 1]) / s_lp,
                wrap(P[lj, 2] - P[li, 2] - ls) / s_ly,
            ]
            if use_anchor:
                parts += [P[starts, 0] / anchor_pos, P[starts, 1] / anchor_pos, wrap(P[starts, 2]) / anchor_yaw]
            return np.concatenate(parts)

        rows, cols = [], []
        r0 = 0

        def block(m, idx_cols):
            nonlocal r0
            rr = np.repeat(np.arange(r0, r0 + m), len(idx_cols))
            cc = np.concatenate([c_[:, None] for c_ in idx_cols], axis=1).ravel() if m else np.zeros(0, np.int64)
            rows.append(rr)
            cols.append(cc)
            r0 += m

        m_o, m_l, m_a = len(oi), len(li), len(starts)
        six = [3 * oi, 3 * oi + 1, 3 * oi + 2, 3 * oj, 3 * oj + 1, 3 * oj + 2]
        block(m_o, six)
        block(m_o, six)
        block(m_o, [3 * oi + 2, 3 * oj + 2])
        four = [3 * li, 3 * li + 1, 3 * lj, 3 * lj + 1]
        block(m_l, four)
        block(m_l, four)
        block(m_l, [3 * li + 2, 3 * lj + 2])
        if use_anchor:
            block(m_a, [3 * starts])
            block(m_a, [3 * starts + 1])
            block(m_a, [3 * starts + 2])
        rr, cc = np.concatenate(rows), np.concatenate(cols)
        S = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(r0, 3 * n)).tocsr()
        sol = least_squares(
            resid, self.P.ravel(), jac_sparsity=S, loss="soft_l1", f_scale=3.0, x_scale="jac",
            max_nfev=max_nfev, method="trf",
        )
        self.P = sol.x.reshape(n, 3)
        self.cost0 = float(np.sqrt(np.mean(resid(self.P.ravel()) ** 2)))
        return self.P
