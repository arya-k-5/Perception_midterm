"""
SPTM agent (Savinov, Dosovitskiy, Koltun, ICLR 2018) rebuilt for this challenge.

    uv run source/sptm_agent.py --local 7 --budget 2000 --verbose
    uv run source/sptm_agent.py --challenge <id>

Memory graph : one node per exploration frame; edges between consecutive frames, plus
               "visual shortcuts" between far-apart frames that a learned network R says are
               the same place for a whole WINDOW of neighbouring frames (not one lucky pair).
Localisation : R scores the camera frame against every node. Search near the previous node
               first; fall back to a global search only when nothing nearby is similar enough.
Planning     : shortest path (unit edges) from the estimated node to the goal node.
Waypoint     : the furthest node along that path that R still says is "the same place" as
               what we see now (so it is close and reachable).
Locomotion   : a learned network L maps (current frame, waypoint frame) -> FORWARD/LEFT/RIGHT.
Last metre   : same RANSAC servo on the four goal photos as my_agent.py, then CHECKIN.

The scorer (R and L) is injected: ``sptm_models.TorchScorer`` for real runs, anything with
``embed / prob / locomote`` for tests. This file needs no torch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from geometry import Matcher, scale_ratio, shift_deg
from my_agent import TURN_DEG, VIEW_OFFSET, _intrinsic_focal
from place_memory import PlaceMemory
from vis_nav_sdk import Action, Agent, Observation, Result, SessionInfo, run
from vlad import load_frames

SIGN = +1  # LEFT x6 shifted the picture by +26 / +23 deg on the real simulator (two mazes)


class SptmAgent(Agent):
    def __init__(
        self,
        data_dir: Path,
        scorer,
        *,
        subsample: int = 1,
        h_min: int = 1,
        h_max: int = 7,
        s_local: float = 0.7,
        s_reach: float = 0.8,
        s_shortcut: float = 0.9,
        local_hops: int = 12,
        shortcut_gap: int = 10,
        shortcut_k: int = 8,
        window: int = 3,
        near_hops: int = 3,
        burst: int = 1,
        geo_min: int = 0,
        scale_tol: float = 0.04,
        budget: int | None = None,
        log_dir: Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.scorer = scorer
        self.geo_min = geo_min
        self.geo_hops = 12
        self.frames, self.bounds = load_frames(Path(data_dir), subsample)
        print(f"{len(self.frames)} frames from {len(self.bounds)} trajectories")
        self.n = len(self.frames)
        self.traj = np.concatenate([[t] * (e - s) for t, (s, e) in enumerate(self.bounds)])
        if hasattr(scorer, "prepare"):
            scorer.prepare(self.frames)
        imgs = [cv2.imread(str(f.path)) for f in self.frames]
        self.emb = np.asarray(scorer.embed(imgs), np.float32)
        del imgs
        if hasattr(scorer, "calibrate"):
            scorer.calibrate(self.emb, self.bounds)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))
        for s, e in self.bounds:
            self.G.add_edges_from((i, i + 1) for i in range(s, e - 1))
        self.matcher = Matcher()
        self._img: dict[int, np.ndarray] = {}
        self.shortcuts = self._shortcuts(shortcut_gap, shortcut_k, window, s_shortcut)
        self.G.add_edges_from(self.shortcuts)
        print(f"graph: {self.G.number_of_edges()} edges, {len(self.shortcuts)} verified shortcuts")
        self.mem = PlaceMemory(self.n, np.array(list(self.G.edges()), dtype=np.int64), window=100)
        marked = self.mem.mark_turnarounds([str(f.action) for f in self.frames], self.bounds)
        print(f"memory: {marked} turn-around spots in the exploration drives marked as costly")
        self.h_min, self.h_max, self.s_local, self.s_reach = h_min, h_max, s_local, s_reach
        self.local_hops, self.near_hops, self.burst = local_hops, near_hops, burst
        self.scale_tol, self.budget = scale_tol, budget
        self.log_dir, self.verbose = log_dir, verbose
        self._local: dict[int, np.ndarray] = {}
        self._reset()

    # ---- graph

    def _prob_pairs(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(self.scorer.prob(a, b), np.float32)

    def _prob_to_all(self, q: np.ndarray) -> np.ndarray:
        return self._prob_pairs(np.broadcast_to(q, self.emb.shape), self.emb)

    def _shortcuts(self, gap: int, k: int, window: int, thr: float) -> list[tuple[int, int]]:
        """Candidates by embedding cosine (cheap), then verified by R over a window of frames:
        the MEDIAN of R(i+d, j+d), d in [-window, window], must beat ``thr`` (paper eq. 2)."""
        e = self.emb / (np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9)
        idx = np.arange(self.n)
        pairs: set[tuple[int, int]] = set()
        for i in range(self.n):
            sim = e @ e[i]
            sim[(self.traj == self.traj[i]) & (np.abs(idx - i) <= gap)] = -2
            for j in np.argsort(-sim)[:k]:
                pairs.add((min(i, int(j)), max(i, int(j))))
        lo = {t: s for t, (s, _) in enumerate(self.bounds)}
        hi = {t: e_ for t, (_, e_) in enumerate(self.bounds)}
        a_idx, b_idx, owner = [], [], []
        for p, (i, j) in enumerate(sorted(pairs)):
            for d in range(-window, window + 1):
                ii, jj = i + d, j + d
                if lo[self.traj[i]] <= ii < hi[self.traj[i]] and lo[self.traj[j]] <= jj < hi[self.traj[j]]:
                    a_idx.append(ii), b_idx.append(jj), owner.append(p)
        if not a_idx:
            return []
        pr = self._prob_pairs(self.emb[a_idx], self.emb[b_idx])
        owner = np.array(owner)
        plist = sorted(pairs)
        out = []
        for p, (i, j) in enumerate(plist):
            if np.median(pr[owner == p]) > thr:
                out.append((i, j))
        if self.geo_min > 0:  # a look-alike corridor fools appearance; shared geometry does not
            keep, fc = [], {}

            def feat(k: int):
                if k not in fc:
                    fc[k] = self.matcher.features(cv2.imread(str(self.frames[k].path)))
                return fc[k]

            for i, j in out:
                n = self.matcher.match(feat(i), feat(j)).n
                if n >= self.geo_min:
                    keep.append((i, j))
            print(f"shortcuts: {len(out)} by appearance, {len(keep)} after RANSAC (>= {self.geo_min} inliers)")
            out = keep
        return out

    def _local_nodes(self, node: int) -> np.ndarray:
        if node not in self._local:
            reach = nx.single_source_shortest_path_length(self.G, node, cutoff=self.local_hops)
            self._local[node] = np.fromiter(reach.keys(), dtype=np.int64)
        return self._local[node]

    # ---- session

    def _reset(self) -> None:
        self.t, self.node, self.hops, self.state, self.log = 0, -1, -1, "init", []
        self.prev_small, self.prev_act, self.stuck, self.queue, self.n_stuck = None, None, 0, [], 0
        self.appr_steps = 0
        self.prev_feat = None
        self.n_loops, self.best_hops, self.best_t = 0, 10**9, 0

    def setup(self, info: SessionInfo) -> None:
        self._reset()
        self.focal = _intrinsic_focal(info)
        if hasattr(self.scorer, "set_focal"):
            self.scorer.set_focal(self.focal)
        self.targets = info.targets
        self.tfeat = [self.matcher.features(t) for t in self.targets]
        tembs = np.asarray(self.scorer.embed(list(self.targets)), np.float32)
        best = (-1, -1, 0)  # (inliers, node, view)
        fallback = (-1.0, -1)
        for k in range(4):
            pk = self._prob_to_all(tembs[k])
            if pk.max() > fallback[0]:
                fallback = (float(pk.max()), int(np.argmax(pk)))
            for i in np.argsort(-pk)[:6]:
                n = self.matcher.match(self.tfeat[k], self._feat(int(i))).n
                if n > best[0]:
                    best = (n, int(i), k)
        self.goal = best[1] if best[0] >= 12 else fallback[1]
        self.mem.set_goal(self.goal)
        self.reach = self.mem.reachable()
        print(f"goal: node {self.goal} (view {best[2]}, {best[0]} inliers; R-best {fallback[1]})")

    def finish(self, result: Result | None) -> None:
        print("result:", result)
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            out = self.log_dir / f"sptm_{int(time.time())}.json"
            out.write_text(json.dumps({"log": self.log, "result": repr(result)}, default=float))

    def _frame_img(self, i: int) -> np.ndarray:
        if i not in self._img:
            self._img[i] = cv2.imread(str(self.frames[i].path))
        return self._img[i]

    def _feat(self, i: int):
        return self.matcher.features(self._frame_img(i))

    # ---- decisions

    def act(self, obs: Observation):
        self.t += 1
        left = obs.steps_left if self.budget is None else min(obs.steps_left, self.budget - obs.step)
        if left <= 12:
            return self._out(Action.CHECKIN, "budget")
        small = cv2.resize(cv2.cvtColor(obs.image, cv2.COLOR_BGR2GRAY), (40, 30)).astype(np.float32)
        cur_feat = None
        if self.prev_act in ("FORWARD", "BACKWARD"):
            blocked = self.prev_small is not None and float(np.abs(small - self.prev_small).mean()) < 1.0
            if self.prev_feat is not None:  # moving forward must make the scene expand
                cur_feat = self.matcher.features(obs.image)
                mm = self.matcher.match(self.prev_feat, cur_feat)
                if mm.n >= 15:
                    sc = scale_ratio(mm)
                    blocked = blocked or not (sc > 1.004 if self.prev_act == "FORWARD" else sc < 0.996)
            self.stuck = self.stuck + 1 if blocked else 0
        self.prev_small = small
        self.prev_feat = cur_feat if cur_feat is not None else self.matcher.features(obs.image)

        if self.queue:
            return self._out(self.queue.pop(0), "recover")
        if self.stuck >= 2:
            self.stuck, self.n_stuck = 0, self.n_stuck + 1
            turn = Action.RIGHT if self.n_stuck % 2 else Action.LEFT
            self.queue = [(turn, 26 + 6 * (self.n_stuck % 3))]  # a decisive 90-130 degree turn
            self.mem.punish([self.node], 3.0, 3)  # remember: being here got us stuck
            self.node = -1  # forget where we thought we were
            return self._out((Action.BACKWARD, 4), "stuck")

        q = np.asarray(self.scorer.embed([obs.image]), np.float32)[0]
        p = self._prob_to_all(q)
        node = -1
        if self.node >= 0:  # temporally consistent: look near the last estimate first
            cand = self._local_nodes(self.node)
            cand = cand[self.reach[cand]]
            if len(cand):
                j = int(cand[np.argmax(p[cand])])
                if p[j] >= self.s_local:
                    node = j
        if node < 0:  # global search
            node = int(np.argmax(np.where(self.reach, p, -1.0)))
        self.node = node
        self.mem.arrive(node)
        path = self.mem.path(node, limit=self.geo_hops + 2)  # node, then frames toward the goal
        self.hops = int(self.mem.hops[node])
        if self.hops < self.best_hops - 2:
            self.best_hops, self.best_t = self.hops, self.t
        if self.mem.stagnating() or self.t - self.best_t > 200:
            return self._escape_loop()

        if self.hops <= self.near_hops:
            r = self._approach(obs)
            if r is not None:
                return r
        self.appr_steps = 0

        l = None
        if hasattr(self.scorer, "pick_waypoint"):  # geometric visibility instead of R
            top = min(self.geo_hops, self.hops, len(path) - 1)
            cands = [(k, path[k]) for k in range(top, self.h_min - 1, -1)]
            l = self.scorer.pick_waypoint(obs.image, cands, self._frame_img)
        if l is None:
            top = min(self.h_max, self.hops, len(path) - 1)
            l = min(self.h_min, len(path) - 1) if self.hops >= self.h_min else 0
            for k in range(top, self.h_min - 1, -1):  # furthest waypoint still "reachable"
                if p[path[k]] >= self.s_reach:
                    l = k
                    break
        wp = path[l]
        name = self.scorer.locomote(obs.image, self._frame_img(wp), wp)
        self.state = f"wp +{l} p={p[wp]:.2f} {getattr(self.scorer, 'dbg', '')}"
        if hasattr(self.scorer, "ticks"):
            n = self.scorer.ticks if name != "FORWARD" else self.burst
        else:
            n = self.burst if name in ("FORWARD", "BACKWARD") else 1
        return self._out((Action[name], n), "follow")

    def _escape_loop(self):
        """Circling a small area, or no progress for a long time: remember the area as bad,
        turn most of the way around, and let the planner choose another route."""
        zone = self.mem.escape_zone()
        self.mem.punish(zone, 6.0, 4)
        cut = self.mem.ban_shortcuts(zone)  # the way out we believed in was a look-alike
        if cut:
            self.G.remove_edges_from(self.mem.banned[-cut:])
            self.reach = self.mem.reachable()
            self._local.clear()
        self.n_loops += 1
        self.best_t = self.t  # give the new route time before judging it
        turn = Action.RIGHT if self.n_loops % 2 else Action.LEFT
        self.queue = [(turn, 40)]  # about 138 degrees
        self.node = -1
        print(f"[{self.t}] loop #{self.n_loops}: {len(zone)} nodes marked costly, turning away, {cut} false shortcuts removed")
        return self._out((Action.BACKWARD, 3), "loop-escape")

    def _approach(self, obs: Observation):
        feat = self.matcher.features(obs.image)
        best = None
        for k, tf in enumerate(self.tfeat):
            m = self.matcher.match(tf, feat)
            if best is None or m.n > best[1].n:
                best = (k, m)
        k, m = best
        if m.n < 12:
            return None
        self.appr_steps += 1
        s = shift_deg(m, self.focal)
        left = SIGN * s + VIEW_OFFSET[k]
        r = scale_ratio(m)
        self.state = f"approach view{k} n={m.n} left={left:+.1f} scale={r:.2f}"
        turn = Action.RIGHT if left > 0 else Action.LEFT
        if k != 0 or abs(left) > 4:
            if k != 0 and abs(left) > 20:
                return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 8))), "approach-turn")
            if abs(left) > 4:
                return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 3))), "approach-turn")
        tol = self.scale_tol if self.appr_steps <= 25 else 2 * self.scale_tol
        if r < 1 - tol:
            return self._out((Action.FORWARD, int(np.clip(round((1 / r - 1) * 12), 1, 3))), "approach-fwd")
        if r > 1 + tol:
            return self._out((Action.BACKWARD, int(np.clip(round((r - 1) * 12), 1, 2))), "approach-back")
        if abs(left) <= 4 and (m.n >= 25 or self.appr_steps > 25):
            return self._out(Action.CHECKIN, "arrived")
        return self._out((Action.FORWARD, 1), "approach-nudge")

    def _out(self, a, why: str):
        act = a[0] if isinstance(a, tuple) else a
        self.prev_act = act.name if act in (Action.FORWARD, Action.BACKWARD) else None
        if self.verbose or why in ("arrived", "budget"):
            print(f"[{self.t}] node {self.node} hops {self.hops} {why} {a} {self.state}")
        self.log.append(dict(t=self.t, node=self.node, hops=self.hops, why=why, act=str(a), state=self.state))
        return a

    def hud(self) -> list[str]:
        return [f"SPTM node {self.node} goal {getattr(self, 'goal', -1)} hops {self.hops}", self.state]


def main() -> None:
    import cli

    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data directory")
    parser.add_argument("--r", type=Path, default=Path("models/sptm_r.pt"))
    parser.add_argument("--l", type=Path, default=Path("models/sptm_l.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--classical", action="store_true", help="no trained networks: VLAD + feature matching")
    parser.add_argument("--hybrid", action="store_true", help="VLAD and the trained R must agree (needs --r/--l)")
    parser.add_argument("--no-l", action="store_true", help="with --hybrid: steer by feature matching, not by L")
    parser.add_argument("--geo-min", type=int, default=None, help="RANSAC inliers a shortcut must have")
    parser.add_argument("--burst", type=int, default=1)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--s-local", type=float, default=0.7)
    parser.add_argument("--s-reach", type=float, default=0.8)
    parser.add_argument("--s-shortcut", type=float, default=0.9)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()
    data = cli.exploration_data(args, args.data)
    if args.classical:
        from classical_scorer import ClassicalScorer

        scorer = ClassicalScorer(Path("cache") / f"sptm-{Path(data).name}")
    elif args.hybrid:
        from classical_scorer import ClassicalScorer
        from hybrid_scorer import HybridScorer
        from sptm_models import TorchScorer

        scorer = HybridScorer(
            ClassicalScorer(Path("cache") / f"sptm-{Path(data).name}"),
            TorchScorer(args.r, args.l, args.device), use_l=not args.no_l,
        )
    else:
        from sptm_models import TorchScorer  # needs torch + torchvision

        scorer = TorchScorer(args.r, args.l, args.device)
    geo = args.geo_min if args.geo_min is not None else (22 if (args.classical or args.hybrid) else 0)
    agent = SptmAgent(
        data, scorer, geo_min=geo, burst=args.burst, budget=args.budget, s_local=args.s_local,
        s_reach=args.s_reach, s_shortcut=args.s_shortcut, log_dir=args.log_dir, verbose=args.verbose,
    )
    if args.local is not None:  # local only: print the TRUE distance to the goal afterwards
        import groundtruth

        groundtruth.install()
    run(agent, cli.challenge(args), **cli.run_options(args))
    if args.local is not None:
        print("ground truth:", groundtruth.summary())
        groundtruth.dump(f"logs/gt/local{args.local}.json")


if __name__ == "__main__":
    main()
