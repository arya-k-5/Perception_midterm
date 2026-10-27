"""
NAV v5 -- a NEW agent (my_agent.py, sptm_agent*.py and everything else are untouched).

Built on my_agent.MyAgent (Bayes localisation over the exploration frames, shortest path on
the topological map, goal-photo servo with RANSAC, then CHECKIN). Four changes, none of them
tied to a particular maze:

  1. Dead-end forgetting. A blocked forward move (image barely changes) removes the one map
     edge we were trying to use for DEAD_END_TICKS ticks, re-plans without it, then restores
     it. The planner stops re-routing through the same wall, and the edge gets retried later.
  2. Image-matching check on the localisation. The chosen node must also match the camera
     frame with at least MIN_NODE_INLIERS RANSAC inliers. Two weak steps in a row reset the
     belief instead of trusting it.
  3. Fewer ticks at the start. Sign calibration uses 3 LEFT + 3 RIGHT ticks (was 6 + 6).
  4. Stable check-in. The goal-photo servo must report "ready" for READY_STEPS steps in a
     row before CHECKIN, so one lucky frame cannot end the run early.

    uv run --no-sync source/nav_v5.py --local 7 --verbose --budget 3000
    uv run --no-sync source/nav_v5.py --challenge <id>
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import cli
from geometry import scale_ratio, shift_deg
from my_agent import TURN_DEG, VIEW_OFFSET, MyAgent
from vis_nav_sdk import Action, Observation, run

DEAD_END_TICKS = 60  # how long a blocked edge stays forgotten
READY_STEPS = 3  # servo must be ready this many steps in a row before CHECKIN
MIN_NODE_INLIERS = 15  # geometric support required for the localised node
WEAK_LIMIT = 2  # consecutive weak localisations before the belief is reset


class NavV5(MyAgent):
    def _reset(self) -> None:
        super()._reset()
        self.removed: list[tuple[int, int, int, dict]] = []  # (restore_at, u, v, edge data)
        self.ready = 0
        self.weak_loc = 0
        self.node_inl = -1

    # -- map edits (dead-end forgetting)

    def _forget_dead_end(self) -> None:
        cur = self.node
        if cur < 0:
            return
        nxt = int(self.topo.next_hop[cur])
        if nxt < 0 or self.topo.graph[cur][nxt].get("visual"):
            return  # only forget a plain drive edge, never a shortcut here
        data = dict(self.topo.graph[cur][nxt])
        self.topo.graph.remove_edge(cur, nxt)
        self.removed.append((self.t + DEAD_END_TICKS, cur, nxt, data))
        self.topo.plan_to(self.goal)
        print(f"[{self.t}] dead end at node {cur}: forgetting edge {cur}->{nxt} for {DEAD_END_TICKS} ticks")

    def _restore_edges(self) -> None:
        keep, restored = [], False
        for at, u, v, data in self.removed:
            if self.t >= at:
                self.topo.graph.add_edge(u, v, **data)
                restored = True
            else:
                keep.append((at, u, v, data))
        self.removed = keep
        if restored:
            self.topo.plan_to(self.goal)

    # -- localisation check

    def _verify_node(self, obs: Observation) -> None:
        m = self.matcher.match(self._node_feat(self.node), self.matcher.features(obs.image))
        self.node_inl = m.n
        if m.n < MIN_NODE_INLIERS:
            self.weak_loc += 1
            if self.weak_loc >= WEAK_LIMIT:
                self.belief.reset()
                self.weak_loc = 0
        else:
            self.weak_loc = 0

    # -- decisions

    def act(self, obs: Observation):
        self.t += 1
        self._restore_edges()
        left = obs.steps_left if self.budget is None else min(obs.steps_left, self.budget - obs.step)
        if left <= 12:
            return self._out(Action.CHECKIN, "budget")
        small = cv2.resize(cv2.cvtColor(obs.image, cv2.COLOR_BGR2GRAY), (40, 30)).astype(np.float32)

        if self.phase == "calib0":
            self.calib_img, self.phase = obs.image, "calib1"
            return self._out((Action.LEFT, 3), "calib")
        if self.phase == "calib1":
            m = self.matcher.match(self.matcher.features(self.calib_img), self.matcher.features(obs.image))
            s = shift_deg(m, self.focal)
            if m.n >= 8 and abs(s) > 3:
                self.sign = 1 if s > 0 else -1
            print(f"calibration: shift {s:+.1f} deg after LEFT x3 ({m.n} matches) -> sign {self.sign:+d}")
            self.phase = "calib2"
            return self._out((Action.RIGHT, 3), "calib")
        self.phase = "nav"

        stuck_now = (
            self.prev_act in ("FORWARD", "BACKWARD")
            and self.prev_small is not None
            and float(np.abs(small - self.prev_small).mean()) < 1.0
        )
        if self.prev_act in ("FORWARD", "BACKWARD"):
            self.stuck = self.stuck + 1 if stuck_now else 0
        self.prev_small = small

        q = self.desc.extract(obs.image)
        self.node = self.belief.update(q)
        self._verify_node(obs)
        self.hops = self.topo.hops(self.node)

        if self.queue:
            return self._out(self.queue.pop(0), "recover")
        if self.stuck >= 2:
            self.stuck, self.n_stuck = 0, self.n_stuck + 1
            self._forget_dead_end()
            turn = Action.RIGHT if self.n_stuck % 2 else Action.LEFT
            self.queue = [(turn, min(8 + 3 * (self.n_stuck % 4), 20))]
            self.belief.reset()
            return self._out((Action.BACKWARD, 4), "stuck")

        if self.hops <= self.near_hops:
            r = self._approach(obs)
            if r is not None:
                return r
        self.appr_steps = 0
        self.ready = 0
        self._check_jump()
        cost = float(self.topo.dist[self.node])
        if cost < self.best_cost - 2:
            self.best_cost, self.best_t = cost, self.t
        if self.unstick > 0:
            self.unstick -= 1
            return self._out((Action.FORWARD, 2), "unstick")
        if self.t - self.best_t > 60:
            self.best_t, self.unstick = self.t, 24
            self.belief.reset()
        name = self.topo.action_toward_goal(self.node, prefer=self.last_follow)
        self.last_follow = name
        if name in ("FORWARD", "BACKWARD"):
            steer = self._steer(obs, name)
            if steer is not None:
                return steer
        n = self.burst if name in ("FORWARD", "BACKWARD") else 2
        return self._out((Action[name], n), "follow")

    def _approach(self, obs: Observation):
        feat = self.matcher.features(obs.image)
        best = None
        for k, tf in enumerate(self.tfeat):
            m = self.matcher.match(tf, feat)
            if best is None or m.n > best[1].n:
                best = (k, m)
        k, m = best
        if m.n < 12:
            self.ready = 0
            return None
        self.appr_steps += 1
        s = shift_deg(m, self.focal)
        left = self.sign * s + VIEW_OFFSET[k]
        r = scale_ratio(m)
        self.state = f"approach view{k} n={m.n} left={left:+.1f} scale={r:.2f} ready={self.ready}"
        turn = Action.RIGHT if left > 0 else Action.LEFT
        if k != 0 or abs(left) > 4:
            self.ready = 0
            if k != 0 and abs(left) > 20:
                return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 8))), "approach-turn")
            if abs(left) > 4:
                return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 3))), "approach-turn")
        tol = self.scale_tol if self.appr_steps <= 25 else 2 * self.scale_tol
        if r < 1 - tol:
            self.ready = 0
            return self._out((Action.FORWARD, int(np.clip(round((1 / r - 1) * 12), 1, 3))), "approach-fwd")
        if r > 1 + tol:
            self.ready = 0
            return self._out((Action.BACKWARD, int(np.clip(round((r - 1) * 12), 1, 2))), "approach-back")
        if abs(left) <= 4 and (m.n >= 25 or self.appr_steps > 25):
            self.ready += 1
            if self.ready >= READY_STEPS:
                return self._out(Action.CHECKIN, "arrived")
            return self._out((Action.FORWARD, 1), "approach-hold")
        self.ready = 0
        return self._out((Action.FORWARD, 1), "approach-nudge")


def main() -> None:
    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data directory")
    parser.add_argument("--descriptor", default="vlad", choices=["vlad", "dino", "both"])
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--head", type=Path, default=None)
    parser.add_argument("--burst", type=int, default=1)
    parser.add_argument("--budget", type=int, default=None, help="own step cap, e.g. 3000")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()
    data = cli.exploration_data(args, args.data)
    agent = NavV5(
        data, descriptor=args.descriptor, subsample=args.subsample, head=args.head, burst=args.burst,
        tau=args.tau, log_dir=args.log_dir, verbose=args.verbose, budget=args.budget,
    )
    run(agent, cli.challenge(args), **cli.run_options(args))


if __name__ == "__main__":
    main()
