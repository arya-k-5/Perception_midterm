"""
NAV v6 -- a NEW file. nav_v5.py, my_agent.py and everything else are untouched.

Built on nav_v5.NavV5. Changes, none tied to a particular maze:

  1. Sequential map only. Visual shortcuts are removed from the graph after it loads. In v5
     the planner kept routing through shortcuts that did not take ("did not take" ~12 times).
  2. Localisation hysteresis. If the belief jumps more than JUMP_IDX frames away from the
     previous node, the jump is accepted only when the previous node no longer matches the
     camera (fewer than MIN_NODE_INLIERS). v5 flipped between far-apart nodes (790, 1252, 2008).
  3. Steering hysteresis. A heading correction needs an error above STEER_DEG on the same side
     for STEER_CONFIRM readings in a row. Per-frame shift noise was the source of the jitter.
  4. Stronger preference for the current move: the planner's margin goes from 2.5 to 4.
  5. Escalating dead-end forgetting. The n-th time the same drive edge is found blocked, it stays
     forgotten for DEAD_END_TICKS * n ticks (v5 always used 60, and the same wall came back).
  6. Jitter counter. A turn that undoes the previous turn within 3 ticks is counted and printed
     at the end, so runs can be compared.

    uv run --no-sync source/nav_v6.py --local 7 --verbose --budget 3000
    uv run --no-sync source/nav_v6.py --challenge <id>
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import cli
from geometry import scale_ratio, shift_deg
from my_agent import TURN_DEG
from nav_v5 import DEAD_END_TICKS, MIN_NODE_INLIERS, NavV5
from vis_nav_sdk import Action, Observation, Result, run

JUMP_IDX = 5  # frames: a belief jump larger than this needs the old node to fail first
STEER_DEG = 8.0  # heading error that can trigger a correction
STEER_CONFIRM = 2  # consecutive same-side readings before a correction fires


class NavV6(NavV5):
    def __init__(self, data_dir: Path, **kw) -> None:
        super().__init__(data_dir, **kw)
        n_before = self.topo.graph.number_of_edges()
        visual = [(u, v) for u, v, d in self.topo.graph.edges(data=True) if d.get("visual")]
        self.topo.graph.remove_edges_from(visual)
        print(f"sequential map: removed {len(visual)} of {n_before} edges (visual shortcuts)")
        # Stronger preference for the current move (TopoMap default margin is 2.5).
        original = self.topo.action_toward_goal
        self.topo.action_toward_goal = lambda node, prefer=None, margin=4.0: original(
            node, prefer=prefer, margin=margin
        )

    def _reset(self) -> None:
        super()._reset()
        self.steer_side, self.steer_run = 0, 0
        self.forget_count: dict[tuple[int, int], int] = {}
        self.last_node = -1
        self.jump_rejects = 0
        self.reversals = 0
        self.last_turn: tuple[Action, int] | None = None

    # -- localisation

    def _verify_node(self, obs: Observation) -> None:
        feat = self.matcher.features(obs.image)
        cand, prev = self.node, self.last_node
        m = self.matcher.match(self._node_feat(cand), feat)
        if prev >= 0 and cand >= 0 and abs(cand - prev) > JUMP_IDX:
            mp = self.matcher.match(self._node_feat(prev), feat)
            if mp.n >= MIN_NODE_INLIERS:
                self.node, self.node_inl = prev, mp.n
                self.jump_rejects += 1
                self.last_node = prev
                self.state = f"kept node {prev} (belief said {cand}, {m.n} inl)"
                return
        self.node_inl = m.n
        if m.n < MIN_NODE_INLIERS:
            self.weak_loc += 1
            if self.weak_loc >= 2:
                self.belief.reset()
                self.weak_loc = 0
        else:
            self.weak_loc = 0
        self.last_node = self.node

    # -- dead ends: escalate the forgetting time

    def _forget_dead_end(self) -> None:
        cur = self.node
        if cur < 0:
            return
        nxt = int(self.topo.next_hop[cur])
        if nxt < 0 or self.topo.graph[cur][nxt].get("visual"):
            return
        key = (cur, nxt)
        n = self.forget_count.get(key, 0) + 1
        self.forget_count[key] = n
        data = dict(self.topo.graph[cur][nxt])
        self.topo.graph.remove_edge(cur, nxt)
        self.removed.append((self.t + DEAD_END_TICKS * n, cur, nxt, data))
        self.topo.plan_to(self.goal)
        print(f"[{self.t}] dead end at node {cur}: forgetting edge {cur}->{nxt} "
              f"for {DEAD_END_TICKS * n} ticks (time {n})")

    # -- steering with hysteresis

    def _steer(self, obs: Observation, name: str):
        tgt, cur = self.node, self.node
        for _ in range(self.lookahead):
            nxt = int(self.topo.next_hop[cur])
            if nxt < 0 or self.topo.edge_action(cur, nxt) != name:
                break
            tgt, cur = nxt, nxt
        if tgt == self.node:
            self.steer_run = 0
            return None
        m = self.matcher.match(self._node_feat(tgt), self.matcher.features(obs.image))
        if m.n < 10:
            self.steer_run = 0
            return None
        left = self.sign * shift_deg(m, self.focal)
        self.state = f"steer n={m.n} left={left:+.1f} run={self.steer_run}"
        side = 0 if abs(left) <= STEER_DEG else (1 if left > 0 else -1)
        if side == 0:
            self.steer_run = 0
            return None
        if side == self.steer_side:
            self.steer_run += 1
        else:
            self.steer_side, self.steer_run = side, 1
        if self.steer_run < STEER_CONFIRM:
            return None
        self.steer_run = 0
        turn = Action.RIGHT if left > 0 else Action.LEFT
        return self._out((turn, int(np.clip(round(abs(left) / TURN_DEG), 1, 3))), "steer")

    # -- logging and jitter count

    def _out(self, a, why: str):
        act = a[0] if isinstance(a, tuple) else a
        if act in (Action.LEFT, Action.RIGHT):
            if self.last_turn is not None and act != self.last_turn[0] and self.t - self.last_turn[1] <= 3:
                self.reversals += 1
            self.last_turn = (act, self.t)
        return super()._out(a, why)

    def finish(self, result: Result | None) -> None:
        print(f"jitter: {self.reversals} turn reversals, {self.jump_rejects} belief jumps rejected, "
              f"{sum(self.forget_count.values())} dead-end forgets")
        super().finish(result)


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
    agent = NavV6(
        data, descriptor=args.descriptor, subsample=args.subsample, head=args.head, burst=args.burst,
        tau=args.tau, log_dir=args.log_dir, verbose=args.verbose, budget=args.budget,
    )
    run(agent, cli.challenge(args), **cli.run_options(args))


if __name__ == "__main__":
    main()
