"""
NAV v11 -- a NEW file, built on nav_v10.NavV10. Everything else is untouched.

Problem: the SptmAgent stuck handler only adds a cost penalty to the stuck node. The planner keeps
the wall-facing drive edge, so the robot walks back into the same dead end (seen on the mock
challenge). Earlier versions (nav_v5/v6) had dead-end forgetting, but those sat on the Bayes path,
not the classical SPTM path.

Change: at a stuck event, remove the drive edge from the last KNOWN node toward the next node the
planner wanted. That edge is forgotten for DEAD_END_TICKS * n ticks, where n counts repeats of the
same edge. Then it is restored. The planner's reachable set is refreshed after each change.

Why the last known node: the estimated node is often -1 at a stuck event (see the logs), so the
blocked edge is taken from PlaceMemory.last instead.

Note: the stuck handler calls mem.punish([node], 3.0, 3). The hook below runs only for that call,
because it is the only caller with amount 3.0 and a single node.

    uv run --no-sync source/nav_v11.py --local 7 --classical --budget 3000 --verbose
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cli
from vis_nav_sdk import Observation, run
from nav_v10 import NavV10
from sptm_agent import SIGN

DEAD_END_TICKS = 60
STUCK_PENALTY = 3.0  # the amount sptm_agent.act passes to mem.punish on a stuck event


class NavV11(NavV10):
    def __init__(self, data_dir: Path, scorer, **kw) -> None:
        super().__init__(data_dir, scorer, **kw)
        self.sign = SIGN  # nav_v10's approach reads self.sign; SPTM keeps it as a module constant
        orig_punish = self.mem.punish

        def punish(nodes, amount, radius=3):
            nodes = list(nodes)
            orig_punish(nodes, amount, radius)
            if amount == STUCK_PENALTY and len(nodes) == 1:
                self._forget_blocked_edge()

        self.mem.punish = punish
        self.forgotten: list[tuple[int, int, int]] = []  # (restore_at, a, b)
        self.forget_count: dict[tuple[int, int], int] = {}

    def _refresh(self) -> None:
        self.mem.set_goal(self.mem.goal)
        self.reach = self.mem.reachable()

    def _forget_blocked_edge(self) -> None:
        base = self.mem.last
        if base < 0:
            return
        path = self.mem.path(base, limit=2)
        if len(path) < 2:
            return
        a, b = sorted((int(path[0]), int(path[1])))
        if b - a != 1:  # only plain drive edges; shortcuts are handled by ban_shortcuts
            return
        mask = (self.mem.u == a) & (self.mem.v == b)
        if not mask.any():
            return
        n = self.forget_count.get((a, b), 0) + 1
        self.forget_count[(a, b)] = n
        self.mem.u, self.mem.v = self.mem.u[~mask], self.mem.v[~mask]
        self.forgotten.append((self.t + DEAD_END_TICKS * n, a, b))
        self._refresh()
        print(f"[{self.t}] dead end: forgetting drive edge {a}-{b} for {DEAD_END_TICKS * n} ticks (time {n})")

    def _restore(self) -> None:
        keep, restored = [], False
        for at, a, b in self.forgotten:
            if self.t >= at:
                self.mem.u = np.append(self.mem.u, np.int64(a))
                self.mem.v = np.append(self.mem.v, np.int64(b))
                restored = True
            else:
                keep.append((at, a, b))
        self.forgotten = keep
        if restored:
            self._refresh()

    def act(self, obs: Observation):
        self._restore()
        return super().act(obs)

    def finish(self, result) -> None:
        print(f"dead-end forgets: {sum(self.forget_count.values())}")
        super().finish(result)


def main() -> None:
    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data directory")
    parser.add_argument("--classical", action="store_true", help="VLAD + feature matching (the only scorer wired here)")
    parser.add_argument("--shortcuts", action="store_true", help="use visual shortcuts (default off)")
    parser.add_argument("--burst", type=int, default=1)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--s-local", type=float, default=0.7)
    parser.add_argument("--s-reach", type=float, default=0.8)
    parser.add_argument("--s-shortcut", type=float, default=0.9)
    parser.add_argument("--geo-min", type=int, default=22, help="RANSAC inliers a shortcut must have")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()
    data = cli.exploration_data(args, args.data)
    from classical_scorer import ClassicalScorer

    scorer = ClassicalScorer(Path("cache") / f"sptm-{Path(data).name}")
    agent = NavV11(
        data, scorer, use_shortcuts=args.shortcuts, geo_min=args.geo_min, burst=args.burst,
        budget=args.budget, s_local=args.s_local, s_reach=args.s_reach, s_shortcut=args.s_shortcut,
        log_dir=args.log_dir, verbose=args.verbose,
    )
    if args.local is not None:
        import groundtruth

        groundtruth.install()
    run(agent, cli.challenge(args), **cli.run_options(args))
    if args.local is not None:
        print("ground truth:", groundtruth.summary())
        groundtruth.dump(f"logs/gt/local{args.local}.json")


if __name__ == "__main__":
    main()
