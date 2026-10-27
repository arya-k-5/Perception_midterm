"""
NAV v12 -- a NEW file, built on nav_v11.NavV11. Everything else is untouched.

Changes:
  1. Permanent ban on a repeat dead end. The first time a drive edge blocks the robot, it is
     forgotten for DEAD_END_TICKS (60). If the same edge blocks it again, it is banned for the
     rest of the run. In nav_v11 the escalating timeouts (60, 120, ... 300) only delayed the
     robot, which came back to the same wall each time.
  2. Distance over time (local runs only). At the end, the closest true distance to the goal is
     printed for each 500-tick window, from the groundtruth trace. Nothing is printed on a server
     run, where there is no ground truth.

    uv run --no-sync source/nav_v12.py --local 7 --classical --budget 3000 --verbose
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cli
from vis_nav_sdk import run
from nav_v11 import NavV11, DEAD_END_TICKS

NEVER = 10**9


class NavV12(NavV11):
    def _forget_blocked_edge(self) -> None:
        base = self.mem.last
        if base < 0:
            return
        path = self.mem.path(base, limit=2)
        if len(path) < 2:
            return
        a, b = sorted((int(path[0]), int(path[1])))
        if b - a != 1:
            return
        mask = (self.mem.u == a) & (self.mem.v == b)
        if not mask.any():
            return
        n = self.forget_count.get((a, b), 0) + 1
        self.forget_count[(a, b)] = n
        self.mem.u, self.mem.v = self.mem.u[~mask], self.mem.v[~mask]
        if n >= 2:
            self.forgotten.append((NEVER, a, b))
            note = "banned for the rest of the run"
        else:
            self.forgotten.append((self.t + DEAD_END_TICKS, a, b))
            note = f"forgotten for {DEAD_END_TICKS} ticks"
        self._refresh()
        print(f"[{self.t}] dead end: drive edge {a}-{b} {note} (time {n})")

    def finish(self, result) -> None:
        super().finish(result)
        try:
            import groundtruth
        except ImportError:
            return
        trace = groundtruth.TRACE
        if not trace:
            return
        print("distance over time (true, closest per 500 ticks):")
        for start in range(0, trace[-1][0] + 1, 500):
            window = [d for t, d, _ in trace if start <= t < start + 500]
            if window:
                print(f"  ticks {start:>5}-{start + 499:<5}  closest {min(window):5.2f} m")


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
    agent = NavV12(
        data, scorer, use_shortcuts=args.shortcuts, geo_min=args.geo_min, burst=args.burst,
        budget=args.budget, s_local=args.s_local, s_reach=args.s_reach, s_shortcut=args.s_shortcut,
        log_dir=args.log_dir, verbose=args.verbose,
    )
    if args.local is not None:
        import groundtruth

        groundtruth.install()
    run(agent, cli.challenge(args), **cli.run_options(args))
    if args.local is not None:
        import groundtruth

        print("ground truth:", groundtruth.summary())
        groundtruth.dump(f"logs/gt/local{args.local}.json")


if __name__ == "__main__":
    main()
