"""
NAV v7 -- a NEW file, built on sptm_agent.SptmAgent. sptm_agent.py and everything else are untouched.

Why this base: sptm_agent.py (classical scoring, no torch) is the version that scored 5/10 on
maze 7 in our earlier runs. v5 and v6 were built on the Bayes filter and failed locally.

Changes, none tied to a particular maze:
  1. Shortcut switch. --shortcuts turns visual shortcuts on. Default is OFF: the planner uses only
     the recorded drives. Run both to see whether shortcuts help or hurt on the same maze.
  2. Jitter counter. A turn that undoes the previous turn within 3 ticks is counted and printed at
     the end, so runs can be compared.

    uv run --no-sync source/nav_v7.py --local 7 --classical --budget 3000 --verbose
    uv run --no-sync source/nav_v7.py --local 7 --classical --shortcuts --budget 3000 --verbose
    uv run --no-sync source/nav_v7.py --challenge <id> --classical
"""

from __future__ import annotations

from pathlib import Path

import cli
from vis_nav_sdk import Action, Result, run
from sptm_agent import SptmAgent


class NavV7(SptmAgent):
    def __init__(self, data_dir: Path, scorer, *, use_shortcuts: bool = False, **kw) -> None:
        self.use_shortcuts = use_shortcuts  # read by _shortcuts(), which runs inside super().__init__
        super().__init__(data_dir, scorer, **kw)

    def _shortcuts(self, gap: int, k: int, window: int, thr: float):
        if not self.use_shortcuts:
            print("shortcuts: OFF (sequential drives only)")
            return []
        return super()._shortcuts(gap, k, window, thr)

    def _reset(self) -> None:
        super()._reset()
        self.reversals = 0
        self.last_turn = None

    def _out(self, a, why: str):
        act = a[0] if isinstance(a, tuple) else a
        if act in (Action.LEFT, Action.RIGHT):
            if self.last_turn is not None and act != self.last_turn[0] and self.t - self.last_turn[1] <= 3:
                self.reversals += 1
            self.last_turn = (act, self.t)
        return super()._out(a, why)

    def finish(self, result: Result | None) -> None:
        print(f"jitter: {self.reversals} turn reversals (shortcuts {'on' if self.use_shortcuts else 'off'})")
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
    agent = NavV7(
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
