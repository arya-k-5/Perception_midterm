"""
NAV v9 -- a NEW file, built on nav_v8.NavV8. Everything else is untouched.

nav_v8 log (maze 7): the goal-photo servo still never fired. The robot came within 0.58 m of the
goal (true), but the best view matched the goal photos with about 24 inliers, one short of the
strict entry bar of 25.

Change: the photo gate uses the sptm_agent threshold for an approach already in progress (12
inliers) instead of the strict entry bar. Still image-driven, still nothing map-specific. The
RANSAC count threshold is a generic matcher setting, not something fitted to this maze.

Risk: a 12-inlier match can be a look-alike view, so the approach may turn toward the wrong view.
The existing appr_lost counter drops the approach after three misses.

    uv run --no-sync source/nav_v9.py --local 7 --classical --budget 3000 --verbose
"""

from __future__ import annotations

from pathlib import Path

import cli
from vis_nav_sdk import Observation, run
from nav_v8 import NavV8


class NavV9(NavV8):
    def act(self, obs: Observation):
        self.t += 1
        left = obs.steps_left if self.budget is None else min(obs.steps_left, self.budget - obs.step)
        if left > 12:
            r = self._approach(obs, weak=True)  # 12 inliers instead of 25
            if r is not None:
                self.appr_on, self.appr_lost = True, 0
                self.state = "photo-gate12 " + self.state
                return r
        self.t -= 1
        return super(NavV8, self).act(obs)  # skip NavV8's gate; NavV7/SptmAgent logic runs


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
    agent = NavV9(
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
