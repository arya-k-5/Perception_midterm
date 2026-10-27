"""
NAV v8 -- a NEW file, built on nav_v7.NavV7 (which is built on sptm_agent.SptmAgent).
sptm_agent.py, nav_v7.py and everything else are untouched.

Problem found in the nav_v7 logs: the goal-photo approach never ran (0 "approach" lines). In
sptm_agent the approach is only tried when the MAP says the goal is within 40 hops, but the
localisation was lost most of the time and the hop count read 150 to 491, even when the robot
was about 1.9 m from the goal.

Change: the goal photos decide, not the map. Every step, before the map logic, the photo
servo is tried with its strict entry test (25 RANSAC inliers against one of the four goal
photos). If it fires, it drives the robot; if not, the normal map logic runs as before.
Nothing here depends on a particular maze.

Caveat: when the photo servo takes over, the stuck detector is not updated for that step.

    uv run --no-sync source/nav_v8.py --local 7 --classical --budget 3000 --verbose
"""

from __future__ import annotations

from pathlib import Path

import cli
from vis_nav_sdk import Action, Observation, run
from nav_v7 import NavV7


class NavV8(NavV7):
    def act(self, obs: Observation):
        self.t += 1
        left = obs.steps_left if self.budget is None else min(obs.steps_left, self.budget - obs.step)
        if left > 12:
            r = self._approach(obs)  # strict entry: needs 25 inliers unless already approaching
            if r is not None:
                self.appr_on, self.appr_lost = True, 0
                self.state = "photo-gate " + self.state
                return r
        self.t -= 1  # super().act() counts this tick itself
        return super().act(obs)


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
    agent = NavV8(
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
