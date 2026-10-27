"""
NAV v10 -- a NEW file, built on nav_v9.NavV9. Everything else is untouched.

nav_v9 log: the 12-inlier photo gate let in weak matches. A side photo (view1) with 12 inliers
gave left=+114 deg, and scale estimates of 0.22 and 1.27 showed up. The robot spun left and
right during the approach, which is the jitter.

Changes (none tied to a maze):
  1. Entry needs ENTER_INLIERS = 20 (was 12). Once an approach is running, KEEP_INLIERS = 12.
  2. Only the front photo (view 0) can start an approach. Side views carry a 90 deg offset,
     so they can only continue one already in progress.
  3. A scale ratio outside 0.6 to 1.6 is treated as no information for that step.
  4. Turns are capped at 3 units (was 8), with a 4 deg dead band.
  5. Check-in needs READY_STEPS = 3 consecutive ready steps (same idea as nav_v5).

    uv run --no-sync source/nav_v10.py --local 7 --classical --budget 3000 --verbose
"""

from __future__ import annotations

from pathlib import Path

import cli
from geometry import scale_ratio, shift_deg
from my_agent import TURN_DEG, VIEW_OFFSET
from vis_nav_sdk import Action, run
from nav_v9 import NavV9

ENTER_INLIERS = 20
KEEP_INLIERS = 12
READY_STEPS = 3
DEAD_BAND_DEG = 4.0


class NavV10(NavV9):
    def _reset(self) -> None:
        super()._reset()
        self.ready = 0

    def _approach(self, obs, weak: bool = False):
        feat = self.matcher.features(obs.image)
        best = None
        for k, tf in enumerate(self.tfeat):
            m = self.matcher.match(tf, feat)
            if best is None or m.n > best[1].n:
                best = (k, m)
        k, m = best
        need = KEEP_INLIERS if self.appr_on else ENTER_INLIERS
        if m.n < need or (not self.appr_on and k != 0):
            self.ready = 0
            return None
        self.appr_steps += 1
        s = shift_deg(m, self.focal)
        left = self.sign * s + VIEW_OFFSET[k]
        r = scale_ratio(m)
        if not (0.6 <= r <= 1.6):
            r = 1.0
        self.state = f"approach view{k} n={m.n} left={left:+.1f} scale={r:.2f} ready={self.ready}"
        if abs(left) > DEAD_BAND_DEG:
            self.ready = 0
            turn = Action.RIGHT if left > 0 else Action.LEFT
            return self._out((turn, int(min(3, max(1, round(abs(left) / TURN_DEG))))), "approach-turn")
        tol = self.scale_tol if self.appr_steps <= 25 else 2 * self.scale_tol
        if r < 1 - tol:
            self.ready = 0
            return self._out((Action.FORWARD, int(min(3, max(1, round((1 / r - 1) * 12))))), "approach-fwd")
        if r > 1 + tol:
            self.ready = 0
            return self._out((Action.BACKWARD, int(min(2, max(1, round((r - 1) * 12))))), "approach-back")
        if m.n >= 25 or self.appr_steps > 25:
            self.ready += 1
            if self.ready >= READY_STEPS:
                return self._out(Action.CHECKIN, "arrived")
            return self._out((Action.FORWARD, 1), "approach-hold")
        self.ready = 0
        return self._out((Action.FORWARD, 1), "approach-nudge")


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
    agent = NavV10(
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
