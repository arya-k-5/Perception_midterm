"""
my_agent_1820lg.py -- my_agent_1820g (Kartik's 1820 + 4-photo goal check) with LightGlue instead of SuperGlue for
EVERY match (shortcut checks, localisation jumps, steering angles, goal approach). NEW FILE.

One-time install (does not touch torch):
    uv pip install --python /Users/arya/vis_nav_player_work/.venv/bin/python git+https://github.com/cvg/LightGlue.git

    uv run source/my_agent_1820lg.py --local 17 --budget 4000

At start-up it prints a calibration: inliers of SuperGlue vs LightGlue on the same exploration photo pairs (same
place / different place). All thresholds in the agent (steer 40, jump 40, goal 20, ...) were tuned on SuperGlue
counts; if LightGlue counts run much higher on DIFFERENT places too, those thresholds let in more false matches.
"""

from __future__ import annotations

import sys

import numpy as np
import sg_matcher
from lg_matcher import LGMatcher
from my_agent import with_defaults
from my_agent_1820g import FLAGS
from my_agent_2767g import GoalAgent
from sptm_agent import main


class LGAgent(GoalAgent):
    def setup(self, info) -> None:
        super().setup(info)
        if not getattr(self, "_calibrated", False):
            self._calibrated = True
            try:
                self._calibrate()
            except Exception as e:  # noqa: BLE001
                print(f"calibration skipped ({e!r})")

    def _calibrate(self) -> None:
        sg = _ORIG_SG(device=None)
        rng = np.random.default_rng(0)
        same = [(i, i + 1) for i in rng.choice(self.n - 2, 15, replace=False) if self.traj[i] == self.traj[i + 1]]
        diff = [tuple(int(x) for x in rng.choice(self.n, 2, replace=False)) for _ in range(15)]
        diff = [(a, b) for a, b in diff if abs(a - b) > 200]

        def count(m, pairs):
            out = []
            for a, b in pairs:
                fa, fb = m.features(self._frame_img(a)), m.features(self._frame_img(b))
                out.append(m.match(fa, fb, self.focal).n)
            return np.array(out)

        s_sg, s_lg = count(sg, same), count(self.gm, same)
        d_sg, d_lg = count(sg, diff), count(self.gm, diff)
        print(f"CALIBRATION inliers  same place: SuperGlue median {np.median(s_sg):.0f}, LightGlue {np.median(s_lg):.0f} | "
              f"different places: SuperGlue median {np.median(d_sg):.0f} (max {d_sg.max()}), "
              f"LightGlue {np.median(d_lg):.0f} (max {d_lg.max()})")


_ORIG_SG = sg_matcher.SGMatcher

if __name__ == "__main__":
    sg_matcher.SGMatcher = LGMatcher  # sptm_agent.main() imports SGMatcher from sg_matcher at call time
    argv = sys.argv[1:]
    GoalAgent.goal_check = "--no-goal-check" not in argv
    argv = [a for a in argv if a != "--no-goal-check"]
    sys.argv[1:] = with_defaults(argv, flags=FLAGS, log_dir="logs/my_agent_1820lg")
    main(LGAgent)
