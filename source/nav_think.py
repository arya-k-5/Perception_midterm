"""
nav_think.py -- nav_merge.py (Kartik's my_agent_2767 + lost detector + damper) with a "think, then
commit" controller. NEW FILE; nothing existing is edited.

Why the robot goes left-right-left in the base agent (read from sptm_agent.py and the maze-8 log):
  1. The waypoint it steers to changes every decision. The agent picks "the furthest of the next 7 photos
     that still looks reachable" (p >= s_reach) afresh each step; near a corner that is +5 one step and
     +3 the next, and the measured angle to them differs by 30 deg (log: "wp +5 yaw=-45" then
     "wp +3 yaw=-14"). The commanded heading flips, so the robot turns, un-turns, turns.
  2. A single noisy measurement can order a turn straight after the opposite turn.
  3. Turn-search and goal-scan spins are long (up to 105 ticks) and always start the same way.

What this file does about it (ideas: lecture 5 robust estimation, lecture 10 filtering, lecture 8 loop closure):
  A. COMMITTED TARGET (pure-pursuit look-ahead): once a target photo is chosen it stays the target until it
     is reached; the look-ahead may grow by at most COMMIT_ADVANCE photos per decision and never jumps back.
     The heading the robot aims for therefore changes smoothly instead of flipping.
  B. SETTLE DEAD-BAND: right after a measured turn the angle must exceed SETTLE_DEG (not STEER_DEG) to
     order another turn -- the robot drives on and measures again instead of correcting a leftover few degrees.
  C. FULL REVERSAL CONFIRMATION: a turn that would undo a turn from a moment ago needs to be asked for twice
     in a row, for turns of up to 6 ticks (nav_merge only did 3), also for the look-around turns.
  D. SHORTER, DIRECTED SEARCH: when the target photo is out of view the robot looks around for at most
     SEARCH_TICKS_NEW ticks, after SEARCH_AFTER_NEW decisions out of view (base: 105 and 4).
  E. OPTIONAL --switch-cost X: the base charges 4.0 for every shortcut; a lower value takes more of the
     verified shortcuts (shorter route). Default 4.0 = unchanged.
  F. Everything nav_merge has: lost detector, jitter counter, "ticks by reason" printout (shows where the
     ticks go; send this line back).

    uv run source/nav_think.py --local 8 --budget 3000
    options (all optional): --no-commit --no-settle --no-search-cut --no-damp --switch-cost 2.0
"""

from __future__ import annotations

import sys

import sptm_agent as base
from my_agent import with_defaults
from nav_merge import FLAGS, MergeAgent, with_budget
from sptm_agent import main
from vis_nav_sdk import Action

COMMIT_ADVANCE = 2  # the look-ahead target may move this many photos further per decision
SETTLE_DEG = 16.0  # dead-band right after a measured turn (base: STEER_DEG = 10)
SEARCH_TICKS_NEW = 60  # base: 105
SEARCH_AFTER_NEW = 5  # base: 4


class ThinkAgent(MergeAgent):
    commit_on = True
    settle_on = True
    max_damp_ticks = 6
    damp_why = ("follow", "steer-turn", "wp-search")

    def _reset(self) -> None:
        super()._reset()
        self.tgt = -1  # committed target photo (a node id on the route)
        self.t_commits = 0

    def _geo_steer(self, obs, wp, l):
        if self.commit_on and self.node >= 0:
            path = self.paths[self.node][::-1]  # node ... goal
            if self.tgt in path[1:]:
                j = path.index(self.tgt)  # where the committed target is now
                l2 = max(j, min(l, j + COMMIT_ADVANCE))
                l2 = min(l2, len(path) - 1)
            else:
                l2 = l
            if l2 != l:
                self.t_commits += 1
            l, wp = l2, path[l2]
            self.tgt = wp
        old = base.STEER_DEG
        if self.settle_on and self.log and self.log[-1]["why"] == "steer-turn":
            base.STEER_DEG = SETTLE_DEG  # parent reads this module constant at call time
        try:
            r = super()._geo_steer(obs, wp, l)
        finally:
            base.STEER_DEG = old
        if r is None:
            self.tgt = -1  # the target is not in view (or too weak): do not hold on to it
        return r

    def finish(self, result) -> None:
        print(f"think: target kept (differs from the base's pick) in {self.t_commits} decisions")
        super().finish(result)


def _pop(argv: list[str], flag: str) -> bool:
    if flag in argv:
        argv.remove(flag)
        return True
    return False


def _pop_val(argv: list[str], flag: str, default: float) -> float:
    if flag in argv:
        i = argv.index(flag)
        v = float(argv[i + 1])
        del argv[i:i + 2]
        return v
    return default


if __name__ == "__main__":
    argv = sys.argv[1:]
    ThinkAgent.commit_on = not _pop(argv, "--no-commit")
    ThinkAgent.settle_on = not _pop(argv, "--no-settle")
    ThinkAgent.damp_on = not _pop(argv, "--no-damp")
    ThinkAgent.lost_on = not _pop(argv, "--no-lost")
    if not _pop(argv, "--no-search-cut"):
        base.SEARCH_TICKS, base.SEARCH_AFTER = SEARCH_TICKS_NEW, SEARCH_AFTER_NEW
    base.SWITCH_COST = _pop_val(argv, "--switch-cost", base.SWITCH_COST)  # read when the graph is built
    sys.argv[1:] = with_defaults(with_budget(argv), flags=FLAGS, log_dir="logs/nav_think")
    main(ThinkAgent)
