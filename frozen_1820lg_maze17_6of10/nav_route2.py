"""
nav_route2.py -- nav_route.py (map first, ticks-priced route, spin guard, lost detector) with three anti-jitter
rules, all taken from runs that reached the goal or from a failure seen in a log.  NEW FILE; nothing edited.

  1. SETTLE DEAD-BAND (from nav_think, on in the maze-8 run that reached the goal in 2,630 ticks): right after a
     measured turn, the next turn needs > 16 deg (not 10) -- the robot drives on and measures again instead of
     correcting the few degrees left over, which was a left-right-left source.
  2. WIDER REVERSAL DAMPER (same run): a turn that undoes a turn from a moment ago must be asked for twice in a
     row, for turns up to 6 ticks, including the look-around turns.
  3. ROUTE LOCK (fix for maze 10, where the robot flipped between two drives at a junction for 700 decisions):
     every decision SuperGlue checks the planned route's photos around the current one; one matching with
     >= 40 inliers is where we are, however similar another place looks. No new links mid-run either.
  Directed steering is off (it made maze 52 worse).

    uv run source/nav_route2.py --local 10 --budget 4000
    options: --no-settle --no-lock, plus nav_route's --no-route --no-damp --no-lost --no-spin
"""

from __future__ import annotations

import re
import sys

import sptm_agent as base
from vis_nav_sdk import Action
from my_agent import with_defaults
from nav_merge import FLAGS, with_budget
from nav_route import RouteAgent
from sptm_agent import main

SETTLE_DEG = 16.0
# 4. LAST-METRE DAMPER (maze 10, steps 603-619: at the goal with 90-118 inliers on the front photo, the angle
#    read -25, +22, -22, +22, -25 ... after each 3-tick turn, and it checked in on one reading at scale 0.88 -> 0 pts).
#    A turn that reverses the previous approach turn first waits one tick (IDLE) and measures again; and a
#    check-in needs the previous reading to be inside the gate too.
GATE_DEG, GATE_SCALE = 10.0, 0.10
CHECKIN_MAX_HOPS = 25  # never check in further than this from the goal photo on the map
MAX_WAITS = 3
RE_APPR = re.compile(r"left=([+-]?[\d.]+) scale=([\d.]+)")
LOCK_BACK, LOCK_AHEAD, LOCK_N = 2, 8, 40


class Route2Agent(RouteAgent):
    directed_on = False
    settle_on = True
    lock_on = True
    max_damp_ticks = 6
    damp_why = ("follow", "steer-turn", "wp-search")

    def _reset(self) -> None:
        super()._reset()
        self.m_lock2, self.m_settle = 0, 0
        self.ap_prev = None  # (left, scale) of the previous approach reading
        self.ap_turn = 0  # sign of the previous approach turn
        self.ap_waits, self.m_waits, self.m_fake = 0, 0, 0

    def _geo_steer(self, obs, wp, l):
        old = base.STEER_DEG
        if self.settle_on and self.log and self.log[-1]["why"] == "steer-turn":
            base.STEER_DEG = SETTLE_DEG  # the base reads this module constant at call time
            self.m_settle += 1
        try:
            return super()._geo_steer(obs, wp, l)
        finally:
            base.STEER_DEG = old

    def _best_match_node(self, obs, q, p):
        node = super()._best_match_node(obs, q, p)
        if not (self.lock_on and self.route_on and self.route):
            return node
        try:
            lo, hi = max(0, self.ri - LOCK_BACK), min(len(self.route), self.ri + LOCK_AHEAD + 1)
            live = self.gm.features(obs.image)
            best_n, best = -1, -1
            for k in self.route[lo:hi]:
                n = self.gm.match(self._node_feat(k), live, self.focal).n
                if n > best_n:
                    best_n, best = n, k
            if best_n >= LOCK_N:
                if best != node:
                    self.m_lock2 += 1
                self.ri = self.route_idx[best]
                return best
        except Exception as e:  # noqa: BLE001
            print(f"[{self.t}] route lock error: {e!r}")
        return node

    def _link(self, a: int, b: int) -> None:
        if self.lock_on and self.route_on and self.route:
            return  # mid-run links pulled the robot onto another drive at a junction (maze 10)
        super()._link(a, b)

    def _out(self, a, why: str):
        if why == "arrived" and getattr(self, "hops", 0) > CHECKIN_MAX_HOPS:
            # the goal photo matches here, but the map says we are far from the goal: a look-alike place
            self.lock, self.side_target = 0, None
            self.m_fake += 1
            self.state += f" -> NOT checking in: {self.hops} hops from the goal on the map (look-alike)"
            return super()._out((Action.IDLE, 1), "fake-goal")
        if why.startswith("approach") or why == "arrived":
            m = RE_APPR.search(self.state or "")
            cur = (float(m.group(1)), float(m.group(2))) if m else None
            act = a[0] if isinstance(a, tuple) else a
            wait = False
            if why == "arrived" and cur is not None and self.ap_waits < MAX_WAITS:
                p = self.ap_prev
                if p is None or abs(p[0]) > GATE_DEG or abs(p[1] - 1.0) > GATE_SCALE:
                    wait = True  # one good reading is not enough: look again from the same spot
            elif why == "approach-turn" and act in (Action.LEFT, Action.RIGHT) and self.ap_waits < MAX_WAITS:
                sgn = 1 if act == Action.LEFT else -1
                if self.ap_turn == -sgn:
                    wait = True  # reverses the last approach turn: measure again before turning back
                self.ap_turn = sgn
            if wait:
                self.ap_waits += 1
                self.m_waits += 1
                self.ap_prev = cur
                return super()._out((Action.IDLE, 1), "approach-wait")
            self.ap_waits = 0  # a real move: waits may be used again
            self.ap_prev = cur if cur is not None else self.ap_prev
        else:
            self.ap_turn = 0
        return super()._out(a, why)

    def finish(self, result) -> None:
        print(f"last metre: {self.m_waits} one-tick waits to re-measure; {self.m_fake} check-ins refused far from the goal")
        print(f"route2: route lock overrode the base {self.m_lock2} times; settle dead-band used {self.m_settle} times")
        super().finish(result)


if __name__ == "__main__":
    argv = sys.argv[1:]
    flags = ("--no-settle", "--no-lock", "--no-route", "--no-damp", "--no-lost", "--no-spin")
    Route2Agent.settle_on = "--no-settle" not in argv
    Route2Agent.lock_on = "--no-lock" not in argv
    Route2Agent.route_on = "--no-route" not in argv
    Route2Agent.damp_on = "--no-damp" not in argv
    Route2Agent.lost_on = "--no-lost" not in argv
    Route2Agent.spin_on = "--no-spin" not in argv
    argv = [a for a in argv if a not in flags]
    base.SEARCH_TICKS, base.SEARCH_AFTER = 45, 5
    sys.argv[1:] = with_defaults(with_budget(argv), flags=FLAGS, log_dir="logs/nav_route2")
    main(Route2Agent)
