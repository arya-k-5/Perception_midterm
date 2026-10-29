"""
my_agent_1820lg2.py -- my_agent_1820lg (Kartik's 1820 + 4-photo goal check + LightGlue; 6/10 on maze 17) with two
fixes for the loops seen on maze 10. NEW FILE; the frozen 1820lg is untouched.

Both loops come from Kartik's "reverse" rule (drive back along a drive by undoing its recorded moves):
  * photos 800-802, ~400 decisions: the undo turned RIGHT (undoing recorded LEFT turns), then at photo 800 the
    look-around turned LEFT and undid it: RIGHT x3, LEFT x2, RIGHT x3, ...
  * photo 2106: "undo LEFT" = RIGHT 2 ticks, 25 times in a row, never matching the previous photo -> spinning.

  1. UNDO TURNS IN ONE MEASURED TURN. When the route goes back over photos where the drive turned on the spot,
     match the camera with each of those photos (LightGlue), take the clearest one, and from its measured angle
     plus the recorded turns compute the whole turn to the last of them; turn that much once. When we are already
     facing it, the agent is told it IS at that photo (so its localisation cannot snap back and start again).
  2. LOOK THE WAY THE ROUTE GOES. When the next route photo is out of view, the look-around turns toward the side
     the recorded route turns to next (incl. while going back), not the network's lean.

    uv run source/my_agent_1820lg2.py --local 10 --budget 4000
  3. NEAR THE GOAL, LOOK FOR IT; 4. NO FLIP-FLOPPING BETWEEN DRIVES; 5. STICK TO THE DRIVE;
  6. LAST-METRE THRESHOLDS SCALED FOR LIGHTGLUE; 7. LAST-METRE DAMPER (constants below).
    --no-undo1 / --no-lookroute / --no-goalnear switch fixes off
"""

from __future__ import annotations

import re
import sys

import numpy as np
import sg_matcher
from geometry import yaw_from_e
from lg_matcher import LGMatcher
from my_agent import with_defaults
from my_agent_1820g import FLAGS
from my_agent_1820lg import LGAgent
from my_agent_2767g import GoalAgent
import sptm_agent as base
from sptm_agent import SIGN, main
from vis_nav_sdk import Action

UNDO_MIN = 30  # inliers to trust a photo's measured angle
FACING_DEG = 6.0  # this close to the target photo's heading = we are there
TURN_DEG = 10.3  # one recorded LEFT/RIGHT photo
LOOK_HOPS = 6  # route hops whose recorded turns decide the look-around side
# 3. NEAR THE GOAL, LOOK FOR IT (maze 67: 12 hops from the goal the route said "back up along the drive"; it backed
#    into a wall ~100 times until the budget ran out; Kartik's approach only starts at <= 3 hops). Within GOAL_NEAR
#    hops on the map, every decision checks the 4 goal photos; a clear match starts the goal approach. Instead of
#    backing up there, it turns on the spot (6 ticks at a time, at most a full circle) looking for them.
GOAL_NEAR = 20
GOAL_SEEN_N = 60  # LightGlue inliers with a goal photo (look-alikes elsewhere reach ~30-45)
NEAR_SCAN_MAX = 18
# 4. NO FLIP-FLOPPING BETWEEN DRIVES (maze 67, steps 900-930: belief 1538 <-> 2480 every few decisions, each asking
#    for the opposite turn). After a jump of > JUMP_FAR photos, another far jump within HOLD decisions needs the
#    new photo to match HOLD_MARGIN more inliers than the one we hold.
JUMP_FAR, HOLD, HOLD_MARGIN = 50, 10, 30
# 5. STICK TO THE DRIVE YOU ARE ON (maze 67: 141 jumps of > 50 photos, e.g. 2200 <-> 52 every decision -- the same
#    spot on two drives, both ~131 hops from the goal; each drive's next photo points another way -> left/right).
#    If the new best photo is on another drive and a photo of the current drive within STICK_K matches nearly as
#    well (place-recognition score within STICK_P), stay on the current drive.
STICK_K, STICK_P = 3, 0.05
# 6. LAST-METRE THRESHOLDS FOR LIGHTGLUE (maze 67: ~150 decisions dancing at the goal on front-photo matches of
#    15-30 inliers, then a check-in on a 22-inlier reading -> 0 pts). Kartik set them for SuperGlue counts; with
#    LightGlue the same place gives ~150 and look-alikes reach 30-45. Navigation thresholds are left as they are.
LG_GOAL_MIN = 45  # base 20: goal-photo inliers the approach acts on
LG_CHECKIN_MIN = 50  # base 18
LG_GOTO_MIN = 70  # base 30
# 7. LAST-METRE DAMPER (maze 67 at the goal: back-photo angle read +150, +214, +150 ... -> LEFT, RIGHT, LEFT): an
#    approach turn (front, side or back photo) that reverses the previous one first waits one tick and measures
#    again; a check-in needs the previous reading inside the gate too. (Ran without problems in nav_route2.)
GATE_DEG, GATE_SCALE, MAX_WAITS = 10.0, 0.10, 3
RE_APPR = re.compile(r"left=([+-]?[\d.]+) scale=([\d.]+)")
LG_CHECKIN_STAGES = [(0.0, 50, 8.0, 0.10), (0.3, 45, 10.0, 0.12), (0.7, 40, 15.0, 0.15)]  # base 16/14/14 inliers


def wrapd(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class LG2Agent(LGAgent):
    lgth_on = True

    def setup(self, info) -> None:
        super().setup(info)
        if self.lgth_on:
            self.goal_min = LG_GOAL_MIN
    undo_on = True
    look_on = True
    goal_on = True

    def _reset(self) -> None:
        super()._reset()
        self._obs = None
        self.force_node, self.forced_at = -1, -10
        self.c_undo, self.c_look, self.c_goal, self.c_hold, self.c_nscan = 0, 0, 0, 0, 0
        self.near_scan, self.last_jump_t = 0, -100
        self.c_stick = 0
        self.ap_prev, self.ap_turn, self.ap_waits, self.c_wait = None, 0, 0, 0

    def act(self, obs):
        self._obs = obs
        return super().act(obs)

    def _best_match_node(self, obs, q, p):
        if self.force_node >= 0:
            n, self.force_node = self.force_node, -1
            return n
        prev = self.node
        node = super()._best_match_node(obs, q, p)
        if prev >= 0 and node >= 0 and self.traj[node] != self.traj[prev]:
            s0, e0 = self.bounds[self.traj[prev]]
            same = [k for k in range(max(s0, prev - STICK_K), min(e0, prev + STICK_K + 1)) if self.reach[k]]
            if same:
                k = max(same, key=lambda j: p[j])
                if p[k] >= p[node] - STICK_P:
                    self.c_stick += 1
                    return k
        if prev >= 0 and node >= 0 and abs(node - prev) > JUMP_FAR:
            if self.t - self.last_jump_t <= HOLD:
                try:
                    live = self.gm.features(obs.image)
                    n_new = self.gm.match(self._node_feat(node), live, self.focal).n
                    n_old = self.gm.match(self._node_feat(prev), live, self.focal).n
                    if n_new < n_old + HOLD_MARGIN:
                        self.c_hold += 1
                        return prev  # just jumped: stay unless the new place is clearly better
                except Exception:  # noqa: BLE001
                    pass
            self.last_jump_t = self.t
        return node

    def _goal_seen(self) -> int:
        live = self.gm.features(self._obs.image)
        return max(self.gm.match(tf, live, self.focal).n for tf in self.tfeat)

    def _turn_of(self, k: int) -> float:
        a = self.frames[k].action
        return TURN_DEG if a == "LEFT" else -TURN_DEG if a == "RIGHT" else 0.0

    def _undo_turns(self):
        node = self.node
        if node < 0 or node not in self.paths or self._obs is None:
            return None
        path = self.paths[node][::-1]
        i = 0
        while (i + 1 < len(path) and path[i + 1] == path[i] - 1 and self.traj[path[i + 1]] == self.traj[node]
               and self.frames[path[i + 1]].action in ("LEFT", "RIGHT")):
            i += 1
        if i == 0:
            return None  # not a run of turns (e.g. backing up after a FORWARD): Kartik's rule does it
        run = path[: i + 1]  # node, node-1, ..., e  (all at one spot, different headings)
        e = run[-1]
        live = self.gm.features(self._obs.image)
        best = None
        for k in run:
            m = self.gm.match(self._node_feat(k), live, self.focal)
            if m.n >= UNDO_MIN and (best is None or m.n > best[1]):
                y = yaw_from_e(m, self.focal)
                if y is not None:
                    best = (k, m.n, float(SIGN * y))
        if best is None:
            return None
        k, n, here = best  # here: + = camera turned left of photo k
        rel = -sum(self._turn_of(j) for j in range(e, k))  # heading of photo e minus heading of photo k
        need = wrapd(rel - here)
        self.c_undo += 1
        if abs(need) <= FACING_DEG:
            if self.t - self.forced_at <= 2:
                return None  # just did this: let Kartik's agent take the next step from photo e
            self.force_node, self.forced_at = e, self.t
            self.state = f"undo turns: facing photo {e} already (via {k}, {n} inliers) -> I am at {e}"
            return (Action.IDLE, 1)
        self.state = f"undo turns {node}->{e} in one go: {need:+.0f} deg (seen {k}, {n} inliers)"
        ticks = int(round(abs(need) / 3.44))
        return (Action.LEFT if need > 0 else Action.RIGHT, int(np.clip(ticks, 1, 52)))

    def _route_turn(self) -> float:
        if self.node not in self.paths:
            return 0.0
        path = self.paths[self.node][::-1]
        total = 0.0
        for a, b in list(zip(path, path[1:]))[:LOOK_HOPS]:
            if self.traj[a] == self.traj[b] and abs(a - b) == 1:
                d = self._turn_of(min(a, b))
                total += d if b > a else -d
        return total

    def _last_metre(self, a, why: str):
        m = RE_APPR.search(self.state or "")
        cur = (float(m.group(1)), float(m.group(2))) if m else None
        act = a[0] if isinstance(a, tuple) else a
        wait = False
        if why == "arrived" and cur is not None and self.ap_waits < MAX_WAITS:
            p = self.ap_prev
            if p is None or abs(p[0]) > GATE_DEG or abs(p[1] - 1.0) > GATE_SCALE:
                wait = True
        elif (why == "approach-turn" or (why.startswith("approach-side") and why.endswith("-turn"))) \
                and act in (Action.LEFT, Action.RIGHT) and self.ap_waits < MAX_WAITS:
            sgn = 1 if act == Action.LEFT else -1
            if self.ap_turn == -sgn:
                wait = True
            self.ap_turn = sgn
        self.ap_prev = cur if cur is not None else self.ap_prev
        if wait:
            self.ap_waits += 1
            self.c_wait += 1
            return (Action.IDLE, 1)
        self.ap_waits = 0
        return None

    def _out(self, a, why: str):
        if self.lgth_on and (why.startswith("approach") or why == "arrived"):
            w = self._last_metre(a, why)
            if w is not None:
                return super()._out(w, "approach-wait")
        elif not why.startswith("approach"):
            self.ap_turn = 0
        near = 0 <= getattr(self, "hops", -1) <= GOAL_NEAR and self._obs is not None
        if near and self.goal_on and getattr(self, "lock", 0) == 0 and not why.startswith("approach") \
                and why not in ("arrived", "budget", "stuck", "recover"):
            try:
                g = self._goal_seen()
            except Exception:  # noqa: BLE001
                g = 0
            if g >= GOAL_SEEN_N:
                self.lock, self.near_scan = 12, 0  # Kartik's approach takes over from the next decision
                self.c_goal += 1
                self.state += f" -> goal photo seen ({g} inliers) {self.hops} hops out: approaching"
                return super()._out((Action.IDLE, 1), "goal-near-seen")
            if why == "reverse" and self.near_scan < NEAR_SCAN_MAX:
                self.near_scan += 1
                self.c_nscan += 1
                self.state = f"{self.hops} hops from the goal: looking for it instead of backing up ({self.near_scan})"
                return super()._out((Action.LEFT, 6), "goal-near-scan")
        if why == "reverse" and self.undo_on:
            act = a[0] if isinstance(a, tuple) else a
            if act in (Action.LEFT, Action.RIGHT):
                try:
                    u = self._undo_turns()
                except Exception as ex:  # noqa: BLE001
                    print(f"[{self.t}] undo error: {ex!r}")
                    u = None
                if u is not None:
                    return super()._out(u, "undo-turns")
        if why == "wp-search" and self.look_on and isinstance(a, tuple):
            plan = self._route_turn()
            if abs(plan) >= TURN_DEG:
                want = Action.LEFT if plan > 0 else Action.RIGHT
                if a[0] != want:
                    self.c_look += 1
                    self.state += f" (looking {want.name}: the route turns {plan:+.0f} deg)"
                    a = (want, a[1])
        return super()._out(a, why)

    def finish(self, result) -> None:
        print(f"fixes: undo-turns {self.c_undo}, look-around side changed {self.c_look}, goal seen near {self.c_goal}, "
              f"near-goal scan steps {self.c_nscan}, far jumps held {self.c_hold}, stayed on the drive {self.c_stick}, "
              f"last-metre waits {self.c_wait}")
        super().finish(result)


if __name__ == "__main__":
    sg_matcher.SGMatcher = LGMatcher
    argv = sys.argv[1:]
    GoalAgent.goal_check = "--no-goal-check" not in argv
    LG2Agent.undo_on = "--no-undo1" not in argv
    LG2Agent.look_on = "--no-lookroute" not in argv
    LG2Agent.goal_on = "--no-goalnear" not in argv
    LG2Agent.lgth_on = "--no-lgthresh" not in argv
    if LG2Agent.lgth_on:
        base.CHECKIN_MIN, base.GOTO_MIN, base.CHECKIN_STAGES = LG_CHECKIN_MIN, LG_GOTO_MIN, LG_CHECKIN_STAGES
    argv = [x for x in argv if x not in ("--no-goal-check", "--no-undo1", "--no-lookroute", "--no-goalnear", "--no-lgthresh")]
    sys.argv[1:] = with_defaults(argv, flags=FLAGS, log_dir="logs/my_agent_1820lg2")
    main(LG2Agent)
