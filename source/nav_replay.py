"""
nav_replay.py -- nav_route.py (map first, ticks-priced shortest path, route-locked localisation) plus
TEACH-AND-REPEAT execution.  NEW FILE; nothing existing is edited.

Why (measured on maze 8 with the true poses, see the chat):
  * the planned route was 370 hops = ~1,110 ticks if every hop is driven exactly as recorded (3 ticks a frame),
  * the run that reached the goal used 2,487 ticks, 1,940 of them turning: the base re-decides every step
    which photo to aim at, measures an angle, turns, searches, ... -- 2.2x the plan.
Teach-and-repeat (Furgale & Barfoot 2010; the "visual teach and repeat" idea in the VO lecture): the exploration
drive already says what to do between two photos -- the recorded action. So when the robot is confidently at
photo c of the route and the route continues FORWARD along the same drive (c -> c+1):
  1. track: of the next 3 route photos, take the one SuperGlue matches best with the camera (>= RP_MIN inliers);
  2. align: if the camera is turned more than RP_ALIGN deg away from that photo's heading, turn back by the
     measured angle (essential matrix yaw, lecture 4/5);
  3. repeat: otherwise do exactly what the drive did there: its recorded action for 3 ticks.
Anything else -- reversing, drive switches (shortcuts), weak matches, the goal approach, stuck/recover -- is
left to the base agent exactly as before.

    uv run source/nav_replay.py --local 8 --budget 3000
    options: --no-replay (= nav_route.py), plus nav_route's --no-route --no-directed --no-damp --no-lost --no-spin
"""

from __future__ import annotations

import sys

import math

import numpy as np
from geometry import yaw_from_e
from my_agent import with_defaults
from nav_merge import FLAGS, with_budget
from nav_route import RouteAgent
from sptm_agent import SIGN, main
from vis_nav_sdk import Action

RP_MIN = 30  # SuperGlue inliers with a route photo to trust the tracker
RP_ALIGN = 15.0  # degrees off the photo's heading before re-aligning (12 re-aligned too often)
RP_TRACK = 3  # route photos ahead the tracker looks at
# shorter path: "skip" edges inside a drive. The exploration drives spin and wander; where a drive comes back
# within SKIP_R metres of where it was (dead reckoning over at most SKIP_K photos, measured on maze 8: 0 wrong
# skips at these settings), the route may jump straight there and the robot only turns by the NET heading change.
SKIP_K = 30  # photos apart at most
SKIP_MIN = 3  # photos apart at least
SKIP_R = 0.15  # metres apart at most (dead reckoning)
SKIP_RISK = 1.0  # hops added to every skip
STEP_M = 0.119  # forward metres per recorded photo (median of the true poses, maze 8)
TURN_DEG = 10.3  # degrees per recorded LEFT / RIGHT photo


def wrapd(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# route lock: at junctions, place recognition jumps to the other drive (maze 10: 1502 <-> 1022 for 700
# decisions, the planned route never left drive 2). Every decision, SuperGlue checks the planned route's photos
# around the current one; if one of them matches clearly, that is where we are -- whatever else looks similar.
LOCK_BACK, LOCK_AHEAD = 2, 8  # route photos behind / ahead of the current one that are checked
LOCK_N = 40  # inliers that keep the robot on the route


BLIND_MAX = 2  # forward steps replayed on trust when the view is briefly weak


RP_FAST_N = 60  # inliers for the fast lane
RP_FAST_DEG = 6.0  # degrees off the photo's heading allowed in the fast lane
RP_FAST_MAX = 3  # recorded steps per decision at most
RP_REPLACE = ("follow", "steer-turn", "steer-fwd", "wp-search", "directed", "damp-fwd")
# look once, then commit: when replay loses the route twice in a row, stop letting the other steering rules
# take turns (maze 10: L "RIGHT 1", look-around "RIGHT 3", directed "LEFT 4", replay "LEFT 3" ... in turn).
# Instead turn on the spot in 6-tick steps, match EVERY route photo ahead at each heading, stop as soon as one
# matches clearly, turn back to the best one by the measured angle, and replay from there.
SCAN_STEP = 6  # ticks per look (~21 deg)
SCAN_STEPS = 9  # at most ~190 deg (a full circle cost ~100 ticks a scan on maze 8: 24 scans, 2,350 ticks)
SCAN_LOOK = 15  # route photos ahead compared at every look
SCAN_GOOD = 45  # inliers that end the scan early
RP_FAIL_SCAN = 4  # replay failures in a row that start a scan
SCAN_COOLDOWN = 40  # decisions after a scan before another may start


class ReplayAgent(RouteAgent):
    replay_on = True

    def _plan_map(self) -> None:
        """Add the skip edges to the map, then let nav_route price and plan the route as before."""
        if self.replay_on and not getattr(self, "skips", None):
            self.skips = set()
            for s0, e0 in self.bounds:
                x = y = h = 0.0
                P = []
                for k in range(s0, e0):
                    P.append((x, y, h))
                    a = self.frames[k].action
                    if a == "FORWARD":
                        x, y = x + STEP_M * math.cos(h), y + STEP_M * math.sin(h)
                    elif a == "BACKWARD":
                        x, y = x - STEP_M * math.cos(h), y - STEP_M * math.sin(h)
                    elif a in ("LEFT", "RIGHT"):
                        h += math.radians(TURN_DEG if a == "LEFT" else -TURN_DEG)
                P = np.asarray(P)
                for i in range(e0 - s0):
                    for j in range(i + SKIP_MIN, min(i + SKIP_K, e0 - s0)):
                        d = math.hypot(P[j, 0] - P[i, 0], P[j, 1] - P[i, 1])
                        if d >= SKIP_R:
                            continue
                        ticks = abs(wrapd(math.degrees(P[j, 2] - P[i, 2]))) / 3.44 + d / 0.04
                        if ticks < 3.0 * (j - i) - 3.0:  # clearly cheaper than driving the drive
                            a, b = s0 + i, s0 + j
                            w = ticks / 3.0 + SKIP_RISK
                            if not self.G.has_edge(a, b) or self.G.edges[a, b]["weight"] > w:
                                self.G.add_edge(a, b, weight=w)
                                self.skips.add((a, b))
            print(f"MAP: added {len(self.skips)} skip edges (drive revisits within {SKIP_R} m, net turn only)")
        super()._plan_map()
        used = [(a, b) for a, b in zip(self.route, self.route[1:]) if (min(a, b), max(a, b)) in self.skips]
        print(f"ROUTE uses {len(used)} skips, saving {sum(abs(b - a) - 1 for a, b in used)} recorded steps")

    def _best_match_node(self, obs, q, p):
        node = super()._best_match_node(obs, q, p)
        if not (self.replay_on and self.route):
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
                    self.m_lock += 1
                self.ri = self.route_idx[best]
                return best
        except Exception as e:  # noqa: BLE001
            print(f"[{self.t}] route lock error: {e!r}")
        return node

    def _link(self, a: int, b: int) -> None:
        """The base links two photos it has just seen as one place and re-plans through the link. On the
        planned route that pulled the robot onto another drive (maze 10, step 904/950): not while we have one."""
        if self.replay_on and self.route:
            return
        super()._link(a, b)

    def _reset(self) -> None:
        super()._reset()
        self.m_skip, self.m_lock, self.m_turnrun, self.m_coast, self.rp_blind = 0, 0, 0, 0, 0
        self.m_rp, self.m_rp_align, self.m_rp_adv, self.m_rp_fast = 0, 0, 0, 0
        self.rp_last = -1  # the last photo whose recorded action we replayed
        self.rp_fail, self.rp_scan, self.rp_cool, self.m_scans = 0, 0, 0, 0
        self.rp_turned, self.rp_best = 0.0, (0, -1, 0.0, 0.0)

    def _replay(self):
        node = self.node
        if node < 0 or node not in self.paths or self._obs is None:
            return None
        path = self.paths[node][::-1]  # node ... goal
        if len(path) < 2:
            return None
        live = self.gm.features(self._obs.image)
        cands = list(path[:RP_TRACK])
        if self.rp_last >= 0:  # also the photos right after the last one replayed (the base's node lags)
            cands += [k for k in (self.rp_last, self.rp_last + 1, self.rp_last + 2) if k in path and k not in cands]
        best_m, c = None, -1
        for k in cands:
            m = self.gm.match(self._node_feat(k), live, self.focal)
            if best_m is None or m.n > best_m.n:
                best_m, c = m, k
        if best_m is None or best_m.n < RP_MIN:
            return None
        y = yaw_from_e(best_m, self.focal)
        if y is None:
            return None
        here = float(SIGN * y)  # + = camera turned left of photo c
        # Already replayed c (and maybe more)? Then we are at the photo after the last one replayed. Turning on
        # the spot, the camera often still matches c best (maze 8: LEFT 3, still "706", LEFT 3 again, then
        # align RIGHT 6 -- for 750 decisions). Measure our heading against c, then shift it by the recorded turns.
        if 0 <= c <= self.rp_last <= c + 60 and self.traj[self.rp_last] == self.traj[c]:
            nc = self.rp_last + 1
            if nc not in path:
                return None
            here = wrapd(here - sum(self._turn_deg(k) for k in range(c, nc)))
            c = nc
            self.m_rp_adv += 1
        i = path.index(c)
        if i + 1 >= len(path):
            return None
        nxt = path[i + 1]
        if (min(c, nxt), max(c, nxt)) in getattr(self, "skips", ()):
            # a skip: same spot, so just turn by the net recorded heading change, once
            lo, hi = min(c, nxt), max(c, nxt)
            dh = sum(self._turn_deg(k) for k in range(lo, hi)) * (1 if nxt > c else -1)
            need = wrapd(dh - here)  # + = turn left
            self.m_skip += 1
            self.rp_last = nxt - 1 if nxt > c else -1
            if nxt < c:
                self.node = nxt
            self.state = f"skip {c}->{nxt}: net turn {need:+.0f} deg (drive turned {dh:+.0f})"
            ticks = int(round(abs(need) / 3.44))
            if ticks == 0:
                return (Action.FORWARD, 1)
            return (Action.LEFT if need > 0 else Action.RIGHT, min(ticks, 52))
        if not (nxt == c + 1 and self.traj[nxt] == self.traj[c]):
            return None  # not a forward hop along a drive: the base handles it
        act = self.frames[c].action
        if act not in ("FORWARD", "LEFT", "RIGHT"):
            return None
        if act in ("LEFT", "RIGHT"):
            # one turn per corner: the whole run of recorded turns on the spot, net, in ONE action, corrected by
            # where the camera points now -- no 10-degree steps with a weak match after each (corner jitter)
            j, total = i, 0.0
            while (j + 1 < len(path) and path[j + 1] == path[j] + 1 and self.traj[path[j + 1]] == self.traj[c]
                   and self.frames[path[j]].action in ("LEFT", "RIGHT")):
                total += self._turn_deg(path[j])
                j += 1
            e = path[j]  # the photo after the turns
            need = wrapd(total - here)
            ticks = int(round(abs(need) / 3.44))
            self.rp_last = e - 1
            self.m_turnrun += 1
            self.state = f"turn {c}->{e}: net {need:+.0f} deg in one go (drive turned {total:+.0f}, here={here:+.0f})"
            if ticks == 0:
                return (Action.FORWARD, 1)
            return (Action.LEFT if need > 0 else Action.RIGHT, min(ticks, 52))
        if abs(here) > RP_ALIGN:
            self.m_rp_align += 1
            self.state = f"replay align to {c}: here={here:+.0f} n={best_m.n}"
            ticks = int(np.clip(round(abs(here) / 3.44), 1, 6))
            return (Action.RIGHT if here > 0 else Action.LEFT, ticks)
        # fast lane: strong match, well aligned, and the drive did the same action again next -> do up to
        # RP_FAST_MAX recorded steps in one decision (same ticks, fewer decisions = less wall-clock time)
        k = 1
        if best_m.n >= RP_FAST_N and abs(here) <= RP_FAST_DEG:
            j = i + 1
            while (k < RP_FAST_MAX and j + 1 < len(path) and path[j + 1] == path[j] + 1
                   and self.traj[path[j]] == self.traj[c] and self.frames[path[j]].action == act):
                k += 1
                j += 1
        self.state = f"replay {c}->{c + k}: {act} x{k} (here={here:+.0f} n={best_m.n})"
        self.rp_last = c + k - 1
        self.m_rp_fast += k - 1
        return (Action[act], 3 * k)

    def _coast(self):
        """Right after a replayed step the view can be briefly weak (just turned into a corridor): trust the
        recording for up to BLIND_MAX forward steps before giving up on replay."""
        if self.rp_blind >= BLIND_MAX or self.rp_last < 0 or not self.log:
            return None
        if self.log[-1]["why"] not in ("replay", "replay-coast"):
            return None
        k = self.rp_last + 1
        if k >= self.n or self.traj[k] != self.traj[self.rp_last] or self.frames[k].action != "FORWARD":
            return None
        self.rp_last, self.rp_blind = k, self.rp_blind + 1
        self.state = f"coast {k}->{k + 1}: FORWARD (view weak, following the recording)"
        return (Action.FORWARD, 3)

    def _turn_deg(self, k: int) -> float:
        """Heading change from photo k to k+1 on its drive (+ = left), from the recorded action."""
        a = self.frames[k].action
        return 10.3 if a == "LEFT" else -10.3 if a == "RIGHT" else 0.0

    # ---- look once, then commit

    def act(self, obs):
        self._obs = obs
        if self.replay_on and self.rp_scan > 0:
            left = obs.steps_left if self.budget is None else min(obs.steps_left, self.budget - obs.step)
            if left > 12:
                self.t += 1
                return self._scan_step(obs)
            self.rp_scan = 0
        return super().act(obs)

    def _near_goal(self) -> bool:
        return getattr(self, "lock", 0) > 0 or 0 <= getattr(self, "hops", 99) <= self.near_hops + 2

    def _scan_step(self, obs):
        live = self.gm.features(obs.image)
        path = self.paths[self.node][::-1] if self.node in self.paths else []
        best = (0, -1, 0.0)
        for k in path[:SCAN_LOOK]:
            m = self.gm.match(self._node_feat(k), live, self.focal)
            if m.n > best[0]:
                y = yaw_from_e(m, self.focal)
                if y is not None:
                    best = (m.n, k, float(SIGN * y))
        if best[0] > self.rp_best[0]:
            self.rp_best = (best[0], best[1], best[2], self.rp_turned)
        if best[0] >= SCAN_GOOD or self.rp_scan <= 1:
            n, k, here, turned_at = self.rp_best
            self.rp_scan, self.rp_best, self.rp_cool = 0, (0, -1, 0.0, 0.0), SCAN_COOLDOWN
            if n < RP_MIN or k < 0:
                self.state = f"scan: no route photo found (best {n})"
                return self._out((Action.FORWARD, 2), "rp-scan-none")
            here_now = here + (self.rp_turned - turned_at)  # we kept turning left after seeing photo k
            self.node = k
            self.rp_last = k - 1 if k > 0 and self.traj[k - 1] == self.traj[k] else -1
            self.state = f"scan: photo {k} ({n} inliers), turn back {-here_now:+.0f} deg"
            if abs(here_now) <= 4:
                return self._out((Action.FORWARD, 1), "rp-scan-commit")
            ticks = int(np.clip(round(abs(here_now) / 3.44), 1, 52))
            return self._out((Action.RIGHT if here_now > 0 else Action.LEFT, ticks), "rp-scan-commit")
        self.rp_scan -= 1
        self.rp_turned += SCAN_STEP * 3.44
        self.state = f"scan: best so far photo {self.rp_best[1]} ({self.rp_best[0]} inliers)"
        return self._out((Action.LEFT, SCAN_STEP), "rp-scan")

    def _out(self, a, why: str):
        if self.rp_cool > 0:
            self.rp_cool -= 1
        if self.replay_on and why in RP_REPLACE:
            try:
                r = self._replay()
            except Exception as e:  # noqa: BLE001
                print(f"[{self.t}] replay error: {e!r}")
                r = None
            if r is None:
                r = self._coast()
                if r is not None:
                    self.m_coast += 1
                    return super()._out(r, "replay-coast")
            else:
                self.rp_blind = 0
            if r is not None:
                a, why = r, "replay"
                self.m_rp += 1
                self.rp_fail = 0
            else:
                self.rp_last = -1  # the base moves the robot now: forget what was replayed
                self.rp_fail += 1
                if (self.rp_fail >= RP_FAIL_SCAN and self.rp_cool == 0 and not self._near_goal()
                        and self._obs is not None and self.node in self.paths):
                    self.rp_fail, self.m_scans = 0, self.m_scans + 1
                    self.rp_scan, self.rp_turned, self.rp_best = SCAN_STEPS, 0.0, (0, -1, 0.0, 0.0)
                    return self._scan_step(self._obs)
        elif why not in ("replay", "replay-coast", "rp-scan", "rp-scan-commit"):
            self.rp_last = -1
        return super()._out(a, why)

    def finish(self, result) -> None:
        print(f"one-go turns: {self.m_turnrun}; coasted steps: {self.m_coast}")
        print(f"look-once scans: {self.m_scans}; skips taken: {self.m_skip}; route lock overrode the base {self.m_lock} times")
        print(f"replay: {self.m_rp} decisions taught-and-repeated ({self.m_rp_align} re-alignments, "
              f"{self.m_rp_adv} advanced past the best-matching photo, {self.m_rp_fast} extra steps in the fast lane)")
        super().finish(result)


if __name__ == "__main__":
    argv = sys.argv[1:]
    flags = ("--no-replay", "--no-route", "--no-damp", "--no-lost", "--no-spin", "--no-directed")
    ReplayAgent.replay_on = "--no-replay" not in argv
    ReplayAgent.route_on = "--no-route" not in argv
    ReplayAgent.damp_on = "--no-damp" not in argv
    ReplayAgent.lost_on = "--no-lost" not in argv
    ReplayAgent.spin_on = "--no-spin" not in argv
    ReplayAgent.directed_on = "--no-directed" not in argv
    argv = [a for a in argv if a not in flags]
    import sptm_agent as base
    base.SEARCH_TICKS, base.SEARCH_AFTER = 45, 5
    sys.argv[1:] = with_defaults(with_budget(argv), flags=FLAGS, log_dir="logs/nav_replay")
    main(ReplayAgent)
