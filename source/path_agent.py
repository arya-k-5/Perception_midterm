"""
PATH AGENT: follow a metric path, and use image matching on the side to make sure we are on it.

    uv run --no-sync source/path_agent.py --classical --local 8 --budget 5000 --verbose --log-dir logs/p_8 2>&1 | tee logs/p_8.txt

Differences from sptm_agent_4.py (which this builds on: same frames, same verified shortcuts, same planner, same
final approach on the four goal photos):

  Map      : every exploration frame gets an (x, y, heading) -- dead reckoning from the recorded actions,
             pinned together by the verified shortcuts (metric_map.py).
  Path     : the planner's route (node sequence) becomes a polyline of metric waypoints.
  Steering : by the robot's own pose estimate against the waypoints (pure pursuit), NOT by matching a
             waypoint photo. So a route that drives a recorded corridor backwards is no problem.
  Pose     : dead reckoning from our own actions, corrected every few steps by matching the camera frame to
             the map frames near the estimate (position from the matched frame, heading from the image shift).
  On-path  : every correction also tells us whether the match agrees with the estimate and with the route;
             a disagreement re-plans from where the picture says we are.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial import cKDTree

import sptm_agent_4 as base
from geometry import scale_ratio, shift_deg
from metric_map import TICK_DEG, TICK_M, MetricMap, wrap
from vis_nav_sdk import Action, Observation, SessionInfo

D2R = math.pi / 180.0
YAW_GAIN = 1.19  # the median horizontal shift overstates a pure turn (LEFT x6 = 20.6 deg showed as +23..26)


class PathAgent(base.SptmAgent):
    def __init__(self, *args, look: float = 0.30, fix_every: int = 3, scene_depth: float = 1.5,
                 anchor_pos: float | None = 0.3, **kw) -> None:
        super().__init__(*args, **kw)
        self.look, self.fix_every, self.scene_depth, self.anchor_pos = look, fix_every, scene_depth, anchor_pos
        self.P = None

    # ------------------------------------------------------------------ map

    def _ticks_per_frame(self, samples: int = 120) -> float:
        """How many ticks pass between two recorded frames: a LEFT/RIGHT frame turns TICK_DEG per tick."""
        idx = [i for s, e in self.bounds for i in range(s, e - 1) if self.frames[i].action in ("LEFT", "RIGHT")]
        if not idx:
            return 3.0
        rng = np.random.default_rng(0)
        pick = rng.choice(idx, size=min(samples, len(idx)), replace=False)
        sh = []
        for i in pick:
            m = self.matcher.match(self._feat(int(i)), self._feat(int(i) + 1))
            if m.n >= 20:
                sh.append(abs(shift_deg(m, self.focal)) / YAW_GAIN)
        if len(sh) < 10:
            return 3.0
        return float(max(1, round(float(np.median(sh)) / TICK_DEG)))

    def build_map(self) -> None:
        tpf = self._ticks_per_frame()
        mm = MetricMap([f.action for f in self.frames], self.bounds, tpf)
        loops = []
        for i, j in self.shortcuts:
            m = self.matcher.match(self._feat(i), self._feat(j))
            if m.n >= 15:
                loops.append((i, j, shift_deg(m, self.focal) / YAW_GAIN * D2R))
        mm.integrate()
        odo = mm.P.copy()
        placed = mm.init_align(loops) if self.anchor_pos is None else len(self.bounds)
        mm.optimise(loops, anchor_pos=self.anchor_pos)
        self.map, self.P, self.odo = mm, mm.P, odo
        self.tree = cKDTree(self.P[:, :2])
        self._img.clear()
        span = np.ptp(self.P[:, :2], axis=0)
        print(f"metric map: {tpf:.0f} ticks per frame, {len(loops)} loop closures, drives placed {placed}/{len(self.bounds)}, "
              f"extent {span[0]:.1f} x {span[1]:.1f} m, fit rms {mm.cost0:.2f}")
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.log_dir / "poses.npy", self.P)
            np.save(self.log_dir / "poses_odometry.npy", odo)

    def setup(self, info: SessionInfo) -> None:
        super().setup(info)  # focal, goal photos, goal node, planner
        if self.P is None:
            self.build_map()
        gx, gy, gt = self.P[self.goal]
        print(f"goal node {self.goal} sits at ({gx:.2f}, {gy:.2f}) facing {math.degrees(gt):.0f} deg in the map")

    def _reset(self) -> None:
        super()._reset()
        self.est, self.sd, self.last, self.fix_t, self.miss, self.blocked = None, 1.0, None, -99, 0, False
        self.n_fix, self.n_jump = 0, 0
        self.route, self.ri, self.n_replan = None, 0, 0  # the route we committed to, and how far along it we are

    def _out(self, a, why: str):
        r = super()._out(a, why)
        act, n = a if isinstance(a, tuple) else (a, 1)
        self.last = (act, n)
        return r

    # ------------------------------------------------------------------ pose estimate

    def _predict(self) -> None:
        """Dead reckoning from the action just executed."""
        if self.last is None or self.est is None:
            return
        act, n = self.last
        x, y, th = self.est
        if act == Action.FORWARD and not self.blocked:
            x, y, self.sd = x + n * TICK_M * math.cos(th), y + n * TICK_M * math.sin(th), self.sd + 0.006 * n
        elif act == Action.BACKWARD and not self.blocked:
            x, y, self.sd = x - n * TICK_M * math.cos(th), y - n * TICK_M * math.sin(th), self.sd + 0.006 * n
        elif act == Action.LEFT:
            th += n * TICK_DEG * D2R
            self.sd += 0.001 * n
        elif act == Action.RIGHT:
            th -= n * TICK_DEG * D2R
            self.sd += 0.001 * n
        self.est = np.array([x, y, float(wrap(th))])
        self.last = None

    def _localise(self, obs: Observation, feat) -> bool:
        """Match the camera frame against map frames near the estimate (or everywhere when lost) and take
        position and heading from the best match."""
        q = np.asarray(self.scorer.embed([obs.image]), np.float32)[0]
        p = self._prob_to_all(q)
        local = self.est is not None and self.sd < 1.2
        cand = np.zeros(0, np.int64)
        if local:
            near = self.tree.query_ball_point(self.est[:2], r=max(0.7, 2.5 * self.sd))
            cand = np.array([c for c in near if self.reach[c]], np.int64)
            need = 18
        if len(cand) < 3:
            local, need = False, 25
            cand = np.flatnonzero(self.reach)
        top = [int(c) for c in cand[np.argsort(-p[cand])[:6]]]
        if local:  # the frames right at the estimate, whatever the global score says, and the next frames of our route
            _, nn = self.tree.query(self.est[:2], k=min(3, self.n))
            top += [int(c) for c in np.atleast_1d(nn) if self.reach[c]]
            if self.route is not None:
                top += [int(c) for c in self.route[self.ri : self.ri + 12]
                        if float(np.hypot(*(self.P[c, :2] - self.est[:2]))) < 0.5]
        ms = []
        for c in dict.fromkeys(top):
            m = self.matcher.match(self._feat(c), feat)
            if m.n >= need:
                ms.append((c, m))
        if not ms:
            self.miss += 1
            return False
        best_n = max(m.n for _, m in ms)
        pos_w, sin_w, cos_w, wsum = np.zeros(2), 0.0, 0.0, 0.0
        for c, m in ms:
            if m.n < 0.7 * best_n:
                continue
            sh = shift_deg(m, self.focal) / YAW_GAIN
            th_c = self.P[c, 2] + sh * D2R
            r = scale_ratio(m)
            delta = float(np.clip(self.scene_depth * (1.0 - 1.0 / r), -0.4, 0.4)) if abs(sh) < 30 else 0.0
            pos_w += m.n * (self.P[c, :2] + delta * np.array([math.cos(th_c), math.sin(th_c)]))
            sin_w, cos_w, wsum = sin_w + m.n * math.sin(th_c), cos_w + m.n * math.cos(th_c), wsum + m.n
        pos, th = pos_w / wsum, math.atan2(sin_w, cos_w)  # many matching frames agree better than one
        c = max(ms, key=lambda cm: cm[1].n)[0]
        if self.est is not None and self.route is not None:  # tie-break toward the node on our own route
            for rc in self.route[self.ri : self.ri + 40]:
                if any(rc == k and mm.n >= 0.8 * best_n for k, mm in ms):
                    c = int(rc)
                    break
        if self.est is None:
            self.est = np.array([pos[0], pos[1], th])
        else:
            jump = float(np.hypot(*(pos - self.est[:2])))
            if jump > 0.8:  # dead reckoning was wrong: trust the picture
                self.n_jump += 1
                if self.verbose:
                    print(f"[{self.t}] fix disagrees with dead reckoning by {jump:.2f} m: trusting the picture (node {c}, n={best_n})")
                self.est = np.array([pos[0], pos[1], th])
            else:  # smooth: single frames are only good to a few tens of cm, dead reckoning is smooth
                xy = self.est[:2] + 0.4 * (pos - self.est[:2])
                self.est = np.array([xy[0], xy[1], float(self.est[2] + 0.6 * wrap(th - self.est[2]))])
        self.est[2] = float(wrap(self.est[2]))
        self.sd, self.miss = 0.15, 0
        self.node, self.fix_t = c, self.t
        self.n_fix += 1
        return True

    # ------------------------------------------------------------------ decisions

    def act(self, obs: Observation):
        self.t += 1
        left = obs.steps_left if self.budget is None else min(obs.steps_left, self.budget - obs.step)
        if left <= 12:
            return self._out(Action.CHECKIN, "budget")
        small = cv2.resize(cv2.cvtColor(obs.image, cv2.COLOR_BGR2GRAY), (40, 30)).astype(np.float32)
        feat = self.matcher.features(obs.image)
        blocked = False
        if self.prev_act in ("FORWARD", "BACKWARD"):
            blocked = self.prev_small is not None and float(np.abs(small - self.prev_small).mean()) < 1.0
            if self.prev_feat is not None:  # moving forward must make the scene expand
                mm = self.matcher.match(self.prev_feat, feat)
                if mm.n >= 15:
                    sc = scale_ratio(mm)
                    blocked = blocked or not (sc > 1.004 if self.prev_act == "FORWARD" else sc < 0.996)
            self.stuck = self.stuck + 1 if blocked else 0
        self.blocked = blocked
        self.prev_small, self.prev_feat = small, feat

        self._predict()

        if self.queue:
            return self._out(self.queue.pop(0), "recover")
        if self.stuck >= 2:
            self.stuck, self.n_stuck = 0, self.n_stuck + 1
            turn = Action.RIGHT if self.n_stuck % 2 else Action.LEFT
            self.queue = [(turn, 26 + 6 * (self.n_stuck % 3))]
            if self.node >= 0:
                self.mem.punish([self.node], 3.0, 3)
            self.route = None  # plan again with the new penalty
            return self._out((Action.BACKWARD, 4), "stuck")

        # the final approach on the four goal photos (latched; identical to sptm_agent_4)
        near_goal = self.est is not None and float(np.hypot(*(self.est[:2] - self.P[self.goal, :2]))) < 1.0
        if self.appr_on or near_goal or 0 <= self.hops <= 15:
            r = self._approach(obs)
            if r is not None:
                self.appr_on, self.appr_lost = True, 0
                return r
            self.appr_lost += 1
            if self.appr_lost >= 3:
                self.appr_on, self.appr_steps, self.ah, self.dead = False, 0, None, 0

        # where are we? dead reckoning, corrected by the picture every few steps
        if self.est is None or self.sd > 0.35 or self.t - self.fix_t >= self.fix_every:
            self._localise(obs, feat)
        if self.est is None or self.node < 0:
            self.state = "lost: looking around"
            return self._out((Action.LEFT, 8), "search")

        self.mem.arrive(self.node)
        off_route = self.route is None or self.ri >= len(self.route) - 1
        if not off_route:
            win = self.route[self.ri : self.ri + 30]
            dd = np.hypot(*(self.P[win, :2] - self.est[:2]).T)
            off_route = float(dd.min()) > 0.6
            if not off_route:  # progress is monotone and slow (a U-turn in the route must not be skipped)
                self.ri += int(np.argmin(dd[:6]))
        if off_route:  # (re)plan only when we have no route, finished it, or the picture says we left it
            if self.route is not None and self.verbose:
                print(f"[{self.t}] re-plan from node {self.node}")
            self.route, self.ri = self.mem.path(self.node, limit=250), 0
            self.n_replan += 1
        path = self.route[self.ri :]
        self.node_on_route = path[0]
        self.hops = int(self.mem.hops[path[0]])
        if self.hops < self.best_hops - 2:
            self.best_hops, self.best_t = self.hops, self.t
        if self.mem.stagnating() or self.t - self.best_t > 200:
            self.est, self.sd, self.route = None, 1.0, None
            return self._escape_loop()
        if self.hops <= self.near_hops:
            r = self._approach(obs, weak=True)
            if r is not None:
                self.appr_on, self.appr_lost = True, 0
                return r
        if not self.appr_on:
            self.appr_steps = 0

        if len(path) == 1:  # standing on the goal frame but the goal photos do not match yet: look around
            return self._out((Action.LEFT, 4), "search-goal")

        # pure pursuit along the route's metric waypoints
        pts = self.P[path, :2]
        k0 = 0
        off = float(np.hypot(*(pts[k0] - self.est[:2])))
        tgt = pts[-1]
        for k in range(k0, min(len(path), k0 + 30)):
            if float(np.hypot(*(pts[k] - self.est[:2]))) >= self.look:
                tgt = pts[k]
                break
        want = math.atan2(tgt[1] - self.est[1], tgt[0] - self.est[0])
        err = math.degrees(float(wrap(want - self.est[2])))
        self.state = (f"at ({self.est[0]:.2f},{self.est[1]:.2f}) {math.degrees(self.est[2]):.0f}deg sd={self.sd:.2f} "
                      f"off-route={off:.2f} err={err:+.0f} fixes={self.n_fix} replans={self.n_replan}")
        if abs(err) > 6:
            units = int(np.clip(round(abs(err) / TICK_DEG), 1, 8 if abs(err) > 30 else 3))
            return self._out((Action.LEFT if err > 0 else Action.RIGHT, units), "follow")
        return self._out((Action.FORWARD, 2 if self.hops > 20 and abs(err) < 4 else 1), "follow")


SptmAgent = PathAgent  # so eval_sptm.py / main() can pick it up like the other versions


def main() -> None:
    base.SptmAgent = PathAgent
    base.main()


if __name__ == "__main__":
    main()
