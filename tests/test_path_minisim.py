"""Path agent on the stand-in maze: the metric map must match the true poses, and the agent must run end to end."""

from __future__ import annotations

import json
import math

import minisim
import numpy as np
from path_agent import PathAgent
from test_sptm_minisim import FakeScorer


def _align_error(est, truth):
    """RMS position error and mean heading error after the best rigid transform of est onto truth."""
    mu_e, mu_t = est[:, :2].mean(0), truth[:, :2].mean(0)
    a, b = est[:, :2] - mu_e, truth[:, :2] - mu_t
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, d]) @ vt
    pos = (a @ r + mu_t) - truth[:, :2]
    dth = np.arctan2(r[0, 1], r[0, 0])
    hd = (est[:, 2] + dth - truth[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return float(np.sqrt((pos**2).sum(1).mean())), float(np.degrees(np.abs(hd)).mean())


def test_metric_map_and_path_agent(tmp_path):
    world, start, goal = minisim.make_episode(5, tmp_path, n=5, length=45)
    agent = PathAgent(tmp_path, FakeScorer(tmp_path), s_local=0.4, s_reach=0.5, s_shortcut=0.6, window=2,
                      anchor_pos=None)
    agent.focal = 92.0
    agent.build_map()
    truth = []
    for f in agent.frames:
        recs = json.loads((f.path.parent / "data_info.json").read_text())
        truth.append(recs[int(f.path.stem)]["pose"])
    truth = np.array(truth)
    e_opt, h_opt = _align_error(agent.P, truth)
    e_odo, h_odo = _align_error(agent.odo, truth)
    print(f"map error: odometry {e_odo:.2f} m / {h_odo:.0f} deg -> optimised {e_opt:.2f} m / {h_opt:.0f} deg")
    assert e_opt <= e_odo + 0.05
    assert e_opt < 0.6 and h_opt < 25
    res = minisim.run_episode(agent, world, start, goal, max_steps=2500)
    print(res, "fixes", agent.n_fix)
    assert res["steps"] > 0 and (res["checked_in"] or res["steps"] >= 2400)
    assert agent.n_fix > 5
