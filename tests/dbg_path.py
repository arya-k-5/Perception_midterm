"""Debug: run the path agent on the stand-in maze and print estimated vs true pose along the way."""
import math, json, sys
import numpy as np
import minisim
from path_agent import PathAgent
from test_sptm_minisim import FakeScorer
from test_path_minisim import _align_error
from pathlib import Path
import tempfile

tmp = Path(tempfile.mkdtemp())
world, start, goal = minisim.make_episode(5, tmp, n=5, length=45)
agent = PathAgent(tmp, FakeScorer(tmp), s_local=0.4, s_reach=0.5, s_shortcut=0.6, window=2, anchor_pos=None, verbose=False)
agent.focal = 92.0
agent.build_map()
truth = np.array([json.loads((f.path.parent / "data_info.json").read_text())[int(f.path.stem)]["pose"] for f in agent.frames])
# map -> world transform
est = agent.P
mu_e, mu_t = est[:, :2].mean(0), truth[:, :2].mean(0)
a, b = est[:, :2] - mu_e, truth[:, :2] - mu_t
u, _, vt = np.linalg.svd(a.T @ b)
r = u @ np.diag([1, np.sign(np.linalg.det(u @ vt))]) @ vt
dth = math.atan2(r[0, 1], r[0, 0])
to_world = lambda p: (np.array(p[:2]) - mu_e) @ r + mu_t
print("start", start, "goal", goal)
orig_act = agent.act
step = [0]
def act(obs):
    out = orig_act(obs)
    step[0] += 1
    if (step[0] % 25 == 0) and agent.est is not None:
        w = to_world(agent.est)
        ex = math.hypot(*(w - np.array([world.x, world.y])))
        eh = math.degrees((agent.est[2] + dth - world.th + math.pi) % (2 * math.pi) - math.pi)
        print(f"act {step[0]:4d} true ({world.x:.2f},{world.y:.2f},{math.degrees(world.th):4.0f}) pose-err {ex:.2f} m {eh:+4.0f} deg node {agent.node} hops {agent.hops} {agent.state[-75:]} -> {out}")
    return out
agent.act = act
res = minisim.run_episode(agent, world, start, goal, max_steps=2500)
print(res)
