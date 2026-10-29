# USE THIS AGENT: `frozen_1820lg/my_agent_1820lg.py`

**Results**
- **Sample challenge (dry run, 2026-10-29): 7/10, perfect tier.** Distance 3, Heading 3, SPL 1. It checked in 0.05 m from the goal and 1.4 deg off heading, after 1,202 steps (SPL 0.63).
- **Local maze 17: 6/10.** 0.02 m, 0.2 deg, 2,525 steps. Every earlier agent scored 0 there.

**What it is:** Kartik's `my_agent_1820` ([kartikparameshwaran/vis_nav_player_work](https://github.com/kartikparameshwaran/vis_nav_player_work), commit `20e6de9`), plus two additions:
1. **4-photo goal check** (`my_agent_2767g.py`): the goal place is chosen by how well all four goal photos match exploration frames within +-20 frames, not by one photo's best match.
2. **SuperPoint + LightGlue instead of SuperGlue** for every match (`lg_matcher.py`).

Every file is frozen as it ran; `frozen_1820lg/MD5SUMS.txt` has the checksums. Logs of both runs are in the folder.

## Setup
Do this inside a checkout of Kartik's repo, which has `models/`, `third_party/SuperGlue` and `source/`:
```bash
git clone https://github.com/kartikparameshwaran/vis_nav_player_work.git
cd vis_nav_player_work
cp -R /path/to/perception_midterm/frozen_1820lg .
uv pip install git+https://github.com/cvg/LightGlue.git     # once; downloads weights on first use
cp .env.example .env    # then set the API key and VIS_NAV_CHALLENGE
```

## Run
```bash
# local maze (free):
python frozen_1820lg/my_agent_1820lg.py --local 17 --budget 4000
# challenge in .env (spends an attempt after you answer y): NO "| tee", NO --yes
python frozen_1820lg/my_agent_1820lg.py
```
On a new maze, Kartik's map step (`map: measuring traj_N ...`) runs with LightGlue and takes 10-40 min before the run starts. It is cached in `cache/odometry/`. No attempt is spent before the `Start attempt N of M?` question.

**Known weaknesses** (local mazes 10 and 67 ran out of the 4,000-step budget with the goal correct):
- Kartik's "reverse along a drive" rule.
- Back-and-forth turning.
- Swapping between the same spot on two drives.

Work on those is in the private working copy (`my_agent_1820lg2.py`, not yet tested).

---

# Visual Navigation Challenge — starter kit

NYU **ROB-UY 3203 Robot Vision** and **ROB-GY 6203 Robot Perception**, run by the
[AI4CE Lab](https://ai4ce.github.io/).

A robot sits in a maze on the course server. Your code receives its camera frames and four
photos of the goal, chooses each move, and checks in when it thinks it has arrived. This
repository is what you fork: a keyboard agent to drive the maze yourself, a place-recognition
baseline, and the skeleton your own agent goes in.

**The tutorial is on the course site:** <https://visual-navigation-challenge.ai4ce.dev/>
Environment setup, a first drive, the agent interface, how the baseline works, and how runs
are scored, step by step. Everything below is the short version.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # Windows (PowerShell): irm https://astral.sh/uv/install.ps1 | iex
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
uv sync
```

## Credentials

Copy `.env.example` to `.env` and fill in your API key (from the course site, under the
account menu) and the challenge id (the last part of a challenge page's URL). Every script
reads the file; git ignores it.

```
VIS_NAV_API_KEY=...
VIS_NAV_CHALLENGE=...
```

## Run

```bash
uv run source/keyboard_agent.py   # drive by hand: arrows move, space checks in, esc quits
uv run source/baseline_agent.py   # drive with the baseline's hints
uv run source/my_agent.py         # your agent (source/my_agent.py)
```

Every script accepts `--challenge` and `--api-key` in place of `.env`, plus `--yes`,
`--no-browser`, `--no-check` and, where relevant, `--data <dir>`. When a new SDK version is
out the scripts say so and offer to update.

## On your own machine

The server's simulator is also a Python package, and `uv sync` installs it. With it, any
script runs on a maze on your machine instead of a challenge: no attempt, no key, no
network, the same code. The first run downloads the texture pack (123 MB).

```bash
uv run source/keyboard_agent.py --local 7    # maze 7: the same maze on every machine
uv run source/my_agent.py --local 7
```

A seed is a maze; share one like a challenge id. The exploration data for a local maze is
recorded on first use into `data/local-<seed>/`, so the baseline works there too.

**Experimental:** `source/rl/` is a reinforcement learning demo on the simulator: a Gymnasium
environment with the robot's true position for the reward (which only exists locally),
skrl's PPO to train on a list of mazes, and the trained policy as an agent that sees only
frames. It shows the plumbing, not a way to score: out of the box it learns to move, not to
find the goal. The baseline in `source/baseline_agent.py` is the course's reference agent;
start there.

```bash
uv sync --group rl          # gymnasium, skrl, and PyPI's torch: CPU on macOS and Windows, CUDA on Linux
uv sync --group rl-cuda     # ...or the CUDA build, for an NVIDIA card on Windows or Linux
uv sync --group rl-cpu      # ...or the CPU build everywhere: the small download
uv run source/rl/env.py --mazes 7,8,9                         # a random policy, to see the environment
uv run source/rl/train.py --mazes 7,8,9,10 --timesteps 200000 # -> models/policy.pt
uv run source/rl/play.py --local 11                           # a maze it never saw
uv run source/rl/play.py --challenge <id>                     # one real attempt
```

## What is here

| file | what |
|---|---|
| `source/my_agent.py` | the skeleton: `__init__`, `setup`, `act`, `finish`, `hud`, `panel`, each with a comment saying when it runs |
| `source/keyboard_agent.py` | drive with the arrow keys |
| `source/baseline_agent.py`, `source/vlad.py` | RootSIFT + VLAD place recognition over the exploration frames, a graph of them, and the next move along the shortest path |
| `source/cli.py` | the shared command line, `.env` loading, `--local` |
| `source/rl/` | experimental RL demo: `env.py` the Gymnasium environment, `models.py` the networks, `train.py` PPO with skrl, `play.py` the policy as an agent |

The SDK's own documentation (`connect()`, `Session`, the REST client) is at
<https://visual-navigation-challenge.ai4ce.dev/sdk>.
