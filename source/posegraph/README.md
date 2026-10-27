# posegraph -- our own pose graph (separate from every other agent)

Nothing outside this folder was edited. Existing modules are imported only.
ORB-SLAM3 is a separate project and is not part of this folder.

## Status

1. **Odometry calibration (done):** `calibrate_odometry.py` measures the true motion from
   the ground-truth poses of the local mazes.
   - Forward: median 0.111 m per recorded step. The map assumes 3 ticks per step, so this is about
     0.037 m per tick, against the 0.03 m in `metric_map.py`.
   - Turns: median 10.4 deg per step, which is 3.46 deg per tick. That matches the assumed 3.44 deg.
   - The forward spread is large, and many steps have no forward motion (blocked by walls).
2. **Pose check with a calibrated forward distance (ready to run):** `check_poses_v2.py`.
3. **Pose-graph localiser for the agent (not started):** use the pose map to localise the
   camera and plan on metric positions, instead of the VLAD node estimate. This is the stage
   that addresses the localisation problem.

## Commands (run from the repo root)

```
uv run --no-sync source/posegraph/calibrate_odometry.py --seeds 7,8,9,10,11,110
uv run --no-sync source/posegraph/check_poses_v2.py --seed 110 --forward 0.037 2>&1 | tee logs/posegraph_110_037.txt
uv run --no-sync source/posegraph/check_poses_v2.py --seed 110 --forward 0.030 2>&1 | tee logs/posegraph_110_030.txt
```

Compare the two logs. The second run is the old constant, so it is the baseline.
