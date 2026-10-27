sample_1_agent: the exact code that scored 3/10 (PARTIAL, 0.15 m, 14 deg, 4,766 steps) on the sample challenge.
It is source/ as of git commit 3120c1d ("Latch goal approach, looser check-in"), frozen so later edits cannot change it.
sample_1_agent.py is an identical copy of sptm_agent.py (md5 758a4054d10cfd78afe89a8439bfbcdd).

Run it from the project root (the folder that has data/, cache/, .env):

  uv run --no-sync sample_1_agent/sample_1_agent.py --classical --verbose --log-dir logs/s1 2>&1 | tee logs/s1.txt

Local maze instead of the challenge:  add  --local 8 --budget 5000
