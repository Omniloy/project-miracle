#!/usr/bin/env python3
"""Re-score existing trajectories against the CURRENT (possibly repaired) task definitions.
Collects every simulation for the given tasks across data_synth/simulations/*/results.json, rebuilds a results.json
whose `tasks` are the current task files, and hands it to relaxed_verify.py. Passing trajectories are written to a
results-shaped file usable by traj_to_sft.py (rewards rewritten to the relaxed score).
Usage (tau2 .venv python, TAU2_DATA_DIR set): rescore.py OUT_PREFIX TASK_ID...  -> OUT_PREFIX.results.json, .relaxed.json, .pass.json"""
import glob, json, os, subprocess, sys
from pathlib import Path
S = Path(__file__).resolve().parents[2]
TASKS = S / "harness/data_synth/tau2/domains/banking_knowledge/tasks"
out, tids = sys.argv[1], sys.argv[2:]
sims, base = [], None
for f in sorted(glob.glob(str(S / "harness/data_synth/simulations/*/results.json"))):
    run = Path(f).parent.name
    if run.startswith("rescore_"): continue
    d = json.load(open(f)); base = base or d
    for s in d["simulations"]:
        if s["task_id"] in tids and s.get("reward_info") is not None:
            s = dict(s); s["_run"] = run; s["trial"] = len(sims); sims.append(s)
tasks = [json.load(open(TASKS / f"task_{t}.json")) for t in tids]
for t in tasks:
    for i, a in enumerate(t["evaluation_criteria"]["actions"]): a.setdefault("action_id", f"{t['id']}_{i}")
res = {k: v for k, v in base.items() if k not in ("tasks", "simulations")}; res["tasks"] = tasks; res["simulations"] = sims
json.dump(res, open(f"{out}.results.json", "w"))
index = {s["trial"]: (s["_run"], s["task_id"]) for s in sims}; json.dump(index, open(f"{out}.index.json", "w"), indent=1)
r = subprocess.run([sys.executable, str(S / "pipeline/banking/relaxed_verify.py"), f"{out}.results.json", "--out", f"{out}.relaxed.json"], capture_output=True, text=True)
print(r.stderr[-6000:], file=sys.stderr)
rel = {(x["task_id"], x["trial"]): x for x in json.load(open(f"{out}.relaxed.json"))}
passing = []
for s in sims:
    x = rel.get((s["task_id"], s["trial"]))
    if x and x.get("relaxed") == 1.0:
        s2 = dict(s); s2["reward_info"] = dict(s2["reward_info"] or {}); s2["reward_info"]["reward"] = 1.0; passing.append(s2)
res["simulations"] = passing; json.dump(res, open(f"{out}.pass.json", "w"))
by = {}
for s in passing: by.setdefault(s["task_id"], []).append(s["_run"])
print("PASS after rescore:", {k: len(v) for k, v in by.items()}, "| tasks with 0 passes:", sorted(set(tids) - set(by)))
