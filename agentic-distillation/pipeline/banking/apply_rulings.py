#!/usr/bin/env python3
"""Apply reviewed GOLD_WRONG rulings (runs/rulings/<task>.json) to the synthetic task files.
Only rulings listed on the command line are applied (review them first!). Records the justification in
description.notes, regenerates action_ids, and writes the task back. Then run validate_task.py + rescore.py.
Usage: apply_rulings.py TASK_ID... """
import json, sys
from pathlib import Path
S = Path(__file__).resolve().parents[2]
TASKS = S / "harness/data_synth/tau2/domains/banking_knowledge/tasks"; RUL = S / "runs/rulings"
for tid in sys.argv[1:]:
    r = json.load(open(RUL / f"{tid}.json"))
    if r["verdict"] != "GOLD_WRONG" or not r.get("repaired_actions"):
        print("skip", tid, r["verdict"]); continue
    p = TASKS / f"task_{tid}.json"; t = json.load(open(p))
    acts = []
    for a in r["repaired_actions"]:
        a = {k: v for k, v in a.items() if k != "action_id"}; a.setdefault("requestor", "assistant")
        args = a.get("arguments") or {}
        if a["name"] == "call_discoverable_agent_tool" and not isinstance(args.get("arguments"), str):
            args["arguments"] = json.dumps(args.get("arguments") or {})
        if a["name"] == "log_verification":
            args.setdefault("time_verified", "2025-11-14 03:40:00 EST")
        acts.append(a)
    t["evaluation_criteria"]["actions"] = acts
    for i, a in enumerate(acts): a["action_id"] = f"{tid}_{i}"
    if r.get("inject_rows"):
        init = t.setdefault("initial_state", {}).setdefault("initialization_data", {}); ad = init.setdefault("agent_data", {}) or {}
        for table, rows in r["inject_rows"].items():
            ad.setdefault(table, {}).setdefault("data", {}).update(rows)
        init["agent_data"] = ad
    if r.get("user_instruction_patch"):
        t.setdefault("user_scenario", {})["instructions"] = r["user_instruction_patch"]
    t["description"]["notes"] = (t["description"].get("notes") or "") + f"\n[REPAIRED 2026-09-04 ruling {r.get('confidence')}] {r.get('policy_basis','')[:400]} :: {r.get('reasoning','')[:400]}"
    json.dump(t, open(p, "w"), indent=1, ensure_ascii=False); print("applied", tid, len(acts), "actions")
