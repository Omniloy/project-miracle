#!/usr/bin/env python3
"""Build one adjudication case file per unsolved synthetic task: scenario, gold write-set, every agent trajectory's
write-set + closing explanation, and the text of the task's required policy documents. Used for manual gold repair.
Usage: case_files.py TASK_IDS... (reads data_synth tasks + all data_synth/simulations/*/results.json)"""
import glob, json, sys, os
from pathlib import Path
S = Path(__file__).resolve().parents[2]
TASKS = S / "harness/data_synth/tau2/domains/banking_knowledge/tasks"
DOCS = S / "harness/tau2-bench/data/tau2/domains/banking_knowledge/documents"
OUT = S / "runs/case_files"; OUT.mkdir(exist_ok=True)
WRITE_SKIP = ("get_", "KB_search", "grep", "get_current_time", "unlock_discoverable_agent_tool", "get_user_information", "search")

def parse_args(x):
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return x
    return x

def writes(msgs):
    out = []
    for m in msgs:
        if m["role"] != "assistant": continue
        for tc in m.get("tool_calls") or []:
            n = tc["name"]; a = tc.get("arguments") or {}
            if n == "call_discoverable_agent_tool":
                t = a.get("agent_tool_name", "")
                if t.startswith("get_") or t.startswith("search"): continue
                out.append((t, parse_args(a.get("arguments"))))
            elif n in ("transfer_to_human_agents", "give_discoverable_user_tool", "call_discoverable_user_tool", "log_verification"):
                out.append((n, {k: v for k, v in a.items() if k not in ("address","email","phone_number","date_of_birth","name")}))
    return out

sims = {}
for f in glob.glob(str(S / "harness/data_synth/simulations/*/results.json")):
    run = Path(f).parent.name
    try: d = json.load(open(f))
    except Exception: continue
    agent = ((d.get("info") or {}).get("agent_info") or {}).get("llm") or run
    for s in d["simulations"]:
        sims.setdefault(s["task_id"], []).append((run, agent, s))

for tid in sys.argv[1:]:
    tf = TASKS / f"task_{tid}.json"
    if not tf.exists(): print("missing", tid); continue
    t = json.load(open(tf))
    L = [f"# {tid}", "", "## Scenario (purpose)", t["description"].get("purpose", ""), "", "## User instructions"]
    us = t.get("user_scenario") or {}
    L.append(json.dumps(us.get("instructions"), indent=1, ensure_ascii=False)[:3000])
    L += ["", "## Injected rows"]
    inj = ((t.get("initial_state") or {}).get("initialization_data") or {}).get("agent_data") or {}
    for table, v in inj.items():
        for rid, row in ((v or {}).get("data") or {}).items():
            L.append(f"- {table}/{rid}: {json.dumps(row, ensure_ascii=False)[:600]}")
    L += ["", "## GOLD actions"]
    for a in t["evaluation_criteria"]["actions"]:
        args = a.get("arguments") or {}
        L.append(f"- {a['name']} {args.get('agent_tool_name') or ''} {json.dumps(parse_args(args.get('arguments')) if 'arguments' in args else {k:v for k,v in args.items() if k in ('user_id','reason')}, ensure_ascii=False)}")
    L += ["", f"## Agent trajectories ({len(sims.get(tid, []))})"]
    for run, agent, s in sims.get(tid, []):
        rw = (s.get("reward_info") or {}).get("reward")
        L.append(f"### {run} / {agent} trial {s.get('trial')} strict={rw}")
        for n, a in writes(s["messages"]):
            L.append(f"- {n} {json.dumps(a, ensure_ascii=False)[:400]}")
        last = [m for m in s["messages"] if m["role"] == "assistant" and (m.get("content") or "").strip()]
        if last: L.append("  FINAL: " + last[-1]["content"].replace("\n", " ")[:700])
    L += ["", "## Required documents"]
    for did in t.get("required_documents") or []:
        p = DOCS / f"{did}.json"
        if p.exists():
            d = json.load(open(p)); L.append(f"### {did}: {d.get('title')}"); L.append((d.get("content") or "")[:3500])
    open(OUT / f"{tid}.md", "w").write("\n".join(L))
    print("wrote", tid, len("\n".join(L)))
