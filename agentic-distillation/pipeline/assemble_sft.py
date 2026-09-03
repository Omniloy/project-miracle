#!/usr/bin/env python3
"""Assemble the SFT training set from verified trajectories (output of traj_to_sft.py).

Rules:
  - only rows with success==True (verifier-passed) are eligible
  - per (task_id): keep at most --per-task trajectories, preferring the SHORTEST passing ones (minimal exact
    write set; an extra unlock/call is noise), and at most one per agent to keep teacher diversity
  - hold out whole tasks (--holdout-frac) as a dev set so dev evaluation never sees a trained trajectory
  - emit Axolotl chat_template format: {"messages":[{role,content,tool_calls?,tool_call_id?}], "tools":[...]}
    (Axolotl masks loss to assistant turns via roles_to_train; our per-message `train` flags are dropped)
Usage: assemble_sft.py OUT_DIR sft_a.jsonl sft_b.jsonl ... [--per-task 3] [--holdout-frac 0.15] [--seed 0]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def clean_msgs(msgs):
    out = []
    for m in msgs:
        e = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            e["tool_calls"] = m["tool_calls"]
        if m["role"] == "tool" and m.get("tool_call_id"):
            e["tool_call_id"] = m["tool_call_id"]
        out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = []
    for f in a.inputs:
        for line in open(f):
            r = json.loads(line)
            if r.get("success") and r.get("reward") == 1.0 and r["n_assistant_turns"] >= 2:
                r["_src"] = Path(f).stem
                rows.append(r)
    by_task = defaultdict(list)
    for r in rows:
        by_task[(r["domain"], r["task_id"])].append(r)

    tasks = sorted(by_task)
    rng.shuffle(tasks)
    n_hold = max(1, int(len(tasks) * a.holdout_frac)) if a.holdout_frac > 0 else 0
    holdout = set(tasks[:n_hold])

    train, dev, stats = [], [], defaultdict(int)
    for key, rs in by_task.items():
        rs.sort(key=lambda r: sum(len(m.get("content") or "") for m in r["messages"]))  # shortest first
        seen_agents, picked = set(), []
        for r in rs:
            ag = (r.get("agent_llm") or r["_src"])
            if ag in seen_agents and len(rs) > a.per_task:
                continue
            seen_agents.add(ag)
            picked.append(r)
            if len(picked) >= a.per_task:
                break
        for r in picked:
            rec = {"messages": clean_msgs(r["messages"]), "tools": r.get("tools") or [], "meta": {"domain": r["domain"], "task_id": r["task_id"], "agent": r.get("agent_llm"), "src": r["_src"]}}
            (dev if key in holdout else train).append(rec)
            stats[r["_src"]] += 1

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng.shuffle(train)
    with open(out / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "dev.jsonl", "w") as f:
        for r in dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"holdout_tasks": sorted(t[1] for t in holdout), "train_tasks": sorted(t[1] for t in tasks if t not in holdout)}, open(out / "split.json", "w"), indent=1)
    tok_est = sum(len(json.dumps(r)) for r in train) // 4
    print(f"eligible rows {len(rows)} over {len(tasks)} tasks | train {len(train)} rows ({len(tasks)-n_hold} tasks, ~{tok_est/1e6:.1f}M tokens est) | dev {len(dev)} rows ({n_hold} held-out tasks) | by source {dict(stats)}", file=sys.stderr)


if __name__ == "__main__":
    main()
