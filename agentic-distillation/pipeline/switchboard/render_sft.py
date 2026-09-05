#!/usr/bin/env python3
"""Render accepted traces to Axolotl chat rows: {"messages": [...], "tools": [...], "meta": {...}}. The system prompt is kept as authored;
tools come from the scenario catalog. Usage: render_sft.py ACCEPTED.jsonl SCENARIOS.jsonl OUT.jsonl"""
import json, sys
acc, scs, out = sys.argv[1:4]
S = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(scs)}
n = 0
with open(out, "w") as f:
    for l in open(acc):
        tr = json.loads(l); sc = S[tr["scenario_id"]]
        msgs = []
        for m in tr["messages"]:
            e = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_calls"): e["tool_calls"] = m["tool_calls"]
            if m["role"] == "tool": e["tool_call_id"] = m.get("tool_call_id")
            msgs.append(e)
        f.write(json.dumps({"messages": msgs, "tools": sc["tools"], "meta": {"scenario_id": sc["scenario_id"], "vertical": sc["vertical"], "language": sc["language"], "style": sc["style"], "intent": sc["intent"], "patterns": sc["target_patterns"]}}, ensure_ascii=False) + "\n"); n += 1
print("rendered", n)
