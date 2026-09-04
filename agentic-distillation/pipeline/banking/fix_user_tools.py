#!/usr/bin/env python3
"""Synthetic tasks that need the customer to call a discoverable USER tool must (like the real tasks) list
`call_discoverable_user_tool` in user_tools; the generator wrote the discoverable tool's own name there, so the user
simulator never had the meta-tool and 0/425 synthetic simulations contained a user tool call. This script:
  - sets user_tools = ["call_discoverable_user_tool"] (+ any legit extras)
  - ensures the gold has the give_discoverable_user_tool / call_discoverable_user_tool pair (restored from the
    pre-ruling backup when a reviewer dropped it) before the first write that consumes the tool's output
  - replaces the last-4 digits in gold with the value the user tool really returns (computed in the env)
Run with tau2 .venv python. Usage: fix_user_tools.py TASK_ID..."""
import json, re, sys
from pathlib import Path
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.environment import get_environment
S = Path(__file__).resolve().parents[2]; T = S / "harness/data_synth/tau2/domains/banking_knowledge/tasks"; B = S / "runs/task_backups"
for tid in sys.argv[1:]:
    p = T / f"task_{tid}.json"; t = json.load(open(p)); acts = t["evaluation_criteria"]["actions"]
    pair = [a for a in acts if a["name"] in ("give_discoverable_user_tool", "call_discoverable_user_tool")]
    if not pair and (B / f"task_{tid}.json").exists():
        pair = [a for a in json.load(open(B / f"task_{tid}.json"))["evaluation_criteria"]["actions"] if a["name"] in ("give_discoverable_user_tool", "call_discoverable_user_tool")]
        # insert before the first agent write that needs the digits (dispute filing), else before first write
        idx = next((i for i, a in enumerate(acts) if a["name"] == "unlock_discoverable_agent_tool" and "dispute" in a["arguments"].get("agent_tool_name", "")), None)
        if idx is None: idx = next((i for i, a in enumerate(acts) if a["name"] == "call_discoverable_agent_tool" and not a["arguments"].get("agent_tool_name", "").startswith("get_")), len(acts))
        acts[idx:idx] = pair; print(tid, "re-inserted user-tool pair at", idx)
    if not pair: print(tid, "no user-tool pair, skip"); continue
    call = next(a for a in pair if a["name"] == "call_discoverable_user_tool")
    name = call["arguments"].get("discoverable_tool_name") or call["arguments"].get("user_tool_name")
    call["arguments"] = {"discoverable_tool_name": name, "arguments": call["arguments"].get("arguments") if isinstance(call["arguments"].get("arguments"), str) else json.dumps(call["arguments"].get("arguments") or {})}
    give = next(a for a in pair if a["name"] == "give_discoverable_user_tool"); give["arguments"] = {"discoverable_tool_name": name}; give["requestor"] = "assistant"; call["requestor"] = "user"
    ut = [x for x in (t.get("user_tools") or []) if x in ("apply_for_credit_card", "submit_referral", "submit_transaction", "request_human_agent_transfer", "call_discoverable_user_tool")]
    if "call_discoverable_user_tool" not in ut: ut.append("call_discoverable_user_tool")
    t["user_tools"] = ut
    # compute the tool's real output
    task = Task(**{k: v for k, v in t.items() if not k.startswith("_")})
    env = get_environment(retrieval_variant="bm25_grep", task=task)
    init = task.initial_state.initialization_data if task.initial_state else None
    env.set_state(initialization_data=init, initialization_actions=None, message_history=[], strict=True)
    env.make_tool_call(tool_name="give_discoverable_user_tool", requestor="assistant", **give["arguments"])
    out = env.make_tool_call(tool_name="call_discoverable_user_tool", requestor="user", **call["arguments"])
    out = str(getattr(out, "content", out)); m = re.search(r"\b(\d{4})\b", out); digits = m.group(1) if m else None
    print(tid, name, "->", out[:120].replace("\n", " "), "| digits", digits)
    if digits and "last_4" in name:
        for a in acts:
            if a["name"] == "call_discoverable_agent_tool":
                d = json.loads(a["arguments"]["arguments"]) if isinstance(a["arguments"].get("arguments"), str) else (a["arguments"].get("arguments") or {})
                if "card_last_4_digits" in d and d["card_last_4_digits"] != digits:
                    print("   fixing card_last_4_digits", d["card_last_4_digits"], "->", digits); d["card_last_4_digits"] = digits; a["arguments"]["arguments"] = json.dumps(d)
    for i, a in enumerate(acts): a["action_id"] = f"{tid}_{i}"
    t["description"]["notes"] = (t["description"].get("notes") or "") + "\n[FIXED 2026-09-04] user_tools -> call_discoverable_user_tool; last-4 digits set to the user tool's real output."
    json.dump(t, open(p, "w"), indent=1, ensure_ascii=False)
