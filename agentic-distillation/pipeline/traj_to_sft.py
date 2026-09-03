#!/usr/bin/env python3
"""Convert tau2-bench results.json files into an SFT dataset (JSONL, chat+tools format).

Filtering follows GLM-4.5/5 practice: keep only environment-verified successes (reward == 1.0),
keep the full multi-turn trajectory, and emit a per-message `train` flag so the trainer can mask
loss to assistant turns only (user, tool and system turns are context, not targets).

Usage:
  traj_to_sft.py OUT.jsonl results1.json [results2.json ...] [--min-reward 1.0] [--keep-failures]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def norm_tool_calls(tcs):
    out = []
    for t in tcs or []:
        args = t.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        out.append({
            "id": t.get("id"),
            "type": "function",
            "function": {"name": t.get("name"), "arguments": args},
        })
    return out


try:  # reproduce the exact system prompt the agent saw: tau2 wraps the domain policy in its agent template
    from tau2.agent.llm_agent import SYSTEM_PROMPT as _SP, AGENT_INSTRUCTION as _AI

    def system_prompt(policy):
        return _SP.format(domain_policy=policy, agent_instruction=_AI)
except Exception:  # running outside the tau2 venv: fall back to the bare policy and say so once
    print("WARNING: tau2 not importable; system prompt = bare domain policy (run with tau2 .venv python for fidelity)", file=sys.stderr)

    def system_prompt(policy):
        return policy


def convert_sim(sim, task, tools, domain, src):
    msgs = []
    if sim.get("policy"):
        msgs.append({"role": "system", "content": system_prompt(sim["policy"]), "train": False})
    for m in sim.get("messages", []):
        role = m.get("role")
        # User-side tool calls (e.g. the customer submitting an application) and their results happen on the
        # user simulator's side; the agent never sees them, so they must not appear in its training context.
        if role == "user" and m.get("tool_calls") and not (m.get("content") or "").strip():
            continue
        if role == "tool" and m.get("requestor") == "user":
            continue
        if role == "assistant":
            entry = {"role": "assistant", "content": m.get("content") or "", "train": True}
            if m.get("tool_calls"):
                entry["tool_calls"] = norm_tool_calls(m["tool_calls"])
            msgs.append(entry)
        elif role == "user":
            msgs.append({"role": "user", "content": m.get("content") or "", "train": False})
        elif role == "tool":
            msgs.append({
                "role": "tool",
                "tool_call_id": m.get("id") or m.get("tool_call_id"),
                "content": m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"), ensure_ascii=False),
                "train": False,
            })
    r = (sim.get("reward_info") or {})
    key = hashlib.sha1(f"{domain}|{sim.get('task_id')}|{sim.get('trial')}|{src}".encode()).hexdigest()[:16]
    return {
        "id": key,
        "domain": domain,
        "task_id": sim.get("task_id"),
        "trial": sim.get("trial"),
        "reward": r.get("reward"),
        "termination": str(sim.get("termination_reason")),
        "agent_llm": (sim.get("info") or {}).get("agent_llm"),
        "n_assistant_turns": sum(1 for x in msgs if x["role"] == "assistant"),
        "tools": tools,
        "messages": msgs,
        "source_file": src,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("results", nargs="+")
    ap.add_argument("--min-reward", type=float, default=1.0)
    ap.add_argument("--keep-failures", action="store_true", help="also emit reward < min (tagged) for RFT/DPO negatives")
    a = ap.parse_args()

    # tau2 results.json leaves environment_info.tool_defs = None; tool schemas are exported per domain
    # (OpenAI function format) by dump from the tau2 registry into pipeline/tools/<domain>.json.
    tools_dir = Path(__file__).parent / "tools"

    def load_tools(domain):
        p = tools_dir / f"{domain}.json"
        return json.load(open(p)) if p.exists() else []

    n_in = n_keep = 0
    with open(a.out, "w") as fo:
        for rf in a.results:
            d = json.load(open(rf))
            info = d.get("info") or {}
            domain = (info.get("environment_info") or {}).get("domain_name") or info.get("domain") or Path(rf).parent.name
            tools = (info.get("environment_info") or {}).get("tool_defs") or load_tools(domain)
            if not tools:
                print(f"WARNING: no tool schemas for domain {domain} ({rf})", file=sys.stderr)
            tasks = {t.get("id"): t for t in d.get("tasks", [])}
            for sim in d.get("simulations", []):
                n_in += 1
                rew = (sim.get("reward_info") or {}).get("reward")
                ok = rew is not None and rew >= a.min_reward
                if not ok and not a.keep_failures:
                    continue
                rec = convert_sim(sim, tasks.get(sim.get("task_id")), tools, domain, rf)
                rec["success"] = bool(ok)
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_keep += 1
    print(f"read {n_in} simulations, wrote {n_keep} -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
