#!/usr/bin/env python3
"""Split long agent trajectories into NON-OVERLAPPING training windows that fit a sequence budget.

Banking trajectories run 30-115K tokens (retrieved policy docs come back as tool results); a 27B bf16 LoRA on one
96 GB card trains comfortably at <=16K tokens. Each window = system prompt + the opening exchange (first user
request and first assistant reply, so the agent always knows what the customer asked) + a consecutive run of
messages. Tool-call/tool-result groups are never split. Every assistant turn inside a window is a training target
(all turns come from verifier-passed trajectories). Char budget ~3.6 chars/token.
Usage: chunk_sft.py IN.jsonl OUT.jsonl [--max-chars 58000]
"""
import argparse, json, sys

def size(m): return len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or ""))

def groups(body):
    """Yield atomic message groups: an assistant tool-call message together with its tool results."""
    i = 0
    while i < len(body):
        g = [body[i]]; i += 1
        if g[0]["role"] == "assistant" and g[0].get("tool_calls"):
            while i < len(body) and body[i]["role"] == "tool": g.append(body[i]); i += 1
        yield g

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--max-chars", type=int, default=58000); ap.add_argument("--tool-cap", type=int, default=8000, help="truncate each tool result to this many chars (context-management approximation)")
    a = ap.parse_args(); n_in = n_out = n_drop = 0
    with open(a.out, "w") as fo:
        for line in open(a.inp):
            r = json.loads(line); n_in += 1
            msgs = r["messages"]
            for m in msgs:  # cap giant retrieval dumps; keep the head (ranked results come first)
                if m["role"] == "tool" and len(m.get("content") or "") > a.tool_cap:
                    m["content"] = m["content"][:a.tool_cap] + f"\n... [truncated {len(m['content'])-a.tool_cap} chars]"
            sysm = msgs[0] if msgs and msgs[0]["role"] == "system" else None
            body = msgs[1:] if sysm else msgs
            gs = list(groups(body))
            # opening = groups up to and including the first user message and the assistant reply after it
            open_n = 0
            for k, g in enumerate(gs):
                if g[0]["role"] == "user": open_n = min(len(gs), k + 2); break
            opening = [m for g in gs[:open_n] for m in g]
            base = (size(sysm) if sysm else 0) + sum(size(m) for m in opening)
            rest = gs[open_n:]
            windows, cur, used = [], [], 0
            for g in rest:
                gsz = sum(size(m) for m in g)
                if gsz + base > a.max_chars: n_drop += 1; continue  # single group too big (huge doc dump)
                if cur and used + gsz + base > a.max_chars:
                    windows.append(cur); cur, used = [], 0
                cur += g; used += gsz
            if cur or not windows: windows.append(cur)
            for w_i, w in enumerate(windows):
                seq = ([sysm] if sysm else []) + opening + w
                if not any(m["role"] == "assistant" for m in seq): continue
                fo.write(json.dumps({"messages": seq, "tools": r.get("tools") or [], "meta": dict(r.get("meta") or {}, window=w_i, n_windows=len(windows))}, ensure_ascii=False) + "\n"); n_out += 1
    print(f"{a.inp}: {n_in} trajectories -> {n_out} windows (dropped {n_drop} oversized groups)", file=sys.stderr)

if __name__ == "__main__": main()
