#!/usr/bin/env python3
"""OpenAI-compatible load generator for the tau2-bench serving benchmark (runs on the Vast box, driven by remote_bench.sh).

Builds N realistic requests from train_turns.jsonl (Axolotl chat rows {"messages","tools"}): for each row the request is
the messages up to (excluding) the LAST assistant turn, i.e. it ends with the last user/tool message that precedes an
assistant turn, plus the row's tools. Sampling matches the eval (temperature 1.0, top_p 0.95, top_k 20 via extra body),
default chat template (thinking on), max_tokens 1200. Runs `concurrency` worker threads that cycle through the request set
until `--seconds` elapse, then prints ONE JSON line (and appends it to --out) with throughput, latency, TTFT (cold/warm),
sanity pass rate (non-empty, tool calls parse, no <tool_call>/<think> leakage), and /metrics deltas (prefix-cache hit rate,
speculative-decoding acceptance) plus nvidia-smi power/clock averages. Never hangs: every HTTP call has a timeout, the
timed phase has a hard cap, and in-flight requests are abandoned after a grace period.

Usage: bench_client.py --model student --concurrency 8 --seconds 170 --config A_bf16_lora [--greedy-compare base] [--out results.jsonl]
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

try:
    import requests
except ImportError:  # both venvs ship requests, but never fail on it
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "requests"], check=False)
    import requests

TERMINAL_ROLES = ("user", "tool")


# ----------------------------------------------------------------------------------------------------------------------
# request construction
# ----------------------------------------------------------------------------------------------------------------------
def clean_msg(m):
    out = {"role": m["role"], "content": m.get("content") if m.get("content") is not None else ""}
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    if m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    return out


def build_requests(path, n, seed):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    reqs = []
    for ri, r in enumerate(rows):
        msgs = r["messages"]
        idx = None
        for i in range(len(msgs) - 1, 0, -1):  # last assistant turn preceded by a user/tool message
            if msgs[i]["role"] == "assistant" and msgs[i - 1]["role"] in TERMINAL_ROLES:
                idx = i
                break
        if idx is None:
            if msgs[-1]["role"] in TERMINAL_ROLES:  # no reference turn: generate the missing next turn
                idx = len(msgs)
            else:
                continue
        ref = msgs[idx] if idx < len(msgs) else None
        prefix = [clean_msg(m) for m in msgs[:idx]]
        if not prefix or prefix[-1]["role"] not in TERMINAL_ROLES:
            continue
        reqs.append({"row": ri, "messages": prefix, "tools": r.get("tools") or [],
                     "ref_kind": None if ref is None else ("tool_call" if ref.get("tool_calls") else "text"),
                     "chars": sum(len(m["content"]) + len(json.dumps(m.get("tool_calls", ""))) for m in prefix) + len(json.dumps(r.get("tools") or []))})
    if not reqs:
        raise SystemExit("no usable rows in " + path)
    import random
    random.Random(seed).shuffle(reqs)
    return reqs[:n]


# ----------------------------------------------------------------------------------------------------------------------
# metrics scraping
# ----------------------------------------------------------------------------------------------------------------------
METRIC_LINE = re.compile(r'^(vllm:[A-Za-z0-9_:]+?)(?:\{(.*)\})?\s+([-+0-9.eEinfa]+)\s*$')


def scrape_metrics(metrics_url):
    """Return {metric_name_without_total: sum over labels} and {metric_name: {position: value}} for per-position metrics."""
    try:
        txt = requests.get(metrics_url, timeout=15).text
    except Exception:
        return None
    tot, per_pos = {}, {}
    for line in txt.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = METRIC_LINE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        name = re.sub(r"_total$", "", name)
        tot[name] = tot.get(name, 0.0) + v
        pm = re.search(r'position="(\d+)"', labels)
        if pm:
            per_pos.setdefault(name, {})[int(pm.group(1))] = per_pos.setdefault(name, {}).get(int(pm.group(1)), 0.0) + v
    return {"tot": tot, "per_pos": per_pos}


def metric_delta(before, after, name):
    if not before or not after:
        return None
    b = sum(v for k, v in before["tot"].items() if k.endswith(name))
    a = sum(v for k, v in after["tot"].items() if k.endswith(name))
    return a - b


def per_pos_delta(before, after, name):
    if not before or not after:
        return None
    out = {}
    for k, pos in after["per_pos"].items():
        if k.endswith(name):
            for p, v in sorted(pos.items()):
                out[p] = v - before["per_pos"].get(k, {}).get(p, 0.0)
    return out or None


# ----------------------------------------------------------------------------------------------------------------------
# GPU sampler (power / clocks) - informative only, never fails the run
# ----------------------------------------------------------------------------------------------------------------------
class GpuSampler(threading.Thread):
    def __init__(self, period=10):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop = period, [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,power.limit,clocks.sm,utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
                self.samples.append([float(x) for x in out.split(",")])
            except Exception:
                pass
            self.stop.wait(self.period)

    def summary(self):
        if not self.samples:
            return {}
        cols = list(zip(*self.samples))
        return {"power_draw_w_mean": round(statistics.mean(cols[0]), 1), "power_limit_w": cols[1][-1],
                "sm_clock_mhz_mean": round(statistics.mean(cols[2])), "gpu_util_pct_mean": round(statistics.mean(cols[3]), 1),
                "gpu_mem_used_mib_max": max(cols[4])}


# ----------------------------------------------------------------------------------------------------------------------
# one streaming request
# ----------------------------------------------------------------------------------------------------------------------
def stream_one(args, req, max_tokens=None):
    body = {"model": args.model, "messages": req["messages"], "tools": req["tools"] or None, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k, "max_tokens": max_tokens or args.max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if not body["tools"]:
        del body["tools"]
    t0 = time.time()
    ttft, content, reasoning, tool_calls, finish, usage = None, [], [], {}, None, None
    with requests.post(args.base_url + "/chat/completions", json=body, headers={"Authorization": "Bearer dummy"},
                       stream=True, timeout=(15, args.timeout)) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode() if isinstance(line, bytes) else line
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                got = False
                if d.get("content"):
                    content.append(d["content"]); got = True
                rc = d.get("reasoning_content") or d.get("reasoning")
                if rc:
                    reasoning.append(rc); got = True
                for tc in d.get("tool_calls") or []:
                    got = True
                    slot = tool_calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if got and ttft is None:
                    ttft = time.time() - t0
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t1 = time.time()
    if finish is None and usage is None:
        raise RuntimeError("stream ended without finish_reason/usage")
    return {"latency": t1 - t0, "ttft": ttft, "content": "".join(content), "reasoning_chars": sum(map(len, reasoning)),
            "tool_calls": [tool_calls[k] for k in sorted(tool_calls)], "finish_reason": finish,
            "completion_tokens": (usage or {}).get("completion_tokens"), "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "t_end": t1}


def sanity(rec, req):
    """Non-empty, tool calls parse (JSON args + known tool name), no raw <tool_call>/<think> leaking into content."""
    content = rec["content"] or ""
    leak = ("<tool_call>" in content) or ("</tool_call>" in content) or ("<think>" in content) or ("</think>" in content)
    known = {t.get("function", {}).get("name") for t in req["tools"]}
    tc_ok = True
    for tc in rec["tool_calls"]:
        if not tc["name"] or (known and tc["name"] not in known):
            tc_ok = False
        try:
            json.loads(tc["arguments"] or "{}")
        except Exception:
            tc_ok = False
    kind = "tool_call" if rec["tool_calls"] else ("text" if content.strip() else "empty")
    passed = (not leak) and ((bool(rec["tool_calls"]) and tc_ok) or (kind == "text"))
    return passed, leak, tc_ok, kind


# ----------------------------------------------------------------------------------------------------------------------
# greedy A/B (guards against the vLLM "LoRA silently no-op" bug #49354): same prompts, model A vs model B, temperature 0
# ----------------------------------------------------------------------------------------------------------------------
def greedy_compare(args, reqs, other_model, n=3):
    def gen(model, req):
        body = {"model": model, "messages": req["messages"], "tools": req["tools"] or None, "temperature": 0.0, "top_p": 1.0,
                "top_k": -1, "seed": 0, "max_tokens": 160, "stream": False}
        if not body["tools"]:
            del body["tools"]
        r = requests.post(args.base_url + "/chat/completions", json=body, headers={"Authorization": "Bearer dummy"}, timeout=(15, args.timeout))
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return json.dumps({"c": msg.get("content"), "r": msg.get("reasoning_content") or msg.get("reasoning"), "t": msg.get("tool_calls")}, sort_keys=True)

    res = {"models": [args.model, other_model], "n": 0, "n_differ": 0, "error": None}
    try:
        for req in reqs[:n]:
            a, b = gen(args.model, req), gen(other_model, req)
            res["n"] += 1
            res["n_differ"] += int(a != b)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    res["verdict"] = "OK_differ" if res["n_differ"] else ("IDENTICAL_check_lora" if res["n"] else "not_run")
    return res


# ----------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="base")
    ap.add_argument("--data", default="/workspace/data_v1/train_turns.jsonl")
    ap.add_argument("--n-requests", type=int, default=12, help="request pool; small so the warmup can prefill it all and the timed phase is warm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=100, help="timed-phase cap; workers stop taking new requests after this")
    ap.add_argument("--grace", type=float, default=45, help="seconds to wait for in-flight requests after the cap (then abandoned)")
    ap.add_argument("--warmup", type=int, default=-1, help="requests run before the timed phase, not counted (default -1 = the whole pool once, "
                    "so the timed phase hits warm prefixes like tau2 turns; 0 disables)")
    ap.add_argument("--warmup-max-tokens", type=int, default=64, help="short generations during warmup: it only needs to prefill the prefix cache")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=300, help="per-request read timeout (s); 1200 tokens at >=5 tok/s fits")
    ap.add_argument("--config", default="unnamed")
    ap.add_argument("--meta", default="{}", help="JSON merged into the output line")
    ap.add_argument("--out", default=None, help="append the JSON result line to this file")
    ap.add_argument("--greedy-compare", default=None, help="other served model name: run 3 greedy prompts on both and report whether outputs differ")
    args = ap.parse_args()

    metrics_url = args.base_url.rsplit("/v1", 1)[0] + "/metrics"
    reqs = build_requests(args.data, args.n_requests, args.seed)
    out = {"config": args.config, "model": args.model, "concurrency": args.concurrency, "n_requests_pool": len(reqs),
           "prompt_chars_mean": round(statistics.mean(r["chars"] for r in reqs)), "max_tokens": args.max_tokens,
           "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}, "seconds_cap": args.seconds}
    try:
        out.update(json.loads(args.meta))
    except Exception:
        out["meta_parse_error"] = args.meta[:200]

    if args.greedy_compare:
        out["lora_effect_check"] = greedy_compare(args, reqs, args.greedy_compare)
        print("greedy compare:", out["lora_effect_check"], file=sys.stderr, flush=True)

    warmup = len(reqs) if args.warmup < 0 else args.warmup
    lock = threading.Lock()
    state = {"next": 0, "seen": set(), "records": [], "warm_records": [], "errors": [], "consec_err": 0, "stop": threading.Event(), "abort": None}

    def take(limit_idx=None):
        with lock:
            i = state["next"]
            if limit_idx is not None and i >= limit_idx:
                return None, None
            state["next"] += 1
            ri = i % len(reqs)
            warm = ri in state["seen"]
            state["seen"].add(ri)
            return ri, warm

    def worker(phase, deadline, limit_idx=None):
        while not state["stop"].is_set() and time.time() < deadline:
            ri, warm = take(limit_idx)
            if ri is None:
                return
            req = reqs[ri]
            try:
                rec = stream_one(args, req, args.warmup_max_tokens if phase == "warmup" else None)
                rec.update({"phase": phase, "req": ri, "warm": warm})
                with lock:
                    state["consec_err"] = 0
                    if phase == "timed" and rec["t_end"] <= deadline + args.grace:
                        state["records"].append((rec, req))
                    elif phase == "warmup":
                        state["warm_records"].append(rec)
            except Exception as e:
                with lock:
                    state["errors"].append(f"{type(e).__name__}: {str(e)[:160]}")
                    state["consec_err"] += 1
                    if state["consec_err"] >= max(6, 3 * args.concurrency):
                        state["abort"] = "too many consecutive errors (server down?)"
                        state["stop"].set()
                time.sleep(1)

    # warmup: prefill the pool (short generations), not counted; bounded by 2 waves of per-request timeouts
    t_w = time.time()
    if warmup > 0:
        ths = [threading.Thread(target=worker, args=("warmup", t_w + 2 * args.timeout, warmup), daemon=True) for _ in range(args.concurrency)]
        [t.start() for t in ths]
        [t.join(timeout=max(0.0, t_w + 2 * args.timeout + 30 - time.time())) for t in ths]
        state["next"] = 0  # timed phase cycles from the start of the pool again (those prefixes are now warm)
    wr = state["warm_records"]; w_wall = max(1e-6, time.time() - t_w)
    out["warmup"] = {"n": len(wr), "requested": warmup, "wall_s": round(w_wall, 1),
                     "ttft_s_mean": round(statistics.mean(r["ttft"] for r in wr if r["ttft"] is not None), 3) if any(r["ttft"] is not None for r in wr) else None,
                     "prompt_tok_s": round(sum(r["prompt_tokens"] or 0 for r in wr) / w_wall, 1) if wr else None,
                     "errors_so_far": len(state["errors"])}
    print(f"warmup done: {out['warmup']}", file=sys.stderr, flush=True)
    if state["abort"]:
        state["stop"].clear(); state["abort"] = None; state["consec_err"] = 0  # give the timed phase its own chance (it aborts again fast if the server is down)

    # timed phase
    sampler = GpuSampler(); sampler.start()
    m_before = scrape_metrics(metrics_url)
    t_start = time.time()
    deadline = t_start + args.seconds
    ths = [threading.Thread(target=worker, args=("timed", deadline), daemon=True) for _ in range(args.concurrency)]
    [t.start() for t in ths]
    for t in ths:
        t.join(timeout=max(0.0, deadline + args.grace - time.time()))
    state["stop"].set()
    m_after = scrape_metrics(metrics_url)
    sampler.stop.set()
    abandoned = sum(1 for t in ths if t.is_alive())

    recs = state["records"]
    if recs:
        t_end = max(r["t_end"] for r, _ in recs)
        window = max(1e-6, t_end - t_start)
        comp = [r["completion_tokens"] or 0 for r, _ in recs]
        prompt = [r["prompt_tokens"] or 0 for r, _ in recs]
        lat = [r["latency"] for r, _ in recs]
        ttft = [r["ttft"] for r, _ in recs if r["ttft"] is not None]
        ttft_cold = [r["ttft"] for r, _ in recs if r["ttft"] is not None and not r["warm"]]
        ttft_warm = [r["ttft"] for r, _ in recs if r["ttft"] is not None and r["warm"]]
        per_req = [c / l for c, l in zip(comp, lat) if l > 0]
        decode = [c / (r["latency"] - r["ttft"]) for (r, _), c in zip(recs, comp) if r["ttft"] is not None and r["latency"] - r["ttft"] > 0.05]
        san = [sanity(r, q) for r, q in recs]
        out.update({
            "status": "ok" if state["abort"] is None else "aborted",
            "abort_reason": state["abort"],
            "requests_completed": len(recs), "requests_abandoned_inflight": abandoned, "errors": len(state["errors"]),
            "error_samples": state["errors"][:3],
            "window_s": round(window, 1),
            "gen_tok_s_aggregate": round(sum(comp) / window, 1),
            "prompt_tok_s_aggregate": round(sum(prompt) / window, 1),
            "completion_tokens_mean": round(statistics.mean(comp), 1),
            "prompt_tokens_mean": round(statistics.mean(prompt), 1),
            "per_request_tok_s_mean": round(statistics.mean(per_req), 2) if per_req else None,
            "per_request_tok_s_median": round(statistics.median(per_req), 2) if per_req else None,
            "decode_tok_s_per_stream_mean": round(statistics.mean(decode), 2) if decode else None,
            "latency_s_mean": round(statistics.mean(lat), 2), "latency_s_median": round(statistics.median(lat), 2),
            "ttft_s_mean": round(statistics.mean(ttft), 3) if ttft else None,
            "ttft_s_median": round(statistics.median(ttft), 3) if ttft else None,
            "ttft_cold_s_mean": round(statistics.mean(ttft_cold), 3) if ttft_cold else None,
            "ttft_warm_s_mean": round(statistics.mean(ttft_warm), 3) if ttft_warm else None,
            "n_warm": len(ttft_warm), "n_cold": len(ttft_cold),
            "sanity_pass_rate": round(sum(1 for s in san if s[0]) / len(san), 3),
            "leak_count": sum(1 for s in san if s[1]),
            "tool_call_rate": round(sum(1 for s in san if s[3] == "tool_call") / len(san), 3),
            "tool_call_parse_fail": sum(1 for s in san if not s[2]),
            "empty_count": sum(1 for s in san if s[3] == "empty"),
            "truncated_rate": round(sum(1 for r, _ in recs if r["finish_reason"] == "length") / len(recs), 3),
            "kind_match_rate": (round(sum(1 for (r, q), s in zip(recs, san) if q["ref_kind"] and s[3] == q["ref_kind"]) / max(1, sum(1 for _, q in recs if q["ref_kind"])), 3)),
            "reasoning_chars_mean": round(statistics.mean(r["reasoning_chars"] for r, _ in recs)),
        })
    else:
        out.update({"status": "failed", "abort_reason": state["abort"] or "no request completed within the cap",
                    "requests_completed": 0, "errors": len(state["errors"]), "error_samples": state["errors"][:5]})

    # /metrics deltas over the timed phase
    q, h = metric_delta(m_before, m_after, "prefix_cache_queries"), metric_delta(m_before, m_after, "prefix_cache_hits")
    out["prefix_cache_hit_rate"] = round(h / q, 3) if (q and h is not None) else None
    drafts = metric_delta(m_before, m_after, "spec_decode_num_drafts")
    dtoks = metric_delta(m_before, m_after, "spec_decode_num_draft_tokens")
    acc = metric_delta(m_before, m_after, "spec_decode_num_accepted_tokens")
    if drafts:
        pp = per_pos_delta(m_before, m_after, "spec_decode_num_accepted_tokens_per_pos")
        out["spec_decode"] = {"drafts": drafts, "draft_tokens": dtoks, "accepted_tokens": acc,
                              "mean_acceptance_length": round(1 + acc / drafts, 3) if acc is not None else None,
                              "draft_acceptance_rate": round(acc / dtoks, 3) if (dtoks and acc is not None) else None,
                              "per_position_acceptance": {str(p): round(v / drafts, 3) for p, v in pp.items()} if pp else None}
    else:
        out["spec_decode"] = None
    out["preemptions"] = metric_delta(m_before, m_after, "num_preemptions")
    out["metrics_scraped"] = bool(m_before and m_after)
    out.update(sampler.summary())
    out["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    line = json.dumps(out, sort_keys=True)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")
    return 0 if recs else 2


if __name__ == "__main__":
    sys.exit(main())
