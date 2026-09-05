#!/usr/bin/env python3
"""Phase-1 Switchboard dataset stats: axis counts, turn/token estimates, verbatim-sentence overlap, judge means."""
import json, re, glob, os, statistics as st
from collections import Counter, defaultdict

D = "/tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/data/switchboard"
T = D + "/traces"
AXES = ["naturalness", "say_do", "identifier_handling", "language"]

scs = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(D + "/scenarios_p1.jsonl")}
final = [json.loads(l) for l in open(D + "/phase1_final.jsonl")]
sft = [json.loads(l) for l in open(D + "/phase1_sft.jsonl")]
ids = [t["scenario_id"] for t in final]

# ---------- funnel ----------
written = sum(1 for f in sorted(glob.glob(T + "/batch_[0-9][0-9].jsonl")) for l in open(f) if l.strip())
vacc = sum(1 for f in sorted(glob.glob(T + "/batch_[0-9][0-9].accepted.jsonl")) for l in open(f) if l.strip())
judged = sum(1 for f in sorted(glob.glob(T + "/batch_[0-9][0-9].judged.jsonl")) for l in open(f) if l.strip())
report = json.load(open(D + "/phase1_final_report.json"))
funnel = {"scenarios_seeded": len(scs), "traces_written": written, "validator_accepted_per_batch": vacc,
          "judge_passed": judged, "deduped_pool": len(final) if False else None,
          "final_validator_rerun_accepted": sum(1 for r in report if not r["errors"]),
          "final_validator_rerun_rejected": sum(1 for r in report if r["errors"]),
          "final_dataset_rows": len(final)}
funnel.pop("deduped_pool")

# ---------- axis counts ----------
def cnt(fn): return dict(Counter(fn(scs[i]) for i in ids).most_common())
counts = {
  "vertical": cnt(lambda s: s["vertical"]),
  "language": cnt(lambda s: s["language"]),
  "style": cnt(lambda s: s["style"]),
  "intent": cnt(lambda s: s["intent"]),
  "search_twist": cnt(lambda s: s["search_twist"]),
  "identifier_kind": dict(Counter(i["kind"] for sid in ids for i in scs[sid]["identifiers"]).most_common()),
  "twist": dict(Counter(i["twist"] for sid in ids for i in scs[sid]["identifiers"]).most_common()),
  "corruption": dict(Counter(i["corruption"] for sid in ids for i in scs[sid]["identifiers"]).most_common()),
  "spoken_style": dict(Counter(i["spoken_style"] for sid in ids for i in scs[sid]["identifiers"]).most_common()),
  "caller_mood": cnt(lambda s: s["caller_persona"]["mood"]),
}
counts["twist_nonnone_traces"] = sum(1 for i in ids if any(x["twist"] != "none" for x in scs[i]["identifiers"]))
counts["corruption_nonnone_traces"] = sum(1 for i in ids if any(x["corruption"] != "none" for x in scs[i]["identifiers"]))
# cross-tab vertical x language
ct = Counter((scs[i]["vertical"], scs[i]["language"]) for i in ids)
counts["vertical_x_language"] = {f"{a}/{b}": n for (a, b), n in sorted(ct.items())}

# ---------- turns / tools / tokens ----------
a_turns = [sum(1 for m in t["messages"] if m["role"] == "assistant") for t in final]
u_turns = [sum(1 for m in t["messages"] if m["role"] == "user") for t in final]
tcalls = [sum(len(m.get("tool_calls") or []) for m in t["messages"]) for t in final]
def row_chars(r):
    n = 0
    for m in r["messages"]:
        n += len(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            n += len(tc["function"]["name"]) + len(tc["function"]["arguments"] if isinstance(tc["function"]["arguments"], str) else json.dumps(tc["function"]["arguments"]))
    return n
msg_chars = [row_chars(r) for r in sft]
tools_chars = [len(json.dumps(r["tools"], ensure_ascii=False)) for r in sft]
def stats(v): return {"mean": round(st.mean(v), 2), "median": st.median(v), "min": min(v), "max": max(v), "total": sum(v)}
sp_words = [len((m.get("content") or "").split()) for t in final for m in t["messages"] if m["role"] == "assistant" and (m.get("content") or "").strip()]
size = {
  "assistant_turns": stats(a_turns), "user_turns": stats(u_turns), "tool_calls": stats(tcalls),
  "assistant_spoken_turn_words": stats(sp_words),
  "chars_messages": stats(msg_chars), "chars_tool_schemas": stats(tools_chars),
  "tokens_est_messages_chars_over_4": {"mean": round(st.mean(msg_chars) / 4, 1), "total": round(sum(msg_chars) / 4)},
  "tokens_est_with_tool_schemas": {"mean": round(st.mean([a + b for a, b in zip(msg_chars, tools_chars)]) / 4, 1),
                                    "total": round(sum(a + b for a, b in zip(msg_chars, tools_chars)) / 4)},
}

# ---------- verbatim assistant-sentence overlap ----------
SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
occ = []            # (norm_sentence, trace_idx)
for ti, t in enumerate(final):
    for m in t["messages"]:
        if m["role"] != "assistant": continue
        for s in SPLIT.split(m.get("content") or ""):
            s2 = re.sub(r"\s+", " ", s.strip().strip("¿¡")).lower()
            if s2: occ.append((s2, ti))
by_sent = defaultdict(set)
for s, ti in occ: by_sent[s].add(ti)
def frac(minw):
    sel = [(s, ti) for s, ti in occ if len(s.split()) >= minw]
    if not sel: return None
    rep = sum(1 for s, ti in sel if len(by_sent[s]) > 1)
    uniq = {s for s, _ in sel}
    return {"sentence_occurrences": len(sel), "unique_sentences": len(uniq),
            "occurrences_shared_across_traces": rep, "fraction_repeated_verbatim": round(rep / len(sel), 4),
            "unique_ratio": round(len(uniq) / len(sel), 4)}
dup = {"all_sentences": frac(1), "sentences_ge_4_words": frac(4), "sentences_ge_8_words": frac(8)}
top = sorted(((len(v), s) for s, v in by_sent.items() if len(s.split()) >= 4), reverse=True)[:15]
dup["most_shared_sentences_ge_4_words"] = [{"traces": n, "text": s} for n, s in top]
# whole-trace duplicate check
sig = Counter(tuple(re.sub(r"\s+", " ", (m.get("content") or "").strip().lower()) for m in t["messages"] if m["role"] == "assistant") for t in final)
dup["identical_assistant_scripts"] = sum(n - 1 for n in sig.values() if n > 1)

# ---------- judge scores ----------
def per_trace(o):
    """Normalize the heterogeneous *.scores.json shapes to [{scenario_id, axes..., verdict}]."""
    rows = []
    blk = o.get("traces") or o.get("per_trace") or o.get("scores") or []
    items = blk if isinstance(blk, list) else [dict(v, scenario_id=k) if isinstance(v, dict) else {"scenario_id": k} for k, v in blk.items()]
    rej = {x if isinstance(x, str) else x.get("scenario_id") for x in (o.get("rejected") or [])}
    for it in items:
        sid = it.get("scenario_id")
        sc = it.get("scores") if isinstance(it.get("scores"), dict) else it
        r = {"scenario_id": sid, "batch": o.get("batch")}
        for a in AXES:
            if isinstance(sc.get(a), (int, float)): r[a] = sc[a]
        v = (it.get("verdict") or "").lower()
        r["passed"] = (sid not in rej) if not v else v in ("accept", "pass", "accepted", "passed")
        rows.append(r)
    return rows

all_rows, shapes = [], {}
for f in sorted(glob.glob(T + "/*.scores.json")):
    o = json.load(open(f))
    shapes[os.path.basename(f)] = next((k for k in ("traces", "per_trace", "scores") if k in o), "none")
    all_rows += per_trace(o)
final_ids = set(ids)
def means(rows):
    out = {}
    for a in AXES:
        v = [r[a] for r in rows if a in r]
        out[a] = round(st.mean(v), 3) if v else None
        out[a + "_n"] = len(v)
    v = [sum(r[a] for a in AXES) / 4 for r in rows if all(a in r for a in AXES)]
    out["all_axes_mean"] = round(st.mean(v), 3) if v else None
    return out
graded = {r["scenario_id"] for r in all_rows}
written_ids = set()
for f in sorted(glob.glob(T + "/batch_[0-9][0-9].jsonl")):
    for l in open(f):
        if l.strip(): written_ids.add(json.loads(l)["scenario_id"])
no_axis = sorted(written_ids - graded)   # judged but no per-axis row (rejected, reason-only)
judge = {
  "traces_submitted_to_judge": len(written_ids),
  "traces_with_axis_scores": len(all_rows),
  "traces_judged_without_axis_scores": no_axis,
  "judge_pass_total": len(final_ids),
  "judge_reject_total": len(written_ids) - len(final_ids),
  "graded_traces": len(all_rows),
  "graded_unique_scenarios": len(graded),
  "judge_pass": sum(1 for r in all_rows if r["passed"]),
  "judge_reject": sum(1 for r in all_rows if not r["passed"]),
  "mean_all_graded": means(all_rows),
  "mean_final_dataset": means([r for r in all_rows if r["scenario_id"] in final_ids]),
  "mean_judge_rejected": means([r for r in all_rows if not r["passed"]]),
  "score_histogram_final": {a: dict(sorted(Counter(r[a] for r in all_rows if r["scenario_id"] in final_ids and a in r).items())) for a in AXES},
  "scores_file_shapes": shapes,
  "final_rows_without_judge_scores": sorted(final_ids - graded),
}

out = {"dataset": "switchboard_phase1", "generated": "phase1_stats.py", "funnel": funnel, "counts": counts,
       "size": size, "duplication": dup, "judge": judge}
json.dump(out, open(D + "/phase1_stats.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "counts"}, ensure_ascii=False, indent=1)[:6000])
print("\nCOUNTS\n", json.dumps(counts, ensure_ascii=False, indent=1))
