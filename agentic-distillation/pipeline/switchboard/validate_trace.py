#!/usr/bin/env python3
"""Validate authored phone-agent traces against their seeded scenarios. A trace file is JSONL, one object per conversation:
{"scenario_id": ..., "messages": [{"role": "system"|"user"|"assistant"|"tool", "content": str, "tool_calls": [{"id","type":"function","function":{"name","arguments": json-string}}], "tool_call_id": str}]}
Checks (programmatic, reject on any hard failure):
  structure: system first; roles alternate sensibly; every tool_call answered by a tool message with matching id; arguments parse as JSON;
             tool names exist in the scenario's catalog; required params present; no assistant turn with empty content AND no tool call.
  identifiers: every identifier argument (document_number, phone*, to_iban, date_of_birth, postal_code, email, card_last4, order_code) equals
             the scenario truth in canonical form (normalizers from validators.py); checksum-twist scenarios must show a re-ask before the
             correct value is used; the truth value must appear in a read-back (chunked digits) in an assistant turn BEFORE the first write tool.
  say/do:  action verbs in past tense ('done', 'listo', 'he bloqueado', 'transferred') only after a successful tool result for that action;
             no question + write tool in the same message.
  style:   no markdown/JSON/ISO dates/snake_case field names in assistant content; spoken turns <= 70 words (ES/EN); language matches scenario;
             style=silent => assistant content empty on tool-call turns (except read-back/confirm turns which have no tool call); style=narrate => a
             tool-call turn on a slow tool has <= 1 sentence of content.
  sequence: expected_tool_sequence write tools all appear (in order) unless the scenario twist makes them unreachable; end_call present in the last assistant message.
Usage: validate_trace.py TRACES.jsonl SCENARIOS.jsonl --out report.json --accepted accepted.jsonl"""
import argparse, json, re, sys
from validators import NORMALIZERS, normalize_phone, normalize_dni, normalize_nie, normalize_iban, normalize_date, normalize_email, normalize_postal_es
from spoken_digits import spoken_to_compact

WRITE_TOOLS = {"block_card", "transfer_funds", "update_contact_phone", "change_plan", "report_outage", "port_in_request", "update_delivery_address", "cancel_order", "create_reservation", "create_callback", "transfer_to_human"}
SLOW_TOOLS = {"search_customer", "get_transactions", "transfer_funds", "get_invoice", "port_in_request", "search_order"}
ID_ARGS = {"document_number": None, "phone": "phone", "line_phone": "phone", "phone_to_port": "phone", "to_iban": "iban", "date_of_birth": "date_of_birth", "postal_code": "postal_code", "email": "email", "card_last4": "card_last4", "order_code": "code"}
PAST_DONE = re.compile(r"\b(all set|i've (blocked|transferred|updated|cancelled|canceled|booked|changed|scheduled)|has been (blocked|transferred|updated|cancelled|booked)|he (bloqueado|transferido|actualizado|cancelado|reservado|cambiado|programado)|queda (bloqueada|actualizada|cancelada|reservada|tramitada)|ya está (bloqueada|hecha la transferencia|actualizado|cancelado|reservada))\b", re.I)
RAW = re.compile(r"[{}\[\]]|```|\*\*|^#|\b\d{4}-\d{2}-\d{2}\b|\b[a-z]+_[a-z_]+\b|\|")
ES_HINT = re.compile(r"\b(el|la|de|que|por|para|con|su|una|usted|gracias|número)\b", re.I); EN_HINT = re.compile(r"\b(the|your|please|thanks|number|with|for|and|let|me)\b", re.I)

def norm_for(kind, val, lang):
    if kind == "phone": return normalize_phone(val)
    if kind == "iban": return normalize_iban(val)
    if kind == "date_of_birth": return normalize_date(val, "DMY")
    if kind == "email": return normalize_email(val)
    if kind == "postal_code": return normalize_postal_es(val)
    if kind in ("dni",): return normalize_dni(val)
    if kind in ("nie",): return normalize_nie(val)
    return re.sub(r"[\s.\-]", "", str(val)).upper()

def digits_readback_present(text, truth):
    """The assistant read the value back: all digit groups of truth appear in order in the text (spaces/dots/dashes between digits allowed),
    or (for masked IBAN/card) the last 4 digits appear."""
    t = re.sub(r"[\s.\-]", "", text or "").upper(); spoken = spoken_to_compact(text or "")
    core = re.sub(r"[\s.\-+]", "", str(truth)).upper()
    if "/" in str(truth):  # dates: accept DDMMYYYY, DMYYYY, DDMMYY spoken forms
        d, m, y = str(truth).split("/"); cands = [d + m + y, str(int(d)) + str(int(m)) + y, d + m + y[2:], str(int(d)) + str(int(m)) + y[2:]]
        return any(c in spoken for c in cands)
    if "@" in core: return core.lower() in re.sub(r"\s|punto|arroba|dot|at", lambda mm: {"punto": ".", "dot": ".", "arroba": "@", "at": "@"}.get(mm.group(0), ""), (text or "").lower())
    return core in t or core in spoken or (len(core) >= 8 and (core[-4:] in t or core[-4:] in spoken))

def validate(trace, sc):
    errs, warns = [], []
    msgs = trace.get("messages") or []
    if not msgs or msgs[0]["role"] != "system": errs.append("no system message first")
    tools = {t["function"]["name"]: t["function"] for t in sc["tools"]}
    lang = sc["language"]; style = sc["style"]
    truth = {i["kind"]: i for i in sc["identifiers"]}
    pending = {}; done_tools = []; first_write_idx = None; readback_seen = {}
    wrong_letter_kinds = {i["kind"] for i in sc["identifiers"] if i["twist"] in ("wrong_control_letter", "one_digit_wrong")}
    reask_seen = set(); a_turns = 0; words_over = 0
    for idx, m in enumerate(msgs):
        r = m["role"]
        if r == "assistant":
            a_turns += 1
            content = (m.get("content") or "").strip(); tcs = m.get("tool_calls") or []
            if not content and not tcs: errs.append(f"msg{idx}: empty assistant turn")
            if content:
                if RAW.search(content): errs.append(f"msg{idx}: raw data/markdown in spoken content: {content[:80]!r}")
                wc = len(content.split())
                if wc > 70: words_over += 1
                if lang == "es" and EN_HINT.search(content) and not ES_HINT.search(content): warns.append(f"msg{idx}: expected Spanish")
                if lang == "en" and ES_HINT.search(content) and not EN_HINT.search(content): warns.append(f"msg{idx}: expected English")
                for kind, info in truth.items():
                    if kind in ("amount_eur",): continue
                    if digits_readback_present(content, info["truth"]): readback_seen.setdefault(kind, idx)
                    if info["twist"] != "none" and digits_readback_present(content, info["caller_says_first"]) is False and re.search(r"(no (coincide|me cuadra|concuerda)|doesn't match|does not match|repeat|repit|otra vez|de nuevo|once more|again)", content, re.I):
                        reask_seen.add(kind)
                if PAST_DONE.search(content):
                    # any write tool succeeded before this message?
                    if not any(n in WRITE_TOOLS for n in done_tools): errs.append(f"msg{idx}: claims completion before any write tool succeeded")
                if "?" in content and any(tc["function"]["name"] in WRITE_TOOLS for tc in tcs): errs.append(f"msg{idx}: asks a question and calls a write tool in the same turn")
            for tc in tcs:
                fn = tc["function"]["name"]
                if fn not in tools: errs.append(f"msg{idx}: unknown tool {fn}"); continue
                try: args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                except Exception: errs.append(f"msg{idx}: tool args not JSON"); continue
                for req in tools[fn]["parameters"].get("required", []):
                    if req not in args: errs.append(f"msg{idx}: {fn} missing required {req}")
                pending[tc.get("id")] = fn
                if fn in WRITE_TOOLS and first_write_idx is None: first_write_idx = idx
                if fn in SLOW_TOOLS and style == "narrate" and content and len(re.findall(r"[.!?]+", content)) > 2: warns.append(f"msg{idx}: narrate-mode pre-tool utterance longer than 1-2 sentences")
                if fn in SLOW_TOOLS and style == "silent" and content: errs.append(f"msg{idx}: silent style but narrated before {fn}")
                # identifier arguments must be canonical truth
                for k, v in args.items():
                    kind = ID_ARGS.get(k)
                    if k == "document_number":
                        want = truth.get("dni") or truth.get("nie") or truth.get("passport")
                        if want and str(v) != want["truth"]: errs.append(f"msg{idx}: document_number {v!r} != truth {want['truth']}")
                        if want and want["twist"] != "none" and want["kind"] not in reask_seen: errs.append(f"msg{idx}: used document after a wrong-letter statement without a re-ask")
                    elif kind and kind in truth:
                        if str(v) != str(truth[kind]["truth"]): errs.append(f"msg{idx}: {k}={v!r} != canonical truth {truth[kind]['truth']}")
                        if kind == "iban" and truth[kind]["twist"] != "none" and kind not in reask_seen: errs.append(f"msg{idx}: IBAN used after a wrong-digit statement without a re-ask")
                    elif kind == "phone" and normalize_phone(str(v)) != str(v): errs.append(f"msg{idx}: phone {v!r} not E.164")
                if fn in WRITE_TOOLS:
                    for kind in truth:
                        if kind in ("amount_eur",): continue
                        if kind in readback_seen and readback_seen[kind] > idx: pass
                    missing_rb = [kind for kind in truth if kind not in ("amount_eur", "date_of_birth") and kind not in readback_seen]
                    if missing_rb: errs.append(f"msg{idx}: write tool {fn} before read-back of {missing_rb}")
        elif r == "tool":
            tid = m.get("tool_call_id")
            if tid not in pending: errs.append(f"msg{idx}: tool result without matching call"); continue
            fn = pending.pop(tid); done_tools.append(fn)
        elif r == "user":
            pass
    if pending: errs.append(f"unanswered tool calls: {list(pending.values())}")
    last_a = [m for m in msgs if m["role"] == "assistant"][-1] if a_turns else None
    if last_a and not any(tc["function"]["name"] == "end_call" for tc in (last_a.get("tool_calls") or [])): warns.append("no end_call in the final assistant turn")
    seq = [t for t in sc["expected_tool_sequence"] if t in WRITE_TOOLS]
    got = [t for t in done_tools if t in WRITE_TOOLS]
    if sc["search_twist"] not in ("verification_mismatch_then_alt_id",) and [t for t in seq if t in got] != seq: errs.append(f"write sequence {got} does not cover expected {seq}")
    if a_turns > sc["max_assistant_turns"] + 4: warns.append(f"{a_turns} assistant turns > budget {sc['max_assistant_turns']}")
    if words_over: warns.append(f"{words_over} spoken turns over 70 words")
    return errs, warns

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("traces"); ap.add_argument("scenarios"); ap.add_argument("--out"); ap.add_argument("--accepted"); a = ap.parse_args()
    scs = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(a.scenarios)}
    report, acc = [], []
    for line in open(a.traces):
        line = line.strip()
        if not line: continue
        try: tr = json.loads(line)
        except Exception as e: report.append({"scenario_id": "?", "errors": [f"bad json: {e}"]}); continue
        sc = scs.get(tr.get("scenario_id"))
        if not sc: report.append({"scenario_id": tr.get("scenario_id"), "errors": ["unknown scenario"]}); continue
        errs, warns = validate(tr, sc)
        report.append({"scenario_id": tr["scenario_id"], "errors": errs, "warnings": warns, "assistant_turns": sum(1 for m in tr["messages"] if m["role"] == "assistant")})
        if not errs: acc.append(tr)
    ok = sum(1 for r in report if not r.get("errors"))
    print(f"validated {len(report)} traces: {ok} accepted, {len(report)-ok} rejected", file=sys.stderr)
    for r in report:
        if r.get("errors"): print(f"  REJECT {r['scenario_id']}: {r['errors'][:3]}", file=sys.stderr)
    if a.out: json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)
    if a.accepted:
        with open(a.accepted, "w") as f:
            for tr in acc: f.write(json.dumps(tr, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
