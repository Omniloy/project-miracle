#!/usr/bin/env python3
"""Contamination check for synthetic Banking tasks against the 97 tau3-Banking test tasks.

The test set is public (tasks.json + tasks/*.json). Anything we train on must not be one of these
tasks or a paraphrase, and must not reuse their specific entities. Three independent checks:
  1. n-gram Jaccard on the user_scenario.instructions text (paraphrase / near-duplicate)
  2. gold-action signature overlap (same ordered (tool, key-args) sequence => same task in disguise)
  3. entity overlap: customer names / emails / ids / account ids / txn ids that appear in test tasks

Usage: decontaminate.py TEST_TASKS_JSON SYNTH_TASKS_JSON [--jaccard 0.5] [--report out.json]
Exit code 1 if any synthetic task is flagged.
"""
import argparse
import json
import re
import sys

WORD = re.compile(r"[a-z0-9$%.@_-]+")


def ngrams(text, n=5):
    toks = WORD.findall((text or "").lower())
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def action_sig(task):
    ec = task.get("evaluation_criteria") or {}
    sig = []
    for a in ec.get("actions") or []:
        args = a.get("arguments") or {}
        # keep the tool and the *shape* of args plus any tool-name argument (discoverable tools)
        keyargs = {k: v for k, v in args.items() if k in ("agent_tool_name", "user_tool_name", "tool_name", "card_type", "account_type")}
        sig.append((a.get("requestor"), a.get("name"), json.dumps(keyargs, sort_keys=True)))
    return tuple(sig)


ENTITY = re.compile(r"(?:[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|(?:chk|sav|biz_chk|cc|dbc|txn|btxn|dsp|ccord|usr|acct|pay|clsr|ref)_[a-z0-9_]+|\b[0-9a-f]{10}\b|\b\d{3}-\d{3}-\d{4}\b)")
PERSON = re.compile(r"(?:named|You are|name is|character:\*\*\s*You are)\s+([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})")


def entities(task):
    """Identifiers unique to a task: emails, DB ids, 10-hex user ids, phone numbers, and customer names
    (names only from explicit persona phrasing — bare capitalized bigrams match street/product names)."""
    blob = json.dumps(task, ensure_ascii=False)
    ents = set(ENTITY.findall(blob))
    instr = ((task.get("user_scenario") or {}).get("instructions")) or ""
    ents |= set(PERSON.findall(instr))
    inj = (((task.get("initial_state") or {}).get("initialization_data") or {}).get("agent_data")) or {}
    for table, v in inj.items():
        for row in ((v or {}).get("data") or {}).values():
            if isinstance(row, dict):
                for k in ("name", "cardholder_name", "full_name", "customer_name"):
                    if row.get(k):
                        ents.add(str(row[k]))
    return ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("synth")
    ap.add_argument("--jaccard", type=float, default=0.5)
    ap.add_argument("--report")
    a = ap.parse_args()

    test = json.load(open(a.test))
    synth = json.load(open(a.synth))
    test_ng = [(t["id"], ngrams(((t.get("user_scenario") or {}).get("instructions")))) for t in test]
    test_sigs = {action_sig(t): t["id"] for t in test}
    test_ents = set().union(*(entities(t) for t in test))
    # Product / policy vocabulary is shared by design (same knowledge base), so anything that appears in the
    # documents corpus ("Sky Blue", "Gold Years Account", ...) is not a customer entity. Only names/ids that are
    # unique to the test tasks count as contamination.
    from pathlib import Path
    docs_dir = Path(a.test).parent / "documents"
    corpus = " ".join(open(p).read() for p in docs_dir.glob("*.json")) if docs_dir.exists() else ""
    test_ents = {e for e in test_ents if e not in corpus}
    test_ents = {e for e in test_ents if not re.match(r"^(Gold|Silver|Bronze|Platinum|Green|Blue|Purple|Beige|Rewards|Card|Account|Rho|Bank)\b", e)}

    flagged = []
    for s in synth:
        reasons = []
        ng = ngrams(((s.get("user_scenario") or {}).get("instructions")))
        best = max(((jaccard(ng, tng), tid) for tid, tng in test_ng), default=(0, None))
        if best[0] >= a.jaccard:
            reasons.append(f"text_jaccard={best[0]:.2f} vs {best[1]}")
        sig = action_sig(s)
        if sig and sig in test_sigs:
            reasons.append(f"identical_gold_action_signature as {test_sigs[sig]}")
        ents = entities(s) & test_ents
        if ents:
            reasons.append(f"shared_entities={sorted(ents)[:5]}")
        if reasons:
            flagged.append({"id": s.get("id"), "reasons": reasons})

    print(f"checked {len(synth)} synthetic tasks against {len(test)} test tasks: {len(flagged)} flagged", file=sys.stderr)
    for f in flagged[:20]:
        print("  FLAG", f["id"], "|", "; ".join(f["reasons"]), file=sys.stderr)
    if a.report:
        json.dump({"flagged": flagged, "n_synth": len(synth), "n_test": len(test)}, open(a.report, "w"), indent=1)
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
