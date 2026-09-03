#!/usr/bin/env python3
"""Generate NON-test tau3-Banking-style tasks with a teacher model via OpenRouter.

Why: the 97 banking_knowledge tasks are the AA test set. We need fresh tasks in the same environment
(same 698 policy docs, same 44 discoverable tools, same DB schema) with NEW customers/accounts/transactions,
teacher-written scenarios and gold actions, so that DB-verified rejection sampling and self-distillation
never touch the test set. Every task is later replay-validated (validate_task.py) and decontaminated.

Usage:
  gen_banking_tasks.py --n 20 --model qwen/qwen3.8-max --out synth/batch1.json [--seed 1] [--workers 4]
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
TAU2 = Path(os.environ.get("TAU2_DIR", HERE.parents[1] / "harness/tau2-bench"))
DOM = TAU2 / "data/tau2/domains/banking_knowledge"

FROZEN_TIME = "2025-11-14 03:40:00 EST"  # what get_current_time() returns in this environment

WRITE_PREFIXES = ("open_", "close_", "file_", "order_", "freeze_", "unfreeze_", "apply_", "submit_", "approve_",
                  "deny_", "pay_", "transfer_", "log_", "reset_", "activate_", "update_", "clear_", "request_",
                  "emergency_", "initial_transfer")


def load():
    tools = json.load(open(HERE / "discoverable_tools.json"))
    tool_doc = json.load(open(HERE / "tool_doc_map.json"))
    schema = json.load(open(HERE / "db_schema_examples.json"))
    test_tasks = json.load(open(DOM / "tasks.json"))
    docs = {}
    for f in (DOM / "documents").glob("*.json"):
        d = json.load(open(f))
        docs[f.stem] = d
    db = json.load(open(DOM / "db.json"))
    return tools, tool_doc, schema, test_tasks, docs, db


def action_templates(test_tasks):
    """Argument shapes actually used by gold actions in the test set, per action name (schema, not content)."""
    tpl = {}
    for t in test_tasks:
        for a in (t.get("evaluation_criteria") or {}).get("actions") or []:
            k = (a.get("requestor"), a.get("name"))
            args = a.get("arguments") or {}
            if k not in tpl:
                tpl[k] = {"requestor": k[0], "name": k[1], "argument_keys": sorted(args.keys()), "example_arguments": args}
    return list(tpl.values())


def existing_entities(db):
    names, ids = set(), set()
    for table, v in db.items():
        for rid, row in ((v or {}).get("data") or {}).items():
            ids.add(rid)
            if isinstance(row, dict):
                for key in ("user_id", "account_id", "card_id", "transaction_id", "referral_id", "dispute_id", "order_id", "payment_id"):
                    if row.get(key):
                        ids.add(str(row[key]))
                if row.get("name"):
                    names.add(row["name"])
                if row.get("cardholder_name"):
                    names.add(row["cardholder_name"])
    return sorted(names), ids


def pick_tools(tools, rng, k):
    agent = tools["agent"]
    writes = [t for t in agent if t["name"].startswith(WRITE_PREFIXES)]
    reads = [t for t in agent if not t["name"].startswith(WRITE_PREFIXES)]
    chosen = [rng.choice(writes)]
    pool = [t for t in agent if t is not chosen[0]]
    chosen += rng.sample(pool, k - 1) if k > 1 else []
    return chosen


def build_prompt(task_idx, chosen, tools, tool_doc, docs, schema, templates, avoid_names, template_task, rng, difficulty):
    doc_ids = []
    for t in chosen:
        doc_ids += tool_doc.get(t["name"], [])[:2]
    # add 1-2 related product docs for policy realism
    extra = [d for d in docs if d not in doc_ids and any(w in d for w in ("credit_cards", "bank_accounts", "checking", "savings"))]
    doc_ids += rng.sample(extra, min(2, len(extra)))
    doc_ids = list(dict.fromkeys(doc_ids))
    doc_blobs = []
    for d in doc_ids:
        c = docs[d]
        content = c.get("content") if isinstance(c, dict) else str(c)
        doc_blobs.append(f"### DOC {d} — {c.get('title','') if isinstance(c, dict) else ''}\n{str(content)[:3500]}")
    all_agent_tool_names = [t["name"] for t in tools["agent"]]
    user_tool_names = [t["name"] for t in tools["user"]]
    rel_tables = ["users", "accounts", "debit_cards", "credit_card_accounts", "bank_account_transaction_history",
                  "credit_card_transaction_history", "transaction_disputes", "payment_history", "credit_card_orders", "referrals"]
    schema_blob = json.dumps({k: schema[k] for k in rel_tables if k in schema}, indent=0)[:9000]
    tmpl = json.loads(json.dumps(template_task))
    tmpl["user_scenario"]["instructions"] = tmpl["user_scenario"]["instructions"][:1200] + " ..."
    tmpl["description"]["notes"] = (tmpl["description"]["notes"] or "")[:800] + " ..."
    tmpl_blob = json.dumps(tmpl, indent=0)[:6000]

    return f"""You are designing ONE new evaluation task for a bank customer-service agent benchmark (tau-bench style, domain "Rho-Bank").
The agent must consult a policy knowledge base (retrieval tools) and call "discoverable" tools whose exact names appear in the policy documents.
The benchmark grades the task by replaying your GOLD ACTIONS on a fresh copy of the database and comparing the final database state (hash) with the agent's.

## Target tools for this task (use these; you may add others from the full list if the scenario needs them)
{json.dumps(chosen, indent=0)[:5000]}

## Full list of discoverable AGENT tool names (exact strings; the agent unlocks then calls them)
{json.dumps(all_agent_tool_names)}
## Discoverable USER tool names (the user can be given these by the agent)
{json.dumps(user_tool_names)}

## Policy documents relevant to these tools (ground your scenario in the REAL policy conditions below; do not invent policy)
{chr(10).join(doc_blobs)[:22000]}

## Database schema (real tables with example rows; note id formats, string amounts like "$3,000.00", date formats)
{schema_blob}

## Gold-action argument shapes used by this benchmark (replicate these key names exactly)
{json.dumps(templates, indent=0)[:6000]}

## Example task in the exact JSON schema (content truncated; copy the STRUCTURE, never the content)
{tmpl_blob}

## Hard requirements
0. Output a JSON object whose TOP-LEVEL keys are exactly: id, description, user_scenario, initial_state, evaluation_criteria, annotations, user_tools, required_documents. Do not nest evaluation_criteria/required_documents/user_tools inside initial_state. The ONLY database tables that exist are: {json.dumps(list(schema.keys()))} — never invent a table; if a tool reads state that has no table, describe it in the scenario instead. required_documents must be NON-EMPTY. description.relevant_policies must be a string (or null), not a list.
1. Invent a NEW customer: new full name (NOT any of: {json.dumps(avoid_names[:60])}), new 10-hex user_id, new account/card/transaction ids following the observed id formats. Put every row the task needs into initial_state.initialization_data.agent_data as {{"<table>": {{"data": {{"<row_id>": {{...full row...}}}}}}}} — the agent's tools can only see what is in the DB.
2. user_scenario.instructions: second person, in the style of the example: character, situation, what they want, constraints, what they know/don't know, and when to end the conversation. The user must NOT mention internal tool names. Difficulty: {difficulty}.
3. description.notes: the designer's hidden rationale — which policy clauses apply, the correct decision, and the traps.
4. evaluation_criteria.actions: the minimal ordered list of DB-changing gold actions with requestor "assistant" or "user". For each discoverable agent tool used: first {{"name":"unlock_discoverable_agent_tool","arguments":{{"agent_tool_name":"<name>"}}}} then {{"name":"call_discoverable_agent_tool", ...}} using the argument shape shown above. Include log_verification if policy requires identity verification. Read-only lookups need not be gold actions. Arguments must reference the ids you created. reward_basis must be ["DB"].
5. required_documents: list the doc ids (from the DOC headers above) the agent must consult.
6. user_tools: list any user tools involved (else []). annotations: null. id: "synth_{task_idx:04d}".
7. Be realistic and self-consistent: amounts, dates (late 2025), eligibility windows and limits must match the policy text so that exactly ONE correct outcome exists.
8. Grading is an exact final-database comparison, so: (a) the environment clock is frozen — get_current_time returns "{FROZEN_TIME}" and every log_verification must use exactly that time_verified; (b) calls to the read-only lookup tools get_all_user_accounts_by_user_id_3847, get_debit_cards_by_account_id_7823, get_bank_account_transactions_9173, get_user_dispute_history_7291, get_payment_history_6183, get_pending_replacement_orders_5765 ARE logged in the database, so include in gold actions exactly the lookups the policy requires the agent to perform (typically: unlock + call get_all_user_accounts_by_user_id_3847 first, then the specific lookup the case needs), and design the scenario so no other lookup is necessary; (c) for optional free-text arguments (e.g. "reason") use the tool's documented default value verbatim, or omit them; prefer tools whose arguments are ids, enums, booleans and amounts.

Return ONLY the task as a single JSON object, no prose, no markdown fences."""


def call_openrouter(model, prompt, api_key, max_tokens=16000, temperature=0.8, retries=4):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "low"},
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/omniloy", "X-Title": "banking-task-gen"},
    )
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            ch = d["choices"][0]; msg = ch["message"]
            txt = msg.get("content") or ""
            u = d.get("usage", {}) or {}; u["finish_reason"] = ch.get("finish_reason")
            return txt, u
        except Exception as e:
            wait = 2 ** i
            print(f"  openrouter error ({e}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("openrouter failed")


def parse_json(txt):
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else None


def normalize(t):
    """Coerce teacher output into tau2's Task schema; return (task, problems)."""
    problems = []
    if isinstance(t.get("task"), dict):  # teacher wrapped it
        t = t["task"]

    # Teachers sometimes nest top-level sections inside another section (e.g. evaluation_criteria under
    # initial_state). Hoist any missing top-level key found deeper in the object.
    def find_key(obj, key, depth=0):
        if depth > 4 or not isinstance(obj, dict):
            return None
        for k, v in obj.items():
            if k == key:
                return v
        for k, v in obj.items():
            if isinstance(v, dict) and k not in ("initialization_data", "agent_data", "data"):
                r = find_key(v, key, depth + 1)
                if r is not None:
                    return r
        return None

    def drop_key(obj, key, depth=0):
        if depth > 4 or not isinstance(obj, dict):
            return
        for k in list(obj.keys()):
            if k == key and depth > 0:
                obj.pop(k)
            elif isinstance(obj[k], dict) and k not in ("initialization_data", "agent_data", "data"):
                drop_key(obj[k], key, depth + 1)

    for key in ("evaluation_criteria", "required_documents", "user_tools", "annotations", "user_scenario", "description"):
        if t.get(key) is None:
            found = find_key(t, key)
            if found is not None:
                t[key] = found
                problems.append(f"hoisted nested {key}")
                drop_key(t, key)
    desc = t.get("description") or {}
    if isinstance(desc.get("relevant_policies"), list):
        desc["relevant_policies"] = "; ".join(map(str, desc["relevant_policies"]))
    desc.setdefault("purpose", f"Task: {t.get('id')}")
    t["description"] = desc
    ec = t.get("evaluation_criteria") or {}
    # common alternative keys
    for alt in ("gold_actions", "golden_actions", "expected_actions"):
        if not ec.get("actions") and t.get(alt):
            ec["actions"] = t.pop(alt)
        if not ec.get("actions") and ec.get(alt):
            ec["actions"] = ec.pop(alt)
    for i, act in enumerate(ec.get("actions") or []):
        act.setdefault("requestor", "assistant")
        act.setdefault("action_id", f"{t.get('id','synth')}_{i}")
        act.setdefault("arguments", {})
        # The environment clock is frozen (get_current_time -> "2025-11-14 03:40:00 EST") and the
        # verification record id is derived from this string, so any other timestamp can never match.
        if act.get("name") == "log_verification" and isinstance(act["arguments"], dict):
            act["arguments"]["time_verified"] = FROZEN_TIME
        # discoverable-tool arguments must be a JSON *string* (that is how the benchmark stores them)
        if act.get("name") == "call_discoverable_agent_tool" and isinstance(act["arguments"], dict):
            inner = act["arguments"].get("arguments")
            if isinstance(inner, dict):
                act["arguments"]["arguments"] = json.dumps(inner)
    ec["reward_basis"] = ["DB"]
    ec.setdefault("env_assertions", None)
    ec.setdefault("communicate_info", [])
    ec.setdefault("nl_assertions", None)
    t["evaluation_criteria"] = ec
    us = t.get("user_scenario") or {}
    if isinstance(us.get("persona"), (dict, list)):  # schema wants a string or null
        us["persona"] = None
    us.setdefault("persona", None)
    t["user_scenario"] = us
    st = t.get("initial_state") or {}
    st.setdefault("initialization_actions", None)
    st.setdefault("message_history", None)
    idata = st.get("initialization_data") or {}
    idata.setdefault("user_data", None)
    st["initialization_data"] = idata
    t["initial_state"] = st
    t.setdefault("annotations", None)
    t.setdefault("user_tools", [])
    if not ec.get("actions"):
        problems.append("no gold actions")
    if not t.get("required_documents"):
        problems.append("no required_documents")
    return t, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="qwen/qwen3.8-max")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tools-per-task", default="1,2,3", help="comma list to sample k from")
    ap.add_argument("--start-idx", type=int, default=0)
    a = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")
    tools, tool_doc, schema, test_tasks, docs, db = load()
    templates = action_templates(test_tasks)
    avoid_names, _ = existing_entities(db)
    # also avoid every customer name that appears in test scenarios
    for t in test_tasks:
        for m in re.findall(r"named ([A-Z][a-z]+ [A-Z][a-z]+)|You are ([A-Z][a-z]+ [A-Z][a-z]+)", (t.get("user_scenario") or {}).get("instructions") or ""):
            avoid_names += [x for x in m if x]
    avoid_names = sorted(set(avoid_names))
    template_task = next(t for t in test_tasks if t["id"] == "task_040")
    ks = [int(x) for x in a.tools_per_task.split(",")]
    rng = random.Random(a.seed)
    difficulties = ["medium: two interdependent policy clauses (e.g. an eligibility window AND a fee/limit rule) that both affect the correct action",
                    "hard: two products whose policies interact, a limit or allowance already partially consumed by injected history, and one ineligible request mixed with eligible ones - the naive action must be wrong",
                    "hard: the customer's stated request is not the right fix; the agent must discover from account history + policy that a different (single) action is required, and must NOT perform the requested one"]

    jobs = []
    for i in range(a.start_idx, a.start_idx + a.n):
        k = rng.choice(ks)
        chosen = pick_tools(tools, rng, k)
        diff = rng.choice(difficulties) if k > 1 else difficulties[0]
        jobs.append((i, build_prompt(i, chosen, tools, tool_doc, docs, schema, templates, avoid_names, template_task, rng, diff), [t["name"] for t in chosen], diff))

    out_tasks, raw, usage_tot = [], [], {"prompt_tokens": 0, "completion_tokens": 0}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(call_openrouter, a.model, p, api_key): (i, tl, diff) for i, p, tl, diff in jobs}
        for f in as_completed(futs):
            i, tl, diff = futs[f]
            try:
                txt, usage = f.result()
            except Exception as e:
                print(f"task {i}: FAILED {e}", file=sys.stderr)
                continue
            for k in usage_tot:
                usage_tot[k] += usage.get(k, 0) or 0
            if usage.get("finish_reason") not in (None, "stop"):
                print(f"task {i}: finish_reason={usage.get('finish_reason')}", file=sys.stderr)
            t = parse_json(txt)
            raw.append({"idx": i, "target_tools": tl, "difficulty": diff, "raw": txt[:20000]})
            if not t:
                print(f"task {i}: unparseable JSON", file=sys.stderr)
                continue
            t["id"] = f"synth_{i:04d}"
            t, probs = normalize(t)
            t["id"] = f"synth_{i:04d}"
            t["_gen"] = {"model": a.model, "target_tools": tl, "difficulty": diff, "seed": a.seed, "normalize_notes": probs}
            out_tasks.append(t)
            n_act = len(((t.get('evaluation_criteria') or {}).get('actions')) or [])
            print(f"task {i}: ok tools={tl} actions={n_act}{' | ' + '; '.join(probs) if probs else ''}", file=sys.stderr)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_tasks, open(a.out, "w"), indent=1, ensure_ascii=False)
    json.dump(raw, open(a.out.replace(".json", ".raw.json"), "w"), indent=1, ensure_ascii=False)
    print(f"wrote {len(out_tasks)}/{a.n} tasks -> {a.out} | tokens {usage_tot}", file=sys.stderr)


if __name__ == "__main__":
    main()
