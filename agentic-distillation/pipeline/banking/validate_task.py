#!/usr/bin/env python3
"""Replay-validate synthetic banking tasks in the real tau2 environment (run with tau2-bench's .venv python).

A task is VALID when:
  - it parses into tau2's Task model,
  - every referenced discoverable tool exists and every required_document id exists,
  - initialization_data applies cleanly (set_state),
  - every gold action replays without raising,
  - the DB hash changes after replay when reward_basis includes DB (otherwise the task is untestable),
  - new entities do not collide with rows already in db.json (fresh customer requirement).

Usage: .venv/bin/python validate_task.py IN_TASKS.json OUT_VALID.json [--report report.json]
"""
import argparse
import json
import sys
from pathlib import Path

from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.environment import get_environment

HERE = Path(__file__).parent
DOM = Path(get_environment.__module__ and __import__("tau2").__file__).parent.parent.parent / "data/tau2/domains/banking_knowledge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--report")
    ap.add_argument("--retrieval", default="bm25_grep")
    a = ap.parse_args()

    tasks = json.load(open(a.inp))
    tools = json.load(open(HERE / "discoverable_tools.json"))
    known_agent = {t["name"] for t in tools["agent"]}
    known_user = {t["name"] for t in tools["user"]}
    doc_ids = {p.stem for p in (DOM / "documents").glob("*.json")}
    base_db = json.load(open(DOM / "db.json"))
    base_ids = set()
    for table, v in base_db.items():
        base_ids.update(((v or {}).get("data") or {}).keys())

    valid, report = [], []
    for raw in tasks:
        t = {k: v for k, v in raw.items() if not k.startswith("_")}
        rid = t.get("id")
        problems = []
        # coerce common teacher slips before schema validation
        desc = t.get("description") or {}
        if isinstance(desc.get("relevant_policies"), list):
            desc["relevant_policies"] = "; ".join(map(str, desc["relevant_policies"]))
        t.setdefault("annotations", None)
        t.setdefault("user_tools", [])
        ec = t.get("evaluation_criteria") or {}
        if not ec.get("actions"):
            report.append({"id": rid, "valid": False, "problems": ["no gold actions (untestable)"]})
            continue
        if "DB" not in [str(b).split(".")[-1] for b in (ec.get("reward_basis") or [])]:
            report.append({"id": rid, "valid": False, "problems": [f"reward_basis must include DB, got {ec.get('reward_basis')}"]})
            continue
        if not t.get("required_documents"):
            report.append({"id": rid, "valid": False, "problems": ["required_documents empty"]})
            continue
        try:
            task = Task(**t)
        except Exception as e:
            report.append({"id": rid, "valid": False, "problems": [f"schema: {str(e)[:300]}"]})
            continue
        # static checks
        for d in t.get("required_documents") or []:
            if d not in doc_ids:
                problems.append(f"unknown document {d}")
        actions = (task.evaluation_criteria.actions if task.evaluation_criteria else None) or []
        for act in actions:
            args = act.arguments or {}
            name = args.get("agent_tool_name")
            if act.name in ("unlock_discoverable_agent_tool", "call_discoverable_agent_tool") and name not in known_agent:
                problems.append(f"unknown agent tool {name}")
            uname = args.get("user_tool_name")
            if act.name in ("give_discoverable_user_tool", "call_discoverable_user_tool") and uname and uname not in known_user:
                problems.append(f"unknown user tool {uname}")
        for ut in t.get("user_tools") or []:
            if ut not in known_user and ut not in ("apply_for_credit_card", "submit_referral", "submit_transaction", "request_human_agent_transfer"):
                problems.append(f"unknown user_tools entry {ut}")
        init = task.initial_state.initialization_data if task.initial_state else None
        inj = (init.agent_data if init else None) or {}
        n_rows = 0
        for table, v in inj.items():
            if table not in base_db:
                problems.append(f"unknown table {table}")
                continue
            for row_id in ((v or {}).get("data") or {}).keys():
                n_rows += 1
                if row_id in base_ids:
                    problems.append(f"row id collides with base db: {row_id}")
        # consistency checks learned from the batch-2 failure analysis
        writes = [(a.name, json.dumps(a.arguments or {}, sort_keys=True)) for a in actions if a.name in ("call_discoverable_agent_tool", "log_verification")]
        if len(writes) != len(set(writes)):
            problems.append("duplicate identical gold calls (creates extra rows no agent will reproduce)")
        names = [((a.arguments or {}).get("agent_tool_name") or "") for a in actions if a.name == "call_discoverable_agent_tool"]
        if any(n.startswith("approve_") for n in names) and any(n.startswith("deny_") for n in names):
            problems.append("gold both approves and denies (contradictory path)")
        # every entity referenced by gold write args must exist in injected rows or the base db
        inj_ids = set(base_ids)
        for table, v in inj.items():
            for row_id, row in ((v or {}).get("data") or {}).items():
                inj_ids.add(row_id)
                if isinstance(row, dict):
                    for k in ("user_id", "account_id", "card_id", "credit_card_account_id", "transaction_id"):
                        if row.get(k):
                            inj_ids.add(str(row[k]))
        cust_ids = {str(r.get("user_id")) for r in ((inj.get("users") or {}).get("data") or {}).values() if isinstance(r, dict)}
        for act in actions:
            args = act.arguments or {}
            inner = args.get("arguments")
            try:
                inner = json.loads(inner) if isinstance(inner, str) else (inner or {})
            except Exception:
                inner = {}
            for k, v in list(inner.items()) + [(k2, v2) for k2, v2 in args.items() if k2 != "arguments"]:
                if k in ("user_id", "account_id", "card_id", "credit_card_account_id", "checking_account_id", "source_account_id", "destination_account_id", "transaction_id") and v and str(v) not in inj_ids:
                    problems.append(f"gold references unknown entity {k}={v}")
            if act.name == "log_verification" and cust_ids and str(args.get("user_id")) not in cust_ids:
                problems.append(f"log_verification user_id {args.get('user_id')} is not the injected customer {sorted(cust_ids)}")
        if problems:
            report.append({"id": rid, "valid": False, "problems": sorted(set(problems))})
            continue
        # dynamic replay
        try:
            env = get_environment(retrieval_variant=a.retrieval, task=task)
            env.set_state(initialization_data=init,
                          initialization_actions=task.initial_state.initialization_actions if task.initial_state else None,
                          message_history=[], strict=True)
            h0 = env.get_db_hash()
            for act in actions:
                env.make_tool_call(tool_name=act.name, requestor=act.requestor, **(act.arguments or {}))
            h1 = env.get_db_hash()
        except Exception as e:
            report.append({"id": rid, "valid": False, "problems": [f"replay: {type(e).__name__}: {str(e)[:400]}"]})
            continue
        basis = (task.evaluation_criteria.reward_basis if task.evaluation_criteria else None) or []
        basis = [str(b).split(".")[-1] for b in basis]
        if "DB" in basis and h0 == h1:
            report.append({"id": rid, "valid": False, "problems": ["gold actions did not change the DB (untestable under DB reward)"]})
            continue
        valid.append(raw)
        report.append({"id": rid, "valid": True, "n_actions": len(actions), "n_injected_rows": n_rows})

    json.dump(valid, open(a.out, "w"), indent=1, ensure_ascii=False)
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)
    nv = len(valid)
    print(f"valid {nv}/{len(tasks)}", file=sys.stderr)
    for r in report:
        if not r["valid"]:
            print(f"  INVALID {r['id']}: {r['problems'][:3]}", file=sys.stderr)


if __name__ == "__main__":
    main()
