#!/usr/bin/env python3
"""Relaxed, reference-based state-diff verifier for TRAINING-DATA filtering (not for benchmark scoring).

The benchmark's strict reward hashes the whole DB, which includes (a) a log table of allowlisted read-tool
calls and (b) verification records keyed by a frozen timestamp. Those make the strict hash fail when an
otherwise-correct agent does one extra lookup or logs a slightly different timestamp. For selecting
trajectories to train on we care about the *business outcome*: did the agent leave the write tables in the
same state the gold actions produce?

For each simulation in a tau2 results.json:
  gold_db  = fresh env + task initial_state + replay gold actions
  agent_db = fresh env + task initial_state + replay the agent's trajectory (tau2's own set_state replay)
  relaxed_reward = 1 iff all tables except IGNORE match, with verification_history compared as a set of
                   (user_id) only, and the strict tau2 reward is recorded alongside.
Also records "extra_reads" (allowlisted read tools the agent called beyond gold) and "missing_writes".

Run with tau2-bench's .venv python:  relaxed_verify.py RESULTS.json [--retrieval bm25_grep] [--out report.json]
Requires TAU2_DATA_DIR to point at the data dir that contains the tasks (e.g. data_synth) when scoring
synthetic runs.
"""
import argparse
import json
import sys

from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.environment import get_environment

IGNORE_TABLES = {"agent_discoverable_tools"}
# free-text fields a correct agent cannot be expected to reproduce byte-for-byte (they never change the business outcome)
FREE_TEXT_FIELDS = {"reason", "notes", "description", "summary", "closure_reason", "decision_reason", "shipping_address", "note", "comment", "memo"}
READ_TOOLS_PREFIX = ("get_",)


def db_dump(env):
    d = env.tools.db.model_dump()
    out = {}
    for table, v in d.items():
        if table in IGNORE_TABLES:
            continue
        rows = (v or {}).get("data") if isinstance(v, dict) else v
        if table == "verification_history" and isinstance(rows, dict):
            out[table] = sorted({str(r.get("user_id")) for r in rows.values() if isinstance(r, dict)})
        elif isinstance(rows, dict):
            # compare rows by content with free-text fields stripped; ids of created rows can depend on call
            # order, so match rows as a multiset of their remaining fields rather than by key
            out[table] = sorted(json.dumps({k: v for k, v in r.items() if k not in FREE_TEXT_FIELDS and not k.endswith("_id") or k in ("user_id","account_id","card_id","credit_card_account_id","transaction_id")}, sort_keys=True) for r in rows.values() if isinstance(r, dict))
        else:
            out[table] = rows
    return out


def diff_tables(a, b):
    return sorted(t for t in set(a) | set(b) if a.get(t) != b.get(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--retrieval", default="bm25_grep")
    ap.add_argument("--out")
    a = ap.parse_args()

    d = json.load(open(a.results))
    tasks = {t["id"]: Task(**t) for t in d["tasks"]}
    report = []
    for sim in d["simulations"]:
        task = tasks[sim["task_id"]]
        init = task.initial_state.initialization_data if task.initial_state else None
        init_actions = task.initial_state.initialization_actions if task.initial_state else None
        strict = (sim.get("reward_info") or {}).get("reward")
        try:
            gold_env = get_environment(retrieval_variant=a.retrieval, task=task)
            gold_env.set_state(initialization_data=init, initialization_actions=init_actions, message_history=[], strict=True)
            for act in (task.evaluation_criteria.actions or []):
                gold_env.make_tool_call(tool_name=act.name, requestor=act.requestor, **(act.arguments or {}))
            gold = db_dump(gold_env)

            from tau2.data_model.message import Message  # noqa: F401  (ensure types are importable)
            agent_env = get_environment(retrieval_variant=a.retrieval, task=task)
            # tau2 replays every tool call in the trajectory when given the message history
            from tau2.evaluator.evaluator_env import EnvironmentEvaluator  # noqa: F401
            from tau2.data_model.simulation import SimulationRun

            run = SimulationRun.model_validate(sim)
            agent_env.set_state(initialization_data=init, initialization_actions=init_actions,
                                message_history=list(run.messages), strict=False)
            agent = db_dump(agent_env)
        except Exception as e:
            report.append({"task_id": sim["task_id"], "trial": sim.get("trial"), "strict": strict, "relaxed": None,
                           "error": f"{type(e).__name__}: {str(e)[:300]}"})
            continue

        mismatched = diff_tables(gold, agent)
        gold_calls = [(x.name, (x.arguments or {}).get("agent_tool_name")) for x in (task.evaluation_criteria.actions or [])
                      if x.name == "call_discoverable_agent_tool"]
        agent_calls = [(tc["name"], (tc.get("arguments") or {}).get("agent_tool_name")) for m in sim["messages"]
                       if m["role"] == "assistant" for tc in (m.get("tool_calls") or []) if tc["name"] == "call_discoverable_agent_tool"]
        gold_set, agent_set = {g[1] for g in gold_calls}, {g[1] for g in agent_calls}
        extra_reads = sorted(x for x in agent_set - gold_set if x and x.startswith(READ_TOOLS_PREFIX))
        missing_writes = sorted(x for x in gold_set - agent_set if x and not x.startswith(READ_TOOLS_PREFIX))
        report.append({"task_id": sim["task_id"], "trial": sim.get("trial"), "strict": strict,
                       "relaxed": 1.0 if not mismatched else 0.0, "mismatched_tables": mismatched,
                       "extra_reads": extra_reads, "missing_writes": missing_writes})

    ok_r = sum(1 for r in report if r.get("relaxed") == 1.0)
    ok_s = sum(1 for r in report if r.get("strict") == 1.0)
    print(f"strict pass {ok_s}/{len(report)} | relaxed pass {ok_r}/{len(report)}", file=sys.stderr)
    for r in report:
        print(f"  {r['task_id']} t{r['trial']}: strict={r['strict']} relaxed={r.get('relaxed')} "
              f"mismatch={r.get('mismatched_tables')} extra_reads={r.get('extra_reads')} missing={r.get('missing_writes')} {r.get('error','')}", file=sys.stderr)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
