#!/usr/bin/env python3
"""Second-round repairs decided by Claude directly (2026-09-04 09:xx UTC) on the tasks the reviewers left AMBIGUOUS or
that stayed unsolved. Each edit records its justification in description.notes."""
import json, os
T=os.path.join(os.environ["S"], "harness/data_synth/tau2/domains/banking_knowledge/tasks")
def load(t): return json.load(open(f"{T}/task_{t}.json"))
def save(t, x, why):
    x["description"]["notes"]=(x["description"].get("notes") or "")+f"\n[REPAIRED-R2 2026-09-04] {why}"
    for i,a in enumerate(x["evaluation_criteria"]["actions"]): a["action_id"]=f"{t}_{i}"
    json.dump(x, open(f"{T}/task_{t}.json","w"), indent=1, ensure_ascii=False); print("repaired", t)
def rows(x, table): return x["initial_state"]["initialization_data"]["agent_data"].setdefault(table, {}).setdefault("data", {})

# synth_1012: apply_savings_account_credit_6831 only accepts class 'savings'; the base DB has no business_savings rows at all.
x=load("synth_1012"); r=rows(x,"accounts")["sav_e7a4c9d2f1_1"]; r["class"]="savings"
save("synth_1012", x, "account class business_savings -> savings (the credit tool rejects any other class; gold math 5.45% vs 5.0% on $40,000 = $15.00 unchanged).")

# synth_4011: the injected prior-dispute row carried an invented last-4 (7742); the user tool deterministically returns 9796.
x=load("synth_4011"); rows(x,"transaction_disputes")["dsp_3c9f1b7e2d_001"]["card_last_4_digits"]="9796"
save("synth_4011", x, "prior dispute row last-4 aligned to the get_card_last_4_digits output (9796) so a reasonable agent is not misled by inconsistent data.")

# synth_1015: (a) DENIED requests carry no cooldown (logistics_005) and only 2 on-time months exist -> insufficient_payment_history;
# (b) add credit_limit so the 50%-cap / utilization checks are possible; (c) no personal checking level charges an overdraft
# fee, so drop the mischarged $35 fee and let the 150% cap ($1,000 -> max $1,500) refuse the $1,600 request cleanly.
x=load("synth_1015")
for a in x["evaluation_criteria"]["actions"]:
    if a["name"]=="call_discoverable_agent_tool" and a["arguments"]["agent_tool_name"].startswith("deny_credit_limit"):
        d=json.loads(a["arguments"]["arguments"]); d["denial_reason"]="insufficient_payment_history"; a["arguments"]["arguments"]=json.dumps(d)
rows(x,"credit_card_accounts")["cc_4c81b3f2a9_silver"]["credit_limit"]="$4,000.00"
rows(x,"bank_account_transaction_history").pop("btxn_4c81b3f2a9_odf", None)
u=x["user_scenario"]["instructions"]
u=u.replace('- If the agent says the debit limit increase can\'t be done because of a recent overdraft fee, push back once: "That overdraft wasn\'t my fault, the bank posted my deposit late!" But if they still refuse, accept it gracefully.',
            '- If the agent says $1,600 exceeds the maximum temporary limit and offers a smaller amount instead, decline politely: you need the full $1,600 or nothing, you will pay the difference another way. Do not accept a partial increase.')
u=u.replace("- If the agent says the credit limit increase was denied because of a cooldown, express disappointment but accept it and ask when you can reapply.",
            "- If the agent says the credit limit increase was denied (for example because you don't have enough payment history yet), express disappointment but accept it and ask when you can reapply.")
x["user_scenario"]["instructions"]=u
save("synth_1015", x, "denial_reason cooldown_period_active -> insufficient_payment_history (logistics_005: denied requests trigger no cooldown; only 2 on-time months, mid tier needs 3); credit_limit $4,000 added; mischarged $35 overdraft fee removed (no personal checking level charges one) so the temporary-limit refusal rests on the 150% cap alone; user told to decline a partial increase.")

# synth_3032: the Diamond card has no credit limit anywhere, so utilization/50%-cap checks are impossible; make the prior
# request APPROVED so the 60-day premium cooldown (logistics_005/007, both readings agree) is the decisive, checkable criterion.
x=load("synth_3032"); r=rows(x,"credit_limit_increase_requests")["cli_a3f7c2b9d4_diamond_001"]; r["status"]="APPROVED"; r.pop("denial_reason",None); r.pop("decision_reason",None)
u=x["user_scenario"]["instructions"]
u=u.replace("You requested a $2,000 increase on your Diamond Elite Card in late October 2025 (around 10/28/2025) and were told it was denied; you may not volunteer this, but if the agent asks whether you've requested an increase before, confirm it honestly.",
            "You requested a $2,000 increase on your Diamond Elite Card in late October 2025 (around 10/28/2025) and it was approved, but the holidays made you want even more room; you may not volunteer this, but if the agent asks whether you've requested an increase before, confirm it honestly.")
x["user_scenario"]["instructions"]=u
save("synth_3032", x, "prior Diamond request set to APPROVED (10/28/2025) so cooldown_period_active is correct under both logistics_005 and _007 (premium 60-day cooldown, eligible 12/27/2025) without needing the undocumented credit limit; user text aligned.")
