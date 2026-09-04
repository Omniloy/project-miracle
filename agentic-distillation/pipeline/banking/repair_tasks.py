#!/usr/bin/env python3
"""Gold repairs for tasks no agent solved, decided by reading the governing policy documents (2026-09-04).
Each repair records its justification in description.notes. Run, then re-validate with validate_task.py."""
import json, os, copy
S=os.environ['S']; T=f"{os.environ['SD']}/tau2/domains/banking_knowledge/tasks"
FROZEN="2025-11-14 03:40:00 EST"
def load(t): return json.load(open(f'{T}/task_{t}.json'))
def save(t, x, why):
    x['description']['notes']=(x['description'].get('notes') or '')+f"\n[REPAIRED 2026-09-04] {why}"
    for i,a in enumerate(x['evaluation_criteria']['actions']): a['action_id']=f"{t}_{i}"
    json.dump(x, open(f'{T}/task_{t}.json','w'), indent=1, ensure_ascii=False); print('repaired', t)
def unlock_call(tool, args): return [{"name":"unlock_discoverable_agent_tool","arguments":{"agent_tool_name":tool},"requestor":"assistant"},{"name":"call_discoverable_agent_tool","arguments":{"agent_tool_name":tool,"arguments":json.dumps(args)},"requestor":"assistant"}]

# synth_1018: KB enumerates late_fee_reversal ("Reversal of a late payment fee") — the specific correct reason for a late fee credit.
x=load('synth_1018')
for a in x['evaluation_criteria']['actions']:
    if a['name']=='call_discoverable_agent_tool': d=json.loads(a['arguments']['arguments']); d['reason']='late_fee_reversal'; a['arguments']['arguments']=json.dumps(d)
save('synth_1018', x, "reason -> late_fee_reversal (doc_credit_cards_credit_cards_(general)_017 lists it as 'Reversal of a late payment fee'; goodwill_adjustment is the generic fallback).")

# synth_1021: doc_credit_cards_credit_card_account_logistics_003 Step 3 requires logging the closure reason before closing.
x=load('synth_1021'); acts=x['evaluation_criteria']['actions']
i=next(k for k,a in enumerate(acts) if a['name']=='unlock_discoverable_agent_tool' and a['arguments']['agent_tool_name']=='close_credit_card_account_7834')
acts[i:i]=unlock_call('log_credit_card_closure_reason_4521', {"credit_card_account_id":"cc_a7c3e91f2b_silver","user_id":"a7c3e91f2b","closure_reason":"not_using_card"})
save('synth_1021', x, "added log_credit_card_closure_reason_4521(not_using_card) before closure per logistics_003 Step 3 ('Ask the customer why... Then log it').")

# synth_1004: doc_031 maps card_not_present_fraud -> card_action close_and_reissue and requires the agent to perform it; liability: reported 3 business days after discovery -> $500 tier.
x=load('synth_1004'); acts=x['evaluation_criteria']['actions']
for a in acts:
    if a['name']=='call_discoverable_agent_tool' and a['arguments']['agent_tool_name']=='file_debit_card_transaction_dispute_6281':
        d=json.loads(a['arguments']['arguments']); d['card_action']='close_and_reissue'; d['customer_max_liability_amount']=500; a['arguments']['arguments']=json.dumps(d)
acts+=unlock_call('close_debit_card_4721', {"card_id":"dbc_a4f9c2e71b","reason":"fraud_suspected"})
acts+=unlock_call('order_debit_card_5739', {"account_id":"chk_a4f9c2e71b_1","user_id":"a4f9c2e71b","delivery_option":"STANDARD","delivery_fee":0,"card_design":"CLASSIC","design_fee":0,"shipping_address":"2201 Barton Springs Road, Austin, TX 78704"})
save('synth_1004', x, "card_action close_and_reissue + close_debit_card + order_debit_card per doc_031 card-action mapping ('agent must separately perform the indicated card action'); liability $500 (report on 11/14 is 3 business days after 11/11 discovery; $50 tier requires <=2).")

# synth_1016: generator omitted the customer rows entirely -> inject them from the task notes.
x=load('synth_1016'); u="4f2b91e7c3"
x['initial_state']['initialization_data']={"agent_data":{
 "users":{"data":{u:{"name":"Priya Nandakumar","user_id":u,"address":"2204 Comal Street, Austin, TX 78702","email":"priya.nandakumar@gmail.com","phone_number":"512-555-0148","date_of_birth":"05/09/1991"}}},
 "accounts":{"data":{f"chk_{u}_1":{"account_id":f"chk_{u}_1","user_id":u,"class":"checking","level":"Blue Account","date_opened":"02/10/2023","status":"OPEN","current_holdings":"4820.00"},
                     f"sav_{u}_2":{"account_id":f"sav_{u}_2","user_id":u,"class":"savings","level":"Gold Account","date_opened":"02/10/2023","status":"OPEN","current_holdings":"3000.00"},
                     f"sav_{u}_0":{"account_id":f"sav_{u}_0","user_id":u,"class":"savings","level":"Gold Account","date_opened":"05/01/2021","status":"CLOSED","current_holdings":"0.00"}}}},"user_data":None}
save('synth_1016', x, "injected the customer, checking ($4,820 OPEN), savings ($3,000 OPEN) and the closed old savings rows described in the notes; the original task had no customer in the DB.")

# synth_2012: same defect -> inject customer + business credit card account.
x=load('synth_2012'); u="v0ictorosei1"
x['initial_state']['initialization_data']={"agent_data":{
 "users":{"data":{u:{"name":"Victor Osei","user_id":u,"address":"2210 Peachtree Crest, Atlanta, GA 30303","email":"victor.osei@oseilogistics.net","phone_number":"404-555-0178","date_of_birth":"06/14/1991"}}},
 "credit_card_accounts":{"data":{f"cc_{u}_bbronze":{"account_id":f"cc_{u}_bbronze","user_id":u,"card_type":"Business Bronze Rewards Card","date_of_account_open":"03/18/2024","current_balance":"$1,120.40","credit_limit":"$8,000.00","reward_points":"3700 points","account_status":"ACTIVE","past_due_amount":"$0.00"}}}},"user_data":None}
save('synth_2012', x, "injected the customer and the Business Bronze Rewards Card account ($37.00 = 3700 points available cash back); the original task had no customer in the DB.")
