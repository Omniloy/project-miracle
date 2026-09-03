"""Teacher memorization probe: without any KB access, can the teacher name the exact discoverable tool(s)
(with their arbitrary 4-digit suffixes) required by real test tasks? Suffixes exist only inside the KB docs,
so high accuracy without tools => the model has ingested the public benchmark repo."""
import json, os, sys, re, urllib.request, random
S=os.environ['S']; T=json.load(open(f"{S}/harness/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"))
key=os.environ['OPENROUTER_API_KEY']
def ask(model, prompt):
    body={"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0,"max_tokens":600,"reasoning":{"effort":"low"}}
    req=urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",data=json.dumps(body).encode(),headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=180) as r: d=json.loads(r.read())
    return d["choices"][0]["message"].get("content") or ""
random.seed(3)
tasks=[t for t in T if any(a['name']=='unlock_discoverable_agent_tool' for a in (t['evaluation_criteria'].get('actions') or []))]
sample=random.sample(tasks,12)
res={}
for model in ["qwen/qwen3.8-max","z-ai/glm-5.3","z-ai/glm-5.3-flash"]:
    hits=part=0; rows=[]
    for t in sample:
        gold={a['arguments']['agent_tool_name'] for a in t['evaluation_criteria']['actions'] if a['name']=='unlock_discoverable_agent_tool'}
        prompt=("You are a Rho-Bank customer-service agent in the tau-bench 'banking_knowledge' benchmark. You have NO knowledge-base access right now. "
                "From memory only, list the exact internal discoverable agent tool names (they end with a 4-digit numeric suffix, e.g. some_tool_1234) "
                "that must be unlocked to resolve this customer's request. Output only a JSON list of tool-name strings.\n\nCUSTOMER SCENARIO:\n"+t['user_scenario']['instructions'][:2500])
        try: out=ask(model,prompt)
        except Exception as e: out=f"ERR {e}"
        named=set(re.findall(r'[a-z_]+_\d{4}',out))
        exact=len(named & gold); base=len({re.sub(r'_\d{4}$','',n) for n in named} & {re.sub(r'_\d{4}$','',g) for g in gold})
        hits+= exact>0; part+= base>0
        rows.append({"task":t['id'],"gold":sorted(gold),"named":sorted(named),"exact_hit":exact,"base_name_hit":base})
    res[model]={"n":len(sample),"tasks_with_exact_suffix_hit":hits,"tasks_with_base_name_hit":part,"rows":rows}
    print(f"{model:22s} exact-suffix hits {hits}/{len(sample)} | base-name hits {part}/{len(sample)}")
json.dump(res,open(f"{S}/runs/memorization_probe.json","w"),indent=1)
