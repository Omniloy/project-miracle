"""Phone-style probe: for each held-out Switchboard trace, cut before 3 assistant turns (early / middle / last) and generate greedily
with thinking off for MODELS (default student,base) on the local vLLM. Saves full outputs + gold to OUT (jsonl)."""
import json, os, sys, time, concurrent.futures as cf, requests
URL=os.environ.get('VLLM','http://localhost:8000')+'/v1/chat/completions'; DATA=sys.argv[1]; OUT=sys.argv[2]; MODELS=os.environ.get('MODELS','student,base').split(',')
rows=[json.loads(l) for l in open(DATA)]; jobs=[]
for k,r in enumerate(rows):
    msgs=r['messages']; idx=[i for i,m in enumerate(msgs) if m['role']=='assistant' and any(x['role']=='user' for x in msgs[:i])]
    if not idx: continue
    for i in sorted({idx[0],idx[len(idx)//2],idx[-1]}):
        for model in MODELS: jobs.append((k,i,model,msgs[:i],r.get('tools') or [],msgs[i],r.get('meta',{})))
def one(j):
    k,i,model,prefix,tools,gold,meta=j; body={'model':model,'messages':prefix,'temperature':0.0,'max_tokens':1024,'chat_template_kwargs':{'enable_thinking':False}}
    if tools: body['tools']=tools
    t0=time.time()
    try:
        r=requests.post(URL,json=body,timeout=600).json(); m=r['choices'][0]['message']
        rec={'trace':k,'cut':i,'model':model,'meta':meta,'finish':r['choices'][0].get('finish_reason'),'usage':r.get('usage'),'wall_s':round(time.time()-t0,1),'content':m.get('content'),'tool_calls':m.get('tool_calls'),'gold':{'content':gold.get('content'),'tool_calls':gold.get('tool_calls')},'last_user':next(((x.get('content') or '')[:300] for x in reversed(prefix) if x['role']=='user'),'')}
    except Exception as e: rec={'trace':k,'cut':i,'model':model,'error':str(e)[:300]}
    with open(OUT,'a') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    return rec
with cf.ThreadPoolExecutor(6) as ex: list(ex.map(one,jobs))
print('PHONE_PROBE_DONE',len(jobs))
