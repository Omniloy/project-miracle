#!/usr/bin/env python3
"""Judge phone_probe.jsonl (student vs base, same prefixes) on PhoneBench-style axes with an LLM judge of a different family
(default z-ai/glm-5.3 via OpenRouter). Per prefix the judge sees the last customer line, the gold reference turn and BOTH candidates
(anonymised A/B, order randomised) and scores each 1-5 on: telephone speaking style, tool-call accuracy vs gold, say/do consistency,
factual grounding, coherence, identifier handling; plus an overall preference. Prints means per model and win rates.
Usage: judge_phone_probe.py PROBE.jsonl OUT.jsonl [--model z-ai/glm-5.3]"""
import json, os, sys, random, re, collections, argparse, concurrent.futures as cf, requests, time
URL='https://openrouter.ai/api/v1/chat/completions'
AX=['speaking_style','tool_call_accuracy','say_do','grounding','coherence','identifier_handling']
PROMPT="""You are grading two candidate ASSISTANT turns from a phone customer-service agent (the agent speaks while operating tools).
Language of the call: {lang}. Style rule of this call: {style} ("narrate" = say one short sentence before a slow tool; "silent" = call tools with no words, report after).
Customer's last line (ASR text): {user}
GOLD reference turn (what a good agent did): text={gold_text} | tool_calls={gold_tools}
Candidate A: text={a_text} | tool_calls={a_tools}
Candidate B: text={b_text} | tool_calls={b_tools}
Score each candidate 1-5 on: speaking_style (sounds like a calm human on the phone: short sentences, no lists/markdown/IDs read aloud, natural in the call's language),
tool_call_accuracy (same tool(s) and canonical arguments as gold when gold calls tools; no tool when gold speaks), say_do (never claims an action the tool did not confirm; no question + write tool in one turn),
grounding (only facts from the conversation/tools), coherence (fits the dialogue), identifier_handling (read-back/normalisation of document numbers, phones, IBANs, dates when relevant; 3 if not relevant).
Return ONLY JSON: {{"A": {{"speaking_style":n,"tool_call_accuracy":n,"say_do":n,"grounding":n,"coherence":n,"identifier_handling":n}}, "B": {{...}}, "preferred": "A"|"B"|"tie", "reason": "<one sentence>"}}"""
def tools_str(tc): return json.dumps([{'name':(t.get('function') or t).get('name'),'args':(t.get('function') or t).get('arguments')} for t in (tc or [])],ensure_ascii=False)[:600]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('probe'); ap.add_argument('out'); ap.add_argument('--model',default='z-ai/glm-5.3'); ap.add_argument('--a',default='student'); ap.add_argument('--b',default='base')
    a=ap.parse_args(); key=os.environ['OPENROUTER_API_KEY']; random.seed(0)
    rs=[json.loads(l) for l in open(a.probe) if l.strip()]; by=collections.defaultdict(dict)
    for r in rs:
        if 'error' in r: continue
        by[(r['trace'],r['cut'])][r['model']]=r
    pairs=[(k,v) for k,v in by.items() if a.a in v and a.b in v]; print('pairs',len(pairs),file=sys.stderr)
    def judge(item):
        k,v=item; s,b=v[a.a],v[a.b]; flip=random.random()<0.5; A,B=(b,s) if flip else (s,b); meta=s.get('meta') or {}
        p=PROMPT.format(lang=meta.get('language'),style=meta.get('style'),user=s.get('last_user',''),gold_text=json.dumps(s['gold'].get('content'),ensure_ascii=False),gold_tools=tools_str(s['gold'].get('tool_calls')),
                        a_text=json.dumps(A.get('content'),ensure_ascii=False),a_tools=tools_str(A.get('tool_calls')),b_text=json.dumps(B.get('content'),ensure_ascii=False),b_tools=tools_str(B.get('tool_calls')))
        for att in range(3):
            try:
                r=requests.post(URL,headers={'Authorization':f'Bearer {key}'},json={'model':a.model,'messages':[{'role':'user','content':p}],'temperature':0,'max_tokens':3000},timeout=180).json()
                j=json.loads(re.search(r'\{.*\}',r['choices'][0]['message']['content'],re.S).group(0)); break
            except Exception as e: err=str(e)+' | '+(json.dumps(r)[:300] if 'r' in dir() else ''); time.sleep(2); j=None
        if not j: return {'key':list(k),'error':err}
        sa,sb=(j['B'],j['A']) if flip else (j['A'],j['B']); pref=j.get('preferred'); 
        if flip and pref in('A','B'): pref={'A':'B','B':'A'}[pref]
        return {'key':list(k),'meta':meta,a.a:sa,a.b:sb,'preferred':{'A':a.a,'B':a.b}.get(pref,'tie'),'reason':j.get('reason'),'flip':flip}
    out=[]; 
    with cf.ThreadPoolExecutor(8) as ex:
        for o in ex.map(judge,pairs): out.append(o)
    with open(a.out,'w') as f:
        for o in out: f.write(json.dumps(o,ensure_ascii=False)+'\n')
    ok=[o for o in out if 'error' not in o]; print(f"judged {len(ok)}/{len(out)}")
    for m in (a.a,a.b): print(f"{m:8}", ' '.join(f"{ax}={sum(o[m].get(ax,0) for o in ok)/max(1,len(ok)):.2f}" for ax in AX), f"| mean={sum(o[m].get(ax,0) for o in ok for ax in AX)/max(1,len(ok)*len(AX)):.2f}")
    pref=collections.Counter(o['preferred'] for o in ok); print('preferred:',dict(pref))
    for lang in ('es','en'):
        sub=[o for o in ok if (o.get('meta') or {}).get('language')==lang]
        if sub: print(f"  {lang}: n={len(sub)}", {m:round(sum(o[m].get(ax,0) for o in sub for ax in AX)/(len(sub)*len(AX)),2) for m in (a.a,a.b)}, dict(collections.Counter(o['preferred'] for o in sub)))
if __name__=='__main__': main()
