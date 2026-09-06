#!/usr/bin/env python3
"""Rewrite the assistant turns of verified banking trajectories into a natural spoken (phone) register, and give every silent
tool-call turn a short spoken "say before do" line. One LLM call per trajectory (Claude Sonnet 5 via OpenRouter); every rewritten
turn is validated: all digit sequences, identifiers (tool names, ids like txn_..., emails) of the original must survive, no markdown,
length <= 1.6x original (+40 chars); failures fall back to the original text with markdown stripped. Silent turns get a preamble
only if it passes the same checks (no numbers invented). Writes OUT.jsonl (same rows, rewritten content) and a stats json.
Usage: rewrite_phone_style.py IN.jsonl OUT.jsonl [--model anthropic/claude-sonnet-5] [--limit N] [--workers 6]"""
import json, re, sys, os, time, argparse, concurrent.futures as cf, requests, collections
URL='https://openrouter.ai/api/v1/chat/completions'
STYLE = """You are rewriting the ASSISTANT turns of a customer-service transcript so they sound like a calm, competent human agent
speaking on the PHONE while operating a computer system. The customer text and the tool calls/results stay exactly as they are; you only
write the assistant's spoken words.

Rules (hard):
- Spoken register: short sentences, contractions where natural, warm but efficient. No lists, no bullets, no markdown (**, #, -, |), no emoji,
  no headings, no tables, no JSON, no field names. Prose only.
- Keep EVERY fact: every amount, date, percentage, count, name, email, account/card name, last-4 digits, and every identifier or tool name
  the original mentions (e.g. txn_..., dsp_..., submit_cash_back_dispute_0589, user ID values) must appear VERBATIM in your rewrite. When the
  original tells the customer how to run a tool on their side (tool name, prefilled arguments), keep those exact strings, spoken plainly
  ("The tool is submit_cash_back_dispute_0589, and it's prefilled with your user ID f9bf8de0be and transaction txn_ba8b473f295d").
- Keep every option, condition, question and policy detail the original gives, just say it the way a person would on a call. If the original
  presents several options, present them in a sentence or two each, still no bullets. Ask at most one question per turn; if the original asks
  several, combine them naturally or keep the most important one and the others in the same sentence.
- Do not add facts, do not change numbers, do not promise things the original didn't.
- Silent turns (the assistant only called tools and said nothing): write ONE short spoken sentence (max 18 words) saying what you're about to do,
  matched to the tool(s) being called, e.g. "Let me check our policy on out-of-network ATM fees for the Light Green account." or
  "One moment, I'm pulling up your accounts." Never state a result, never ask a question, never invent a number. Vary the wording; do not
  start every one with "Let me". If several tools are called, one sentence covers them.
- Turns that already contain text AND tool calls: rewrite the text in the spoken register (same fact rules).
- Length: about the same as the original or shorter; never longer than 1.5x.

Output: ONLY a JSON object {"turns": [{"i": <message index>, "content": "<spoken text>"}, ...]} with one entry for EVERY assistant message
index listed in the request, in order. No commentary."""
def strip_md(t):
    t=re.sub(r'\*\*(.*?)\*\*',r'\1',t); t=re.sub(r'^\s*[-*•]\s+','',t,flags=re.M); t=re.sub(r'^#+\s*','',t,flags=re.M); t=re.sub(r'\s*\|\s*',' ',t)
    t=re.sub(r'[\U0001F300-\U0001FAFF✅❌✔✖⚠⭐]','',t); return re.sub(r'\n{2,}','\n',t).strip()
NUM=re.compile(r'\d[\d,]*(?:\.\d+)?'); IDENT=re.compile(r'\b(?:[a-z]+_[a-z0-9_]+\d[a-z0-9_]*|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|[a-z]{2,4}_[a-f0-9]{6,})\b',re.I)
def _norm(x):
    x=x.replace(',','')
    try: f=float(x); return str(int(f)) if f==int(f) else ('%g'%f)
    except ValueError: return x
def nums(t): return collections.Counter(_norm(x) for x in NUM.findall(t))
def idents(t): return set(IDENT.findall(t))
def valid(orig, new, silent):
    if not new or not new.strip(): return 'empty'
    if re.search(r'\*\*|^#|^\s*[-*•] |\|',new,flags=re.M): return 'markdown'
    if re.search(r'[\U0001F300-\U0001FAFF✅❌]',new): return 'emoji'
    if silent:
        if len(new)>200: return 'too_long_preamble'
        if nums(new): return 'preamble_number'
        if '?' in new: return 'preamble_question'
        return None
    if len(new)>1.6*len(orig)+40: return 'too_long'
    on,nn=nums(orig),nums(new)
    # spoken dates: "11/14/2025" may become "November 14th, 2025" (or "November 14") -> release those numbers if month name + day are present
    MON=['january','february','march','april','may','june','july','august','september','october','november','december']
    low=new.lower()
    for mo,dd,yy in re.findall(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b',orig):
        mi=int(mo)
        if 1<=mi<=12 and MON[mi-1] in low and re.search(r'\b'+str(int(dd))+r'(?:st|nd|rd|th)?\b',low):
            for k in (str(int(mo)),str(int(dd)),yy,str(int(yy))): on.pop(k,None)
    WORDS={'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine','10':'ten','11':'eleven','12':'twelve','13':'thirteen','14':'fourteen','15':'fifteen','16':'sixteen','17':'seventeen','18':'eighteen','19':'nineteen','20':'twenty'}
    for k in on:
        if k not in nn and not (k in WORDS and re.search(r'\b'+WORDS[k]+r'\b',low)): return f'missing_number:{k}'
    for k in idents(orig):
        if k not in new: return f'missing_ident:{k}'
    return None
def build_prompt(row):
    msgs=row['messages']; lines=[]; targets=[]
    for i,m in enumerate(msgs):
        r=m['role']; c=(m.get('content') or '')
        if r=='system': lines.append(f"[{i}] SYSTEM: (customer-service policy prompt, omitted)"); continue
        if r=='tool': lines.append(f"[{i}] TOOL RESULT: {c[:500].replace(chr(10),' ')}{' ...' if len(c)>500 else ''}"); continue
        if r=='user': lines.append(f"[{i}] CUSTOMER: {c[:1200]}"); continue
        tcs=[(t.get('function') or t) for t in (m.get('tool_calls') or [])]
        calls='; '.join(f"{t.get('name')}({json.dumps(t.get('arguments'))[:300]})" for t in tcs)
        silent=bool(tcs) and not c.strip(); targets.append((i,silent))
        lines.append(f"[{i}] ASSISTANT{' (SILENT, tool calls: '+calls+')' if silent else (' (text + tool calls: '+calls+')' if tcs else '')}: {c if c.strip() else '<no words>'}")
    idx=', '.join(str(i) for i,_ in targets)
    return "Transcript:\n"+"\n".join(lines)+f"\n\nRewrite these assistant message indices: {idx}", targets
def call(model, prompt, key):
    for attempt in range(4):
        try:
            r=requests.post(URL,headers={'Authorization':f'Bearer {key}'},json={'model':model,'messages':[{'role':'system','content':STYLE},{'role':'user','content':prompt}],'temperature':0.3,'max_tokens':12000,'response_format':{'type':'json_object'}},timeout=600)
            j=r.json(); txt=j['choices'][0]['message']['content']; usage=j.get('usage',{})
            m=re.search(r'\{.*\}',txt,re.S); return json.loads(m.group(0)), usage
        except Exception as e:
            err=str(e); time.sleep(3*(attempt+1))
    raise RuntimeError(err)
def process(row, model, key):
    prompt,targets=build_prompt(row)
    out,usage=call(model,prompt,key)
    got={int(t['i']):t['content'] for t in out.get('turns',[]) if 'i' in t}
    stats=collections.Counter(); msgs=row['messages']
    for i,silent in targets:
        orig=msgs[i].get('content') or ''; new=got.get(i)
        why=valid(orig,new or '',silent) if new is not None else 'missing'
        if why is None:
            msgs[i]['content']=new.strip(); stats['rewritten_silent' if silent else 'rewritten_text']+=1
        else:
            stats['fallback:'+why.split(':')[0]]+=1
            if not silent: msgs[i]['content']=strip_md(orig)
    row.setdefault('meta',{})['phone_style']=dict(stats); return row,stats,usage
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--model',default='anthropic/claude-sonnet-5'); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--workers',type=int,default=6)
    a=ap.parse_args(); key=os.environ['OPENROUTER_API_KEY']
    rows=[json.loads(l) for l in open(a.inp)]; rows=rows[:a.limit] if a.limit else rows
    done={}
    if os.path.exists(a.out):
        for l in open(a.out):
            r=json.loads(l); done[r['meta'].get('task_id','')+'|'+str(r['meta'].get('src'))+'|'+str(r['meta'].get('k',''))]=r
    tot=collections.Counter(); usage_tot=collections.Counter(); outf=open(a.out,'a')
    todo=[]
    for k,r in enumerate(rows):
        r.setdefault('meta',{})['k']=k
        key_=r['meta'].get('task_id','')+'|'+str(r['meta'].get('src'))+'|'+str(k)
        if key_ in done: continue
        todo.append(r)
    print(f'{len(rows)} rows, {len(done)} already done, {len(todo)} to do',file=sys.stderr)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for row,stats,usage in ex.map(lambda r: process(r,a.model,key), todo):
            outf.write(json.dumps(row,ensure_ascii=False)+'\n'); outf.flush(); tot.update(stats)
            for k2 in ('prompt_tokens','completion_tokens'): usage_tot[k2]+=usage.get(k2,0)
            print(dict(stats),file=sys.stderr)
    print('TOTAL',dict(tot),dict(usage_tot),file=sys.stderr)
    json.dump({'stats':dict(tot),'usage':dict(usage_tot)},open(a.out+'.stats.json','w'),indent=1)
if __name__=='__main__': main()
