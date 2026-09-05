#!/usr/bin/env python3
"""Trace-level diff of two tau2 results.json files (BASE vs MODEL) on the tasks whose outcome differs.
For every failed action check of the losing run it says what actually happened in the transcript:
never_called_<tool> | wrong_tool_suffix | wrong_number:<args> | wrong_string:<args> | missing_arg | matched_but_extra_or_order,
plus whether the conversation ended in ###TRANSFER### and the write-call count delta. Also flags repetition (max repeated 40-char chunk)
and the largest single-turn completion so 'loop' claims are checked against the text, not inferred from throughput.
Usage: python3 pipeline/trace_diff.py BASE.json MODEL.json [--all]"""
import json,re,sys,collections
def load(p): return {s['task_id']:s for s in json.load(open(p))['simulations']}
def passed(s): return bool(s.get('reward_info')) and s['reward_info']['reward']>=1
def norm(a):
    if isinstance(a,str):
        try: a=json.loads(a)
        except Exception: return a
    return {k:norm(v) for k,v in a.items()} if isinstance(a,dict) else a
def calls(s):
    return [(m['role'],tc.get('name'),norm(tc.get('arguments'))) for m in s['messages'] if m['role'] in('assistant','user') for tc in (m.get('tool_calls') or [])]
READS=('get_','KB_','grep','unlock_','list_','search_')
def n_writes(s): return sum(1 for r,n,a in calls(s) if r=='assistant' and not str(n).startswith(READS))
def ended_transfer(s): return any('###TRANSFER###' in (m.get('content') or '') for m in s['messages'] if m['role']=='user')
def flat(d,p=''):
    out={}
    for k,v in (d or {}).items():
        if isinstance(v,dict): out.update(flat(v,p+k+'.'))
        else: out[p+k]=v
    return out
def classify(s):
    cats=[]; cs=calls(s)
    for a in s['reward_info'].get('action_checks') or []:
        if a['action_match']: continue
        exp=a['action']; en=exp['name']; ea=norm(exp['arguments']) or {}
        inner=ea.get('agent_tool_name') or ea.get('discoverable_tool_name') if isinstance(ea,dict) else None
        def iname(c): return (c[2].get('agent_tool_name') or c[2].get('discoverable_tool_name')) if isinstance(c[2],dict) else None
        cands=[c for c in cs if c[1]==en and (inner is None or iname(c)==inner)]
        if not cands:
            fam=[c for c in cs if c[1]==en and inner and re.sub(r'_\d+$','',str(iname(c) or ''))==re.sub(r'_\d+$','',inner)]
            cats.append('wrong_tool_suffix' if fam else 'never_called_'+re.sub(r'_\d+$','',inner or en)); continue
        ef=flat(ea if isinstance(ea,dict) else {}); best=None
        for c in cands:
            cf=flat(c[2] if isinstance(c[2],dict) else {}); diff=[k for k in ef if str(ef[k])!=str(cf.get(k))]
            if best is None or len(diff)<len(best[0]): best=(diff,cf)
        diff,cf=best
        if not diff: cats.append('matched_but_extra_or_order'); continue
        kinds=set()
        for k in diff:
            ev,cv=ef[k],cf.get(k)
            if isinstance(ev,(int,float)) or re.fullmatch(r'[\d.]+',str(ev)): kinds.add('wrong_number')
            elif cv is None: kinds.add('missing_arg')
            else: kinds.add('wrong_string')
        cats.append('+'.join(sorted(kinds))+':'+','.join(k.split('.')[-1] for k in diff)[:40])
    return cats
def loopstats(s):
    asst=[m for m in s['messages'] if m['role']=='assistant']
    rep=0
    for m in asst:
        t=m.get('content') or ''
        if len(t)>=400:
            c=collections.Counter(t[i:i+40] for i in range(0,len(t)-40,20)); rep=max(rep,c.most_common(1)[0][1])
    comp=max([(m.get('usage') or {}).get('completion_tokens') or 0 for m in asst] or [0])
    return rep,comp
def main():
    B,M=load(sys.argv[1]),load(sys.argv[2]); allt='--all' in sys.argv
    lost=[t for t in M if t in B and passed(B[t]) and not passed(M[t])]; gained=[t for t in M if t in B and passed(M[t]) and not passed(B[t])]
    agg=collections.Counter()
    print(f"LOST (base pass, model fail): {len(lost)}")
    for t in lost:
        c=classify(M[t]); rep,comp=loopstats(M[t]); tr=ended_transfer(M[t])
        print(f"  {t} transfer M/B {int(tr)}/{int(ended_transfer(B[t]))} writes M-B {n_writes(M[t])-n_writes(B[t]):+d} maxrep {rep} maxcomp {comp} | {c}")
        for x in c: agg[x.split(':')[0]]+=1
        if tr and not ended_transfer(B[t]): agg['ENDED_IN_TRANSFER(model only)']+=1
    print('aggregate:',dict(agg))
    print(f"\nGAINED (model pass, base fail): {len(gained)}")
    for t in gained: print(f"  {t} base transfer {int(ended_transfer(B[t]))} | base fails: {classify(B[t]) if B[t].get('reward_info') else 'infra'}")
    if allt:
        print("\nALL model sims: repetition / longest turn"); 
        for t,s in sorted(M.items(), key=lambda kv:-loopstats(kv[1])[1])[:10]: print(' ',t,loopstats(s))
if __name__=='__main__': main()
