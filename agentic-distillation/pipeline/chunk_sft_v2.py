#!/usr/bin/env python3
"""v2 window builder. Splits each trajectory into windows that fit MAX_TOK tokens as rendered by the real Qwen3.8 chat template
(enable_thinking=False, tools included). Unlike chunk_sft.py, a window that starts mid-conversation keeps a COMPRESSED history of
everything before it (tool results shortened to HIST_TOOL chars, assistant/user text kept), so the model always sees how the call
began. Tool results inside the window are capped at TOOL_CAP chars. Every assistant turn inside the window is a target.
Also verifies the prefix property used for loss masking: template(prefix, add_generation_prompt, enable_thinking=False) must be a
token-prefix of template(prefix + assistant turn). Usage: chunk_sft_v2.py IN.jsonl OUT.jsonl --tok DIR [--max-tok 16384]"""
import json, sys, argparse, copy
from transformers import AutoTokenizer
DUMMY_USER={'role':'user','content':'(call in progress)'}
def render(tok, msgs, tools, gen=False):
    if not any(m['role']=='user' for m in msgs): msgs=msgs+[DUMMY_USER]   # the template raises without a user query
    txt=tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
    return tok.encode(txt, add_special_tokens=False)
def size_ok(tok, msgs, tools, max_tok):
    return len(render(tok, msgs, tools)) <= max_tok
def groups(body):
    """assistant turn + its tool results form one group; a user message is its own group"""
    out=[]; cur=[]
    for m in body:
        if m['role'] in ('user','assistant'):
            if cur: out.append(cur)
            cur=[m]
        else: cur.append(m)
    if cur: out.append(cur)
    return out
def norm_args(m):
    """tau2 stores tool_call arguments as JSON strings; the Qwen template needs a mapping"""
    m=copy.deepcopy(m)
    for tc in m.get('tool_calls') or []:
        f=tc.get('function') or tc
        a=f.get('arguments')
        if isinstance(a,str):
            try: f['arguments']=json.loads(a) if a.strip() else {}
            except Exception: f['arguments']={'_raw':a}
        if 'function' not in tc:  # tau2 flat form {id,name,arguments} -> OpenAI form
            tc['type']='function'; tc['function']={'name':tc.pop('name'),'arguments':f['arguments']}; tc.pop('arguments',None)
    return m
def cap(m, n):
    m=copy.deepcopy(m)
    if m['role']=='tool' and len(m.get('content') or '')>n: m['content']=m['content'][:n]+f"\n... [truncated {len(m['content'])-n} chars]"
    return m
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--tok',required=True); ap.add_argument('--max-tok',type=int,default=16384)
    ap.add_argument('--tool-cap',type=int,default=8000); ap.add_argument('--hist-tool',type=int,default=300); ap.add_argument('--verify',type=int,default=30)
    a=ap.parse_args(); tok=AutoTokenizer.from_pretrained(a.tok)
    n_in=n_out=n_drop=0; n_tokens=0; n_targets=0; verified=0; bad=0
    with open(a.out,'w') as fo:
        for line in open(a.inp):
            r=json.loads(line); n_in+=1; msgs=r['messages']; tools=r.get('tools') or []
            sysm=[m for m in msgs if m['role']=='system'][:1]; body=[norm_args(cap(m,a.tool_cap)) for m in msgs if m['role']!='system']
            gs=groups(body); windows=[]; start=0
            while start<len(gs):
                hist=[cap(m,a.hist_tool) for g in gs[:start] for m in g]
                end=start
                while end<len(gs):
                    cand=sysm+hist+[m for g in gs[start:end+1] for m in g]
                    if size_ok(tok,cand,tools,a.max_tok): end+=1
                    else: break
                if end==start:  # even one group does not fit with the history: drop history first, then the group
                    if size_ok(tok,sysm+[m for m in gs[start]],tools,a.max_tok): hist=[]; end=start+1
                    else: n_drop+=1; start+=1; continue
                # a window must contain at least one assistant target
                seq=sysm+hist+[m for g in gs[start:end] for m in g]
                if any(m['role']=='assistant' for g in gs[start:end] for m in g):
                    windows.append((seq,start,end))
                start=end
            for w_i,(seq,s0,e0) in enumerate(windows):
                n_out+=1; n_tokens+=len(render(tok,seq,tools))
                n_targets+=sum(1 for m in seq for _ in [0] if m['role']=='assistant')
                if verified<a.verify:  # prefix-property check on the last assistant turn
                    idx=max(i for i,m in enumerate(seq) if m['role']=='assistant')
                    p=render(tok,seq[:idx],tools,gen=True)
                    f=render(tok,seq[:idx+1],tools)
                    verified+=1
                    if f[:len(p)]!=p: bad+=1
                fo.write(json.dumps({'messages':seq,'tools':tools,'meta':dict(r.get('meta') or {},window=w_i,n_windows=len(windows),hist_groups=s0)},ensure_ascii=False)+'\n')
    print(f"{a.inp}: {n_in} trajectories -> {n_out} windows, {n_tokens/1e6:.2f}M tokens, {n_targets} assistant targets, dropped {n_drop} oversized groups; prefix-property checks {verified} bad {bad}",file=sys.stderr)
if __name__=='__main__': main()
