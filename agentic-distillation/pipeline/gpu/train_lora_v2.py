#!/usr/bin/env python3
"""Custom bf16 LoRA SFT loop for Qwen3.8-27B (HF transformers + peft), replacing Axolotl (v0: 85 tok/s; HF probe: 950 tok/s).
- Windows: {"messages","tools"} rows; every assistant turn is a target. Loss masks come from the REAL chat template with
  enable_thinking=False: target span = render(prefix, add_generation_prompt) .. render(prefix+turn); the empty <think></think>
  block is part of the prompt, so the model learns to answer right after it (exactly what vLLM feeds at inference).
- Chunked cross-entropy over the hidden states (no full-vocab logits tensor: 248k vocab x 16k tokens would be 16 GB).
- Token-budgeted gradient accumulation (ACC_TOKENS per optimizer step), cosine LR with warmup, eval loss on dev windows.
Usage: train_lora_v2.py --model DIR --train a.jsonl [b.jsonl ...] --dev d.jsonl --out DIR [--epochs 2] [--lr 1e-4] [--max-tok 16384]
       [--acc-tokens 65536] [--chunk 2048] [--dry-run]"""
import argparse, json, math, os, random, sys, time, torch, torch.nn.functional as F
from transformers import AutoTokenizer
DUMMY_USER={'role':'user','content':'(call in progress)'}
def render(tok, msgs, tools, gen=False):
    if not any(m['role']=='user' for m in msgs): msgs=msgs+[DUMMY_USER]
    return tok.encode(tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, add_generation_prompt=gen, enable_thinking=False), add_special_tokens=False)
def encode_window(tok, row, max_tok):
    msgs=row['messages']; tools=row.get('tools') or []
    ids=render(tok, msgs, tools); labels=[-100]*len(ids); n_t=0
    first_user=next((i for i,m in enumerate(msgs) if m['role']=='user'), None)
    if first_user is None: return None
    for i,m in enumerate(msgs):
        if m['role']!='assistant' or i<first_user: continue   # the opening greeting precedes any user turn (tau2 sends it fixed) -> not a target
        p=render(tok, msgs[:i], tools, gen=True); f=render(tok, msgs[:i+1], tools)
        if f[:len(p)]!=p or ids[:len(f)]!=f: return None  # prefix property violated -> skip window
        end=len(f)-1 if tok.decode(f[-1:])=='\n' else len(f)   # keep <|im_end|>, drop the trailing newline
        for j in range(len(p), end): labels[j]=ids[j]
        n_t+=1
    if len(ids)>max_tok or n_t==0: return None
    return ids, labels, n_t
def load_rows(paths):
    rows=[]
    for p in paths:
        for l in open(p): rows.append(json.loads(l))
    return rows
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--train',nargs='+',required=True); ap.add_argument('--dev',nargs='*',default=[])
    ap.add_argument('--out',required=True); ap.add_argument('--epochs',type=float,default=2); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--max-tok',type=int,default=16384)
    ap.add_argument('--acc-tokens',type=int,default=65536); ap.add_argument('--chunk',type=int,default=2048); ap.add_argument('--warmup',type=float,default=0.05); ap.add_argument('--r',type=int,default=64); ap.add_argument('--alpha',type=int,default=128)
    ap.add_argument('--eval-every',type=int,default=25); ap.add_argument('--save-every',type=int,default=50); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--status',default='/workspace/status/train_v2.jsonl')
    a=ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    tok=AutoTokenizer.from_pretrained(a.model)
    t0=time.time(); train=[]; skipped=0
    for r in load_rows(a.train):
        e=encode_window(tok,r,a.max_tok)
        if e: train.append(e)
        else: skipped+=1
    dev=[e for e in (encode_window(tok,r,a.max_tok) for r in load_rows(a.dev)) if e] if a.dev else []
    ntok=sum(len(e[0]) for e in train); ntgt=sum(sum(1 for x in e[1] if x!=-100) for e in train)
    print(f"windows train {len(train)} (skipped {skipped}) dev {len(dev)} | tokens {ntok/1e6:.2f}M, trainable {ntgt/1e6:.2f}M ({100*ntgt/max(1,ntok):.1f}%), assistant targets {sum(e[2] for e in train)} | prep {time.time()-t0:.0f}s",flush=True)
    steps_per_epoch=math.ceil(ntok/a.acc_tokens); total_steps=math.ceil(steps_per_epoch*a.epochs); print(f"~{steps_per_epoch} optimizer steps/epoch, {total_steps} total",flush=True)
    if a.dry_run:
        e=train[0]; ids,lab,_=e; s=next(i for i,x in enumerate(lab) if x!=-100); print('first target span starts with:',repr(tok.decode(ids[s-12:s])),'->',repr(tok.decode([x for x in lab[s:s+30] if x!=-100]))); return
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda'); model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','linear_attn.in_proj_qkv','linear_attn.in_proj_z','linear_attn.out_proj']
    model=get_peft_model(model,LoraConfig(r=a.r,lora_alpha=a.alpha,lora_dropout=0.05,target_modules=TARGETS,task_type='CAUSAL_LM')); model.print_trainable_parameters()
    base=model.base_model.model  # Qwen3_5ForCausalLM
    backbone=base.model; lm_head=base.lm_head
    def loss_fn(ids, labels):
        x=torch.tensor([ids],device='cuda'); y=torch.tensor([labels],device='cuda')
        h=backbone(input_ids=x).last_hidden_state[0]            # [T, d]
        y=y[0,1:]; h=h[:-1]                                      # predict token t+1
        tot=torch.zeros((),device='cuda',dtype=torch.float32); n=(y!=-100).sum()
        for s in range(0,h.shape[0],a.chunk):
            hs=h[s:s+a.chunk]; ys=y[s:s+a.chunk]
            if (ys!=-100).any():
                def f(hs_,ys_): return F.cross_entropy(lm_head(hs_).float(),ys_,ignore_index=-100,reduction='sum')
                tot=tot+torch.utils.checkpoint.checkpoint(f,hs,ys,use_reentrant=False)
        return tot, n
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=0.0,betas=(0.9,0.999))
    warm=max(1,int(total_steps*a.warmup))
    def lr_at(s): return a.lr*s/warm if s<warm else a.lr*0.5*(1+math.cos(math.pi*min(1.0,(s-warm)/max(1,total_steps-warm))))
    os.makedirs(a.out,exist_ok=True); os.makedirs(os.path.dirname(a.status),exist_ok=True)
    def save_adapter():
        """peft saves keys as base_model.model.model.layers.* (Qwen3_5ForCausalLM); vLLM's Qwen3_5ForConditionalGeneration
        LoRA loader needs base_model.model.model.language_model.layers.* (adapter_v0 had those) - otherwise the adapter is
        loaded but SILENTLY never applied (v2 lesson: student == base on every probe)."""
        model.save_pretrained(a.out)
        from safetensors.torch import load_file, save_file
        f=os.path.join(a.out,'adapter_model.safetensors'); t=load_file(f)
        t={k.replace('base_model.model.model.layers.','base_model.model.model.language_model.layers.',1):v for k,v in t.items()}
        save_file(t,f,metadata={'format':'pt'})
    def log(d):
        d['t']=round(time.time(),1); print(json.dumps(d),flush=True); open(a.status,'a').write(json.dumps(d)+'\n')
    @torch.no_grad()
    def evaluate():
        model.eval(); tot=0.0; n=0
        for ids,lab,_ in dev:
            l,c=loss_fn(ids,lab); tot+=l.item(); n+=c.item()
        model.train(); return tot/max(1,n)
    if dev: log({'step':0,'eval_loss':round(evaluate(),4)})
    step=0; acc_tok=0; acc_loss=0.0; acc_n=0; t_start=time.time(); tok_seen=0; order=[]
    ep=0
    while step<total_steps:
        if not order: order=list(range(len(train))); random.shuffle(order); ep+=1
        i=order.pop(); ids,lab,_=train[i]
        l,n=loss_fn(ids,lab); (l/ max(1,a.acc_tokens/16)).backward()   # scale: ~per-target-token loss, keeps grads O(1)
        acc_loss+=l.item(); acc_n+=n.item(); acc_tok+=len(ids); tok_seen+=len(ids)
        if acc_tok>=a.acc_tokens or (not order and ep>=a.epochs):
            for g in opt.param_groups: g['lr']=lr_at(step)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            opt.step(); opt.zero_grad(set_to_none=True); step+=1
            el=time.time()-t_start
            log({'step':step,'epoch':round(ep-len(order)/len(train),3),'loss':round(acc_loss/max(1,acc_n),4),'lr':lr_at(step),'tok_s':round(tok_seen/el,1),'tokens':tok_seen,'elapsed_min':round(el/60,1),'eta_min':round(el/60*(total_steps-step)/max(1,step),1),'mem_gb':round(torch.cuda.max_memory_allocated()/1e9,1)})
            acc_tok=0; acc_loss=0.0; acc_n=0
            if dev and step%a.eval_every==0: log({'step':step,'eval_loss':round(evaluate(),4)})
            if step%a.save_every==0: save_adapter(); log({'step':step,'saved':a.out})
    if dev: log({'step':step,'eval_loss':round(evaluate(),4),'final':True})
    save_adapter(); log({'step':step,'saved':a.out,'done':True}); print('TRAIN_DONE',flush=True)
if __name__=='__main__': main()
