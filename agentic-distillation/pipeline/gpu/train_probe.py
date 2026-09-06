"""Measure fwd+bwd throughput of Qwen3.8-27B + LoRA (same targets as the SFT yaml) on one GPU, without Axolotl, for several
attention implementations, and profile the slowest ops. Writes /workspace/status/train_probe.json. SEQ env = tokens per step (4096)."""
import torch, time, json, os, sys, traceback
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
mp=sys.argv[1]; seq=int(os.environ.get('SEQ','4096')); res={'seq':seq,'torch':torch.__version__}
TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','linear_attn.in_proj_qkv','linear_attn.in_proj_z','linear_attn.out_proj']
for attn in os.environ.get('ATTNS','sdpa,eager,flex_attention').split(','):
    try:
        t0=time.time(); model=AutoModelForCausalLM.from_pretrained(mp,dtype=torch.bfloat16,attn_implementation=attn,device_map='cuda'); load=time.time()-t0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); model.config.use_cache=False
        m=get_peft_model(model,LoraConfig(r=64,lora_alpha=128,lora_dropout=0.05,target_modules=TARGETS)); m.train()
        n_tr=sum(p.numel() for p in m.parameters() if p.requires_grad)
        ids=torch.randint(1000,200000,(1,seq),device='cuda'); rates=[]
        for i in range(4):
            torch.cuda.synchronize(); t0=time.time(); out=m(input_ids=ids,labels=ids); out.loss.backward(); m.zero_grad(set_to_none=True); torch.cuda.synchronize(); dt=time.time()-t0; rates.append(seq/dt)
            print(attn,'step',i,round(seq/dt,1),'tok/s',round(dt,1),'s',flush=True)
        res[attn]={'tok_s_last':round(rates[-1],1),'tok_s':[round(r,1) for r in rates],'load_s':round(load,1),'mem_gb':round(torch.cuda.max_memory_allocated()/1e9,1),'trainable_params':n_tr}
        if attn==os.environ.get('PROFILE','sdpa'):
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CUDA,ProfilerActivity.CPU]) as prof:
                out=m(input_ids=ids,labels=ids); out.loss.backward(); m.zero_grad(set_to_none=True); torch.cuda.synchronize()
            res['profile_'+attn]=prof.key_averages().table(sort_by='cuda_time_total',row_limit=30)
            print(res['profile_'+attn],flush=True)
        # no-checkpointing variant for the first attn only
        if attn=='sdpa':
            try:
                m.gradient_checkpointing_disable(); torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); t0=time.time(); out=m(input_ids=ids,labels=ids); out.loss.backward(); m.zero_grad(set_to_none=True); torch.cuda.synchronize(); dt=time.time()-t0
                res['sdpa_no_ckpt']={'tok_s':round(seq/dt,1),'mem_gb':round(torch.cuda.max_memory_allocated()/1e9,1)}; print('sdpa no-ckpt',round(seq/dt,1),'tok/s',flush=True)
            except Exception as e: res['sdpa_no_ckpt']={'error':str(e)[:300]}
        del m,model,out; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        res[attn]={'error':str(e)[:600],'tb':traceback.format_exc()[-800:]}; print(attn,'ERROR',str(e)[:300],flush=True)
        try: del model
        except Exception: pass
        torch.cuda.empty_cache()
    json.dump(res,open('/workspace/status/train_probe.json','w'),indent=1)
print('PROBE_DONE')
