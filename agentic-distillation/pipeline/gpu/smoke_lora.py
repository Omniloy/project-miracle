#!/usr/bin/env python3
"""GPU smoke test for LoRA fine-tuning Qwen3.8-27B (hybrid Gated-DeltaNet + Gated-Attention, model_type qwen3_5).

Answers the three questions that decide the training plan, in ~10 minutes on one 80-96 GB card:
  1. Do the linear-attention kernels (flash-linear-attention / causal-conv1d) import and run on this GPU/arch?
  2. Does a bf16 LoRA (attention + MLP, and optionally the DeltaNet in_proj/out_proj) fit and train (loss decreases)?
  3. Peak memory and tokens/sec at seq 4096 -> extrapolate to our trajectories (8-32K).

Usage (on the Vast box):  python smoke_lora.py --model Qwen/Qwen3.8-27B --steps 12 --seq 4096 [--target-deltanet]
Writes smoke_result.json.
"""
import argparse
import json
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--target-deltanet", action="store_true", help="also LoRA the Gated-DeltaNet projections")
    ap.add_argument("--data", default=None, help="optional SFT jsonl (our converter format) to build real samples from")
    a = ap.parse_args()

    res = {"gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0), "torch": torch.__version__}
    print(res)

    # 1) kernels
    for mod in ("fla", "causal_conv1d", "flash_attn"):
        try:
            m = __import__(mod)
            res[f"import_{mod}"] = getattr(m, "__version__", "ok")
        except Exception as e:
            res[f"import_{mod}"] = f"FAIL: {type(e).__name__}: {str(e)[:120]}"
    print({k: v for k, v in res.items() if k.startswith("import_")})

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from peft import LoraConfig, get_peft_model

    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    res["model_type"] = getattr(cfg, "model_type", None)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    res["load_s"] = round(time.time() - t0, 1)
    res["mem_after_load_GB"] = round(torch.cuda.memory_allocated() / 1e9, 1)
    print("loaded", res["load_s"], "s;", res["mem_after_load_GB"], "GB")

    # discover linear-module names to pick LoRA targets
    names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    res["linear_module_names"] = sorted(names)
    targets = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if n in names]
    if a.target_deltanet:
        targets += [n for n in names if any(k in n for k in ("in_proj", "out_proj", "a_proj", "b_proj", "ba_proj", "qkv", "qkvz"))]
    res["lora_targets"] = targets
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    res["trainable_params_M"] = round(trainable / 1e6, 1)

    # 2) data: real SFT rows if given (chat template), else repeated text
    if a.data and os.path.exists(a.data):
        rows = [json.loads(l) for l in open(a.data)][:16]
        texts = []
        for r in rows:
            msgs = [{k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id")} for m in r["messages"]]
            try:
                texts.append(tok.apply_chat_template(msgs, tools=r.get("tools") or None, tokenize=False))
            except Exception as e:
                res.setdefault("template_errors", []).append(str(e)[:120])
        res["template_ok_rows"] = len(texts)
    else:
        texts = ["The quick brown fox jumps over the lazy dog. " * 400] * 8
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=a.seq, padding="max_length")
    ids = enc["input_ids"].to(0)
    attn = enc["attention_mask"].to(0)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    losses, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for step in range(a.steps):
        i = step % ids.shape[0]
        out = model(input_ids=ids[i:i + 1], attention_mask=attn[i:i + 1], labels=ids[i:i + 1])
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(round(out.loss.item(), 4))
        print(f"step {step} loss {losses[-1]}  mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    dt = time.time() - t0
    res.update({"losses": losses, "loss_decreased": losses[-1] < losses[0], "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                "tokens_per_s": round(a.steps * a.seq / dt, 1), "seq": a.seq})
    json.dump(res, open("smoke_result.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
