#!/bin/bash
# Unattended eval job on a Vast box: serve Qwen3.8-27B (+ optional LoRA adapter from HF) with vLLM, run tau2-bench
# banking_knowledge with the AA config against it, upload results to HF. Env: HF_TOKEN, OPENROUTER_API_KEY, WORK_REPO,
# ADAPTER (HF path in WORK_REPO, e.g. adapter_v0, or "none"), EVAL_SET (dev|test|both), TRIALS (default 1)
# Serving knobs (research/inference_speed.md, 2026-09-04). All optional; with SERVE_MODE unset the job behaves exactly as before
# (bf16 base + LoRA kernels, tau2 concurrency 8):
#   SERVE_MODE=lora|merged   lora = base bf16 weights + adapter via --enable-lora (served as "student" on top of "base");
#                            merged = merge the adapter into the base weights first (PEFT-equivalent W += alpha/r*B@A, fp32 on GPU, written bf16,
#                            keeps the mtp.* tensors) and serve /workspace/merged as "student" (needs DISK_GB >= 180: +56 GB).
#   QUANT=none|fp8           fp8 = vLLM online FP8 (--quantization fp8; W8A8, halves weight bytes per decode step: ~1.8x per stream measured).
#   SPEC=none|mtp|ngram      mtp = native Qwen3.8 MTP head, k=MTP_K (default 2; k=3 crashed on 0.28.0/sm_120); ngram = prompt-lookup (measured net loss).
#   CONC                     tau2 --max-concurrency (default 16 when SERVE_MODE is set, 8 otherwise = old behaviour).
#   MAX_NUM_SEQS (32)        vLLM --max-num-seqs; must stay <= the Mamba cache block count (573 on 96 GB bf16); keep 32 with MTP.
#   THINKING=default|off     off = pass chat_template_kwargs {"enable_thinking": false} to the agent (adapter_v0 was trained with thinking off).
#   CUDAGRAPH                optional vLLM cudagraph_mode override (e.g. PIECEWISE) for the merged-bf16 paths that segfaulted in full-graph replay.
# Recommended (measured 393 tok/s aggregate / 48.8 tok/s per stream at 8 streams, 505 / 35.0 at 16 vs 120 / 15 today):
#   -e SERVE_MODE=merged -e QUANT=fp8 -e SPEC=mtp -e CONC=16
set -uo pipefail; export HF_HUB_ENABLE_HF_TRANSFER=1
W=/workspace; mkdir -p $W/status $W/eval; cd $W; LOG=$W/status/remote_eval.log; exec > >(tee -a $LOG) 2>&1
STATUS_TAG=${STATUS_TAG:-eval}; MAXLEN=${MAXLEN:-262144}; TEMP=${TEMP:-0}; KVDTYPE=${KVDTYPE:-auto}; GPU_UTIL=${GPU_UTIL:-0.90}; SERVE_MODE=${SERVE_MODE:-}; QUANT=${QUANT:-none}; SPEC=${SPEC:-none}; MTP_K=${MTP_K:-2}; MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}; THINKING=${THINKING:-default}
if [ -z "$SERVE_MODE" ]; then CONC=${CONC:-8}; else CONC=${CONC:-16}; fi
echo "=== remote_eval start $(date -u) MAXLEN=$MAXLEN KVDTYPE=$KVDTYPE GPU_UTIL=$GPU_UTIL ADAPTER=${ADAPTER:-none} EVAL_SET=${EVAL_SET:-both} TRIALS=${TRIALS:-1} SERVE_MODE=${SERVE_MODE:-<unset:lora>} QUANT=$QUANT SPEC=$SPEC MTP_K=$MTP_K CONC=$CONC MAX_NUM_SEQS=$MAX_NUM_SEQS THINKING=$THINKING ==="
case "${SERVE_MODE:-lora}" in lora|merged) ;; *) echo "bad SERVE_MODE=$SERVE_MODE"; exit 2;; esac
case "$QUANT" in none|fp8) ;; *) echo "bad QUANT=$QUANT"; exit 2;; esac
case "$SPEC" in none|mtp|ngram) ;; *) echo "bad SPEC=$SPEC"; exit 2;; esac
if [ "$SERVE_MODE" = merged ] && [ "${ADAPTER:-none}" = none ]; then echo "SERVE_MODE=merged needs ADAPTER"; exit 2; fi
up() { python - "$@" <<'PY' || true
import sys,os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); repo=os.environ["WORK_REPO"]; src,dst=sys.argv[1],sys.argv[2]
(api.upload_folder if os.path.isdir(src) else api.upload_file)(**({"folder_path":src} if os.path.isdir(src) else {"path_or_fileobj":src}), path_in_repo=dst, repo_id=repo, repo_type="dataset"); print("uploaded",src,"->",dst)
PY
}
trap 'echo "=== remote_eval exit $(date -u) ==="; up $W/status status_${STATUS_TAG:-eval}' EXIT
python -m pip install -q huggingface_hub hf_transfer 2>&1 | tail -1; echo "ALIVE $(date -u) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)" > $W/status/step_eval.txt; up $W/status status_${STATUS_TAG:-eval}
heartbeat() { while true; do sleep 300; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > $W/status/gpu.txt; date -u >> $W/status/gpu.txt; [ -f $W/status/vllm.log ] && tail -c 4000 $W/status/vllm.log > $W/status/vllm_tail.log; up $W/status status_${STATUS_TAG:-eval}; done; }
heartbeat & HB=$!; trap 'kill $HB 2>/dev/null; echo "=== remote_eval exit $(date -u) ==="; up $W/status status_${STATUS_TAG:-eval}' EXIT
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["WORK_REPO"], repo_type="dataset", local_dir="/workspace", token=os.environ["HF_TOKEN"], allow_patterns=["bundle/*","data/*","data_v*/*","adapter_v*/**"])  # bundle includes task_tiers.json
snapshot_download("Qwen/Qwen3.8-27B", local_dir="/workspace/Qwen3.8-27B", token=os.environ["HF_TOKEN"])
PY
echo "=== vLLM in its own venv ==="; python -m pip install -q uv 2>&1 | tail -1; uv venv /workspace/vvenv -q --python 3.12 2>&1 | tail -1 || python -m venv /workspace/vvenv
uv pip install -q --python /workspace/vvenv/bin/python vllm==0.28.0 2>&1 | tail -2; VPY=/workspace/vvenv/bin/python; $VPY -c "import vllm,torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
echo "STEP vllm installed $(date -u)" >> $W/status/step_eval.txt; up $W/status status_${STATUS_TAG:-eval}
MODEL=base; [ "${ADAPTER:-none}" != "none" ] && MODEL=student
# ---- merge (SERVE_MODE=merged): same merge_lora.py as remote_bench.sh; keeps shard layout, config, tokenizer and the 15 mtp.* tensors ----
cat > $W/merge_lora.py <<'PY'
import json, os, sys, time, shutil, glob, math
import torch
from safetensors import safe_open
from safetensors.torch import save_file
base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time(); dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
assert cfg.get("peft_type") == "LORA" and not cfg.get("use_dora") and cfg.get("bias", "none") == "none", cfg
r, alpha = cfg["r"], cfg["lora_alpha"]; scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
assert not cfg.get("rank_pattern") and not cfg.get("alpha_pattern"), "per-module rank/alpha not handled"
A, B = {}, {}
with safe_open(os.path.join(adapter, "adapter_model.safetensors"), "pt", device=dev) as f:
    for k in f.keys():
        mod = k.replace("base_model.model.", "", 1)
        if k.endswith(".lora_A.weight"): A[mod[:-len(".lora_A.weight")]] = f.get_tensor(k)
        elif k.endswith(".lora_B.weight"): B[mod[:-len(".lora_B.weight")]] = f.get_tensor(k)
        else: raise SystemExit("unexpected adapter key " + k)
assert set(A) == set(B) and A, "A/B mismatch"
targets = {m + ".weight" for m in A}; print("adapter modules:", len(A), "scale:", scale, "device:", dev)
os.makedirs(out, exist_ok=True)
for f in os.listdir(base):  # everything except weights verbatim (config, tokenizer, chat template, index)
    if not f.endswith(".safetensors") and os.path.isfile(os.path.join(base, f)): shutil.copy(os.path.join(base, f), os.path.join(out, f))
idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
missing = targets - set(idx); assert not missing, f"{len(missing)} adapter targets not in base: {sorted(missing)[:5]}"
mtp = [k for k in idx if k.startswith("mtp.")]; print("mtp tensors in base index:", len(mtp))
n_mod, max_d, sum_d, n_el = 0, 0.0, 0.0, 0
for shard in sorted(set(idx.values())):
    tensors = {}
    with safe_open(os.path.join(base, shard), "pt", device="cpu") as f:
        meta = f.metadata()
        for k in f.keys():
            t = f.get_tensor(k)
            if k in targets:
                m = k[:-len(".weight")]; a, b = A[m], B[m]
                assert t.shape == (b.shape[0], a.shape[1]), (k, t.shape, a.shape, b.shape)
                delta = (b.float() @ a.float()) * scale
                w = t.to(dev).float() + delta
                d = delta.abs(); max_d = max(max_d, d.max().item()); sum_d += d.sum().item(); n_el += d.numel()
                t = w.to(t.dtype).cpu(); n_mod += 1; del delta, w, d
            tensors[k] = t
    save_file(tensors, os.path.join(out, shard), metadata=meta or {"format": "pt"}); del tensors
    print("wrote", shard, f"{time.time()-t0:.0f}s", flush=True)
assert n_mod == len(targets), (n_mod, len(targets))
out_idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
for k, s in out_idx.items():  # every indexed tensor (incl. mtp.*) must exist in the merged dir
    with safe_open(os.path.join(out, s), "pt", device="cpu") as f: assert k in f.keys(), k
size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, "*")))
res = {"config": "merge", "kind": "merge", "status": "ok", "modules_merged": n_mod, "scale": scale, "merge_s": round(time.time()-t0, 1),
       "merged_gb": round(size/1e9, 1), "max_abs_delta": max_d, "mean_abs_delta": sum_d/max(1, n_el), "mtp_tensors_kept": len(mtp), "device": dev}
print(json.dumps(res)); json.dump(res, open("/workspace/status/merge.json", "w"))
PY
SERVE_PATH=$W/Qwen3.8-27B; SERVED=base; LORA=""
if [ "$SERVE_MODE" = merged ]; then
  echo "=== merge $ADAPTER into base -> $W/merged ==="; df -h $W | tail -1
  if timeout 2400 $VPY $W/merge_lora.py $W/Qwen3.8-27B $W/$ADAPTER $W/merged && [ -f $W/status/merge.json ]; then
    cat $W/status/merge.json; du -sh $W/merged; echo "STEP merge done $(date -u)" >> $W/status/step_eval.txt; SERVE_PATH=$W/merged; SERVED=$MODEL
  else
    echo "STEP merge FAILED $(date -u)" >> $W/status/step_eval.txt; up $W/status status_${STATUS_TAG:-eval}; exit 1
  fi
  up $W/status status_${STATUS_TAG:-eval}
else
  [ "${ADAPTER:-none}" != "none" ] && LORA="--enable-lora --lora-modules student=$W/${ADAPTER} --max-lora-rank 64"
fi
EXTRA=""
[ "$QUANT" = fp8 ] && EXTRA="$EXTRA --quantization fp8"
case "$SPEC" in
  mtp)   EXTRA="$EXTRA --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K}" ;;   # native Qwen3_5MTP head, no "model" key needed
  ngram) EXTRA="$EXTRA --speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":3,\"prompt_lookup_max\":5,\"prompt_lookup_min\":2}" ;;
esac
[ -n "${CUDAGRAPH:-}" ] && EXTRA="$EXTRA --compilation-config {\"cudagraph_mode\":\"$CUDAGRAPH\"}"
echo "vllm serve $SERVE_PATH --served-model-name $SERVED --max-num-seqs $MAX_NUM_SEQS $LORA $EXTRA"
nohup $VPY -m vllm.entrypoints.openai.api_server --model $SERVE_PATH --served-model-name $SERVED --port 8000 --max-model-len $MAXLEN --max-num-seqs $MAX_NUM_SEQS --gpu-memory-utilization $GPU_UTIL --kv-cache-dtype $KVDTYPE \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching $LORA $EXTRA > $W/status/vllm.log 2>&1 &
VPID=$!; UP=0; for i in $(seq 1 150); do curl -sf localhost:8000/v1/models >/dev/null && { UP=1; break; }; kill -0 $VPID 2>/dev/null || break; sleep 10; done
if [ "$UP" != 1 ]; then echo "STEP vllm FAILED $(date -u)" >> $W/status/step_eval.txt; grep -E "Error|error" $W/status/vllm.log | tail -20; up $W/status status_${STATUS_TAG:-eval}; exit 1; fi
curl -s localhost:8000/v1/models | head -c 400; echo; echo "STEP vllm up $(date -u)" >> $W/status/step_eval.txt
grep -E "KV cache size|Maximum concurrency|Mamba cache|SpeculativeConfig|quantization|Using default LoRA" $W/status/vllm.log | tail -8; tail -c 3000 $W/status/vllm.log > $W/status/vllm_tail.log; up $W/status status_${STATUS_TAG:-eval}

# ============================ PROBE: what does the model actually emit? ============================
# Greedy (temperature 0) generations, thinking off and on, student vs base, on real multi-turn prefixes from the held-out
# dev_turns.jsonl and train_turns.jsonl (same domain policy + tool schemas as the tau2 eval). Full text is kept, nothing
# is truncated, so 'loop' / 'degenerate' claims can be checked against the output itself. max_tokens 8192 like the capped eval.
echo "=== PROBE start $(date -u) ==="
cat > $W/probe.py <<'PY'
import json, time, os, sys, random, collections, concurrent.futures as cf, requests
W='/workspace'; URL='http://localhost:8000/v1/chat/completions'; OUT=f'{W}/status/probe.jsonl'
NPRE=int(os.environ.get('N_PREFIX','10')); MAXTOK=int(os.environ.get('MAXTOK','8192'))
def rows(p):
    try: return [json.loads(l) for l in open(p)]
    except Exception as e: print('no',p,e); return []
def prefixes(rs, n, tag):
    out=[]; random.seed(0)
    for r in rs:
        msgs=r['messages']; idx=[i for i,m in enumerate(msgs) if m['role']=='assistant' and i>0]
        if not idx: continue
        # one shallow (first assistant turn after the user) and one deep (last assistant turn) prefix per row
        for i in {idx[0], idx[len(idx)//2], idx[-1]}:
            out.append({'tag':tag,'row':rs.index(r),'cut':i,'prev_role':msgs[i-1]['role'],'n_msgs':i,'messages':msgs[:i],'tools':r.get('tools') or [],
                        'gold':{'content':msgs[i].get('content'),'tool_calls':msgs[i].get('tool_calls')}})
    random.shuffle(out); out.sort(key=lambda p: sum(len(json.dumps(m)) for m in p['messages']))
    # spread over depth: take evenly spaced by prompt size
    step=max(1,len(out)//n); return out[::step][:n]
dev=rows(f'{W}/data_v1/dev_turns.jsonl'); tr=rows(f'{W}/data_v1/train_turns.jsonl')
P=prefixes(dev, NPRE, 'dev')+prefixes(tr, NPRE//2, 'train'); print('prefixes',len(P),flush=True)
def one(p, model, thinking):
    body={'model':model,'messages':p['messages'],'temperature':0.0,'max_tokens':MAXTOK,'stream':False}
    if p['tools']: body['tools']=p['tools']
    if thinking=='off': body['chat_template_kwargs']={'enable_thinking':False}
    t0=time.time()
    try:
        r=requests.post(URL,json=body,timeout=1800); j=r.json(); ch=j['choices'][0]; m=ch['message']
        content=m.get('content') or ''; reasoning=m.get('reasoning_content') or m.get('reasoning') or ''
        def rep(t):
            if len(t)<300: return 0
            c=collections.Counter(t[i:i+40] for i in range(0,len(t)-40,20)); return c.most_common(1)[0][1]
        rec={'tag':p['tag'],'row':p['row'],'cut':p['cut'],'prev_role':p['prev_role'],'n_msgs':p['n_msgs'],'model':model,'thinking':thinking,'ok':True,
             'finish':ch.get('finish_reason'),'usage':j.get('usage'),'wall_s':round(time.time()-t0,1),'content_chars':len(content),'reasoning_chars':len(reasoning),
             'content_repeat40':rep(content),'reasoning_repeat40':rep(reasoning),'tool_calls':m.get('tool_calls'),'content':content,'reasoning':reasoning,'gold':p['gold']}
    except Exception as e:
        rec={'tag':p['tag'],'row':p['row'],'cut':p['cut'],'model':model,'thinking':thinking,'ok':False,'error':str(e)[:500],'wall_s':round(time.time()-t0,1)}
    with open(OUT,'a') as f: f.write(json.dumps(rec)+'\n')
    print(f"{model:8}{thinking:8} {p['tag']}#{p['row']}@{p['cut']:<3} finish={rec.get('finish')} comp_tok={(rec.get('usage') or {}).get('completion_tokens')} content={rec.get('content_chars')} rep={rec.get('content_repeat40')} reasoning={rec.get('reasoning_chars')} tc={len(rec.get('tool_calls') or [])} wall={rec['wall_s']}s",flush=True)
    return rec
# order: the case under suspicion first, then its controls; 4 in flight (the eval's short tier ran 10)
jobs=[(p,'student','off') for p in P]+[(p,'base','off') for p in P]+[(p,'student','default') for p in P]
with cf.ThreadPoolExecutor(4) as ex: list(ex.map(lambda a: one(*a), jobs))
print('=== PROBE done ===',flush=True)
PY
$VPY -m pip install -q requests 2>&1 | tail -1
# heartbeat already uploads status/ every 5 min (probe.jsonl grows there)
timeout 3600 $VPY $W/probe.py; echo "STEP probe done $(date -u)" >> $W/status/step_eval.txt
curl -s localhost:8000/metrics | grep -E "^vllm:(num_preemptions_total|request_generation_tokens_(sum|count)|request_prompt_tokens_(sum|count))" > $W/status/metrics_tail.txt
echo "PROBE_COMPLETE $(date -u)" >> $W/status/step_eval.txt; up $W/status status_${STATUS_TAG:-eval}
echo "=== PROBE end $(date -u) ==="
