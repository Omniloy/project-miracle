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
MAXLEN=${MAXLEN:-262144}; KVDTYPE=${KVDTYPE:-auto}; GPU_UTIL=${GPU_UTIL:-0.90}; SERVE_MODE=${SERVE_MODE:-}; QUANT=${QUANT:-none}; SPEC=${SPEC:-none}; MTP_K=${MTP_K:-2}; MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}; THINKING=${THINKING:-default}
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
trap 'echo "=== remote_eval exit $(date -u) ==="; up $W/status status_eval' EXIT
python -m pip install -q huggingface_hub hf_transfer 2>&1 | tail -1; echo "ALIVE $(date -u) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)" > $W/status/step_eval.txt; up $W/status status_eval
heartbeat() { while true; do sleep 300; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > $W/status/gpu.txt; date -u >> $W/status/gpu.txt; [ -f $W/status/vllm.log ] && tail -c 4000 $W/status/vllm.log > $W/status/vllm_tail.log; up $W/status status_eval; done; }
heartbeat & HB=$!; trap 'kill $HB 2>/dev/null; echo "=== remote_eval exit $(date -u) ==="; up $W/status status_eval' EXIT
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["WORK_REPO"], repo_type="dataset", local_dir="/workspace", token=os.environ["HF_TOKEN"], allow_patterns=["bundle/*","data/*","data_v*/*","adapter_v*/**"])
snapshot_download("Qwen/Qwen3.8-27B", local_dir="/workspace/Qwen3.8-27B", token=os.environ["HF_TOKEN"])
PY
echo "=== vLLM in its own venv ==="; python -m pip install -q uv 2>&1 | tail -1; uv venv /workspace/vvenv -q --python 3.12 2>&1 | tail -1 || python -m venv /workspace/vvenv
uv pip install -q --python /workspace/vvenv/bin/python vllm==0.28.0 2>&1 | tail -2; VPY=/workspace/vvenv/bin/python; $VPY -c "import vllm,torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
echo "STEP vllm installed $(date -u)" >> $W/status/step_eval.txt; up $W/status status_eval
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
    echo "STEP merge FAILED $(date -u)" >> $W/status/step_eval.txt; up $W/status status_eval; exit 1
  fi
  up $W/status status_eval
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
if [ "$UP" != 1 ]; then echo "STEP vllm FAILED $(date -u)" >> $W/status/step_eval.txt; grep -E "Error|error" $W/status/vllm.log | tail -20; up $W/status status_eval; exit 1; fi
curl -s localhost:8000/v1/models | head -c 400; echo; echo "STEP vllm up $(date -u)" >> $W/status/step_eval.txt
grep -E "KV cache size|Maximum concurrency|Mamba cache|SpeculativeConfig|quantization|Using default LoRA" $W/status/vllm.log | tail -8; tail -c 3000 $W/status/vllm.log > $W/status/vllm_tail.log; up $W/status status_eval
echo "=== tau2-bench ==="; cd $W; [ -d tau2-bench ] || git clone -q --depth 1 https://github.com/sierra-research/tau2-bench.git; cd tau2-bench; pip install -q uv 2>/dev/null; uv sync -q --extra knowledge 2>&1 | tail -1
export HOSTED_VLLM_API_BASE=http://localhost:8000/v1 HOSTED_VLLM_API_KEY=dummy
# synthetic dev set = all synthetic tasks not used for training
mkdir -p $W/data_synth/tau2/domains/banking_knowledge/tasks; for f in data/tau2/domains/banking_knowledge/*; do b=$(basename $f); [ "$b" = tasks ] || [ "$b" = tasks.json ] || ln -sfn $PWD/$f $W/data_synth/tau2/domains/banking_knowledge/$b; done
for f in data/tau2/*; do b=$(basename $f); [ "$b" = domains ] || ln -sfn $PWD/$f $W/data_synth/tau2/$b; done; for d in airline retail telecom mock; do ln -sfn $PWD/data/tau2/domains/$d $W/data_synth/tau2/domains/$d; done
python - <<'PY'
import json,glob,shutil,os
# dev = the PINNED held-out tasks of the training split (10 tasks), not every task outside v0's train set (88 tasks, ~4 h)
sp=json.load(open('/workspace/'+os.environ.get('DATA_DIR','data')+'/split.json')); hold=set(sp.get('holdout_tasks') or [])
n=0
for f in glob.glob('/workspace/bundle/data_synth_tasks/task_*.json'):
    tid=json.load(open(f))['id']
    if tid in hold: shutil.copy(f,'/workspace/data_synth/tau2/domains/banking_knowledge/tasks/'); n+=1
print('dev synthetic tasks:', n)
PY
if [ "$THINKING" = off ]; then AGENT_ARGS='{"temperature":1.0,"top_p":0.95,"extra_body":{"top_k":20,"chat_template_kwargs":{"enable_thinking":false}}}'
else AGENT_ARGS='{"temperature":1.0,"top_p":0.95,"extra_body":{"top_k":20}}'; fi
echo "agent args: $AGENT_ARGS  concurrency: $CONC  model: hosted_vllm/$MODEL"
if [ "${EVAL_SET:-both}" != "test" ]; then
  echo "=== DEV eval (synthetic held-out, Flash user-sim) ==="; TAU2_DATA_DIR=$W/data_synth .venv/bin/tau2 run --domain banking_knowledge --retrieval-config bm25_grep --num-trials 2 --max-steps 200 --seed 5 \
    --agent-llm hosted_vllm/$MODEL --agent-llm-args "$AGENT_ARGS" --user-llm openrouter/z-ai/glm-5.3-flash --max-concurrency $CONC --save-to dev_${MODEL} 2>&1 | grep -E 'Average Reward|Pass\^1|Infra|Error' | tail -4
  cp -r $W/data_synth/simulations/dev_${MODEL} $W/eval/ 2>/dev/null
fi
if [ "${EVAL_SET:-both}" != "dev" ]; then
  # RUNS = comma-separated model:thinking pairs evaluated sequentially on the same server, e.g. "base:default,student:default,student:off"
  # (default: the single MODEL with $THINKING). Each run uploads its own results so partial progress survives.
  RUNS=${RUNS:-$MODEL:$THINKING}; RUNS=${RUNS//,/ }; RUNS=${RUNS//_/ }
  for run in $RUNS; do
    RM=${run%%:*}; RT=${run##*:}; [ "$RM" = "$run" ] && RT=$THINKING
    if [ "$RT" = off ]; then RA='{"temperature":1.0,"top_p":0.95,"extra_body":{"top_k":20,"chat_template_kwargs":{"enable_thinking":false}}}'; else RA='{"temperature":1.0,"top_p":0.95,"extra_body":{"top_k":20}}'; fi
    TAG=test_${RM}_think${RT}
    echo "=== TEST eval $TAG (97 real tasks, AA config, gpt-5.4-mini medium) $(date -u) ==="; .venv/bin/tau2 run --domain banking_knowledge --retrieval-config bm25_grep --num-trials ${TRIALS:-1} --max-steps 200 --seed 300 \
      --agent-llm hosted_vllm/$RM --agent-llm-args "$RA" --user-llm openrouter/openai/gpt-5.4-mini --user-llm-args '{"reasoning_effort":"medium"}' --max-concurrency $CONC --save-to $TAG 2>&1 | grep -E 'Average Reward|Pass\^1|Infra|Error' | tail -4
    cp -r data/simulations/$TAG $W/eval/ 2>/dev/null; echo "STEP $TAG done $(date -u)" >> $W/status/step_eval.txt; up $W/eval/$TAG eval_${MODEL}/$TAG; up $W/status status_eval
  done
fi
echo "{\"serve_mode\":\"${SERVE_MODE:-lora}\",\"quant\":\"$QUANT\",\"spec\":\"$SPEC\",\"mtp_k\":$MTP_K,\"conc\":$CONC,\"max_num_seqs\":$MAX_NUM_SEQS,\"thinking\":\"$THINKING\",\"kv_dtype\":\"$KVDTYPE\",\"maxlen\":$MAXLEN,\"adapter\":\"${ADAPTER:-none}\"}" > $W/eval/serving_config.json
up $W/eval eval_${MODEL}; echo "EVAL_COMPLETE $(date -u)" | tee -a $W/status/step_eval.txt; up $W/status status_eval
