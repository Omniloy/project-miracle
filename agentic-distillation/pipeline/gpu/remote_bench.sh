#!/bin/bash
# Unattended serving benchmark on a Vast box (same conventions as remote_eval.sh): download base model + adapter_v0 +
# bench_client.py, build the vLLM 0.28.0 venv, then bring up vLLM once per candidate config, drive it with
# bench_client.py at concurrency 8 and 16, kill it, append JSON lines to /workspace/status/bench_results.jsonl and
# upload status/ to HF status_bench/ after every config. Ends with BENCH_COMPLETE in status_bench/step_bench.txt.
# Env: HF_TOKEN, WORK_REPO (required; set by launch_vast.sh). Optional: ADAPTER (default adapter_v0),
#   CONFIGS (default "A B C D E F C3"), CLIENT_SECS (timed seconds per concurrency level, default 100), CONCS (default
#   "8 16"), N_REQ (request pool, default 12: the client first prefills the whole pool with 64-token generations, ~2 min
#   cold on the first level and seconds on the second, so the timed phases hit warm prefixes like tau2 turns), MTP_K
#   (default 2), STARTUP_TIMEOUT (s, default 1200: covers a cold torch.compile + CUDA-graph capture), BENCH_DEADLINE_MIN
#   (default 85: remaining configs are skipped once elapsed wall time exceeds this; a config started just before it ends
#   ~10-12 min later). Expected: ~12 min setup + ~5 min merge + ~11-13 min per config -> A..E in ~80 min, F/C3 if time.
# Disk: base bf16 (54 GB) + merged bf16 (54 GB) + venv -> launch with DISK_GB >= 180.
# NOTE: bench_client.py must be in HF bundle/ (upload it with launch_vast.sh's uploader or the same python one-liner)
# before launching, since --onstart-cmd only fetches this script.
# Configs: A bf16+LoRA (current eval), B merged bf16, C merged+MTP(k=MTP_K), D merged+online FP8, E merged FP8+MTP,
#   F merged+ngram, C3 merged+MTP k=3 (optional, last); AM = bf16+LoRA+MTP (LoRA x spec-decode is not in the official
#   0.28.0 compatibility matrix, so it is only run as a fallback when the merge fails, or if listed in CONFIGS).
set -uo pipefail; export HF_HUB_ENABLE_HF_TRANSFER=1
W=/workspace; mkdir -p $W/status $W/bundle; cd $W; LOG=$W/status/remote_bench.log; exec > >(tee -a $LOG) 2>&1
T0=$(date +%s); STATUS=status_bench; STEP=$W/status/step_bench.txt; RES=$W/status/bench_results.jsonl; VLOG=$W/status/vllm.log
ADAPTER=${ADAPTER:-adapter_v0}; CONFIGS=${CONFIGS:-"A B C D E F C3"}; CLIENT_SECS=${CLIENT_SECS:-100}; CONCS=${CONCS:-"8 16"}
N_REQ=${N_REQ:-12}; MTP_K=${MTP_K:-2}; STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1200}; BENCH_DEADLINE_MIN=${BENCH_DEADLINE_MIN:-85}
echo "=== remote_bench start $(date -u) host=$(hostname) ADAPTER=$ADAPTER CONFIGS=$CONFIGS CLIENT_SECS=$CLIENT_SECS CONCS=$CONCS MTP_K=$MTP_K ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader
up() { python - "$@" <<'PY' || true
import sys,os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); repo=os.environ["WORK_REPO"]; src,dst=sys.argv[1],sys.argv[2]
(api.upload_folder if os.path.isdir(src) else api.upload_file)(**({"folder_path":src} if os.path.isdir(src) else {"path_or_fileobj":src}), path_in_repo=dst, repo_id=repo, repo_type="dataset"); print("uploaded",src,"->",dst)
PY
}
elapsed_min() { echo $(( ($(date +%s) - T0) / 60 )); }
step() { echo "STEP $* $(date -u) (+$(elapsed_min)m)" >> $STEP; up $W/status $STATUS; }
rec() { echo "$1" >> $RES; }  # append one JSON line
VPID=""; SERVER_RE='^[^ ]*vvenv/bin/python[^ ]* -m vllm\.entrypoints'   # anchored: matches only the API server, never this script/wrappers
server_alive() { { [ -n "$VPID" ] && kill -0 $VPID 2>/dev/null; } || pgrep -f "$SERVER_RE" >/dev/null; }
kill_vllm() {  # kill the API server + its EngineCore workers (same process group via setsid), wait until the GPU is free (bounded)
  if [ -n "$VPID" ]; then kill -TERM -- -$VPID 2>/dev/null || kill -TERM $VPID 2>/dev/null; fi
  pkill -TERM -f "$SERVER_RE" 2>/dev/null; pkill -TERM -f "^VLLM::" 2>/dev/null
  for i in $(seq 1 12); do sleep 5; server_alive || pgrep -f "^VLLM::" >/dev/null || break; done
  [ -n "$VPID" ] && kill -KILL -- -$VPID 2>/dev/null; pkill -KILL -f "$SERVER_RE" 2>/dev/null; pkill -KILL -f "^VLLM::" 2>/dev/null; VPID=""
  for i in $(seq 1 12); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${m:-0}" -lt 4000 ] && break; sleep 5; done
  echo "vllm down, GPU mem used: ${m:-?} MiB"
}
trap 'echo "=== remote_bench exit $(date -u) ==="; up $W/status $STATUS' EXIT
python -m pip install -q huggingface_hub hf_transfer 2>&1 | tail -1; echo "ALIVE $(date -u) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)" > $STEP; up $W/status $STATUS
heartbeat() { while true; do sleep 300; nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader > $W/status/gpu.txt; date -u >> $W/status/gpu.txt; [ -f $VLOG ] && tail -c 4000 $VLOG > $W/status/vllm_tail.log; up $W/status $STATUS; done; }
heartbeat & HB=$!; trap 'kill $HB 2>/dev/null; kill_vllm; echo "=== remote_bench exit $(date -u) (+$(elapsed_min)m) ==="; up $W/status $STATUS' EXIT

echo "=== downloads ==="
timeout 3600 python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
t=time.time(); ad=os.environ.get("ADAPTER","adapter_v0")
snapshot_download(os.environ["WORK_REPO"], repo_type="dataset", local_dir="/workspace", token=os.environ["HF_TOKEN"],
                  allow_patterns=["bundle/*", "data_v1/train_turns.jsonl", f"{ad}/*.json", f"{ad}/*.safetensors", f"{ad}/*.jinja"],
                  ignore_patterns=["*/checkpoint-*/*"])
snapshot_download("Qwen/Qwen3.8-27B", local_dir="/workspace/Qwen3.8-27B", token=os.environ["HF_TOKEN"])
print("downloads done in", round(time.time()-t), "s")
PY
CLIENT=$W/bundle/bench_client.py; DATA=$W/data_v1/train_turns.jsonl
for f in $CLIENT $DATA $W/$ADAPTER/adapter_config.json $W/$ADAPTER/adapter_model.safetensors $W/Qwen3.8-27B/config.json; do
  [ -f $f ] || { echo "BENCH_FAILED missing $f $(date -u)" | tee -a $STEP; up $W/status $STATUS; exit 1; }; done
du -sh $W/Qwen3.8-27B $W/$ADAPTER; df -h $W | tail -1; step downloads done

echo "=== vLLM in its own venv ==="; python -m pip install -q uv 2>&1 | tail -1; uv venv /workspace/vvenv -q --python 3.12 2>&1 | tail -1 || python -m venv /workspace/vvenv
uv pip install -q --python /workspace/vvenv/bin/python vllm==0.28.0 2>&1 | tail -2; VPY=/workspace/vvenv/bin/python
$VPY -c "import vllm,torch,requests,safetensors; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cap', torch.cuda.get_device_capability())" || { echo "BENCH_FAILED vllm install $(date -u)" | tee -a $STEP; exit 1; }
$VPY -m py_compile $CLIENT || { echo "BENCH_FAILED client does not compile $(date -u)" | tee -a $STEP; exit 1; }
step vllm installed

# ------------------------------------------------------------------------------------------------------------------
# merge adapter into the base weights (PEFT-equivalent W += alpha/r * B@A, computed on the GPU tensor-by-tensor over the
# original safetensors shards). Done this way instead of transformers+PEFT merge_and_unload because the transformers
# Qwen3_5 classes drop the mtp.* tensors on load (research finding) and re-key/split the checkpoint; this keeps the exact
# shard layout, config and the MTP head so configs C/E can use MTP on the merged weights.
# ------------------------------------------------------------------------------------------------------------------
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
run_merge() {
  echo "=== merge $ADAPTER into base -> $W/merged ==="
  if timeout 2400 $VPY $W/merge_lora.py $W/Qwen3.8-27B $W/$ADAPTER $W/merged && [ -f $W/status/merge.json ]; then
    rec "$(cat $W/status/merge.json)"; du -sh $W/merged; df -h $W | tail -1; MERGED=1
  else
    echo "MERGE FAILED rc=$?"; rec "{\"config\":\"merge\",\"kind\":\"merge\",\"status\":\"failed\"}"; MERGED=0; rm -rf $W/merged
  fi
  step merge status=$MERGED
}
MERGED=-1  # -1 = not attempted yet

# ------------------------------------------------------------------------------------------------------------------
# serving helpers
# ------------------------------------------------------------------------------------------------------------------
COMMON=(--served-model-name base --port 8000 --max-model-len 65536 --max-num-seqs 32 --gpu-memory-utilization 0.90
        --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching)
start_vllm() {  # $1 = model path, rest = extra flags. Sets VPID and STARTUP_S; returns 1 on failure (never hangs)
  local model=$1; shift; : > $VLOG
  echo "vllm serve $model ${COMMON[*]} $*"
  setsid nohup $VPY -m vllm.entrypoints.openai.api_server --model $model "${COMMON[@]}" "$@" > $VLOG 2>&1 &   # setsid: own process group -> engine workers die with it
  VPID=$!; local t=$(date +%s); UP=0
  while [ $(( $(date +%s) - t )) -lt $STARTUP_TIMEOUT ]; do
    curl -sf -m 10 localhost:8000/v1/models >/dev/null && { UP=1; break; }
    server_alive || { echo "vllm process exited"; break; }
    sleep 10
  done
  STARTUP_S=$(( $(date +%s) - t ))
  if [ "$UP" != 1 ]; then echo "vllm startup FAILED after ${STARTUP_S}s"; grep -E "Error|error|Traceback|NotImplemented|exceeds" $VLOG | tail -15; tail -c 3000 $VLOG > $W/status/vllm_tail.log; return 1; fi
  echo "vllm up in ${STARTUP_S}s"; grep -E "KV cache size|Maximum concurrency|Mamba|SpeculativeConfig|Speculative|Resolved architecture|quantization|Using default LoRA" $VLOG | tail -12; return 0
}
json_str() { python -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }  # stdin -> JSON string literal
server_info() {  # $1 config name, $2 status: one JSON line with startup time + relevant vLLM log lines for this config
  local alive=0; server_alive && alive=1
  local lines; lines=$(grep -E "KV cache size|Maximum concurrency|Mamba cache|SpecDecoding metrics|SpeculativeConfig|Resolved architecture|quantization|Using default LoRA|Error|Traceback" $VLOG | tail -12 | cut -c1-400 | json_str)
  rec "{\"config\":\"$1\",\"kind\":\"server_info\",\"status\":\"$2\",\"startup_s\":${STARTUP_S:-null},\"server_alive_after_load\":$alive,\"elapsed_min\":$(elapsed_min),\"vllm_log\":$lines}"
}
run_clients() {  # $1 config name, $2 served model name for the client, $3 extra client args (e.g. --greedy-compare base)
  local name=$1 model=$2 extra=${3:-}
  for c in $CONCS; do
    echo "--- client config=$name model=$model concurrency=$c ${CLIENT_SECS}s ---"
    local meta="{\"startup_s\":${STARTUP_S:-null},\"gpu\":\"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\"}"
    # worst case inside the client: greedy A/B (~2 min) + warmup (<= 2 x 300 s) + timed cap + 45 s drain -> hard-capped here
    timeout $(( CLIENT_SECS + 900 )) $VPY $CLIENT --base-url http://localhost:8000/v1 --model $model --data $DATA --n-requests $N_REQ \
      --concurrency $c --seconds $CLIENT_SECS --grace 45 --timeout 300 --config $name --meta "$meta" --out $RES $extra 2>&1 | tail -4
    local rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && { echo "client rc=$rc"; rec "{\"config\":\"$name\",\"concurrency\":$c,\"status\":\"client_failed\",\"rc\":$rc}"; }
    extra=""  # greedy compare only once
    grep "SpecDecoding metrics" $VLOG | tail -1 | cut -c1-300
    server_alive || { echo "vllm died during client run"; break; }
  done
}
bench_config() {  # $1 name, $2 model path, $3 client model name, $4 client extra args, $5.. vllm extra flags
  local name=$1 model=$2 cmodel=$3 cextra=$4; shift 4
  if [ $(elapsed_min) -ge $BENCH_DEADLINE_MIN ]; then echo "SKIP $name: deadline ${BENCH_DEADLINE_MIN}m reached (+$(elapsed_min)m)"; rec "{\"config\":\"$name\",\"kind\":\"server_info\",\"status\":\"skipped_deadline\",\"elapsed_min\":$(elapsed_min)}"; return; fi
  echo; echo "=============== CONFIG $name (+$(elapsed_min)m) ==============="; nvidia-smi --query-gpu=memory.used --format=csv,noheader
  if start_vllm "$model" "$@"; then
    run_clients $name $cmodel "$cextra"; server_info $name ok
  else
    server_info $name startup_failed
  fi
  kill_vllm; cp $VLOG $W/status/vllm_$name.log 2>/dev/null; gzip -f $W/status/vllm_$name.log 2>/dev/null; tail -c 4000 $VLOG > $W/status/vllm_tail.log
  step config $name done
}
need_merged() {  # merge lazily (once), never past the deadline
  if [ $MERGED -eq -1 ]; then
    if [ $(elapsed_min) -ge $((BENCH_DEADLINE_MIN - 10)) ]; then echo "SKIP merge: too close to deadline (+$(elapsed_min)m)"; MERGED=0; else run_merge; fi
  fi
  [ $MERGED -eq 1 ] && return 0
  echo "SKIP $1: merged weights unavailable"; rec "{\"config\":\"$1\",\"kind\":\"server_info\",\"status\":\"skipped_no_merge\"}"; return 1
}
MTP="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K}"   # vLLM 0.28.0 native Qwen3_5MTP; no "model" key needed
MTP3='--speculative-config {"method":"mtp","num_speculative_tokens":3}'
NGRAM='--speculative-config {"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}'
# Online FP8 (--quantization fp8, dynamic per-tensor W8A8) on sm_120: supported in principle (Blackwell has FP8 tensor cores);
# research flagged that offline block-128 FP8 is the vetted production path and online FP8 had a degenerate-output report for
# merged models on Blackwell -> D/E are measured here (sanity_pass_rate/leak_count expose degeneration) but startup failure just skips them.
FP8="--quantization fp8"

echo "=== benchmark loop: $CONFIGS ==="; : > $RES
for cfg in $CONFIGS; do
  case $cfg in
    A)  bench_config A_bf16_lora $W/Qwen3.8-27B student "--greedy-compare base" --enable-lora --lora-modules student=$W/$ADAPTER --max-lora-rank 64 ;;
    B)  need_merged B && bench_config B_merged_bf16 $W/merged base "" ;;
    C)  need_merged C && bench_config C_merged_mtp${MTP_K} $W/merged base "" $MTP ;;
    C3) need_merged C3 && { [ "$MTP_K" = 3 ] && echo "SKIP C3: same as C" || bench_config C3_merged_mtp3 $W/merged base "" $MTP3; } ;;
    D)  need_merged D && bench_config D_merged_fp8 $W/merged base "" $FP8 ;;
    E)  need_merged E && bench_config E_merged_fp8_mtp${MTP_K} $W/merged base "" $FP8 $MTP ;;
    F)  need_merged F && bench_config F_merged_ngram $W/merged base "" $NGRAM ;;
    AM) bench_config AM_bf16_lora_mtp${MTP_K} $W/Qwen3.8-27B student "" --enable-lora --lora-modules student=$W/$ADAPTER --max-lora-rank 64 $MTP; RAN_AM=1 ;;
    *)  echo "unknown config $cfg" ;;
  esac
done
# fallback: if the merge failed, LoRA+MTP is the only way to measure MTP on the student (works in 0.28.0 code, not in the official matrix)
if [ $MERGED -eq 0 ] && [ "${RAN_AM:-0}" != 1 ]; then echo "merge failed -> fallback config AM (bf16 + LoRA + MTP k=$MTP_K)"; bench_config AM_bf16_lora_mtp${MTP_K} $W/Qwen3.8-27B student "" --enable-lora --lora-modules student=$W/$ADAPTER --max-lora-rank 64 $MTP; fi
echo "=== results ==="; cat $RES
echo "BENCH_COMPLETE $(date -u) (+$(elapsed_min)m)" | tee -a $STEP; up $W/status $STATUS
