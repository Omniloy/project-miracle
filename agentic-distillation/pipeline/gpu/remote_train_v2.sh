#!/bin/bash
# Runs unattended on the Vast box (launched from --onstart-cmd). No SSH is available from the controller, so this
# script is the whole job: fetch bundle from HF -> install kernels -> download model -> smoke test -> LoRA SFT
# -> upload adapter/logs. Progress is mirrored to HF every few minutes (status/*.txt) and to `vastai logs`.
# Required env: HF_TOKEN, WORK_REPO (dataset repo holding bundle/ and data/), STAGE (smoke|train|all)
set -uo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
W=/workspace; mkdir -p $W/status $W/data $W/bundle; cd $W
LOG=$W/status/remote_main.log; exec > >(tee -a $LOG) 2>&1
echo "=== remote_main start $(date -u) STAGE=${STAGE:-all} host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

up() { # upload a path to the work repo (never fails the job)
  python - "$@" <<'EOF' || true
import sys,os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); repo=os.environ["WORK_REPO"]
src,dst=sys.argv[1],sys.argv[2]
if os.path.isdir(src): api.upload_folder(folder_path=src, path_in_repo=dst, repo_id=repo, repo_type="dataset")
else: api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="dataset")
print("uploaded", src, "->", dst)
EOF
}
heartbeat() { while true; do sleep 240; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > $W/status/gpu.txt; date -u >> $W/status/gpu.txt; up $W/status status_${STATUS_TAG:-train_v2}; done; }
heartbeat & HB=$!
trap 'kill $HB 2>/dev/null; echo "=== remote_main exit $(date -u) ==="; up $W/status status' EXIT

echo "=== fetch bundle + data from $WORK_REPO ==="
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["WORK_REPO"], repo_type="dataset", local_dir="/workspace", token=os.environ["HF_TOKEN"], allow_patterns=["bundle/*","data/*",os.environ.get("DATA_DIR","data_v2")+"/*"])
EOF
ls -la $W/bundle $W/data

echo "=== kernels ==="
python -c "import fla; print('fla', fla.__version__)" 2>/dev/null || pip install -q flash-linear-attention 2>&1 | tail -1
# causal-conv1d (DeltaNet conv) and flash-attn (full-attention layers) were missing on the v0 box -> ~85 tok/s.
# Try prebuilt wheels with a time cap; training falls back to slower kernels if they are unavailable for this GPU.
python - <<'EOF'
import importlib, torch, transformers, peft
print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)
for m in ('fla','causal_conv1d','flash_attn'):
    try: print(m, 'OK', getattr(importlib.import_module(m),'__version__',''))
    except Exception as e: print(m, 'MISSING', type(e).__name__)
EOF
# env-driven training config: DATA_DIR (default data), RUN_NAME (default q38_27b_lora_v0), EPOCHS
CFG=$W/bundle/axolotl_lora_q38_27b.yaml
sed -i "s#/workspace/data/train_turns.jsonl#/workspace/${DATA_DIR:-data}/train_turns.jsonl#; s#/workspace/data/dev_turns.jsonl#/workspace/${DATA_DIR:-data}/dev_turns.jsonl#; s#q38_27b_lora_v0#${RUN_NAME:-q38_27b_lora_v0}#g" $CFG
[ -n "${EPOCHS:-}" ] && sed -i "s#^num_epochs: .*#num_epochs: ${EPOCHS}#" $CFG
python -c "import flash_attn" 2>/dev/null || sed -i "s#attn_implementation: flash_attention_2#attn_implementation: sdpa#" $CFG
grep -E 'train_turns|output_dir|num_epochs|attn_implementation' $CFG
pip list 2>/dev/null | grep -iE '^(axolotl|flash-attn|flash_attn|flash-linear-attention|causal-conv1d|bitsandbytes|vllm) '
echo "STEP kernels done" > $W/status/step.txt; up $W/status status_${STATUS_TAG:-train_v2}

echo "=== model download ==="
python - <<'EOF'
import os, time
from huggingface_hub import snapshot_download
t=time.time(); snapshot_download("Qwen/Qwen3.8-27B", local_dir="/workspace/Qwen3.8-27B", token=os.environ["HF_TOKEN"]); print("model downloaded in", round(time.time()-t), "s")
EOF
du -sh $W/Qwen3.8-27B; echo "STEP model done" > $W/status/step.txt; up $W/status status_${STATUS_TAG:-train_v2}

# ============================ TRAIN v2 (custom HF+peft loop) ============================
DATA_DIR=${DATA_DIR:-data_v2}; RUN_NAME=${RUN_NAME:-v2}; EPOCHS=${EPOCHS:-2}; LR=${LR:-1e-4}; MAX_TOK=${MAX_TOK:-16384}; ACC_TOKENS=${ACC_TOKENS:-65536}
echo "=== TRAIN $RUN_NAME $(date -u) DATA_DIR=$DATA_DIR EPOCHS=$EPOCHS LR=$LR MAX_TOK=$MAX_TOK ACC_TOKENS=$ACC_TOKENS ==="; echo "STEP train start $(date -u)" >> $W/status/step.txt; up $W/status status_${STATUS_TAG:-train_v2}
ls -la $W/$DATA_DIR
TRAIN_FILES=$(ls $W/$DATA_DIR/train*.jsonl | tr '\n' ' '); DEV_FILES=$(ls $W/$DATA_DIR/dev*.jsonl 2>/dev/null | tr '\n' ' ')
( while true; do sleep 240; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > $W/status/gpu.txt; date -u >> $W/status/gpu.txt; tail -c 6000 $W/status/train_v2.log > $W/status/train_tail.log; up $W/status status_${STATUS_TAG:-train_v2}; done ) & HB2=$!
python $W/bundle/train_lora_v2.py --model $W/Qwen3.8-27B --train $TRAIN_FILES --dev $DEV_FILES --out $W/outputs/adapter_$RUN_NAME --epochs $EPOCHS --lr $LR --max-tok $MAX_TOK --acc-tokens $ACC_TOKENS --status $W/status/train_v2.jsonl > $W/status/train_v2.log 2>&1
RC=$?; kill $HB2 2>/dev/null; tail -5 $W/status/train_v2.log
if [ $RC -ne 0 ] || ! grep -q TRAIN_DONE $W/status/train_v2.log; then echo "STEP train FAILED rc=$RC $(date -u)" >> $W/status/step.txt; up $W/status status_${STATUS_TAG:-train_v2}; exit 1; fi
echo "STEP train done $(date -u)" >> $W/status/step.txt; ls -la $W/outputs/adapter_$RUN_NAME; up $W/outputs/adapter_$RUN_NAME adapter_$RUN_NAME; up $W/status status_${STATUS_TAG:-train_v2}
echo "TRAIN_COMPLETE $(date -u)" >> $W/status/step.txt; up $W/status status_${STATUS_TAG:-train_v2}
# ============================ chained eval: short-tier trace (thinking off, greedy) + phone probe ============================
if [ "${CHAIN_EVAL:-1}" = 1 ]; then
  echo "=== chained eval $(date -u) ==="; mkdir -p $W/adapter_$RUN_NAME; cp -r $W/outputs/adapter_$RUN_NAME/* $W/adapter_$RUN_NAME/
  curl -sSL -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/$WORK_REPO/resolve/main/bundle/remote_trace.sh -o $W/remote_trace.sh
  export ADAPTER=adapter_$RUN_NAME EVAL_SET=test KVDTYPE=fp8 GPU_UTIL=0.95 SERVE_MODE=lora SPEC=mtp MTP_K=2 TEMP=0 MAXTOK=8192 STATUS_TAG=trace_$RUN_NAME
  export TRACE_TASKS=${TRACE_TASKS:-task_026_task_072_task_008_task_078_task_027_task_044_task_050_task_021_task_046_task_014_task_093_task_020_task_002_task_016_task_095_task_019_task_033_task_006_task_031_task_035_task_015_task_032_task_023_task_034_task_017_task_004_task_005_task_010_task_098}
  export TRACE_CONC=${TRACE_CONC:-10} TRACE_MINUTES=${TRACE_MINUTES:-75} TRACE_MODEL=student TRACE_THINKING=off
  bash $W/remote_trace.sh
  echo "=== phone probe $(date -u) ==="; curl -sSL -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/$WORK_REPO/resolve/main/bundle/phone_probe.py -o $W/phone_probe.py
  rm -f $W/status/phone_probe.jsonl; /workspace/vvenv/bin/python -m pip install -q requests 2>&1 | tail -1
  MODELS=student,base /workspace/vvenv/bin/python $W/phone_probe.py $W/$DATA_DIR/dev_sw.jsonl $W/status/phone_probe.jsonl 2>&1 | tail -2
  echo "STEP phone probe done $(date -u)" >> $W/status/step.txt; up $W/status status_${STATUS_TAG:-train_v2}
fi
echo "ALL_DONE $(date -u)" >> $W/status/step.txt; up $W/status status_${STATUS_TAG:-train_v2}
