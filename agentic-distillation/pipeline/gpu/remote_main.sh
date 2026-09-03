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
heartbeat() { while true; do sleep 240; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > $W/status/gpu.txt; date -u >> $W/status/gpu.txt; up $W/status status; done; }
heartbeat & HB=$!
trap 'kill $HB 2>/dev/null; echo "=== remote_main exit $(date -u) ==="; up $W/status status' EXIT

echo "=== fetch bundle + data from $WORK_REPO ==="
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["WORK_REPO"], repo_type="dataset", local_dir="/workspace", token=os.environ["HF_TOKEN"], allow_patterns=["bundle/*","data/*"])
EOF
ls -la $W/bundle $W/data

echo "=== kernels ==="
python -c "import fla; print('fla', fla.__version__)" 2>/dev/null || pip install -q flash-linear-attention 2>&1 | tail -1
# causal-conv1d (DeltaNet conv) and flash-attn (full-attention layers) were missing on the v0 box -> ~85 tok/s.
# Try prebuilt wheels with a time cap; training falls back to slower kernels if they are unavailable for this GPU.
python -c "import causal_conv1d" 2>/dev/null || timeout 900 pip install -q causal-conv1d --no-build-isolation 2>&1 | tail -1
python -c "import flash_attn" 2>/dev/null || timeout 1500 pip install -q flash-attn --no-build-isolation 2>&1 | tail -1
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
echo "STEP kernels done" > $W/status/step.txt; up $W/status status

echo "=== model download ==="
python - <<'EOF'
import os, time
from huggingface_hub import snapshot_download
t=time.time(); snapshot_download("Qwen/Qwen3.8-27B", local_dir="/workspace/Qwen3.8-27B", token=os.environ["HF_TOKEN"]); print("model downloaded in", round(time.time()-t), "s")
EOF
du -sh $W/Qwen3.8-27B; echo "STEP model done" > $W/status/step.txt; up $W/status status

if [ "${STAGE:-all}" != "train" ]; then
  echo "=== SMOKE: LoRA attn+mlp ==="; cd $W/bundle
  python smoke_lora.py --model $W/Qwen3.8-27B --steps 8 --seq 4096 --data $W/data/train_turns.jsonl 2>&1 | tail -30; cp smoke_result.json $W/status/smoke_attn_mlp.json 2>/dev/null
  echo "=== SMOKE: + DeltaNet projections ==="
  python smoke_lora.py --model $W/Qwen3.8-27B --steps 8 --seq 4096 --data $W/data/train_turns.jsonl --target-deltanet 2>&1 | tail -30; cp smoke_result.json $W/status/smoke_deltanet.json 2>/dev/null
  echo "STEP smoke done" > $W/status/step.txt; up $W/status status; cd $W
fi

if [ "${STAGE:-all}" != "smoke" ]; then
  echo "=== TRAIN: axolotl LoRA ==="
  # if the DeltaNet smoke failed, fall back to attention+MLP targets only
  if [ -f $W/status/smoke_deltanet.json ] && ! python -c "import json,sys; d=json.load(open('$W/status/smoke_deltanet.json')); sys.exit(0 if d.get('loss_decreased') else 1)"; then
    echo "DeltaNet LoRA smoke failed -> removing linear_attn targets"; sed -i '/linear_attn\./d' $W/bundle/axolotl_lora_q38_27b.yaml
  fi
  axolotl preprocess $W/bundle/axolotl_lora_q38_27b.yaml 2>&1 | tail -15
  echo "STEP preprocess done" > $W/status/step.txt; up $W/status status
  axolotl train $W/bundle/axolotl_lora_q38_27b.yaml 2>&1 | tee $W/status/train.log | grep -E "loss|eval|Error|error|Traceback|saving|Saving|it/s|s/it" | tail -400
  echo "STEP train done rc=$?" > $W/status/step.txt
  OUT=$W/outputs/${RUN_NAME:-q38_27b_lora_v0}; ls -la $OUT | head; up $OUT adapter_${RUN_NAME#q38_27b_lora_}
fi
echo "JOB_COMPLETE $(date -u)" >> $W/status/step.txt; up $W/status status
