#!/bin/bash
# Unattended eval job on a Vast box: serve Qwen3.8-27B (+ optional LoRA adapter from HF) with vLLM, run tau2-bench
# banking_knowledge with the AA config against it, upload results to HF. Env: HF_TOKEN, OPENROUTER_API_KEY, WORK_REPO,
# ADAPTER (HF path in WORK_REPO, e.g. adapter_v0, or "none"), EVAL_SET (dev|test|both), TRIALS (default 1)
set -uo pipefail; export HF_HUB_ENABLE_HF_TRANSFER=1
W=/workspace; mkdir -p $W/status $W/eval; cd $W; LOG=$W/status/remote_eval.log; exec > >(tee -a $LOG) 2>&1
echo "=== remote_eval start $(date -u) ADAPTER=${ADAPTER:-none} EVAL_SET=${EVAL_SET:-both} TRIALS=${TRIALS:-1} ==="
up() { python - "$@" <<'PY' || true
import sys,os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); repo=os.environ["WORK_REPO"]; src,dst=sys.argv[1],sys.argv[2]
(api.upload_folder if os.path.isdir(src) else api.upload_file)(**({"folder_path":src} if os.path.isdir(src) else {"path_or_fileobj":src}), path_in_repo=dst, repo_id=repo, repo_type="dataset"); print("uploaded",src,"->",dst)
PY
}
trap 'echo "=== remote_eval exit $(date -u) ==="; up $W/status status_eval' EXIT
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["WORK_REPO"], repo_type="dataset", local_dir="/workspace", token=os.environ["HF_TOKEN"], allow_patterns=["bundle/*","data/*","adapter_v*/**"])
snapshot_download("Qwen/Qwen3.8-27B", local_dir="/workspace/Qwen3.8-27B", token=os.environ["HF_TOKEN"])
PY
echo "=== vLLM (pip if missing) ==="; python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || pip install -q vllm 2>&1 | tail -1
LORA=""; [ "${ADAPTER:-none}" != "none" ] && LORA="--enable-lora --lora-modules student=$W/${ADAPTER} --max-lora-rank 64"
nohup python -m vllm.entrypoints.openai.api_server --model $W/Qwen3.8-27B --served-model-name base --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching $LORA > $W/status/vllm.log 2>&1 &
for i in $(seq 1 90); do curl -sf localhost:8000/v1/models >/dev/null && break; sleep 10; done; curl -s localhost:8000/v1/models | head -c 300; echo
MODEL=base; [ "${ADAPTER:-none}" != "none" ] && MODEL=student
echo "=== tau2-bench ==="; cd $W; [ -d tau2-bench ] || git clone -q --depth 1 https://github.com/sierra-research/tau2-bench.git; cd tau2-bench; pip install -q uv 2>/dev/null; uv sync -q --extra knowledge 2>&1 | tail -1
export HOSTED_VLLM_API_BASE=http://localhost:8000/v1 HOSTED_VLLM_API_KEY=dummy
# synthetic dev set = all synthetic tasks not used for training
mkdir -p $W/data_synth/tau2/domains/banking_knowledge/tasks; for f in data/tau2/domains/banking_knowledge/*; do b=$(basename $f); [ "$b" = tasks ] || [ "$b" = tasks.json ] || ln -sfn $PWD/$f $W/data_synth/tau2/domains/banking_knowledge/$b; done
for f in data/tau2/*; do b=$(basename $f); [ "$b" = domains ] || ln -sfn $PWD/$f $W/data_synth/tau2/$b; done; for d in airline retail telecom mock; do ln -sfn $PWD/data/tau2/domains/$d $W/data_synth/tau2/domains/$d; done
python - <<'PY'
import json,glob,shutil,os
train=set(json.load(open('/workspace/data/split.json'))['train_tasks'])
n=0
for f in glob.glob('/workspace/bundle/data_synth_tasks/task_*.json'):
    tid=json.load(open(f))['id']
    if tid not in train: shutil.copy(f,'/workspace/data_synth/tau2/domains/banking_knowledge/tasks/'); n+=1
print('dev synthetic tasks:', n)
PY
AGENT_ARGS='{"temperature":1.0,"top_p":0.95,"extra_body":{"top_k":20}}'
if [ "${EVAL_SET:-both}" != "test" ]; then
  echo "=== DEV eval (synthetic held-out, Flash user-sim) ==="; TAU2_DATA_DIR=$W/data_synth .venv/bin/tau2 run --domain banking_knowledge --retrieval-config bm25_grep --num-trials 2 --max-steps 200 --seed 5 \
    --agent-llm hosted_vllm/$MODEL --agent-llm-args "$AGENT_ARGS" --user-llm openrouter/z-ai/glm-5.3-flash --max-concurrency 8 --save-to dev_${MODEL} 2>&1 | grep -E 'Average Reward|Pass\^1|Infra|Error' | tail -4
  cp -r $W/data_synth/simulations/dev_${MODEL} $W/eval/ 2>/dev/null
fi
if [ "${EVAL_SET:-both}" != "dev" ]; then
  echo "=== TEST eval (97 real tasks, AA config, gpt-5.4-mini medium) ==="; .venv/bin/tau2 run --domain banking_knowledge --retrieval-config bm25_grep --num-trials ${TRIALS:-1} --max-steps 200 --seed 300 \
    --agent-llm hosted_vllm/$MODEL --agent-llm-args "$AGENT_ARGS" --user-llm openrouter/openai/gpt-5.4-mini --user-llm-args '{"reasoning_effort":"medium"}' --max-concurrency 8 --save-to test_${MODEL} 2>&1 | grep -E 'Average Reward|Pass\^1|Infra|Error' | tail -4
  cp -r data/simulations/test_${MODEL} $W/eval/ 2>/dev/null
fi
up $W/eval eval_${MODEL}; echo "EVAL_COMPLETE $(date -u)" | tee -a $W/status/step_eval.txt; up $W/status status_eval
