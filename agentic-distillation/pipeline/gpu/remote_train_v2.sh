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
fi
echo "ALL_DONE $(date -u)" >> $W/status/step.txt; up $W/status status_${STATUS_TAG:-train_v2}
