#!/bin/bash
# Cheap-GPU trainability smoke of the REAL training stack (Axolotl chat_template dataset type) on a SMALL model.
# Purpose: prove that switchboard phase1_sft.jsonl rows are accepted by axolotl preprocess (tool schemas + tool_calls
# render through the tokenizer chat template) and that a LoRA step actually reduces loss, before spending 96 GB-hours.
# Runs unattended from --onstart-cmd (no SSH); all progress is mirrored to HF status_smoke/ like remote_main.sh.
# Env: HF_TOKEN, WORK_REPO (dataset repo), optional MODEL (default Qwen/Qwen3-1.7B), MAX_STEPS (30), SEQ (4096).
set -uo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
W=/workspace; mkdir -p $W/status $W/data_sw; cd $W
LOG=$W/status/remote_smoke.log; exec > >(tee -a $LOG) 2>&1
MODEL=${MODEL:-Qwen/Qwen3-1.7B}; MAX_STEPS=${MAX_STEPS:-30}; SEQ=${SEQ:-4096}; RUN=smoke_small
OUT=$W/outputs/$RUN; CFG=$W/axolotl_lora_smoke.yaml; STEPF=$W/status/step_smoke.txt
echo "=== remote_smoke_small start $(date -u) MODEL=$MODEL MAX_STEPS=$MAX_STEPS SEQ=$SEQ host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

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
die() { # die "stage" logfile  -> STEP FAILED marker + last 40 log lines, upload, exit
  { echo "STEP FAILED $1 $(date -u)"; echo "--- last 40 lines of ${2:-$LOG} ---"; tail -40 "${2:-$LOG}"; } > $STEPF
  echo "STEP FAILED $1"; up $W/status status_smoke; exit 1
}
python -m pip install -q huggingface_hub hf_transfer 2>&1 | tail -1
echo "ALIVE $(date -u) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)" > $STEPF; up $W/status status_smoke
heartbeat() { while true; do sleep 240; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > $W/status/gpu.txt; date -u >> $W/status/gpu.txt
  [ -f $W/status/train.log ] && tail -c 20000 $W/status/train.log > $W/status/train_tail.log; up $W/status status_smoke; done; }
heartbeat & HB=$!
trap 'kill $HB 2>/dev/null; echo "=== remote_smoke_small exit $(date -u) ==="; up $W/status status_smoke' EXIT

echo "=== data: switchboard/phase1/phase1_sft.jsonl from $WORK_REPO ==="
python - <<'EOF'
import os, json
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["WORK_REPO"], "switchboard/phase1/phase1_sft.jsonl", repo_type="dataset",
                    local_dir="/workspace/data_sw", token=os.environ["HF_TOKEN"])
rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
print("rows:", len(rows))
# deterministic held-out: last 8 rows = eval, rest = train (rows are already batch-shuffled upstream)
tr, ev = rows[:-8], rows[-8:]
for name, part in (("train_smoke", tr), ("dev_smoke", ev)):
    with open(f"/workspace/data_sw/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(part))
# how the raw rows encode tool call arguments (axolotl json.loads()es strings; the bare chat template does not)
argt = {}
for r in rows:
    for m in r["messages"]:
        for tc in (m.get("tool_calls") or []): argt[type(tc["function"]["arguments"]).__name__] = argt.get(type(tc["function"]["arguments"]).__name__, 0) + 1
print("tool_call argument types:", argt)
EOF
[ $? -ne 0 ] && die data_download $LOG
ls -la $W/data_sw

echo "=== model download: $MODEL ==="
python - <<EOF
import os, time
from huggingface_hub import snapshot_download
t=time.time(); snapshot_download("$MODEL", local_dir="/workspace/base_model", token=os.environ["HF_TOKEN"])
print("model downloaded in", round(time.time()-t), "s")
EOF
[ $? -ne 0 ] && die model_download $LOG
du -sh $W/base_model; echo "STEP model done $(date -u)" >> $STEPF; up $W/status status_smoke

# ---- config: derived from bundle/axolotl_lora_q38_27b.yaml (the real 27B run). Deltas, all forced by the small box:
#      base_model -> small dense model (no Gated-DeltaNet -> linear_attn.* LoRA targets dropped),
#      sequence_len 16384->4096, lora_r 64->16 / alpha 128->32, grad_accum 8->4, num_epochs 2 -> max_steps 30,
#      evals_per_epoch/saves_per_epoch -> eval_steps 10 / save_steps 30, datasets -> the switchboard phase1 split above.
#      Everything that decides whether the DATA is trainable is unchanged: chat_template tokenizer_default with
#      enable_thinking false, type chat_template, field_messages/field_tools, roles_to_train assistant, train_on_eos turn,
#      message_property_mappings (tool_calls + tool_call_id), sample_packing false, bf16 LoRA, CCE plugin.
cat > $CFG <<EOF
base_model: /workspace/base_model

plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
strict: false

chat_template: tokenizer_default
chat_template_kwargs:
  enable_thinking: false

datasets:
  - path: /workspace/data_sw/train_smoke.jsonl
    type: chat_template
    field_messages: messages
    field_tools: tools
    roles_to_train: ["assistant"]
    train_on_eos: turn
    message_property_mappings:
      role: role
      content: content
      tool_calls: tool_calls
      tool_call_id: tool_call_id
test_datasets:
  - path: /workspace/data_sw/dev_smoke.jsonl
    type: chat_template
    split: train
    field_messages: messages
    field_tools: tools
    roles_to_train: ["assistant"]
    train_on_eos: turn
    message_property_mappings:
      role: role
      content: content
      tool_calls: tool_calls
      tool_call_id: tool_call_id
val_set_size: 0

output_dir: $OUT
dataset_prepared_path: /workspace/last_run_prepared

sequence_len: $SEQ
sample_packing: false
pad_to_sequence_len: false

adapter: lora
load_in_4bit: false
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

gradient_accumulation_steps: 4
micro_batch_size: 1
num_epochs: 1
max_steps: $MAX_STEPS
optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 0.0001
warmup_ratio: 0.03
weight_decay: 0.0

bf16: auto
tf32: false          # transformers is_torch_tf32_available() rejected tf32 on the RTX 4090 box; the 27B config keeps tf32: true
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
attn_implementation: flash_attention_2

logging_steps: 1
eval_strategy: steps
eval_steps: 10
save_strategy: steps
save_steps: $MAX_STEPS
save_total_limit: 1
special_tokens:
EOF
python -c "import cut_cross_entropy" 2>/dev/null || { echo "no cut_cross_entropy -> dropping plugin"; sed -i '/CutCrossEntropyPlugin/d; /^plugins:$/d' $CFG; }
python -c "import flash_attn" 2>/dev/null || { echo "no flash_attn -> sdpa"; sed -i "s#attn_implementation: flash_attention_2#attn_implementation: sdpa#" $CFG; }
cp $CFG $W/status/axolotl_lora_smoke.yaml
python -c "import axolotl, transformers, torch; print('axolotl', axolotl.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)" 2>&1 | tail -2

echo "=== preprocess (RAW rows: tool_calls arguments are JSON strings) ==="
timeout 1800 axolotl preprocess $CFG > $W/status/preprocess.log 2>&1; RC=$?
tr '\r' '\n' < $W/status/preprocess.log | grep -aE "Success!|Error|Traceback" | tail -8
echo "preprocess rc=$RC"
FIXED=0
if [ "$RC" -ne 0 ]; then
  echo "=== preprocess FAILED on raw rows -> retry with arguments JSON-string -> dict ==="
  cp $W/status/preprocess.log $W/status/preprocess_raw_failed.log
  python - <<'EOF'
import json
for name in ("train_smoke", "dev_smoke"):
    p = f"/workspace/data_sw/{name}.jsonl"; n = 0
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    for r in rows:
        for m in r["messages"]:
            for tc in (m.get("tool_calls") or []):
                a = tc["function"]["arguments"]
                if isinstance(a, str): tc["function"]["arguments"] = json.loads(a); n += 1
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, "converted", n)
EOF
  rm -rf $W/last_run_prepared
  timeout 1800 axolotl preprocess $CFG > $W/status/preprocess.log 2>&1; RC=$?; FIXED=1
  echo "preprocess (fixed) rc=$RC"
fi
[ "$RC" -ne 0 ] && die preprocess $W/status/preprocess.log
echo "STEP preprocess done $(date -u) raw_accepted=$([ $FIXED -eq 0 ] && echo yes || echo no)" >> $STEPF
# sample counts / token lengths straight out of the prepared arrow datasets
python - <<'EOF' 2>&1 | tee $W/status/prepared_stats.txt
import glob, json, os
from datasets import load_from_disk
out = {}
for d in sorted(glob.glob("/workspace/last_run_prepared/*")):
    if not os.path.isdir(d): continue
    try: ds = load_from_disk(d)
    except Exception as e: print("skip", d, e); continue
    L = [len(x) for x in ds["input_ids"]]
    T = [sum(1 for t in x if t != -100) for x in ds["labels"]]
    out[os.path.basename(d)] = {"n": len(ds), "tok_min": min(L), "tok_max": max(L), "tok_mean": round(sum(L)/len(L), 1),
                                "trainable_tok_total": sum(T), "trainable_pct": round(100*sum(T)/sum(L), 2),
                                "rows_all_masked": sum(1 for t in T if t == 0)}
print(json.dumps(out, indent=2)); json.dump(out, open("/workspace/status/prepared_stats.json", "w"), indent=2)
EOF
up $W/status status_smoke

echo "=== train ==="
timeout 2400 axolotl train $CFG > $W/status/train.log 2>&1; RC=$?
tr '\r' '\n' < $W/status/train.log | grep -aE "'"'"'loss'"'"'|eval_loss|Error|Traceback|Saving|train_runtime" | tail -60
echo "train rc=$RC"
[ "$RC" -ne 0 ] && die train $W/status/train.log
echo "STEP train done $(date -u) rc=$RC" >> $STEPF
ls -la $OUT | head -30
# trainer_state.json (final dir or last checkpoint) + parsed metrics
python - <<'EOF' 2>&1 | tee -a $W/status/prepared_stats.txt
import glob, json, os, shutil
out="/workspace/outputs/smoke_small"
cands=[os.path.join(out,"trainer_state.json")]+sorted(glob.glob(os.path.join(out,"checkpoint-*/trainer_state.json")))
src=[c for c in cands if os.path.exists(c)]
if not src: print("NO trainer_state.json found"); raise SystemExit
src=src[-1]; shutil.copy(src,"/workspace/status/trainer_state.json"); print("trainer_state from", src)
st=json.load(open(src)); h=st.get("log_history",[])
m={"train_loss_by_step":[(e["step"],e["loss"]) for e in h if "loss" in e],
   "eval_loss_by_step":[(e["step"],e["eval_loss"]) for e in h if "eval_loss" in e],
   "final":{k:v for e in h for k,v in e.items() if k in ("train_runtime","train_samples_per_second","train_steps_per_second","total_flos","train_loss")}}
try:
    ps=json.load(open("/workspace/status/prepared_stats.json"))
    tr=[v for k,v in ps.items() if "train" in k or "dev" not in k]
    if tr: m["train_tokens_per_sample_mean"]=tr[0]["tok_mean"]
except Exception as e: print("prepared_stats:", e)
rt=m["final"].get("train_runtime"); sps=m["final"].get("train_samples_per_second")
if rt and m.get("train_tokens_per_sample_mean") and sps: m["tokens_per_second"]=round(sps*m["train_tokens_per_sample_mean"],1)
json.dump(m, open("/workspace/status/smoke_metrics.json","w"), indent=2); print(json.dumps(m, indent=2)[:3000])
EOF
up $OUT switchboard/phase1_smoke_adapter
echo "JOB_COMPLETE $(date -u)" >> $STEPF
up $W/status status_smoke
