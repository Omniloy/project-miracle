#!/bin/bash
# Persist work from the ephemeral session: (1) code + STATE.md -> git branch, (2) data/results -> private HF repo.
# Usage: save_state.sh [--loop MINUTES]   (env: SCRATCH=<scratchpad dir>, HF_TOKEN)
set -uo pipefail
REPO=/home/user/project-miracle; DIR=$REPO/agentic-distillation
SCRATCH=${SCRATCH:-/tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad}
BRANCH=claude/multiverse-model-compression-74j82i
[ -f $SCRATCH/.env ] && set -a && . $SCRATCH/.env && set +a

save_once() {
  echo "=== save $(date -u +%FT%TZ) ==="
  # refresh code copies (never secrets)
  mkdir -p $DIR/pipeline/banking $DIR/pipeline/gpu
  cp $SCRATCH/pipeline/*.py $DIR/pipeline/ 2>/dev/null; cp $SCRATCH/pipeline/banking/*.py $DIR/pipeline/banking/ 2>/dev/null
  cp $SCRATCH/pipeline/gpu/*.py $SCRATCH/pipeline/gpu/*.yaml $SCRATCH/pipeline/gpu/*.sh $DIR/pipeline/gpu/ 2>/dev/null
  if grep -rlE 'sk-or-v1-[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{30}|[0-9a-f]{64}' $DIR >/dev/null 2>&1; then echo "!! possible secret in $DIR — not committing"; else
    cd $REPO && git add agentic-distillation && if ! git diff --cached --quiet; then
      git -c user.name="${GIT_AUTHOR_NAME:-Claude}" -c user.email="${GIT_AUTHOR_EMAIL:-noreply@anthropic.com}" commit -q -m "agentic-distillation: save state $(date -u +%F' '%H:%M) UTC

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013aZpgjdaPi3GjqVfDvLRn4" && for i in 1 2 3 4; do git push -q -u origin $BRANCH && { echo "pushed to $BRANCH"; break; } || sleep $((2**i)); done
    else echo "git: nothing new"; fi
  fi
  # data + results -> HF (idempotent uploads; skip huge/unchanged via upload_folder's hashing)
  [ -n "${HF_TOKEN:-}" ] && python3 - <<'EOF'
import os, glob
from huggingface_hub import HfApi
S=os.environ["SCRATCH"]; api=HfApi(token=os.environ["HF_TOKEN"]); repo="Enriqueag26/agentic-distill-work"
def up_folder(src, dst, pat=None):
    if os.path.isdir(src):
        api.upload_folder(folder_path=src, path_in_repo=dst, repo_id=repo, repo_type="dataset", allow_patterns=pat); print("  hf <-", dst)
def up_file(src, dst):
    if os.path.isfile(src): api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="dataset"); print("  hf <-", dst)
up_folder(f"{S}/pipeline/banking/synth", "synthetic_tasks/all_batches", ["*.json"])
up_folder(f"{S}/harness/data_synth/tau2/domains/banking_knowledge/tasks", "bundle/data_synth_tasks", ["*.json"])
for f in glob.glob(f"{S}/runs/sft_*.jsonl"): up_file(f, f"trajectories/{os.path.basename(f)}")
for d in glob.glob(f"{S}/runs/sft_v*"):
    if os.path.isdir(d): up_folder(d, f"data_{os.path.basename(d).split('_',1)[1]}")
for f in glob.glob(f"{S}/runs/*_relaxed.json") + glob.glob(f"{S}/runs/*_split.json") + [f"{S}/runs/memorization_probe.json"]: up_file(f, f"results/{os.path.basename(f)}")
for d in glob.glob(f"{S}/harness/data_synth/simulations/*") + glob.glob(f"{S}/harness/tau2-bench/data/simulations/*"):
    r=os.path.join(d,"results.json")
    if os.path.isfile(r): up_file(r, f"results/simulations/{os.path.basename(d)}/results.json")
for f in glob.glob(f"{S}/runs/*.md"): up_file(f, f"research/{os.path.basename(f)}")
up_file("/home/user/project-miracle/agentic-distillation/STATE.md", "STATE.md")
EOF
  echo "=== save done ==="
}

if [ "${1:-}" = "--loop" ]; then while true; do save_once; sleep $(( ${2:-20} * 60 )); done; else save_once; fi
