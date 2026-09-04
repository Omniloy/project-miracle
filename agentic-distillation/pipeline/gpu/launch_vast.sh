#!/bin/bash
# Launch an unattended Vast job. Usage: launch_vast.sh OFFER_ID SCRIPT_NAME DISK_GB "EXTRA -e K=V ..." [IMAGE]
# The script (pipeline/gpu/SCRIPT_NAME) is uploaded to HF bundle/ first; the onstart cmd curls it and runs it detached.
# Default image = axolotlai/axolotl-cloud:main-latest, the only image on which our --onstart-cmd has actually run
# (vllm/vllm-openai:v0.28.0 and pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel both sat idle: no onstart, ~$8 lost).
set -euo pipefail; S=${SCRATCH:-$(dirname "$0")/../..}; set -a; . $S/.env; set +a
OFFER=$1; SCRIPT=$2; DISK=${3:-150}; EXTRA=${4:-}; IMAGE=${5:-axolotlai/axolotl-cloud:main-latest}
REPO=Enriqueag26/agentic-distill-work
python3 - "$S/pipeline/gpu/$SCRIPT" <<'PY'
import os,sys
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=sys.argv[1], path_in_repo="bundle/"+os.path.basename(sys.argv[1]), repo_id="Enriqueag26/agentic-distill-work", repo_type="dataset"); print("uploaded", sys.argv[1])
PY
ONSTART="mkdir -p /workspace && curl -sSL -H 'Authorization: Bearer $HF_TOKEN' https://huggingface.co/datasets/$REPO/resolve/main/bundle/$SCRIPT -o /workspace/$SCRIPT && chmod +x /workspace/$SCRIPT && nohup bash /workspace/$SCRIPT > /workspace/onstart.out 2>&1 &"
vastai create instance $OFFER --image "$IMAGE" --disk $DISK --ssh --direct \
  --env "-e HF_TOKEN=$HF_TOKEN -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY -e WORK_REPO=$REPO $EXTRA" \
  --onstart-cmd "$ONSTART" --raw | tee $S/runs/vast_create_last.json
