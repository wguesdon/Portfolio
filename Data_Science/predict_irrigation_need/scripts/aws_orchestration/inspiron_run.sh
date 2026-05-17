#!/bin/bash
# Launch a training run on the Inspiron home server via SSH + Podman.
#
# Usage:
#   ./inspiron_run.sh <dockerfile-suffix> <script-name>
#
# Examples:
#   ./inspiron_run.sh lgb train_lgb_ote_deep.py
#   ./inspiron_run.sh xgb train_xgb_ote_magic.py
#
# One-time setup required on the Inspiron:
#   ssh Inspiron "mkdir -p ~/kaggle/PS6E4/{data,code,predictions,logs}"
#   rsync -avz data/raw/ Inspiron:~/kaggle/PS6E4/data/
#
# See docs/inspiron_workflow.md for full setup instructions.

set -euo pipefail

HOST="Inspiron"
REMOTE_BASE="~/kaggle/PS6E4"
LOCAL_BASE="/mnt/data/Github/Kaggle/Playground_Series/PS6E4"

DOCKERFILE_SUFFIX="${1:?Usage: $0 <dockerfile-suffix> <script-name>}"
SCRIPT="${2:?Usage: $0 <dockerfile-suffix> <script-name>}"

IMAGE="kaggle-ps6e4-${DOCKERFILE_SUFFIX}"
LOG_NAME="${SCRIPT%.py}_$(date +%Y%m%d_%H%M%S).log"
SESSION="kaggle_${DOCKERFILE_SUFFIX}_$$"

echo "=================================================="
echo "Inspiron training launcher"
echo "=================================================="
echo "Host:        $HOST"
echo "Dockerfile:  Dockerfile.${DOCKERFILE_SUFFIX}"
echo "Script:      $SCRIPT"
echo "Image:       $IMAGE"
echo "Session:     $SESSION"
echo "Log:         ${REMOTE_BASE}/logs/${LOG_NAME}"
echo ""

echo "=== Syncing code ==="
rsync -avz --delete \
    "${LOCAL_BASE}/aws/" "${HOST}:${REMOTE_BASE}/code/aws/"
rsync -avz --delete \
    "${LOCAL_BASE}/scripts/" "${HOST}:${REMOTE_BASE}/code/scripts/"

echo ""
echo "=== Building image (cached layers reused) ==="
ssh "$HOST" "cd ${REMOTE_BASE}/code/aws && \
    podman build -f Dockerfile.${DOCKERFILE_SUFFIX} -t ${IMAGE} ."

echo ""
echo "=== Launching training in detached tmux session ==="
ssh "$HOST" "tmux new -d -s ${SESSION} 'cd ${REMOTE_BASE} && \
    podman run --rm \
        -v \$(pwd)/data:/opt/ml/input/data/training:ro \
        -v \$(pwd)/predictions:/opt/ml/model:rw \
        -e SM_CHANNEL_TRAINING=/opt/ml/input/data/training \
        -e SM_MODEL_DIR=/opt/ml/model \
        ${IMAGE} \
        python ${SCRIPT} 2>&1 | tee logs/${LOG_NAME}; \
    echo; echo \"=== Done. Press any key to close tmux. ===\"; read'"

echo ""
echo "=================================================="
echo "Training running in tmux session: ${SESSION}"
echo ""
echo "Monitor log:"
echo "  ssh ${HOST} 'tail -f ${REMOTE_BASE}/logs/${LOG_NAME}'"
echo ""
echo "Attach interactive:"
echo "  ssh -t ${HOST} 'tmux attach -t ${SESSION}'"
echo ""
echo "List running sessions:"
echo "  ssh ${HOST} 'tmux ls'"
echo ""
echo "Kill session if needed:"
echo "  ssh ${HOST} 'tmux kill-session -t ${SESSION}'"
echo ""
echo "When done, pull results:"
echo "  rsync -avz ${HOST}:${REMOTE_BASE}/predictions/ ${LOCAL_BASE}/predictions/"
echo "=================================================="
