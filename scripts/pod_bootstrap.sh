#!/usr/bin/env bash
# Runs on the RunPod pod. Self-observing: pushes a heartbeat marker to branch `r0-run`
# at each stage (so we have observability without SSH), runs R0, ALWAYS pushes
# results/log on exit, then self-terminates the pod.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"
BRANCH="r0-run"
REMOTE="https://x-access-token:${GH_TOKEN}@github.com/ratnaditya-j/trust-geometry"
mkdir -p results
git config user.email "pod@trust-geometry"; git config user.name "tg-pod"
git checkout -B "$BRANCH" >/dev/null 2>&1 || true

heartbeat() {  # $1 = stage label
  echo "$(date -u +%H:%M:%S) STAGE=$1" | tee -a results/heartbeat.txt
  git add -A results/ >/dev/null 2>&1 || true
  git commit -q -m "hb: $1" >/dev/null 2>&1 || true
  git push -f "$REMOTE" "$BRANCH" >/dev/null 2>&1 || true
}

self_terminate() {
  if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    curl -s -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" >/dev/null 2>&1 || true
  fi
}
finish() { heartbeat "EXIT"; self_terminate; }
trap finish EXIT

heartbeat "STARTED"
echo "[bootstrap] $(date) on $(hostname)"; nvidia-smi | tee results/nvidia_smi.txt || true
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 | tee -a results/heartbeat.txt || true
heartbeat "GPU_OK"

echo "[bootstrap] installing deps..."
pip install -q --upgrade pip >/dev/null 2>&1
pip install -q "transformers>=4.56.0" "accelerate>=1.0.0" "huggingface_hub>=0.25" \
    scikit-learn scipy numpy safetensors hf_transfer "kernels>=0.4.0" triton 2>&1 | tail -6
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -c "import transformers,sklearn;print('transformers',transformers.__version__,'sklearn',sklearn.__version__)" 2>&1 | tee -a results/heartbeat.txt
heartbeat "DEPS_DONE"

echo "[bootstrap] running R0 driver..."
python3 scripts/run_r0.py 2>&1 | tee -a results/boot.log
echo "[bootstrap] R0 exit ${PIPESTATUS[0]}"
heartbeat "R0_DONE"
