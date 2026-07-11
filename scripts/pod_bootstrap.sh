#!/usr/bin/env bash
# Runs on the RunPod pod. Installs deps, runs R0, and ALWAYS pushes results/log back
# to the `r0-run` branch (even on failure) so we have observability without SSH.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"
BRANCH="r0-run"

push_back() {
  cd "$REPO_DIR"
  git config user.email "pod@trust-geometry"
  git config user.name "tg-pod"
  git checkout -B "$BRANCH" >/dev/null 2>&1 || true
  cp /workspace/boot.log results/boot.log 2>/dev/null || true
  nvidia-smi > results/nvidia_smi.txt 2>&1 || true
  git add -A results/ >/dev/null 2>&1 || true
  git commit -m "R0 pod run (auto)" >/dev/null 2>&1 || true
  git push -f "https://x-access-token:${GH_TOKEN}@github.com/ratnaditya-j/trust-geometry" "$BRANCH" 2>&1 | tail -3 || true
  echo "[bootstrap] pushed results to branch $BRANCH"
}
trap push_back EXIT

echo "[bootstrap] $(date) starting on $(hostname)"
nvidia-smi || true
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || true

echo "[bootstrap] installing deps..."
pip install -q --upgrade pip
# recent transformers for gpt_oss (GptOssForCausalLM); accelerate; kernels for MXFP4 dequant
pip install -q "transformers>=4.56.0" "accelerate>=1.0.0" "huggingface_hub>=0.25" \
    scikit-learn scipy numpy safetensors hf_transfer "kernels>=0.4.0" 2>&1 | tail -5
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "[bootstrap] python/lib versions:"
python3 - <<'PY'
import transformers, sklearn, numpy, scipy
print("transformers", transformers.__version__, "| sklearn", sklearn.__version__)
PY

echo "[bootstrap] running R0 driver..."
python3 scripts/run_r0.py
echo "[bootstrap] R0 driver exit code: $?"
