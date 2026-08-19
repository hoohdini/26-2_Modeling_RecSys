#!/bin/bash
set -euo pipefail
BASE=$HOME/recsys
MC=$HOME/miniconda3
mkdir -p "$BASE/logs" "$HOME/downloads"

if [ ! -x "$MC/bin/conda" ]; then
  echo "[1/4] downloading miniconda"
  cd "$HOME/downloads"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "$MC"
else
  echo "[1/4] miniconda already present"
fi

source "$MC/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if [ ! -d "$MC/envs/grid" ]; then
  echo "[2/4] creating env grid (python 3.10)"
  conda create -y --prefix "$MC/envs/grid" python=3.10
else
  echo "[2/4] env grid already exists"
fi
conda activate "$MC/envs/grid"

echo "[3/4] installing torch (cu124)"
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu124

echo "[4/4] installing GRID deps"
pip install -q lightning==2.5.0 transformers==4.47.0 sentencepiece tokenizers \
  hydra-core==1.3.2 hydra-colorlog omegaconf rootutils python-dotenv rich \
  pandas pyarrow tensorflow-cpu==2.18.0 google-cloud-bigquery psutil "protobuf<5"

python - <<'PY'
import torch, transformers, lightning, tensorflow as tf
print("torch", torch.__version__, "cuda_build", torch.version.cuda)
print("transformers", transformers.__version__, "lightning", lightning.__version__, "tf", tf.__version__)
PY
echo "SETUP_DONE"
