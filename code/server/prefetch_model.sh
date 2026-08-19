#!/bin/bash
set -euo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/grid
export HF_HOME=$HOME/hf_home
mkdir -p "$HF_HOME"
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "google/flan-t5-xl",
    allow_patterns=["config.json", "generation_config.json", "*.safetensors",
                    "*.safetensors.index.json", "spiece.model", "tokenizer.json",
                    "tokenizer_config.json", "special_tokens_map.json"],
    max_workers=8,
)
print("MODEL_AT", p)
PY
echo "PREFETCH_DONE"
