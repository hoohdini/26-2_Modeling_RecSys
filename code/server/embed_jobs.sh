#!/bin/bash
#SBATCH --job-name=embed_jobs
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/embed_jobs_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/embed_jobs_%j.err

# ============================================================================
# 구인구직 프로필 텍스트 → flan-t5-xl 인코더 임베딩 (N x 2048) · 실측 약 4분
#
#   sbatch --export=ALL,TEXTS=/mnt/data1/dsl05/recsys/jobs_v4/profile_text.tsv,\
#                   OUT=/mnt/data1/dsl05/recsys/jobs_v4/jobs_text_emb.pt \
#          --job-name=embed_jobs_v4_<이름> embed_jobs.sh
#
#   TEXTS  id<TAB>text (item_id 0..N-1 조밀)     기본 jobs/profile_text.tsv (v3)
#   OUT    저장할 .pt                              기본 jobs/jobs_text_emb.pt
#
# fp32 · max_length 128 · attention-mask 평균 풀링 = Beauty 임베딩과 같은 규약 (data_gen/embed_jobs.py)
# ============================================================================
set -euo pipefail
BASE=/mnt/data1/dsl05
PY=$BASE/miniconda3/envs/grid/bin/python
export HF_HOME=$BASE/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TEXTS=${TEXTS:-$BASE/recsys/jobs/profile_text.tsv}
OUT=${OUT:-$BASE/recsys/jobs/jobs_text_emb.pt}
echo "=== node: $(hostname) ==="
echo "TEXTS=$TEXTS OUT=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
test -f "$TEXTS" || { echo "텍스트 없음: $TEXTS"; exit 1; }
[ -f "$OUT" ] && { echo "이미 존재: $OUT (지우고 다시 제출)"; exit 1; }
mkdir -p "$(dirname "$OUT")"

$PY $BASE/recsys/embed_jobs.py --texts "$TEXTS" --out "$OUT" --batch 64
echo "JOB_DONE"
