#!/bin/bash
# v4 준비 + 제출을 마스터 노드에서 한 번에 (nohup 으로 실행, PC 와 무관하게 끝까지 간다)
#   nohup bash ~/recsys/prep_and_submit_v4.sh > ~/recsys/logs/prep_v4.log 2>&1 &
#
#   [1] G-SID 임베딩 2종 (CPU) — v3 텍스트 임베딩 재사용(md5 동일 확인). 튜닝식 β₀=1.0
#   [2] TFRecord (CPU, tensorflow-cpu)
#   [3] SID 3종 sbatch (GPU 각 4분, L4)
#   [4] 학습 9셀 + 덤프 + 마무리 sbatch — SID 3종 afterok 의존
set -u
LCD=/data1/dsl05/recsys
PY=/data1/dsl05/miniconda3/envs/grid/bin/python
ME=${ME:-minchan}
V4=$LCD/jobs_v4
CV4=/mnt/data1/dsl05/recsys/jobs_v4
DATA=/mnt/data1/dsl05/recsys/grid_data/jobs_A_v4
cd $LCD/jobs || exit 1
echo "=== v4 prep 시작 $(date '+%F %T') ==="

if [ ! -f "$V4/jobs_gsid_b10_emb.pt" ]; then
  echo "[1a] G-SID(β₀=1.0) 임베딩"
  $PY gen_gstar_beta.py --beta 1.0 --embedding_path $V4/jobs_text_emb.pt \
      --edges_path $V4/jobs_edges_final.csv --interactions_path $V4/jobs_interactions_long.csv \
      --out $V4/jobs_gsid_b10_emb.pt || { echo "FAIL gen_gstar b10"; exit 1; }
fi
if [ ! -f "$V4/jobs_rewired_b10_emb.pt" ]; then
  echo "[1b] 무작위 그래프(G1) 임베딩"
  $PY gen_gstar_beta.py --beta 1.0 --embedding_path $V4/jobs_text_emb.pt \
      --edges_path $V4/jobs_edges_rewired.csv --interactions_path $V4/jobs_interactions_long.csv \
      --out $V4/jobs_rewired_b10_emb.pt || { echo "FAIL gen_gstar rewired"; exit 1; }
fi
ls -la $V4/*.pt

if [ ! -d "$LCD/grid_data/jobs_A_v4" ]; then
  echo "[2] TFRecord"
  cd $LCD
  $PY to_grid_v2.py $V4/Jobs_split_A.pkl $LCD/grid_data/jobs_A_v4 || { echo "FAIL to_grid"; exit 1; }
fi
ls $LCD/grid_data/jobs_A_v4

echo "[3] SID 3종 제출"
cd $LCD
S1=$(sbatch --export=ALL,TAG=jobs_v4_text,EMB=$CV4/jobs_text_emb.pt,DIM=2048,LEVELS=4,DATA=$DATA --job-name=${ME}_v4_sid_text sid_jobs.sh | awk '{print $NF}')
S2=$(sbatch --export=ALL,TAG=jobs_v4_gsid_b10,EMB=$CV4/jobs_gsid_b10_emb.pt,DIM=2048,LEVELS=4,DATA=$DATA --job-name=${ME}_v4_sid_gsid sid_jobs.sh | awk '{print $NF}')
S3=$(sbatch --export=ALL,TAG=jobs_v4_rewired,EMB=$CV4/jobs_rewired_b10_emb.pt,DIM=2048,LEVELS=4,DATA=$DATA --job-name=${ME}_v4_sid_rew sid_jobs.sh | awk '{print $NF}')
echo "SID 작업: $S1 $S2 $S3"

echo "[4] 학습 9셀 제출 (SID afterok)"
ME=$ME AFTER=$S1:$S2:$S3 bash $LCD/run_jobs_v4.sh train
echo "=== v4 prep 완료 $(date '+%F %T') ==="
squeue -u dsl05 -o "%.8i %.30j %.2t %.8M %.16E" | tail -25
echo "PREP_DONE"
