#!/bin/bash
#SBATCH --job-name=finish_v4
#SBATCH --partition=partition1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/finish_v4_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/finish_v4_%j.err
# v4 덤프가 끝나면 자동 실행. 채점 → 판정표(있는 시드만) → V4_보고.txt
set -u
B=/mnt/data1/dsl05; [ -d "$B" ] || B=/data1/dsl05
cd $B/recsys || exit 1
PY=$B/miniconda3/envs/eval/bin/python
[ -x "$PY" ] || PY=$B/miniconda3/envs/grid/bin/python
{
  echo "=== finish_v4 시작 $(date '+%F %T') ==="
  bash ./score_jobs_v4.sh
  echo; echo "################ 구인구직 v4 · 시드 42,7,13 (+3,21,99 있으면) ################"
  JOBS_VER=v4 SEEDS=42,7,13,3,21,99 $PY ./verdict_v3.py
  echo; echo "=== 셀별 소요 ==="
  sacct -u dsl05 -S "$(date -d '4 days ago' +%F)" -X --format=JobName%30,Elapsed%11,State%10 | grep -E "_t4_|_d4_|v4_sid" || true
  echo "=== finish_v4 완료 $(date '+%F %T') ==="
} 2>&1 | tee V4_보고.txt
