#!/bin/bash
#SBATCH --job-name=finish_gsid_week
#SBATCH --partition=partition1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/finish_gsid_week_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/finish_gsid_week_%j.err
#
# P1·P3 덤프가 전부 끝나면(실패 포함) Slurm 이 자동 실행. GPU 를 쓰지 않는다.
# 채점 → Beauty n=5 판정(verdict_gsid_week.py) → 구인구직 n=6 판정(verdict_v3.py) → GSID_WEEK_보고.txt
set -u
B=/mnt/data1/dsl05; [ -d "$B" ] || B=/data1/dsl05
cd $B/recsys || exit 1
PY=$B/miniconda3/envs/eval/bin/python
[ -x "$PY" ] || PY=$B/miniconda3/envs/grid/bin/python
R=GSID_WEEK_보고.txt
{
  echo "=== finish_gsid_week 시작 $(date '+%F %T') ==="
  bash ./score_gsid_week.sh
  echo
  echo "################ Beauty LOO n=5 (P1) ################"
  $PY ./verdict_gsid_week.py
  echo
  echo "################ 구인구직 n=6 (P3) ################"
  JOBS_VER=v3 SEEDS=42,7,13,3,21,99 $PY ./verdict_v3.py
  echo
  echo "=== 셀별 소요 ==="
  sacct -u dsl05 -S "$(date -d '3 days ago' +%F)" -X --format=JobName%28,Elapsed%11,State%10 | grep -E "_t_ns_|_t_jobs_|_d_" || true
  echo "=== finish_gsid_week 완료 $(date '+%F %T') ==="
} 2>&1 | tee "$R"
echo "보고서: ~/recsys/$R"
