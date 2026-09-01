#!/bin/bash
#SBATCH --job-name=finish_v3
#SBATCH --partition=partition1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/finish_v3_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/finish_v3_%j.err
#
# v3 덤프 6건이 끝나면 Slurm 이 이걸 자동 실행한다. GPU 를 쓰지 않는다.
# 채점 -> 3시드 판정표 -> ~/recsys/V3_아침보고.txt
# 로컬 PC 가 꺼지거나 재시작해도 무관하다.
set -u
B=/mnt/data1/dsl05; [ -d "$B" ] || B=/data1/dsl05
cd $B/recsys || exit 1
echo "=== finish_v3 시작 $(date '+%F %T') ==="
bash ./score_jobs_v3.sh
echo
PY=$B/miniconda3/envs/eval/bin/python
[ -x "$PY" ] || PY=$B/miniconda3/envs/grid/bin/python
$PY ./verdict_v3.py
echo
echo "=== finish_v3 완료 $(date '+%F %T') ==="
echo "보고서: ~/recsys/V3_아침보고.txt"
