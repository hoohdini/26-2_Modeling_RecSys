#!/bin/bash
#SBATCH --job-name=finish_p4p5
#SBATCH --partition=partition1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/finish_p4p5_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/finish_p4p5_%j.err
#
# P4·P5 덤프가 전부 끝나면(실패 포함) Slurm 이 자동 실행. GPU 를 쓰지 않는다.
# 채점 → 규칙 D 판정 + 접두사 제약 표(verdict_p4_p5.py) → P4P5_보고.txt
set -u
B=/mnt/data1/dsl05; [ -d "$B" ] || B=/data1/dsl05
cd $B/recsys || exit 1
PY=$B/miniconda3/envs/eval/bin/python
[ -x "$PY" ] || PY=$B/miniconda3/envs/grid/bin/python
R=P4P5_보고.txt
{
  echo "=== finish_p4p5 시작 $(date '+%F %T') ==="
  bash ./score_p4_p5.sh
  echo
  echo "################ P4 예산 민감도 (규칙 D) · P5 접두사 제약 ################"
  $PY ./verdict_p4_p5.py
  echo
  echo "=== 셀별 소요 · 멈춘 스텝 ==="
  sacct -u dsl05 -S "$(date -d '4 days ago' +%F)" -X --format=JobName%30,Elapsed%11,State%10 | grep -E "_t4b_|_d4b_|_d5_" || true
  for d in tiger_out/p4_*; do
    [ -d "$d" ] || continue
    printf '  %-18s 최적 ckpt %s · 마지막 스텝 %s\n' "$(basename $d)" \
      "$(ls $d/checkpoints 2>/dev/null | grep -o 'step=[0-9]*' | head -1)" \
      "$(tail -1 $d/csv/*/metrics.csv 2>/dev/null | cut -d, -f2)"
  done
  echo "=== finish_p4p5 완료 $(date '+%F %T') ==="
} 2>&1 | tee "$R"
echo "보고서: ~/recsys/$R"
