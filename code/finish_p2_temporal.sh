#!/bin/bash
#SBATCH --job-name=finish_p2
#SBATCH --partition=partition1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/finish_p2_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/finish_p2_%j.err
#
# P2 덤프가 전부 끝나면(실패 포함) Slurm 이 자동 실행. GPU 를 쓰지 않는다.
# 채점 → Temporal W4·W5 규칙 C 판정(verdict_p2_temporal.py) → P2_보고.txt
set -u
B=/mnt/data1/dsl05; [ -d "$B" ] || B=/data1/dsl05
cd $B/recsys || exit 1
PY=$B/miniconda3/envs/eval/bin/python
[ -x "$PY" ] || PY=$B/miniconda3/envs/grid/bin/python
R=P2_보고.txt
{
  echo "=== finish_p2 시작 $(date '+%F %T') ==="
  bash ./score_p2_temporal.sh
  echo
  echo "################ Beauty Temporal W4·W5 n=3 (P2 · 규칙 C) ################"
  $PY ./verdict_p2_temporal.py
  echo
  echo "=== 셀별 소요 · 멈춘 스텝 ==="
  sacct -u dsl05 -S "$(date -d '4 days ago' +%F)" -X --format=JobName%28,Elapsed%11,State%10 | grep -E "_t2_|_d2_" || true
  for d in tiger_out/tp_* tiger_out/tg_text_W4 tiger_out/tg_b10_W4 tiger_out/tg_b10_W4_s7 tiger_out/tg_text_W5; do
    [ -d "$d" ] || continue
    printf '  %-16s 최적 ckpt %s · 마지막 스텝 %s\n' "$(basename $d)" \
      "$(ls $d/checkpoints 2>/dev/null | grep -o 'step=[0-9]*' | head -1)" \
      "$(tail -1 $d/csv/*/metrics.csv 2>/dev/null | cut -d, -f2)"
  done
  echo "=== finish_p2 완료 $(date '+%F %T') ==="
} 2>&1 | tee "$R"
echo "보고서: ~/recsys/$R"
