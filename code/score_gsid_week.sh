#!/bin/bash
# P1(Beauty 시드 3·21) + P3(구인구직 시드 3·21·99) 덤프 채점. 여러 번 돌려도 안전.
#   Beauty → results_gsid_week/ns_*.json   (기존 42·7·13 결과 9개도 같은 폴더에 복사해 둔다)
#   Jobs   → results_jobs_v3/jobs_*_v3_s{3,21,99}.json  (v3 결과와 한 폴더)
B=/mnt/data1/dsl05; [ -d "$B" ] || B=/data1/dsl05
cd $B/recsys || exit 1
PY=$B/miniconda3/envs/eval/bin/python
[ -x "$PY" ] || PY=$B/miniconda3/envs/grid/bin/python
S(){ echo sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt; }
mkdir -p results_gsid_week results_jobs_v3
score(){  # TAG SIDNAME SPLIT OUTDIR
  D=tiger_out/$1/eval_dump_k200/test_predictions_rank0.pkl
  O=$4/$1.json
  [ -f "$O" ] && { echo "  skip $1"; return; }
  [ -f "$D" ] || { echo "  대기 $1 (덤프 없음)"; return; }
  echo "  채점 $1"
  $PY tiger_to_eval.py --pred "$D" --sid "$(S $2)" --split "$3" --label "$1" --out "$O" 2>&1 | tail -2
}
for s in 3 21; do
  score ns_base_s$s  baseline                Beauty_split_A.pkl results_gsid_week
  score ns_b10_s$s   sid_noTau_k0_b10        Beauty_split_A.pkl results_gsid_week
  score ns_a07_s$s   sid_blend_centered_a07  Beauty_split_A.pkl results_gsid_week
done
for s in 3 21 99; do
  score jobs_text_v3_s$s      jobs_text      jobs/Jobs_split_A.pkl results_jobs_v3
  score jobs_gsid_b10_v3_s$s  jobs_gsid_b10  jobs/Jobs_split_A.pkl results_jobs_v3
done
echo "--- results_gsid_week ---"; ls -1 results_gsid_week
echo "--- results_jobs_v3 ---";   ls -1 results_jobs_v3
