#!/bin/bash
# v4 덤프 채점. 여러 번 돌려도 안전(이미 있는 결과는 건너뜀). score_jobs_v3.sh 와 같은 방식.
B=/mnt/data1/dsl05; [ -d "$B" ] || B=/data1/dsl05
cd $B/recsys || exit 1
PY=$B/miniconda3/envs/eval/bin/python
[ -x "$PY" ] || PY=$B/miniconda3/envs/grid/bin/python
[ -x "$PY" ] || { echo "python 없음: $PY"; exit 1; }
S(){ echo sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt; }
mkdir -p results_jobs_v4
score(){
  D=tiger_out/$1/eval_dump_k200/test_predictions_rank0.pkl
  O=results_jobs_v4/$1.json
  [ -f "$O" ] && { echo "  skip $1"; return; }
  [ -f "$D" ] || { echo "  대기 $1 (덤프 없음)"; return; }
  echo "  채점 $1"
  $PY tiger_to_eval.py --pred "$D" --sid "$(S $2)" --split jobs_v4/Jobs_split_A.pkl --label "$1" --out "$O" 2>&1 | tail -3
}
for s in 42 7 13 3 21 99; do
  score jobs_text_v4_s$s      jobs_v4_text
  score jobs_gsid_b10_v4_s$s  jobs_v4_gsid_b10
done
for s in 42 7 13; do
  score jobs_rewired_v4_s$s   jobs_v4_rewired
done
echo "--- 완료된 결과 ---"; ls -1 results_jobs_v4/ 2>/dev/null
