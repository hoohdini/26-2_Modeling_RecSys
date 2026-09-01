#!/bin/bash
# v2 덤프 채점. 여러 번 돌려도 안전(이미 있는 결과는 건너뜀).
cd /data1/dsl05/recsys
PY=/data1/dsl05/miniconda3/envs/eval/bin/python
[ -x "$PY" ] || PY=/data1/dsl05/miniconda3/envs/grid/bin/python
S(){ echo sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt; }
mkdir -p results_jobs_v2
score(){  # $1=TAG  $2=SIDNAME
  D=tiger_out/$1/eval_dump_k200/test_predictions_rank0.pkl
  O=results_jobs_v2/$1.json
  [ -f "$O" ] && { echo "  skip $1"; return; }
  [ -f "$D" ] || { echo "  대기 $1 (덤프 없음)"; return; }
  echo "  채점 $1"
  $PY tiger_to_eval.py --pred "$D" --sid "$(S $2)" --split jobs/Jobs_split_A.pkl --label "$1" --out "$O" 2>&1 | tail -3
}
score jobs_text_v2_s42      jobs_text
score jobs_gsid_b10_v2_s42  jobs_gsid_b10
score jobs_text_v2_s7       jobs_text
score jobs_gsid_b10_v2_s7   jobs_gsid_b10
score jobs_rewired_v2_s42   jobs_gsid_rewired
score jobs_text_v2_s13      jobs_text
score jobs_gsid_b10_v2_s13  jobs_gsid_b10
echo "--- 완료된 결과 ---"; ls -1 results_jobs_v2/ 2>/dev/null
