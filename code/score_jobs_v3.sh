#!/bin/bash
# v3 덤프 채점. 여러 번 돌려도 안전(이미 있는 결과는 건너뜀).
# 계산노드는 /mnt/data1, 로그인노드는 /data1 로 같은 저장소를 본다.
B=/mnt/data1/dsl05; [ -d "$B" ] || B=/data1/dsl05
cd $B/recsys || exit 1
PY=$B/miniconda3/envs/eval/bin/python
[ -x "$PY" ] || PY=$B/miniconda3/envs/grid/bin/python
[ -x "$PY" ] || { echo "python 없음: $PY"; exit 1; }
S(){ echo sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt; }
mkdir -p results_jobs_v3
score(){
  D=tiger_out/$1/eval_dump_k200/test_predictions_rank0.pkl
  O=results_jobs_v3/$1.json
  [ -f "$O" ] && { echo "  skip $1"; return; }
  [ -f "$D" ] || { echo "  대기 $1 (덤프 없음)"; return; }
  echo "  채점 $1"
  $PY tiger_to_eval.py --pred "$D" --sid "$(S $2)" --split jobs/Jobs_split_A.pkl --label "$1" --out "$O" 2>&1 | tail -3
}
score jobs_text_v3_s42      jobs_text
score jobs_gsid_b10_v3_s42  jobs_gsid_b10
score jobs_text_v3_s7       jobs_text
score jobs_gsid_b10_v3_s7   jobs_gsid_b10
score jobs_text_v3_s13      jobs_text
score jobs_gsid_b10_v3_s13  jobs_gsid_b10
score jobs_rewired_v3_s42   jobs_gsid_rewired
score jobs_rewired_v3_s7    jobs_gsid_rewired
score jobs_rewired_v3_s13   jobs_gsid_rewired
echo "--- 완료된 결과 ---"; ls -1 results_jobs_v3/ 2>/dev/null
