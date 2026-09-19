#!/bin/bash
# P4(예산 민감도 8셀) + P5(접두사 제약 덤프 6개) 채점. 여러 번 돌려도 안전.
#   P4 → results_p4/p4_{base,b10}_{15k,45k}_s{42,7}.json
#   P5 → results_p5/nsP_{base,b10}_s{42,7,13}.json   (제약 OFF 원본은 results_gsid_week/ns_*.json 그대로)
B=/mnt/data1/dsl05; [ -d "$B" ] || B=/data1/dsl05
cd $B/recsys || exit 1
PY=$B/miniconda3/envs/eval/bin/python
[ -x "$PY" ] || PY=$B/miniconda3/envs/grid/bin/python
S(){ echo sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt; }
mkdir -p results_p4 results_p5
score(){  # LABEL SIDNAME DUMP OUTDIR
  O=$4/$1.json
  [ -f "$O" ] && { echo "  skip $1"; return; }
  [ -f "$3" ] || { echo "  대기 $1 (덤프 없음: $3)"; return; }
  echo "  채점 $1  <- $3"
  $PY tiger_to_eval.py --pred "$3" --sid "$(S $2)" --split Beauty_split_A.pkl --label "$1" --out "$O" 2>&1 | tail -2
}
echo "[P4]"
for st in 15k 45k; do for s in 42 7; do
  score p4_base_${st}_s$s  baseline          tiger_out/p4_base_${st}_s$s/eval_dump_k200/test_predictions_rank0.pkl  results_p4
  score p4_b10_${st}_s$s   sid_noTau_k0_b10  tiger_out/p4_b10_${st}_s$s/eval_dump_k200/test_predictions_rank0.pkl   results_p4
done; done
echo "[P5]"
score nsP_base_s42  baseline          tiger_out/clip_L4/eval_dump_k200_prefix/test_predictions_rank0.pkl                    results_p5
score nsP_base_s7   baseline          tiger_out/clip_L4_seed7/eval_dump_k200_prefix/test_predictions_rank0.pkl              results_p5
score nsP_base_s13  baseline          tiger_out/clip_L4_seed13/eval_dump_k200_prefix/test_predictions_rank0.pkl             results_p5
score nsP_b10_s42   sid_noTau_k0_b10  tiger_out/tiger_noTau_k0_b10/eval_dump_k200_prefix/test_predictions_rank0.pkl         results_p5
score nsP_b10_s7    sid_noTau_k0_b10  tiger_out/tiger_noTau_k0_b10_seed7/eval_dump_k200_prefix/test_predictions_rank0.pkl   results_p5
score nsP_b10_s13   sid_noTau_k0_b10  tiger_out/tiger_noTau_k0_b10_seed13/eval_dump_k200_prefix/test_predictions_rank0.pkl  results_p5
echo "--- results_p4 ---"; ls -1 results_p4; echo "--- results_p5 ---"; ls -1 results_p5
