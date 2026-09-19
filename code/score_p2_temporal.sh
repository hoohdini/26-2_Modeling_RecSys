#!/bin/bash
# P2(Beauty Temporal W4·W5 × text/b10 × 시드 42·7·13) 덤프 채점. 여러 번 돌려도 안전.
#   → results_p2_temporal/tp_*.json   (라벨은 새 TAG. 재사용 셀은 기존 tiger_out 덤프를 새 라벨로 채점한다)
#   Split B 는 work/eval/Beauty_split_B.pkl (저장소 Beauty_split_B.pkl 과 md5 동일: b82c6ae4…). --window 로 윈도우를 준다.
B=/mnt/data1/dsl05; [ -d "$B" ] || B=/data1/dsl05
cd $B/recsys || exit 1
PY=$B/miniconda3/envs/eval/bin/python
[ -x "$PY" ] || PY=$B/miniconda3/envs/grid/bin/python
SPLIT=work/eval/Beauty_split_B.pkl
S(){ echo sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt; }
mkdir -p results_p2_temporal
score(){  # TAG WINDOW SIDNAME OLDTAG
  O=results_p2_temporal/$1.json
  [ -f "$O" ] && { echo "  skip $1"; return; }
  D=tiger_out/$1/eval_dump_k200/test_predictions_rank0.pkl
  SRC=$1
  if [ ! -f "$D" ] && [ "$4" != "-" ]; then D=tiger_out/$4/eval_dump_k200/test_predictions_rank0.pkl; SRC=$4; fi
  [ -f "$D" ] || { echo "  대기 $1 (덤프 없음)"; return; }
  echo "  채점 $1  <- tiger_out/$SRC  ($2)"
  $PY tiger_to_eval.py --pred "$D" --sid "$(S $3)" --split "$SPLIT" --window "$2" --label "$1" --out "$O" 2>&1 | tail -2
}
# 셀 표는 run_p2_temporal.sh 와 같다
for s in 42 7 13; do
  case $s in 42) OT=tg_text_W4; OB=tg_b10_W4;;  7) OT=-; OB=tg_b10_W4_s7;;  *) OT=-; OB=-;; esac
  score tp_text_W4_s$s  W4  baseline          $OT
  score tp_b10_W4_s$s   W4  sid_noTau_k0_b10  $OB
  case $s in 42) OT=tg_text_W5;; *) OT=-;; esac
  score tp_text_W5_s$s  W5  baseline          $OT
  score tp_b10_W5_s$s   W5  sid_noTau_k0_b10  -
done
echo "--- results_p2_temporal ---"; ls -1 results_p2_temporal
