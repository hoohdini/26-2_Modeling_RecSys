#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# MaskGR 파이프라인 러너 — SID 하나를 받아 세 트랙 지표까지 한 번에.
#
#   서버에서:  ./run_maskgr.sh <SID이름> [트랙]
#
#   <SID이름>  sid_out/<이름>/L4/sid_tensor.pt 를 가리키는 이름
#              text | gsid_a01 | gsid_a03 | crab | crab_gsid ...
#   [트랙]     loo | temporal | all   (기본 loo)
#
# 예)
#   ./run_maskgr.sh text                 # ④번 칸 — 베이스라인
#   ./run_maskgr.sh gsid_a01             # ⑥번 칸
#   ./run_maskgr.sh crab temporal        # ⑤번 칸을 Temporal 트랙으로
#
# 이 스크립트는 sbatch 를 "제출"만 하고 기다리지 않는다. 의존성이 있는 단계
# (학습 -> 덤프)는 --dependency 로 묶는다. GPU 동시 2장 제한은 슬럼이 알아서 큐잉한다.
#
# 다양성 트랙은 별도 실행이 아니다 — LOO/Temporal 덤프에 같은 지표를 얹는 것이라
# 아래 3단계에서 자동으로 같이 나온다. (docs/HANDOFF_트랙C_다양성평가.md)
# ============================================================================
set -euo pipefail

BASE=${BASE:-/mnt/data1/dsl05}
RECSYS=$BASE/recsys
SIDNAME=${1:?SID 이름이 필요합니다 (예: text, gsid_a01, crab)}
TRACK=${2:-loo}

SID=${SID:-$RECSYS/sid_out/$SIDNAME/L4/sid_tensor.pt}
NH=${NH:-5}
CAND=${CAND:-200}
SEED=${SEED:-42}
WINDOWS=${WINDOWS:-W5}          # Temporal 트랙에서 돌릴 윈도우 (공백 구분: "W3 W4 W5")

test -f "$SID" || { echo "SID 없음: $SID"; exit 1; }

submit_one () {   # $1=태그  $2=데이터경로
  local TAG=$1 DATA=$2
  echo "--- $TAG  (data=$DATA)"

  local J1
  J1=$(sbatch --parsable --export=ALL,TAG=$TAG,SID=$SID,NH=$NH,DATA=$DATA,SEED=$SEED \
        $RECSYS/maskgr_train.sh)
  echo "    학습      job $J1"

  # 검증용 얕은 덤프 — 학습 로그와 대조해 덤프 경로가 옳은지 먼저 확인한다
  local J2
  J2=$(sbatch --parsable --dependency=afterok:$J1 \
        --export=ALL,TAG=$TAG,SID=$SID,NH=$NH,DATA=$DATA,CAND=10 \
        $RECSYS/maskgr_eval_dump.sh)
  echo "    덤프 c10  job $J2   (학습 로그 val/recall@5 와 대조용)"

  # 본 덤프
  local J3
  J3=$(sbatch --parsable --dependency=afterok:$J2 \
        --export=ALL,TAG=$TAG,SID=$SID,NH=$NH,DATA=$DATA,CAND=$CAND \
        $RECSYS/maskgr_eval_dump.sh)
  echo "    덤프 c$CAND job $J3"
}

echo "=== MaskGR 파이프라인 · SID=$SIDNAME · 트랙=$TRACK ==="
echo "    $SID"

case "$TRACK" in
  loo|all)
    submit_one "mg_${SIDNAME}_A" "$RECSYS/grid_data/beauty_A"
    ;;&
  temporal|all)
    for W in $WINDOWS; do
      D=$RECSYS/grid_data/beauty_B/$W
      if [ ! -d "$D" ]; then
        echo "    [건너뜀] $W — 데이터 없음: $D"
        echo "             먼저: python3 code/to_grid_b.py Beauty_split_B.pkl grid_data/beauty_B -w $W"
        continue
      fi
      submit_one "mg_${SIDNAME}_B_${W}" "$D"
    done
    ;;
esac

cat <<EOF

=== 덤프가 끝나면 로컬에서 (세 트랙 지표) ===

# LOO 트랙 + 다양성 트랙 (같은 덤프에서 함께 나옵니다)
python code/tiger_to_eval.py \\
  --pred tiger_runs/mg_${SIDNAME}_A/eval_dump_c${CAND}_t0.01_s${NH}/test_predictions_rank0.pkl \\
  --sid sid/${SIDNAME}/L4/sid_tensor.pt --label mg_${SIDNAME}

# Temporal 트랙 (윈도우별)
python code/tiger_to_eval.py \\
  --pred tiger_runs/mg_${SIDNAME}_B_W5/eval_dump_c${CAND}_t0.01_s${NH}/test_predictions_rank0.pkl \\
  --sid sid/${SIDNAME}/L4/sid_tensor.pt --split Beauty_split_B.pkl --window W5 \\
  --label mg_${SIDNAME}

# 한 표로 (TIGER 열과 나란히)
python code/compare_table.py results/baseline_splitA_loo.json results/tiger_clip_L4_loo.json \\
                             results/tiger_mg_${SIDNAME}_loo.json --k 10

# 확산 다이얼 곡선 — TEMP 를 훑어 덤프를 여러 개 만든 뒤
for T in 0.01 0.1 0.3 0.5 1.0 2.0; do
  sbatch --export=ALL,TAG=mg_${SIDNAME}_A,SID=\$SID,NH=$NH,CAND=$CAND,TEMP=\$T maskgr_eval_dump.sh
done
EOF
