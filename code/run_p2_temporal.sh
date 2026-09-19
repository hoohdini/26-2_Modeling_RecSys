#!/bin/bash
# 실험 트랙 P2 제출 — Beauty Temporal 예산 통일 (노션 "1. 그래프 SID 신뢰도 실험 계획" 3절 P2, 판정 규칙 C)
#
#   P2  Beauty Temporal  W4, W5 × 텍스트(baseline) · 튜닝식 β₀1.0(sid_noTau_k0_b10) × 시드 42, 7, 13 = 12셀
#       목적: W5 가 8k(tgfix) 레시피였던 것을 LOO 와 같은 30k 레시피로 재실행해 세 트랙(LOO·W4·W5) 3시드를 완성한다.
#
# 레시피는 **LOO 시드 셀과 동일** (같은 레시피여야 "세 트랙 재현" 주장이 성립한다. run_gsid_week.sh 와 같은 값)
#   MAX_STEPS=30000 PATIENCE=8 LIMIT_VAL=1.0 VAL_EVERY=1000 NH=5 WARMUP=1500 CLIP=1.0 SCHED=1 LR=0.001 VAL_SHUFFLE=true
#   DATA 만 grid_data/beauty_B/W4 또는 W5 (학습과 덤프에 같은 값). 덤프 배치 32, TOPK 200.
#
# 알아 둘 것 (서버 실측 2026-09-16)
#   · 같은 레시피로 이미 돌아 있는 셀 4개가 있다: tg_text_W4(s42) · tg_b10_W4(s42) · tg_b10_W4_s7 · tg_text_W5(s42).
#     hydra overrides 가 위 레시피와 글자 단위로 같아(30000 스텝·patience 8·LIMIT_VAL 1.0·시드) 그대로 쓴다.
#     P1 이 기존 LOO 3시드를 재사용한 것과 같은 규칙이다. REUSE=0 이면 4셀도 새 TAG 로 다시 돌린다.
#   · Temporal 은 내부 검증 정답이 placeholder 라(to_grid_b.py 주석) patience 8 조기종료가 14k~17k 스텝에서 걸리고
#     최적 체크포인트는 6k~9k 에 잡힌다. 이것은 레시피의 일부이므로 바꾸지 않는다. 보고서에 실제 멈춘 스텝을 같이 적는다.
#   · 큐 순서(사용자 승인): P1+P3 → v4 train → v4 seeds6 → **P2** → P4·P5·G2. seeds6 마무리 작업 뒤에 붙이려면 AFTER=<jid>.
#
#   bash run_p2_temporal.sh              # 제출 (없는 셀만)
#   bash run_p2_temporal.sh --dry        # 명령만 출력
#   ME=이름 AFTER=<jid> REUSE=1 bash run_p2_temporal.sh
#
# 덤프는 학습 afterok, 마지막에 CPU 마무리 작업(finish_p2_temporal.sh)을 모든 덤프 afterany 로 건다.
# → 채점(score_p2_temporal.sh) 과 판정표(verdict_p2_temporal.py) 가 자동으로 나온다 (~/recsys/P2_보고.txt).

set -u
DRY=${1:-}
BASE=/mnt/data1/dsl05
LBASE=/data1/dsl05
CD=$BASE/recsys
LCD=$LBASE/recsys
ME=${ME:-minchan}
AFTER=${AFTER:-}
REUSE=${REUSE:-1}

S()  { echo "$CD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }
LS() { echo "$LCD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }
sub() {
  if [ "$DRY" = "--dry" ]; then echo "DRY$RANDOM"; echo "sbatch $*" >&2; return; fi
  sbatch "$@" | awk '{print $NF}'
}

# TAG            윈도우  SID폴더            SEED  재사용할 기존 TAG(- 면 없음)
# W5 를 앞에 둔다 — 8k 레시피를 30k 로 바꾸는 것이 P2 의 핵심이라 W5 가 먼저 완성돼야 한다.
CELLS="
tp_b10_W5_s42    W5  sid_noTau_k0_b10   42  -
tp_text_W5_s7    W5  baseline            7  -
tp_b10_W5_s7     W5  sid_noTau_k0_b10    7  -
tp_text_W5_s13   W5  baseline           13  -
tp_b10_W5_s13    W5  sid_noTau_k0_b10   13  -
tp_text_W4_s7    W4  baseline            7  -
tp_text_W4_s13   W4  baseline           13  -
tp_b10_W4_s13    W4  sid_noTau_k0_b10   13  -
tp_text_W5_s42   W5  baseline           42  tg_text_W5
tp_text_W4_s42   W4  baseline           42  tg_text_W4
tp_b10_W4_s42    W4  sid_noTau_k0_b10   42  tg_b10_W4
tp_b10_W4_s7     W4  sid_noTau_k0_b10    7  tg_b10_W4_s7
"

DEP=""; [ -n "$AFTER" ] && DEP="--dependency=afterany:$AFTER"
DUMPS=""; NSUB=0; NREUSE=0
echo "제출 $(date '+%F %T') · ME=$ME · REUSE=$REUSE${AFTER:+ · AFTER=$AFTER}"
while read -r TAG W SIDN SEED OLD; do
  [ -z "$TAG" ] && continue
  SID=$(S "$SIDN"); DATA=$CD/grid_data/beauty_B/$W
  [ -f "$(LS "$SIDN")" ] || [ "$DRY" = "--dry" ] || { echo "SID 없음: $(LS "$SIDN")"; exit 1; }
  [ -d "$LCD/grid_data/beauty_B/$W" ] || [ "$DRY" = "--dry" ] || { echo "DATA 없음: beauty_B/$W"; exit 1; }
  if [ -d "$LCD/tiger_out/$TAG" ]; then echo "이미 존재: tiger_out/$TAG — 건너뜀"; continue; fi
  if [ "$REUSE" = "1" ] && [ "$OLD" != "-" ] && [ -f "$LCD/tiger_out/$OLD/eval_dump_k200/test_predictions_rank0.pkl" ]; then
    printf '%-16s 재사용 tiger_out/%s (같은 레시피 · seed %s · %s)\n' "$TAG" "$OLD" "$SEED" "$W"
    NREUSE=$((NREUSE+1)); continue
  fi
  JID=$(sub $DEP --export=ALL,TAG=$TAG,SID=$SID,DATA=$DATA,SEED=$SEED,\
MAX_STEPS=30000,PATIENCE=8,LIMIT_VAL=1.0,VAL_EVERY=1000,\
NH=5,WARMUP=1500,CLIP=1.0,SCHED=1,LR=0.001,VAL_SHUFFLE=true \
        --job-name=${ME}_t2_$TAG $LCD/tiger_beauty.sh)
  DID=$(sub --dependency=afterok:$JID \
        --export=ALL,TAG=$TAG,SID=$SID,DATA=$DATA,TOPK=200,BATCH=32 \
        --job-name=${ME}_d2_$TAG $LCD/tiger_eval_dump.sh)
  DUMPS="${DUMPS:+$DUMPS:}$DID"; NSUB=$((NSUB+1))
  printf '%-16s 학습 %-8s → 덤프 %-8s (SID %s · seed %s · %s)\n' "$TAG" "$JID" "$DID" "$SIDN" "$SEED" "$W"
done <<< "$CELLS"

if [ -n "$DUMPS" ]; then
  FID=$(sub --dependency=afterany:$DUMPS --job-name=${ME}_finish_p2 $LCD/finish_p2_temporal.sh)
  echo "마무리(채점·판정) 작업 $FID — 모든 덤프가 끝나면(실패 포함) 자동 실행 → ~/recsys/P2_보고.txt"
fi
echo
echo "새로 제출 $NSUB셀 · 재사용 $NREUSE셀 · GPU 2장 · 셀당 약 2~3.5시간(조기종료 시 짧음) + 덤프 4분"
echo "확인: squeue -u dsl05 -o \"%.8i %.26j %.2t %.10M\"   채점만: bash $LCD/score_p2_temporal.sh"
