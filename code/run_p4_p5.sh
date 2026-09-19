#!/bin/bash
# 실험 트랙 P4 + P5 제출 (노션 "1. 그래프 SID 신뢰도 실험 계획" 3절 P4·P5, 판정 규칙 D)
#
#   P4  학습 예산 민감도   Beauty LOO · 텍스트(baseline) · 튜닝식 β₀1.0(sid_noTau_k0_b10) × 15,000·45,000스텝 × 시드 42, 7 = 8셀
#       규칙 D: 15k, 45k 각각에서 (그래프 − 텍스트) 가 2/2 부호 일치 → "이득이 예산과 무관". 못 하면 30k 에서만 확인됐다고 한정.
#       30k 참고선은 P1 의 n=5 결과(results_gsid_week)를 그대로 쓴다.
#       코사인 스케줄이 총 스텝에 맞춰 늘어나므로(scheduler_steps=MAX_STEPS) 45k 실행의 30k 시점은 30k 실행과 같지 않다. 별도 실행이다.
#   P5  접두사 제약 덤프 (학습 없음)   기존 LOO 3시드 체크포인트에 PREFIX=1 TOPK=200 으로 덤프만 다시 뜬다 = 6개
#       텍스트 = clip_L4 · clip_L4_seed7 · clip_L4_seed13   (results_gsid_week 의 ns_base_s42·s7·s13 원천, 2026-09-18 재채점으로 확인)
#       튜닝식 = tiger_noTau_k0_b10 · _seed7 · _seed13       (ns_b10_s42·s7·s13 원천)
#       구인구직 §8 과 같은 표(bucket_recall@50 · zero-shot · 무효 SID 비율)를 Beauty 에서 만든다.
#
# 레시피(P4)는 LOO 시드 셀과 동일하고 MAX_STEPS 만 바꾼다:
#   PATIENCE=8 LIMIT_VAL=1.0 VAL_EVERY=1000 NH=5 WARMUP=1500 CLIP=1.0 SCHED=1 LR=0.001 VAL_SHUFFLE=true · 덤프 배치 32, TOPK 200
#
#   bash run_p4_p5.sh              # 제출 (없는 것만)
#   bash run_p4_p5.sh --dry        # 명령만 출력
#   ME=이름 AFTER=<jid> bash run_p4_p5.sh     # AFTER 작업이 끝난 뒤(afterany) 시작
#
# 순서: P5 덤프 6개(각 4분) → P4 15k 4셀(각 약 1시간 40분) → P4 45k 4셀(각 약 5시간). 마무리(finish_p4_p5.sh)는 전부 afterany.
# → 채점(score_p4_p5.sh) 과 판정표(verdict_p4_p5.py) 가 자동으로 나온다 (~/recsys/P4P5_보고.txt).

set -u
DRY=${1:-}
BASE=/mnt/data1/dsl05
LBASE=/data1/dsl05
CD=$BASE/recsys
LCD=$LBASE/recsys
ME=${ME:-minchan}
AFTER=${AFTER:-}
DATA=$CD/grid_data/beauty_A

S()  { echo "$CD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }
LS() { echo "$LCD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }
sub() {
  if [ "$DRY" = "--dry" ]; then echo "DRY$RANDOM"; echo "sbatch $*" >&2; return; fi
  sbatch "$@" | awk '{print $NF}'
}
DEP=""; [ -n "$AFTER" ] && DEP="--dependency=afterany:$AFTER"
ALL=""
echo "제출 $(date '+%F %T') · ME=$ME${AFTER:+ · AFTER=$AFTER}"

# ---- P5: 접두사 제약 덤프 (학습 없음) ----
# 기존TAG                 SID폴더
P5="
clip_L4                  baseline
clip_L4_seed7            baseline
clip_L4_seed13           baseline
tiger_noTau_k0_b10       sid_noTau_k0_b10
tiger_noTau_k0_b10_seed7 sid_noTau_k0_b10
tiger_noTau_k0_b10_seed13 sid_noTau_k0_b10
"
echo "[P5] 접두사 제약 덤프"
while read -r TAG SIDN; do
  [ -z "$TAG" ] && continue
  [ -d "$LCD/tiger_out/$TAG/checkpoints" ] || [ "$DRY" = "--dry" ] || { echo "체크포인트 없음: tiger_out/$TAG"; exit 1; }
  if [ -f "$LCD/tiger_out/$TAG/eval_dump_k200_prefix/test_predictions_rank0.pkl" ]; then echo "  이미 존재: $TAG/eval_dump_k200_prefix — 건너뜀"; continue; fi
  DID=$(sub $DEP --export=ALL,TAG=$TAG,SID=$(S $SIDN),DATA=$DATA,TOPK=200,BATCH=32,PREFIX=1 \
        --job-name=${ME}_d5_$TAG $LCD/tiger_eval_dump.sh)
  ALL="${ALL:+$ALL:}$DID"
  printf '  %-26s 덤프 %-8s (PREFIX=1 · SID %s)\n' "$TAG" "$DID" "$SIDN"
done <<< "$P5"

# ---- P4: 예산 민감도 ----
# TAG               SID폴더            STEPS  SEED
P4="
p4_base_15k_s42   baseline           15000  42
p4_b10_15k_s42    sid_noTau_k0_b10   15000  42
p4_base_15k_s7    baseline           15000   7
p4_b10_15k_s7     sid_noTau_k0_b10   15000   7
p4_base_45k_s42   baseline           45000  42
p4_b10_45k_s42    sid_noTau_k0_b10   45000  42
p4_base_45k_s7    baseline           45000   7
p4_b10_45k_s7     sid_noTau_k0_b10   45000   7
"
echo "[P4] 예산 민감도"
while read -r TAG SIDN STEPS SEED; do
  [ -z "$TAG" ] && continue
  [ -f "$(LS "$SIDN")" ] || [ "$DRY" = "--dry" ] || { echo "SID 없음: $(LS "$SIDN")"; exit 1; }
  if [ -d "$LCD/tiger_out/$TAG" ]; then echo "  이미 존재: tiger_out/$TAG — 건너뜀"; continue; fi
  JID=$(sub $DEP --export=ALL,TAG=$TAG,SID=$(S $SIDN),DATA=$DATA,SEED=$SEED,\
MAX_STEPS=$STEPS,PATIENCE=8,LIMIT_VAL=1.0,VAL_EVERY=1000,\
NH=5,WARMUP=1500,CLIP=1.0,SCHED=1,LR=0.001,VAL_SHUFFLE=true \
        --job-name=${ME}_t4b_$TAG $LCD/tiger_beauty.sh)
  DID=$(sub --dependency=afterok:$JID \
        --export=ALL,TAG=$TAG,SID=$(S $SIDN),DATA=$DATA,TOPK=200,BATCH=32 \
        --job-name=${ME}_d4b_$TAG $LCD/tiger_eval_dump.sh)
  ALL="${ALL:+$ALL:}$DID"
  printf '  %-18s 학습 %-8s → 덤프 %-8s (SID %s · %s스텝 · seed %s)\n' "$TAG" "$JID" "$DID" "$SIDN" "$STEPS" "$SEED"
done <<< "$P4"

if [ -n "$ALL" ]; then
  FID=$(sub --dependency=afterany:$ALL --job-name=${ME}_finish_p4p5 $LCD/finish_p4_p5.sh)
  echo "마무리(채점·판정) 작업 $FID — 모든 덤프가 끝나면(실패 포함) 자동 실행 → ~/recsys/P4P5_보고.txt"
fi
echo
echo "GPU 2장 · P5 6덤프 약 30분 · P4 15k 2웨이브 약 3시간 반 · 45k 2웨이브 약 10시간 → 약 14시간"
echo "확인: squeue -u dsl05 -o \"%.8i %.26j %.2t %.10M\"   채점만: bash $LCD/score_p4_p5.sh"
