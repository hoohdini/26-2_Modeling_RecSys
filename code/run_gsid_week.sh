#!/bin/bash
# 실험 트랙 P1 + P3 제출 (노션 "1. 그래프 SID 신뢰도 실험 계획" 3절) — 2026-09-16 제출
#
#   P1  Beauty LOO 시드 확장   텍스트(baseline) · 튜닝식 β₀1.0(sid_noTau_k0_b10) · 고전 α0.7(sid_blend_centered_a07)
#                              × 시드 3, 21  = 6셀 → n=5 (기존 42·7·13 과 합침)
#   P3  구인구직 시드 확장     jobs_text · jobs_gsid_b10 × 시드 3, 21, 99 = 6셀 → n=6
#
# 레시피는 **기존 시드 셀과 동일**하게 맞춘다 (같은 레시피여야 시드 평균이 성립한다).
#   Beauty  (tiger_out/tiger_noTau_k0_b10_seed13 의 .hydra/overrides.yaml 실측)
#           MAX_STEPS=30000 PATIENCE=8 LIMIT_VAL=1.0 VAL_EVERY=1000 NH=5 WARMUP=1500 CLIP=1.0 SCHED=1 LR=0.001
#           ※ 노션 7절 예시는 PATIENCE=1000 이라 적혀 있지만 기존 3시드가 PATIENCE=8 로 돌았으므로 8 을 쓴다.
#   Jobs    (run_jobs_v3.sh 와 동일)  MAX_STEPS=30000 PATIENCE=1000 LIMIT_VAL=150
#
# 덤프는 학습 afterok, 마지막에 CPU 마무리 작업(finish_gsid_week.sh)을 모든 덤프 afterany 로 건다.
# → 채점과 판정표가 자동으로 나온다 (results_gsid_week/, results_jobs_v3/, GSID_WEEK_보고.txt).
#
#   bash run_gsid_week.sh          # 제출
#   bash run_gsid_week.sh --dry    # 명령만 출력
#   ME=이름 bash run_gsid_week.sh  # 작업 이름 접두사 (공용 계정 규칙)

set -u
DRY=${1:-}
BASE=/mnt/data1/dsl05
LBASE=/data1/dsl05
CD=$BASE/recsys
LCD=$LBASE/recsys
ME=${ME:-mc}

S()  { echo "$CD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }
LS() { echo "$LCD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }
sub() {
  if [ "$DRY" = "--dry" ]; then echo "DRY$RANDOM"; echo "sbatch $*" >&2; return; fi
  sbatch "$@" | awk '{print $NF}'
}

# TAG  SID폴더  SEED  DATA  LIMIT_VAL  PATIENCE  DUMP_BATCH
CELLS="
ns_base_s3            baseline               3   beauty_A  1.0  8     32
ns_b10_s3             sid_noTau_k0_b10       3   beauty_A  1.0  8     32
ns_a07_s3             sid_blend_centered_a07 3   beauty_A  1.0  8     32
ns_base_s21           baseline               21  beauty_A  1.0  8     32
ns_b10_s21            sid_noTau_k0_b10       21  beauty_A  1.0  8     32
ns_a07_s21            sid_blend_centered_a07 21  beauty_A  1.0  8     32
jobs_text_v3_s3       jobs_text              3   jobs_A    150  1000  64
jobs_gsid_b10_v3_s3   jobs_gsid_b10          3   jobs_A    150  1000  64
jobs_text_v3_s21      jobs_text              21  jobs_A    150  1000  64
jobs_gsid_b10_v3_s21  jobs_gsid_b10          21  jobs_A    150  1000  64
jobs_text_v3_s99      jobs_text              99  jobs_A    150  1000  64
jobs_gsid_b10_v3_s99  jobs_gsid_b10          99  jobs_A    150  1000  64
"

DUMPS=""
echo "제출 $(date '+%F %T') · ME=$ME"
while read -r TAG SIDN SEED DATAN LV PAT DB; do
  [ -z "$TAG" ] && continue
  SID=$(S "$SIDN"); DATA=$CD/grid_data/$DATAN
  [ -f "$(LS "$SIDN")" ] || [ "$DRY" = "--dry" ] || { echo "SID 없음: $(LS "$SIDN")"; exit 1; }
  [ -d "$LCD/grid_data/$DATAN" ] || [ "$DRY" = "--dry" ] || { echo "DATA 없음: $DATAN"; exit 1; }
  if [ -d "$LCD/tiger_out/$TAG" ]; then echo "이미 존재: tiger_out/$TAG — 건너뜀"; continue; fi
  JID=$(sub --export=ALL,TAG=$TAG,SID=$SID,DATA=$DATA,SEED=$SEED,\
MAX_STEPS=30000,PATIENCE=$PAT,LIMIT_VAL=$LV,VAL_EVERY=1000,\
NH=5,WARMUP=1500,CLIP=1.0,SCHED=1,LR=0.001,VAL_SHUFFLE=true \
        --job-name=${ME}_t_$TAG $LCD/tiger_beauty.sh)
  DID=$(sub --dependency=afterok:$JID \
        --export=ALL,TAG=$TAG,SID=$SID,DATA=$DATA,TOPK=200,BATCH=$DB \
        --job-name=${ME}_d_$TAG $LCD/tiger_eval_dump.sh)
  DUMPS="${DUMPS:+$DUMPS:}$DID"
  printf '%-22s 학습 %-8s → 덤프 %-8s (SID %s · seed %s · %s · LV %s · PAT %s)\n' "$TAG" "$JID" "$DID" "$SIDN" "$SEED" "$DATAN" "$LV" "$PAT"
done <<< "$CELLS"

if [ -n "$DUMPS" ]; then
  FID=$(sub --dependency=afterany:$DUMPS --job-name=${ME}_finish_gsid_week $LCD/finish_gsid_week.sh)
  echo "마무리(채점·판정) 작업 $FID — 모든 덤프가 끝나면(실패 포함) 자동 실행"
fi
echo
echo "GPU 2장 · 12셀 × 약 3시간 40분 + 덤프(Beauty 4분, Jobs 27분) ≈ 벽시계 24시간"
echo "확인: squeue -u dsl05 -o \"%.8i %.26j %.2t %.10M\"   보고서: ~/recsys/GSID_WEEK_보고.txt"
