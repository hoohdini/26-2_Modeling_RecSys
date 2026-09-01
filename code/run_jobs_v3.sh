#!/bin/bash
# 구인구직 v3 — H1 재실행 (6셀). **데이터는 v2 그대로**, 학습 프로토콜만 고정한다.
#
# 왜 다시 도는가
# --------------
# v2 는 조기종료 기본값(PATIENCE=8)을 그대로 써서 셀마다 다른 step 에서 멈췄다.
#     text  s42 best 5,000 / s7 3,000 / s13 17,000
#     gsid  s42 best 6,000 / s7 3,000 / s13  4,000
# text 기준선만 시드마다 3,000~17,000 으로 흩어졌고, 늦게 멈춘 s13 은 질적으로
# 다른 해로 수렴했다(coverage@10 0.196 vs 0.05). 그 결과 3시드 노이즈 바닥이
# 폭발해 전 지표가 판정 불가가 됐다. 2시드 PASS 는 42·7 이 우연히 둘 다 일찍
# 멈춘 데서 나온 것이다.
#
# 무엇을 고정하나
# --------------
#   MAX_STEPS=30000   Beauty 117135 와 동일 (그쪽 best 는 step 28,000)
#   PATIENCE=1000     사실상 비활성 — 모든 셀이 30,000 step 을 끝까지 돈다
#   LIMIT_VAL=150     검증 표본 150x64=9,600 명 (전체 18,741 의 51%).
#                     기본 20 은 1,280 명(6.8%)이라 best 체크포인트 선택 자체가
#                     잡음이었다. 검증 1회 1.5s -> 11s, 30회로 5분 추가에 그친다.
#
# 비교 공정성: 6셀 전부 같은 창(30,000 step)에서 같은 방식으로 best 를 고른다.
#
# SID 3종과 grid_data/jobs_A 는 v2 것을 그대로 쓴다 (데이터는 G-L 통과본).
#
# 사용:  bash run_jobs_v3.sh          # 제출
#        bash run_jobs_v3.sh --dry    # 명령만 출력

set -u
BASE=/mnt/data1/dsl05     # 계산노드 기준
LBASE=/data1/dsl05        # 로그인노드 기준
CD=$BASE/recsys
LCD=$LBASE/recsys
DATA=$CD/grid_data/jobs_A
LDATA=$LCD/grid_data/jobs_A
DRY=${1:-}

S()  { echo "$CD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }
LS() { echo "$LCD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }

CELLS="
jobs_text_v3_s42       jobs_text      42
jobs_gsid_b10_v3_s42   jobs_gsid_b10  42
jobs_text_v3_s7        jobs_text       7
jobs_gsid_b10_v3_s7    jobs_gsid_b10   7
jobs_text_v3_s13       jobs_text      13
jobs_gsid_b10_v3_s13   jobs_gsid_b10  13
jobs_rewired_v3_s42    jobs_gsid_rewired  42
jobs_rewired_v3_s7     jobs_gsid_rewired   7
jobs_rewired_v3_s13    jobs_gsid_rewired  13
"

sub() {
  if [ "$DRY" = "--dry" ]; then echo "DRYRUN"; echo "sbatch $*" >&2; return; fi
  sbatch "$@" | awk '{print $NF}'
}

echo "DATA=$DATA"
[ -d "$LDATA" ] || { echo "grid_data/jobs_A 없음 ($LDATA)"; exit 1; }
echo "고정: MAX_STEPS=30000  PATIENCE=1000  LIMIT_VAL=150"
echo

echo "$CELLS" | while read -r TAG SIDN SEED; do
  [ -z "$TAG" ] && continue
  SID=$(S "$SIDN")
  [ -f "$(LS "$SIDN")" ] || { echo "SID 없음: $(LS "$SIDN")"; exit 1; }
  [ -d "$LCD/tiger_out/$TAG" ] && { echo "이미 존재: tiger_out/$TAG — 건너뜀"; continue; }

  JID=$(sub --export=ALL,TAG=$TAG,SID=$SID,DATA=$DATA,SEED=$SEED,\
MAX_STEPS=30000,PATIENCE=1000,LIMIT_VAL=150,VAL_EVERY=1000,\
NH=5,WARMUP=1500,CLIP=1.0,SCHED=1,LR=0.001,VAL_SHUFFLE=true \
        --job-name=t3_$TAG $LCD/tiger_beauty.sh)

  DID=$(sub --dependency=afterok:$JID \
        --export=ALL,TAG=$TAG,SID=$SID,DATA=$DATA,TOPK=200 \
        --job-name=d3_$TAG $LCD/tiger_eval_dump.sh)

  printf '%-24s 학습 %-8s → 덤프 %-8s (SID %s, seed %s)\n' "$TAG" "$JID" "$DID" "$SIDN" "$SEED"
done

echo
echo "GPU 2장 · 셀당 30,000 step ≈ 3시간 40분 · 3웨이브 ≈ 11시간 30분"
echo "채점:  bash $LCD/score_jobs_v3.sh"
