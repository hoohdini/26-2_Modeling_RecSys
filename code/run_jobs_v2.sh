#!/bin/bash
# 구인구직 v2 재학습 — 5셀 + eval dump 를 의존성으로 묶어 제출한다.
#
# v1 은 시뮬레이터 TEMP=16 때문에 학습 신호가 없어 폐기했다.
# v2 는 TEMP=0.15 GAMMA=0.20 CAND_POOL=12000 (OBS_NOISE=14.0 불변) 로 재생성했고
# G-L 게이트를 통과했다. 근거: docs/JOBS_학습실패_원인분석.md
#
# **제출 전 반드시** grid_data/jobs_A 가 v2 데이터로 갱신됐는지 확인할 것.
# SID 3종은 v1 것을 그대로 쓴다 (SID 는 텍스트·그래프에서 나오고 상호작용을 보지 않는다).
#
# 사용:  bash run_jobs_v2.sh          # 제출
#        bash run_jobs_v2.sh --dry    # 명령만 출력

set -u
# 계산노드는 /mnt/data1 로, 로그인노드는 /data1 로 같은 저장소를 본다.
# sbatch 에 넘기는 경로는 계산노드 기준(BASE), 존재 확인은 로그인노드 기준(LBASE).
BASE=/mnt/data1/dsl05
LBASE=/data1/dsl05
CD=$BASE/recsys
LCD=$LBASE/recsys
DATA=$CD/grid_data/jobs_A
LDATA=$LCD/grid_data/jobs_A
DRY=${1:-}

S()  { echo "$CD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }
LS() { echo "$LCD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }

# 셀 정의:  TAG  SID이름  SEED
CELLS="
jobs_text_v2_s42       jobs_text          42
jobs_gsid_b10_v2_s42   jobs_gsid_b10      42
jobs_text_v2_s7        jobs_text           7
jobs_gsid_b10_v2_s7    jobs_gsid_b10       7
jobs_rewired_v2_s42    jobs_gsid_rewired  42
jobs_text_v2_s13       jobs_text          13
jobs_gsid_b10_v2_s13   jobs_gsid_b10      13
"

sub() {  # echo 또는 실제 제출. stdout 은 JOBID 만.
  if [ "$DRY" = "--dry" ]; then echo "DRYRUN" ; echo "sbatch $*" >&2; return; fi
  sbatch "$@" | awk '{print $NF}'
}

echo "DATA=$DATA"
[ -d "$LDATA" ] || { echo "grid_data/jobs_A 없음 ($LDATA)"; exit 1; }
echo

echo "$CELLS" | while read -r TAG SIDN SEED; do
  [ -z "$TAG" ] && continue
  SID=$(S "$SIDN")
  [ -f "$(LS "$SIDN")" ] || { echo "SID 없음: $(LS "$SIDN")"; exit 1; }
  [ -d "$LCD/tiger_out/$TAG" ] && { echo "이미 존재: tiger_out/$TAG — 건너뜀"; continue; }

  JID=$(sub --export=ALL,TAG=$TAG,SID=$SID,DATA=$DATA,SEED=$SEED,\
STEPS=30000,NH=5,WARMUP=1500,CLIP=1.0,SCHED=1,LR=0.001,VAL_SHUFFLE=true \
        --job-name=tiger_$TAG $LCD/tiger_beauty.sh)

  DID=$(sub --dependency=afterok:$JID \
        --export=ALL,TAG=$TAG,SID=$SID,DATA=$DATA,TOPK=200 \
        --job-name=dump_$TAG $LCD/tiger_eval_dump.sh)

  printf '%-24s 학습 %-8s → 덤프 %-8s (SID %s, seed %s)\n' "$TAG" "$JID" "$DID" "$SIDN" "$SEED"
done

echo
echo "GPU 한도 2장이라 2셀씩 진행합니다. 셀당 학습 2.5~3시간 + 덤프 30분."
echo "채점:  bash $LCD/score_jobs_v2.sh   (덤프 나온 것만, 여러 번 돌려도 안전)"
