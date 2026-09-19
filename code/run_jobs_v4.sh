#!/bin/bash
# 구인구직 v4 — 캐글 분포로 재보정한 상호작용 위에서 H1 을 다시 잰다.
#
# v3 와 무엇이 같고 다른가
#   같다   프로필 12,000 · 텍스트 · 그래프(jobs_edges_final.csv) · G1 대조군(rewired) · 학습 프로토콜
#          (MAX_STEPS=30000 PATIENCE=1000 LIMIT_VAL=150 · run_jobs_v3.sh 와 동일)
#   다르다 상호작용만 다시 생성 (data_gen/spec/v4_params.json 의 temp/gamma/k 값)
#          → Split A/B 가 바뀌므로 grid_data 와 SID 도 v4 이름으로 새로 만든다.
#            (텍스트 임베딩은 텍스트가 같으니 v3 것을 재사용해도 되지만, G-SID 는 인기도
#             head/tail 가중치가 상호작용에서 오므로 반드시 다시 만든다. 셋 다 v4 로 통일한다.)
#
# 단계
#   bash run_jobs_v4.sh prep          임베딩(GPU 4분) → G-SID 임베딩(CPU) → SID 3종(GPU 각 4분) → TFRecord(CPU)
#   bash run_jobs_v4.sh train         본 실험 9셀 (text/gsid/rewired × 시드 42,7,13) + 덤프
#   bash run_jobs_v4.sh seeds6        추가 시드 3,21,99 × text/gsid = 6셀 (n=6, 주지표 검정력)
#   bash run_jobs_v4.sh <단계> --dry  명령만 출력
#
# 전제 (로컬 → 서버 업로드, 마스터 노드 경로 기준)
#   ~/recsys/jobs_v4/profile_text.tsv           v3 과 같은 파일
#   ~/recsys/jobs_v4/jobs_edges_final.csv       v3 과 같은 파일
#   ~/recsys/jobs_v4/jobs_edges_rewired.csv     v3 과 같은 파일
#   ~/recsys/jobs_v4/jobs_interactions_long.csv v4 생성본 (user_id,position,item_id,split)
#   ~/recsys/jobs_v4/Jobs_split_A.pkl           v4 생성본
#   ~/recsys/embed_jobs.py · sid_jobs.sh(DATA 인자 지원 확인) · ~/recsys/jobs/gen_gstar_beta.py · graph_sid_augment.py 는 v3 때 올린 것
#   ~/recsys/embed_jobs_v4.sh 는 저장소 code/server/embed_jobs.sh 를 올린 것 (서버의 embed_jobs.sh 는 v3 경로가 박혀 있어 쓰지 않는다)
#
# 제출 전 확인: 로컬에서 G-L 게이트를 통과한 Split 만 올린다 (python data_gen/gate_learnability.py).

set -u
STAGE=${1:-}
DRY=${2:-}
[ -z "$STAGE" ] && { sed -n '2,30p' "$0"; exit 1; }

BASE=/mnt/data1/dsl05     # 계산노드
LBASE=/data1/dsl05        # 로그인노드
CD=$BASE/recsys
LCD=$LBASE/recsys
V4=$CD/jobs_v4
LV4=$LCD/jobs_v4
DATA=$CD/grid_data/jobs_A_v4
LDATA=$LCD/grid_data/jobs_A_v4
PY=$LBASE/miniconda3/envs/grid/bin/python     # 마스터 노드에서 CPU 작업용
ME=${ME:-$(whoami)}                           # 공용 계정이라 작업 이름에 사람 이름을 넣는다. ME=이름 으로 지정
AFTER=${AFTER:-}                              # train: SID 작업이 끝난 뒤 시작하게 하려면 AFTER=jid:jid:jid

S()  { echo "$CD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }
LS() { echo "$LCD/sid_out/$1/L4/infer/pickle/merged_predictions_tensor.pt"; }
sub() {
  if [ "$DRY" = "--dry" ]; then echo "DRYRUN"; echo "sbatch $*" >&2; return; fi
  sbatch "$@" | awk '{print $NF}'
}
run() {
  if [ "$DRY" = "--dry" ]; then echo "\$ $*" >&2; return; fi
  "$@"
}

case "$STAGE" in
prep)
  for f in profile_text.tsv jobs_edges_final.csv jobs_edges_rewired.csv jobs_interactions_long.csv Jobs_split_A.pkl; do
    [ -f "$LV4/$f" ] || { echo "없음: $LV4/$f"; exit 1; }
  done
  echo "[1/4] 텍스트 임베딩 (GPU, 약 4분)"
  J1=$(sub --export=ALL,TEXTS=$V4/profile_text.tsv,OUT=$V4/jobs_text_emb.pt \
           --job-name=${ME}_v4_embed $LCD/embed_jobs_v4.sh)   # 저장소 code/server/embed_jobs.sh 를 이 이름으로 업로드 (서버의 embed_jobs.sh 는 v3 경로 고정본)
  echo "      job $J1 — 끝나면 아래를 마스터 노드에서 실행:"
  cat <<EOS
  # [2/4] G-SID 임베딩 (CPU, 튜닝식 β₀=1.0 · 승자 파라미터) — v3 과 같은 생성기
  cd $LCD/jobs      # v3 때 gen_gstar_beta.py · graph_sid_augment.py 를 여기 두고 썼다 (서버 실측 2026-09-16)
  $PY gen_gstar_beta.py --beta 1.0 \\
      --embedding_path $LV4/jobs_text_emb.pt --edges_path $LV4/jobs_edges_final.csv \\
      --interactions_path $LV4/jobs_interactions_long.csv --out $LV4/jobs_gsid_b10_emb.pt
  $PY gen_gstar_beta.py --beta 1.0 \\
      --embedding_path $LV4/jobs_text_emb.pt --edges_path $LV4/jobs_edges_rewired.csv \\
      --interactions_path $LV4/jobs_interactions_long.csv --out $LV4/jobs_rewired_b10_emb.pt
  # [3/4] SID 3종 (GPU 각 4분, L4 만)
  sbatch --export=ALL,TAG=jobs_v4_text,EMB=$V4/jobs_text_emb.pt,DIM=2048,LEVELS=4,DATA=$DATA --job-name=${ME}_v4_sid_text $LCD/sid_jobs.sh
  sbatch --export=ALL,TAG=jobs_v4_gsid_b10,EMB=$V4/jobs_gsid_b10_emb.pt,DIM=2048,LEVELS=4,DATA=$DATA --job-name=${ME}_v4_sid_gsid $LCD/sid_jobs.sh
  sbatch --export=ALL,TAG=jobs_v4_rewired,EMB=$V4/jobs_rewired_b10_emb.pt,DIM=2048,LEVELS=4,DATA=$DATA --job-name=${ME}_v4_sid_rew $LCD/sid_jobs.sh
  # [4/4] TFRecord (CPU, tensorflow-cpu 는 grid env 에 있음) — SID 보다 먼저 있어야 하므로 [2/4] 직후 실행
  $PY $LCD/to_grid_v2.py $LV4/Jobs_split_A.pkl $LDATA
EOS
  ;;
train|seeds6)
  if [ "$STAGE" = "train" ]; then
    CELLS="
jobs_text_v4_s42       jobs_v4_text      42
jobs_gsid_b10_v4_s42   jobs_v4_gsid_b10  42
jobs_text_v4_s7        jobs_v4_text       7
jobs_gsid_b10_v4_s7    jobs_v4_gsid_b10   7
jobs_text_v4_s13       jobs_v4_text      13
jobs_gsid_b10_v4_s13   jobs_v4_gsid_b10  13
jobs_rewired_v4_s42    jobs_v4_rewired   42
jobs_rewired_v4_s7     jobs_v4_rewired    7
jobs_rewired_v4_s13    jobs_v4_rewired   13
"
  else
    CELLS="
jobs_text_v4_s3        jobs_v4_text       3
jobs_gsid_b10_v4_s3    jobs_v4_gsid_b10   3
jobs_text_v4_s21       jobs_v4_text      21
jobs_gsid_b10_v4_s21   jobs_v4_gsid_b10  21
jobs_text_v4_s99       jobs_v4_text      99
jobs_gsid_b10_v4_s99   jobs_v4_gsid_b10  99
"
  fi
  [ -d "$LDATA" ] || [ "$DRY" = "--dry" ] || { echo "grid_data/jobs_A_v4 없음 ($LDATA) — prep 먼저"; exit 1; }
  echo "DATA=$DATA · 고정: MAX_STEPS=30000 PATIENCE=1000 LIMIT_VAL=150 · 시드 확장 (3,21,99) 는 사전 고정값"
  DEP=""; [ -n "$AFTER" ] && DEP="--dependency=afterok:$AFTER"
  DUMPS=""
  while read -r TAG SIDN SEED; do
    [ -z "$TAG" ] && continue
    SID=$(S "$SIDN")
    [ -f "$(LS "$SIDN")" ] || [ "$DRY" = "--dry" ] || [ -n "$AFTER" ] || { echo "SID 없음: $(LS "$SIDN")"; exit 1; }
    [ -d "$LCD/tiger_out/$TAG" ] && { echo "이미 존재: tiger_out/$TAG — 건너뜀"; continue; }
    JID=$(sub $DEP --export=ALL,TAG=$TAG,SID=$SID,DATA=$DATA,SEED=$SEED,\
MAX_STEPS=30000,PATIENCE=1000,LIMIT_VAL=150,VAL_EVERY=1000,\
NH=5,WARMUP=1500,CLIP=1.0,SCHED=1,LR=0.001,VAL_SHUFFLE=true \
          --job-name=${ME}_t4_$TAG $LCD/tiger_beauty.sh)
    DID=$(sub --dependency=afterok:$JID \
          --export=ALL,TAG=$TAG,SID=$SID,DATA=$DATA,TOPK=200 \
          --job-name=${ME}_d4_$TAG $LCD/tiger_eval_dump.sh)
    DUMPS="${DUMPS:+$DUMPS:}$DID"
    printf '%-24s 학습 %-8s → 덤프 %-8s (SID %s, seed %s)\n' "$TAG" "$JID" "$DID" "$SIDN" "$SEED"
  done <<< "$CELLS"
  if [ -n "$DUMPS" ]; then
    FID=$(sub --dependency=afterany:$DUMPS --job-name=${ME}_finish_v4_$STAGE $LCD/finish_jobs_v4.sh)
    echo "마무리(채점·판정) 작업 $FID — 덤프가 모두 끝나면 자동 실행 → ~/recsys/V4_보고.txt"
  fi
  echo
  echo "GPU 2장 · 셀당 약 3시간 40분 + 덤프 27분 · 9셀 = 5웨이브 ≈ 21시간 · 6셀 = 3웨이브 ≈ 13시간"
  echo "채점:  bash $LCD/score_jobs_v4.sh   판정:  JOBS_VER=v4 SEEDS=42,7,13,3,21,99 $PY $LCD/verdict_v3.py"
  ;;
*)
  echo "단계는 prep | train | seeds6"; exit 1;;
esac
