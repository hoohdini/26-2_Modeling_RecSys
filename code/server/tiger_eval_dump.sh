#!/bin/bash
#SBATCH --job-name=tiger_evaldump
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys/GRID
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/evaldump_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/evaldump_%j.err

# ============================================================================
# 트랙 C — 학습 없이 test 단계만 돌려 "정답이 가려진" 추천 결과를 덤프한다.
#
#   sbatch --export=ALL,TAG=clip_L4,TOPK=50,CKPT=<경로> tiger_eval_dump.sh
#
#   TAG    학습 산출물 폴더 이름 (tiger_out/$TAG)
#   CKPT   체크포인트 경로. 생략하면 tiger_out/$TAG/checkpoints 에서 자동 탐색
#   TOPK   빔 폭 = 유저당 생성 후보 수 (기본 50)
#   OUT    덤프 폴더 (기본 tiger_out/$TAG/eval_dump_k$TOPK)
#
#   LIMIT  test 배치 수 제한 (스모크용). 0 이면 전체
#   PREFIX 1 이면 접두사 제약 디코딩(무효 SID 제거). 기본 0
#
# 왜 src.inference 도 src.train 도 아닌 src.eval_dump 인가
# --------------------------------------------------------
#   · src.inference(predict_step) 은 정답을 입력에서 가리지 않는다.
#     test_step -> eval_step 은 test 데이터로더가 NextKTokenMasking 으로
#     마지막 K 토큰을 마스킹해 주므로 이쪽이 올바른 평가 경로다.
#   · src.train train=False 는 ckpt_path 를 설정에서 읽지 않는다(ModelCheckpoint
#     콜백의 best_model_path 만 본다). 학습을 건너뛰면 랜덤 가중치로 평가한다.
#   자세한 근거는 eval_dump.py / prediction_dumper.py 주석 참고.
#
# 반드시 확인할 것
# ----------------
#   · TOPK 는 빔 폭이라 값이 커지면 상위 10개 결과도 미세하게 달라진다.
#     TOPK=10 으로 한 번 돌려 metrics.csv 의 test/recall@5,@10 과 일치하는지
#     먼저 확인하면 덤프 경로가 옳다는 것이 증명된다. (권장 순서)
#   · 학습과 동일한 NH / VOCAB / SID 를 줘야 한다. 다르면 체크포인트가 안 실린다.
# ============================================================================
set -euo pipefail
BASE=/mnt/data1/dsl05
PY=$BASE/miniconda3/envs/grid/bin/python
DATA=$BASE/recsys/grid_data/beauty_A

TAG=${TAG:?TAG 필요}
SID=${SID:-$BASE/recsys/sid_out/baseline/L4/infer/pickle/merged_predictions_tensor.pt}
NH=${NH:-5}
WIDTH=${WIDTH:-256}
SEQ_LEN=${SEQ_LEN:-120}
TOPK=${TOPK:-50}
BATCH=${BATCH:-64}
LIMIT=${LIMIT:-0}
PREFIX=${PREFIX:-0}
VOCAB=$(( NH * WIDTH ))
LIMIT_OVERRIDE=""
if [ "$LIMIT" != "0" ]; then LIMIT_OVERRIDE="+trainer.limit_test_batches=$LIMIT"; fi

# PREFIX=1 은 접두사 제약 디코딩. 생성 도중 실제 SID 로 이어지지 않는 가지를 쳐내므로
# 무효 SID 가 사라지지만, 지금까지 보고한 지표는 전부 false 기준이다. 켜면 재측정 필요.
PREFIX_OVERRIDE="+model.should_check_prefix=false"
SUFFIX=""
if [ "$PREFIX" = "1" ]; then
  PREFIX_OVERRIDE="+model.should_check_prefix=true"
  SUFFIX="_prefix"
fi

RUND=$BASE/recsys/tiger_out/${TAG}
CKPT=${CKPT:-$(ls -1t $RUND/checkpoints/*.ckpt 2>/dev/null | head -1)}
OUT=${OUT:-$RUND/eval_dump_k${TOPK}${SUFFIX}}

export HF_HOME=$BASE/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) ==="
echo "TAG=$TAG TOPK=$TOPK NH=$NH VOCAB=$VOCAB BATCH=$BATCH PREFIX=$PREFIX"
echo "CKPT=$CKPT"
echo "OUT=$OUT"
test -n "$CKPT" -a -f "$CKPT" || { echo "체크포인트 없음: '$CKPT'"; exit 1; }
test -f "$SID" || { echo "SID 없음: $SID"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

rm -rf "$OUT"; mkdir -p "$OUT"
cd $BASE/recsys/GRID

DL_TR="data_loading.train_dataloader_config.dataloader"
DL_VA="data_loading.val_dataloader_config.dataloader"
DL_TE="data_loading.test_dataloader_config.dataloader"

# hydra.run.dir 는 학습 산출물을 덮지 않도록 덤프 폴더 아래로 보낸다.
$PY -m src.eval_dump \
  experiment=tiger_train_flat \
  data_dir=$DATA \
  semantic_id_path=$SID \
  num_hierarchies=$NH \
  sequence_length=$SEQ_LEN \
  data_loading.features_config.features.0.semantic_ids=$SID \
  model.huggingface_model.config.vocab_size=$VOCAB \
  +model.top_k_for_generation=$TOPK \
  $PREFIX_OVERRIDE \
  "ckpt_path='$CKPT'" \
  hydra.run.dir=$OUT/hydra \
  trainer.devices=1 \
  trainer.max_epochs=-1 \
  +callbacks.pred_dump._target_=src.callbacks.prediction_dumper.TigerPredictionDumper \
  +callbacks.pred_dump.out_dir=$OUT \
  $LIMIT_OVERRIDE \
  $DL_TE.batch_size_per_device=$BATCH \
  $DL_TR.num_workers=0 $DL_TR.persistent_workers=false $DL_TR.timeout=0 \
  $DL_VA.num_workers=0 $DL_VA.persistent_workers=false $DL_VA.timeout=0 \
  $DL_TE.num_workers=0 $DL_TE.persistent_workers=false $DL_TE.timeout=0

echo ""
echo "=== 덤프 산출물 ==="
ls -la "$OUT"/*.pkl 2>/dev/null || echo "(pkl 없음 — 로그에서 [dumper] 줄을 확인할 것)"
echo "JOB_DONE"
