#!/bin/bash
#SBATCH --job-name=sid
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys/GRID
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/sid_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/sid_%j.err

# ============================================================================
# RQ-KMeans 로 아이템별 Semantic ID (SID) 생성
#   - 코드북 레벨: 3단계 / 4단계 (LEVELS 로 변경 가능)
#   - 레벨당 코드북 크기: 256 (WIDTH)
#   - 충돌: 마지막 자리에 0부터 시작하는 일련번호
#     (src/utils/tensor_utils.py 의 deduplicate_rows_in_tensor 를 0-base 로 수정해 둠)
#
# ★ dsl05 는 팀 공용 계정입니다. 남의 결과를 덮어쓰지 않도록 TAG 를 꼭 지정하세요.
#
#   sbatch --export=ALL,TAG=gsid_홍길동,EMB=/mnt/data1/dsl05/recsys/work/홍길동/my_emb.pt,DIM=512 \
#          --job-name=sid_홍길동 sid_beauty.sh
#
#   TAG   출력 폴더 이름          (기본 baseline)  → sid_out/$TAG/L{3,4}/
#   EMB   입력 임베딩 .pt 경로     (기본 텍스트 임베딩)
#   DIM   임베딩 차원             (기본 2048)
#   LEVELS 만들 레벨 목록          (기본 "3 4")
#   FORCE=1 이면 기존 출력 폴더를 지우고 다시 만듦 (기본은 덮어쓰기 거부)
# ============================================================================

set -euo pipefail
BASE=/mnt/data1/dsl05
PY=$BASE/miniconda3/envs/grid/bin/python
DATA=$BASE/recsys/grid_data/beauty_A

TAG=${TAG:-baseline}
EMB=${EMB:-$BASE/recsys/embeddings/beauty_A/merged_predictions_tensor.pt}
DIM=${DIM:-2048}
WIDTH=${WIDTH:-256}
LEVELS=${LEVELS:-"3 4"}
STEPS_PER_LAYER=${STEPS_PER_LAYER:-30}
FORCE=${FORCE:-0}

export HF_HOME=$BASE/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) ==="
echo "TAG=$TAG  EMB=$EMB  DIM=$DIM  WIDTH=$WIDTH  LEVELS=$LEVELS"
nvidia-smi --query-gpu=name,memory.total --format=csv
test -f "$EMB" || { echo "임베딩 파일 없음: $EMB"; exit 1; }

cd $BASE/recsys/GRID

# items/ TFRecord 가 파일 1개라 워커를 2개 이상 쓰면 빈 이터레이터로 죽는다 → num_workers=0 고정
no_workers () { echo "$1.num_workers=0 $1.persistent_workers=false $1.timeout=0"; }
DL_TRAIN="data_loading.datamodule.train_dataloader_config"
DL_VAL="data_loading.datamodule.val_dataloader_config"
DL_TEST="data_loading.datamodule.test_dataloader_config"
DL_PRED="data_loading.datamodule.predict_dataloader_config"

for L in $LEVELS; do
  # GRID 는 steps_per_layer = max_steps // n_layers 로 나눠 쓴다.
  # max_steps 를 고정하면 레벨 수마다 레벨당 스텝이 달라져 조건이 어긋나므로 곱해서 준다.
  MAX_STEPS=$(( STEPS_PER_LAYER * L ))
  OUTD=$BASE/recsys/sid_out/${TAG}/L${L}

  echo ""
  echo "###################################################################"
  echo "###  [$TAG] num_hierarchies=$L  width=$WIDTH  max_steps=$MAX_STEPS"
  echo "###  출력: $OUTD"
  echo "###################################################################"

  # ★ 공용 계정이라 남의 결과를 지우지 않도록 방어
  if [ -d "$OUTD" ]; then
    if [ "$FORCE" = "1" ]; then
      echo "경고: 기존 출력 폴더를 지우고 다시 만듭니다 (FORCE=1) — $OUTD"
      rm -rf "$OUTD"
    else
      echo "중단: 출력 폴더가 이미 있습니다 — $OUTD"
      echo "      다른 TAG 를 쓰거나, 정말 덮어쓸 거면 FORCE=1 을 주세요."
      exit 1
    fi
  fi
  mkdir -p "$OUTD"

  echo "--- [L=$L] 1/2 코드북 학습 (RQ-KMeans) ---"
  $PY -m src.train \
    experiment=rkmeans_train_flat \
    data_dir=$DATA \
    embedding_path=$EMB \
    embedding_dim=$DIM \
    num_hierarchies=$L \
    codebook_width=$WIDTH \
    trainer.max_steps=$MAX_STEPS \
    hydra.run.dir=$OUTD/train \
    trainer.devices=1 \
    $(no_workers $DL_TRAIN) \
    $(no_workers $DL_VAL) \
    $(no_workers $DL_TEST)

  CKPT=$(ls -t $OUTD/train/checkpoints/*.ckpt | head -1)
  echo "--- [L=$L] checkpoint: $CKPT ---"

  echo "--- [L=$L] 2/2 아이템별 SID 부여 ---"
  $PY -m src.inference \
    experiment=rkmeans_inference_flat \
    data_dir=$DATA \
    embedding_path=$EMB \
    embedding_dim=$DIM \
    num_hierarchies=$L \
    codebook_width=$WIDTH \
    ckpt_path=$CKPT \
    callbacks.bq_writer=null \
    hydra.run.dir=$OUTD/infer \
    trainer.devices=1 \
    $(no_workers $DL_PRED)

  echo "--- [L=$L] 산출물 ---"
  ls -la $OUTD/infer/pickle/
done

echo ""
echo "JOB_DONE"
