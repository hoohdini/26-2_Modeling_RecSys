#!/bin/bash
#SBATCH --job-name=grid_sid_beauty
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys/GRID
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/sid_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/sid_%j.err

# RQ-KMeans 로 아이템별 Semantic ID (SID) 생성.
#   - 코드북 레벨: 3단계 / 4단계 두 가지
#   - 레벨당 코드북 크기: 256
#   - 충돌: 마지막 자리에 0부터 시작하는 일련번호를 붙여 구분
#     (src/utils/tensor_utils.py 의 deduplicate_rows_in_tensor 를 0-base 로 수정해 둠)
#
# ★ 레벨당 학습 스텝을 고정한다.
#   GRID 는 steps_per_layer = trainer.max_steps // n_layers 로 나눠 쓰기 때문에,
#   max_steps 를 기본값 30 으로 두면 L=3 은 레벨당 10스텝, L=4 는 7스텝이 되어 조건이 달라진다.
#   그래서 max_steps = STEPS_PER_LAYER * L 로 준다.
#   아이템이 12,101개 · 배치 2048 이라 1 에폭 ≈ 6스텝 → 30스텝은 레벨당 약 5 에폭.
#
# 경로 주의: 계산 노드에서는 공유 저장소가 /mnt/data1, conda activate 는 깨지므로 python 절대경로 사용.

set -euo pipefail
BASE=/mnt/data1/dsl05
PY=$BASE/miniconda3/envs/grid/bin/python
DATA=$BASE/recsys/grid_data/beauty_A
EMB=$BASE/recsys/embeddings/beauty_A/merged_predictions_tensor.pt
DIM=2048
WIDTH=256
STEPS_PER_LAYER=30

export HF_HOME=$BASE/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
test -f "$EMB" || { echo "임베딩 파일 없음: $EMB"; exit 1; }

cd $BASE/recsys/GRID

# items/ TFRecord 가 파일 1개라 워커를 2개 이상 쓰면 빈 이터레이터로 죽는다 → num_workers=0 고정
no_workers () {   # $1 = 데이터로더 config 경로
  echo "$1.num_workers=0 $1.persistent_workers=false $1.timeout=0"
}
DL_TRAIN="data_loading.datamodule.train_dataloader_config"
DL_VAL="data_loading.datamodule.val_dataloader_config"
DL_TEST="data_loading.datamodule.test_dataloader_config"
DL_PRED="data_loading.datamodule.predict_dataloader_config"

for L in 3 4; do
  MAX_STEPS=$(( STEPS_PER_LAYER * L ))
  echo ""
  echo "###################################################################"
  echo "###  num_hierarchies=$L  codebook_width=$WIDTH  max_steps=$MAX_STEPS"
  echo "###  (레벨당 $STEPS_PER_LAYER 스텝)"
  echo "###################################################################"
  OUTD=$BASE/recsys/sid_out/L${L}
  rm -rf "$OUTD"
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
