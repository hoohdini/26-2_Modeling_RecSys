#!/bin/bash
#SBATCH --job-name=maskgr_train
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys/MaskGR
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/maskgr_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/maskgr_%j.err

# ============================================================================
# MaskGR (마스크 확산) 학습 — 실험표 오른쪽 열.
#
#   sbatch --export=ALL,TAG=mg_clip_L4,SID=<sid.pt> maskgr_train.sh
#   sbatch --export=ALL,TAG=mg_gsid_a01,SID=<gsid.pt> maskgr_train.sh   # G-SID
#
#   TAG   출력 폴더 이름 (maskgr_out/$TAG)
#   SID   SID 텐서 절대경로. **주소만 바꿔 끼우면 다른 칸이 된다**
#   NH    num_hierarchies = SID 텐서의 행 수 (4단계 SID -> 5)
#
# TIGER 와 반드시 맞춰야 하는 것 (안 맞추면 4x2 표의 좌우 비교가 깨진다)
# --------------------------------------------------------------------
#   · 같은 SID 파일         (같은 행에서는 동일해야 함)
#   · 같은 TFRecord         grid_data/beauty_A — training/evaluation/testing 규약 공유
#   · 같은 seq_len 120
#   · 전수 검증 LIMIT_VAL=1.0  ← 표본 검증은 곡선을 왜곡한다 (노션 11-4절)
#   · 시드를 바꿔 최소 2회   ← 노이즈 바닥(NDCG@10 0.0016)보다 큰 차이만 주장할 수 있다
#
# 함정 (GRID 에서 겪은 것이 그대로 재현된다)
# ------------------------------------------
#   1) TFRecord 가 split 당 1파일이라 num_workers>0 이면 워커가 굶어 죽는다.
#      num_workers=0 + timeout=0 + persistent_workers=false 세트로 눌러 둔다.
#      (MaskGR 기본값은 num_workers=8 / timeout=60 / persistent=true 이라 반드시 덮어쓸 것)
#   2) MaskGR 기본 batch_size 는 2048 이다. 우리 GPU(48GB) 와 데이터 규모에는 과하니
#      BATCH 로 낮춰 잡는다.
#   3) wandb 가 기본 로거다. 서버에서 오프라인이면 죽으므로 logger=csv 로 바꾼다.
# ============================================================================
set -euo pipefail
BASE=/mnt/data1/dsl05
PY=$BASE/miniconda3/envs/maskgr/bin/python
# DATA 를 바꾸면 트랙이 바뀐다:
#   LOO      grid_data/beauty_A
#   Temporal grid_data/beauty_B/W5  (윈도우 하나가 데이터셋 하나)
DATA=${DATA:-$BASE/recsys/grid_data/beauty_A}

TAG=${TAG:?TAG 필요}
SID=${SID:?SID 필요 — SID 텐서 절대경로}
NH=${NH:-5}
SEQ_LEN=${SEQ_LEN:-120}
BATCH=${BATCH:-256}
MAX_STEPS=${MAX_STEPS:-30000}     # TIGER 와 같은 예산에서 출발한다
VAL_EVERY=${VAL_EVERY:-1000}
LIMIT_VAL=${LIMIT_VAL:-1.0}       # 전수 검증. 표본은 곡선을 속인다
LR=${LR:-0.001}
SEED=${SEED:-42}
CLIP=${CLIP:-1.0}
SMOKE=${SMOKE:-0}
FORCE=${FORCE:-0}

if [ "$SMOKE" = "1" ]; then MAX_STEPS=50; VAL_EVERY=25; LIMIT_VAL=0.05; fi

OUTD=$BASE/recsys/maskgr_out/${TAG}
if [ -d "$OUTD" ]; then
  if [ "$FORCE" = "1" ]; then echo "경고: 기존 출력 삭제 (FORCE=1)"; rm -rf "$OUTD"
  else echo "중단: 출력 폴더가 이미 있습니다 — $OUTD (다른 TAG 를 쓰거나 FORCE=1)"; exit 1; fi
fi
mkdir -p "$OUTD"

export HF_HOME=$BASE/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

echo "=== node: $(hostname) ==="
echo "TAG=$TAG NH=$NH BATCH=$BATCH STEPS=$MAX_STEPS SEED=$SEED SMOKE=$SMOKE"
echo "SID=$SID"
test -f "$SID" || { echo "SID 없음: $SID"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd $BASE/recsys/MaskGR

DL_TR="data_loading.train_dataloader_config.dataloader"
DL_VA="data_loading.val_dataloader_config.dataloader"
DL_TE="data_loading.test_dataloader_config.dataloader"

$PY -m src.train \
  experiment=discrete_diffusion_train \
  paths.data_dir=$DATA \
  sid_data_path=$SID \
  model.num_hierarchies=$NH \
  seq_len=$SEQ_LEN \
  batch_size=$BATCH \
  exp_id=$TAG \
  seed=$SEED \
  hydra.run.dir=$OUTD \
  logger=csv \
  trainer=gpu \
  trainer.devices=1 \
  trainer.max_steps=$MAX_STEPS \
  trainer.max_epochs=-1 \
  trainer.val_check_interval=$VAL_EVERY \
  +trainer.limit_val_batches=$LIMIT_VAL \
  +trainer.gradient_clip_val=$CLIP \
  optim.optimizer.lr=$LR \
  callbacks.model_checkpoint.dirpath=$OUTD/checkpoints \
  $DL_TR.batch_size_per_device=$BATCH \
  $DL_VA.batch_size_per_device=$BATCH \
  $DL_TE.batch_size_per_device=$BATCH \
  $DL_TR.num_workers=0 $DL_TR.persistent_workers=false $DL_TR.timeout=0 \
  $DL_VA.num_workers=0 $DL_VA.persistent_workers=false $DL_VA.timeout=0 \
  $DL_TE.num_workers=0 $DL_TE.persistent_workers=false $DL_TE.timeout=0

echo ""
echo "=== 체크포인트 ==="
ls -la "$OUTD"/checkpoints/*.ckpt 2>/dev/null || echo "(체크포인트 없음 — 로그 확인)"
echo "JOB_DONE"
