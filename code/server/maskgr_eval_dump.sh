#!/bin/bash
#SBATCH --job-name=maskgr_evaldump
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys/MaskGR
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/maskgr_evaldump_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/maskgr_evaldump_%j.err

# ============================================================================
# MaskGR — 학습 없이 test 만 돌려 "정답이 가려진" 추천 결과를 덤프한다.
# 산출 pkl 은 GRID 덤프와 스키마가 같아서 트랙 C 하네스가 그대로 읽는다.
#
#   sbatch --export=ALL,TAG=mg_clip_L4,SID=<sid.pt>,CAND=200 maskgr_eval_dump.sh
#
#   TAG    학습 산출물 폴더 이름 (maskgr_out/$TAG)
#   SID    SID 텐서 절대경로. **G-SID 로 바꿀 때 여기만 바꾸면 된다**
#   NH     num_hierarchies = SID 텐서의 행 수 (4단계 SID -> 5)
#   CKPT   체크포인트. 생략하면 maskgr_out/$TAG/checkpoints 에서 자동 탐색
#   CAND   후보 수 = diffusion_config.num_candidates (기본 200)
#
#   확산 다이얼 (파레토 곡선용 — 이게 트랙 C 가 기다린 "생성 자체의 손잡이"다)
#   TEMP   unmasking_temperature   기본 0.01 (사실상 argmax). 키우면 다양해진다
#   STEPS  num_steps               기본 NH
#   UNMASK unmasking_type          top-prob | random | top-prob-dup-last | left-to-right
#   SCHED  noise_schedule          uniform | edm | last-token-ar
#   ITYPE  inference_type          beam-search-generation | constrained-beam-search-generation
#
# 반드시 확인할 것
# ----------------
#   · 먼저 CAND=10 (기본값)으로 한 번 돌려 학습 로그의 val/recall@5 와 대조할 것.
#     일치하면 덤프 경로가 평가 경로를 바꾸지 않았다는 뜻이다. (GRID 에서 쓴 방법)
#   · CAND 는 빔 폭이라 값이 커지면 상위 10개 결과도 미세하게 달라진다.
#     실험 간 비교할 때는 고정할 것.
#   · TFRecord 가 split 당 1파일이라 num_workers>0 이면 데이터로더가 멈춘다.
#     아래 오버라이드로 0 으로 눌러 둔다. (GRID 에서 겪은 함정과 동일)
# ============================================================================
set -euo pipefail
BASE=/mnt/data1/dsl05
PY=$BASE/miniconda3/envs/maskgr/bin/python
# 학습 때 쓴 DATA 와 반드시 같아야 한다. 다르면 다른 분할로 채점하게 된다.
DATA=${DATA:-$BASE/recsys/grid_data/beauty_A}

TAG=${TAG:?TAG 필요}
SID=${SID:?SID 필요 — SID 텐서 절대경로}
NH=${NH:-5}
CAND=${CAND:-200}
BATCH=${BATCH:-256}
SEQ_LEN=${SEQ_LEN:-120}

TEMP=${TEMP:-0.01}
STEPS=${STEPS:-$NH}
UNMASK=${UNMASK:-top-prob}
SCHED=${SCHED:-uniform}
ITYPE=${ITYPE:-beam-search-generation}

RUND=$BASE/recsys/maskgr_out/${TAG}
CKPT=${CKPT:-$(ls -1t $RUND/checkpoints/*.ckpt 2>/dev/null | head -1)}
SUFFIX="_c${CAND}_t${TEMP}_s${STEPS}"
OUT=${OUT:-$RUND/eval_dump${SUFFIX}}

export HF_HOME=$BASE/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) ==="
echo "TAG=$TAG NH=$NH CAND=$CAND TEMP=$TEMP STEPS=$STEPS UNMASK=$UNMASK SCHED=$SCHED"
echo "SID=$SID"
echo "CKPT=$CKPT"
echo "OUT=$OUT"
test -n "$CKPT" -a -f "$CKPT" || { echo "체크포인트 없음: '$CKPT'"; exit 1; }
test -f "$SID" || { echo "SID 없음: $SID"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

rm -rf "$OUT"; mkdir -p "$OUT"
cd $BASE/recsys/MaskGR

DL_TR="data_loading.train_dataloader_config.dataloader"
DL_VA="data_loading.val_dataloader_config.dataloader"
DL_TE="data_loading.test_dataloader_config.dataloader"

$PY -m src.maskgr_eval_dump \
  experiment=discrete_diffusion_train \
  paths.data_dir=$DATA \
  sid_data_path=$SID \
  model.num_hierarchies=$NH \
  seq_len=$SEQ_LEN \
  model.diffusion_config.num_candidates=$CAND \
  model.diffusion_config.unmasking_temperature=$TEMP \
  model.diffusion_config.num_steps=$STEPS \
  model.diffusion_config.unmasking_type=$UNMASK \
  model.diffusion_config.noise_schedule=$SCHED \
  model.diffusion_config.inference_type=$ITYPE \
  "ckpt_path='$CKPT'" \
  hydra.run.dir=$OUT/hydra \
  trainer=gpu \
  trainer.devices=1 \
  logger=csv \
  +callbacks.pred_dump._target_=src.callbacks.maskgr_dumper.MaskGRPredictionDumper \
  +callbacks.pred_dump.out_dir=$OUT \
  $DL_TE.batch_size_per_device=$BATCH \
  $DL_TR.num_workers=0 $DL_TR.persistent_workers=false $DL_TR.timeout=0 \
  $DL_VA.num_workers=0 $DL_VA.persistent_workers=false $DL_VA.timeout=0 \
  $DL_TE.num_workers=0 $DL_TE.persistent_workers=false $DL_TE.timeout=0

echo ""
echo "=== 덤프 산출물 ==="
ls -la "$OUT"/*.pkl 2>/dev/null || echo "(pkl 없음 — 로그에서 [dumper] 줄을 확인할 것)"
cat "$OUT/dump_meta.json" 2>/dev/null || true
echo "JOB_DONE"
