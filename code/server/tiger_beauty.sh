#!/bin/bash
#SBATCH --job-name=tiger
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys/GRID
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/tiger_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/tiger_%j.err

# ============================================================================
# TIGER 학습 + 전체 테스트셋 평가 (Split A / Leave-One-Out)
#
#   sbatch --export=ALL,TAG=clip_L4,CLIP=1.0 --job-name=tiger_clip tiger_beauty.sh
#
#   TAG         출력 폴더 (tiger_out/$TAG)        기본 baseline_L4
#   SID         SID 텐서 경로                     기본 4단계 baseline
#   NH          num_hierarchies                   기본 5 (코드 4자리 + 충돌 구분자)
#   MAX_STEPS   학습 스텝                         기본 30000
#   VAL_EVERY   검증 주기(스텝)                   기본 1000
#   LIMIT_VAL   검증 배치 수                      기본 20
#   PATIENCE    조기 종료(검증 횟수)              기본 8
#   SCHED       1=warmup+cosine, 0=고정 lr        기본 1
#   WARMUP      warmup 스텝                       기본 1000
#   CLIP        gradient_clip_val (0=클리핑 없음) 기본 0
#   VAL_SHUFFLE 검증셋 셔플                       기본 true
#   SMOKE=1     60스텝만 돌려 설정 검증
#   FORCE=1     기존 출력 폴더 덮어쓰기
#
# ─ 반드시 알아야 할 설정 함정 5가지 (전부 실제로 겪음) ───────────────────────
# 1) num_hierarchies 는 SID 텐서의 행 수와 같아야 한다. 모델이 codebooks.shape[0] 에서
#    유도하고(SemanticIDEncoderDecoder.__init__), 데이터는 id_map[:NH] 로 자른다.
#    4단계 SID 텐서는 (5, 12101) 이므로 NH=5. 4로 두면 충돌 구분자가 빠진다.
# 2) vocab_size 는 NH x 코드북크기. GRID 는 임베딩 테이블 하나를 계층별 오프셋으로
#    나눠 쓴다(계층 h 의 코드 c -> h*256+c). 기본값 256 이면 인덱스 범위를 벗어난다.
# 3) 데이터셋이 UnboundedSequenceIterable(무한 반복) 이라 에폭이 끝나지 않는다.
#    max_epochs 로 제어하면 검증도 체크포인트도 영원히 안 생긴다. 스텝 기준으로 줄 것.
# 4) TFRecord 가 split 당 파일 1개라 num_workers>=2 면 워커 하나가 빈 이터레이터를
#    받아 StopIteration 으로 죽는다. num_workers=0 + timeout=0 + persistent=false 세트.
# 5) GRID 설정 어디에도 gradient_clip_val 이 없다(= 클리핑 없음). 클리핑 없이 Adam 으로
#    오래 돌리면 loss 최저점 직후 발산한다. 실측: 고정 lr 은 step 18k, warmup+cosine 은
#    step 7k 에서 동일 패턴(loss 급등 + recall 붕괴). CLIP=1.0 을 권장한다.
#
# ─ 검증셋 주의 ──────────────────────────────────────────────────────────────
#   VAL_SHUFFLE=false 로 두면 파일 앞쪽 유저만 검증에 쓰이는데, 이들은 시퀀스가
#   길어서(중앙값 7 vs 전체 5) 검증 지표가 낙관적으로 나온다. 기본 true 로 둘 것.
# ============================================================================

set -euo pipefail
BASE=/mnt/data1/dsl05
PY=$BASE/miniconda3/envs/grid/bin/python
# DATA 를 바꾸면 트랙이 바뀐다. 학습과 덤프에 반드시 같은 값을 줄 것.
#   LOO      grid_data/beauty_A
#   Temporal grid_data/beauty_B/W5   (윈도우 하나가 데이터셋 하나)
DATA=${DATA:-$BASE/recsys/grid_data/beauty_A}

TAG=${TAG:-baseline_L4}
SID=${SID:-$BASE/recsys/sid_out/baseline/L4/infer/pickle/merged_predictions_tensor.pt}
NH=${NH:-5}
WIDTH=${WIDTH:-256}
SEQ_LEN=${SEQ_LEN:-120}
BATCH=${BATCH:-256}
VAL_BATCH=${VAL_BATCH:-64}
SMOKE=${SMOKE:-0}
FORCE=${FORCE:-0}
CLIP=${CLIP:-0}
VAL_SHUFFLE=${VAL_SHUFFLE:-true}
SCHED=${SCHED:-1}
WARMUP=${WARMUP:-1000}
# GRID 기본 lr=0.001. 이 데이터에서는 step 7~8k 부근에서 주기적으로 진동한다.
# 그래디언트 클리핑으로는 못 잡았다 — Adam 이 이미 그래디언트를 정규화해서
# norm 이 1.0 을 넘는 일이 드물고, 클리핑이 사실상 작동하지 않는다(실측: 클리핑
# 유무만 다른 두 실행이 같은 step 8000 에서 거의 같은 값으로 붕괴).
LR=${LR:-0.001}
SEED=${SEED:-42}   # 시드 변동폭 측정용. 실험 비교 시 시드를 바꿔 여러 번 돌릴 것

VOCAB=$(( NH * WIDTH ))

if [ "$SMOKE" = "1" ]; then
  MAX_STEPS=60; VAL_EVERY=30; LIMIT_VAL=3; PATIENCE=100
else
  MAX_STEPS=${MAX_STEPS:-30000}
  VAL_EVERY=${VAL_EVERY:-1000}
  LIMIT_VAL=${LIMIT_VAL:-20}
  PATIENCE=${PATIENCE:-8}
fi

if [ "$SCHED" = "1" ]; then
  SCHED_OVERRIDE="++optim.scheduler={_target_:src.components.scheduler.WarmupCosineSchedulerNonzeroMin,_partial_:true,warmup_steps:${WARMUP},scheduler_steps:${MAX_STEPS},min_ratio:0.1}"
else
  SCHED_OVERRIDE="optim.scheduler=null"
fi

export HF_HOME=$BASE/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

OUTD=$BASE/recsys/tiger_out/${TAG}
echo "=== node: $(hostname) ==="
echo "TAG=$TAG NH=$NH VOCAB=$VOCAB STEPS=$MAX_STEPS CLIP=$CLIP SCHED=$SCHED WARMUP=$WARMUP VAL_SHUFFLE=$VAL_SHUFFLE"
echo "SID=$SID SEED=$SEED LR=$LR"
nvidia-smi --query-gpu=name,memory.total --format=csv
test -f "$SID" || { echo "SID 파일 없음: $SID"; exit 1; }
$PY -c "
import torch;t=torch.load('$SID',map_location='cpu')
print('SID tensor:',tuple(t.shape),'max code:',int(t.max()))
assert t.shape[0]==$NH, f'NH($NH) 와 SID 행 수({t.shape[0]}) 불일치'
assert int(t.max())< $WIDTH, 'WIDTH 보다 큰 코드가 있음'
"

if [ -d "$OUTD" ]; then
  if [ "$FORCE" = "1" ]; then echo "경고: 기존 출력 삭제 (FORCE=1)"; rm -rf "$OUTD"
  else echo "중단: 출력 폴더가 이미 있습니다 — $OUTD (다른 TAG 를 쓰거나 FORCE=1)"; exit 1; fi
fi
mkdir -p "$OUTD"

cd $BASE/recsys/GRID

DL_TR="data_loading.train_dataloader_config.dataloader"
DL_VA="data_loading.val_dataloader_config.dataloader"
DL_TE="data_loading.test_dataloader_config.dataloader"

$PY -m src.train \
  experiment=tiger_train_flat \
  data_dir=$DATA \
  semantic_id_path=$SID \
  num_hierarchies=$NH \
  sequence_length=$SEQ_LEN \
  data_loading.features_config.features.0.semantic_ids=$SID \
  model.huggingface_model.config.vocab_size=$VOCAB \
  hydra.run.dir=$OUTD \
  trainer.devices=1 \
  trainer.max_steps=$MAX_STEPS \
  trainer.max_epochs=-1 \
  trainer.accumulate_grad_batches=1 \
  trainer.val_check_interval=$VAL_EVERY \
  +trainer.limit_val_batches=$LIMIT_VAL \
  +trainer.gradient_clip_val=$CLIP \
  callbacks.early_stopping.patience=$PATIENCE \
  optim.optimizer.lr=$LR \
  seed=$SEED \
  "$SCHED_OVERRIDE" \
  $DL_VA.should_shuffle_rows=$VAL_SHUFFLE \
  $DL_TR.batch_size_per_device=$BATCH \
  $DL_VA.batch_size_per_device=$VAL_BATCH \
  $DL_TE.batch_size_per_device=$VAL_BATCH \
  $DL_TR.num_workers=0 $DL_TR.persistent_workers=false $DL_TR.timeout=0 \
  $DL_VA.num_workers=0 $DL_VA.persistent_workers=false $DL_VA.timeout=0 \
  $DL_TE.num_workers=0 $DL_TE.persistent_workers=false $DL_TE.timeout=0

echo ""
echo "=== 산출물 ==="
ls -la $OUTD/checkpoints/ 2>/dev/null || true
echo "JOB_DONE"
