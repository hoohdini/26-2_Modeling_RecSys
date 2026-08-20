#!/bin/bash
#SBATCH --job-name=tiger_infer
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys/GRID
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/infer_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/infer_%j.err

# 학습된 TIGER 체크포인트로 testing 셋 전체에 대해 SID 를 생성한다.
# 산출물: tiger_out/<TAG>/infer/pickle/merged_predictions.pkl
#         (user_id -> 생성된 semantic_ids top-10)
# 지표 코드를 거치지 않고 예측을 직접 확인하기 위한 것.
#
#   sbatch --export=ALL,TAG=baseline_L4,CKPT=<체크포인트경로> tiger_infer.sh

# ★ 체크포인트 파일명에 '=' 가 들어 있어(checkpoint_epoch=000_step=018000.ckpt)
#   hydra 오버라이드 파서가 "mismatched input '='" 로 죽는다.
#   값을 작은따옴표로 감싸 hydra 에 문자열로 넘겨야 한다.
# ★ tiger_inference_flat 의 predict 데이터로더는 labels 가 null 이라
#   마지막 아이템(정답)을 가리지 않는다. 그대로 쓰면 정답을 입력에 넣은 채
#   '그 다음'을 예측하게 되어 평가가 무의미해진다.
#   학습 스크립트의 test 단계와 동일하게 NextKTokenMasking 을 넣어야 한다.
# ★★ masking_token 이 학습(-1)과 추론 config(1)에서 다르다.
#    NextKTokenMasking 은 마지막 K토큰을 [마스킹토큰 1개 + 패딩 K-1개]로 바꾸는데,
#    모델은 학습 내내 -1 을 '여기를 예측하라' 신호로 배웠다. 1 을 넣으면 실제 코드값
#    (계층0의 코드 1)으로 읽혀 완전히 다른 입력이 된다. 반드시 -1 로 맞출 것.
set -euo pipefail
BASE=/mnt/data1/dsl05
PY=$BASE/miniconda3/envs/grid/bin/python
DATA=$BASE/recsys/grid_data/beauty_A
SID=${SID:-$BASE/recsys/sid_out/baseline/L4/infer/pickle/merged_predictions_tensor.pt}
NH=${NH:-5}
WIDTH=${WIDTH:-256}
VOCAB=$(( NH * WIDTH ))
TAG=${TAG:?TAG 필요}
CKPT=${CKPT:?CKPT 필요}
OUTD=$BASE/recsys/tiger_out/${TAG}/infer

export HF_HOME=$BASE/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) ==="
echo "TAG=$TAG NH=$NH VOCAB=$VOCAB"
echo "CKPT=$CKPT"
test -f "$CKPT" || { echo "체크포인트 없음"; exit 1; }
rm -rf "$OUTD"; mkdir -p "$OUTD"

cd $BASE/recsys/GRID
DL="data_loading.predict_dataloader_config.dataloader"

$PY -m src.inference \
  experiment=tiger_inference_flat \
  data_dir=$DATA \
  semantic_id_path=$SID \
  num_hierarchies=$NH \
  "ckpt_path='$CKPT'" \
  model.huggingface_model.config.vocab_size=$VOCAB \
  "++$DL.labels={sequence_data:{transform:{_target_:src.data.loading.components.label_function.NextKTokenMasking,next_k:$NH}}}" \
  $DL.masking_token=-1 $DL.padding_token=-1 \
  hydra.run.dir=$OUTD \
  trainer.devices=1 \
  $DL.num_workers=0 $DL.persistent_workers=false $DL.timeout=0

echo ""
ls -la $OUTD/pickle/ 2>/dev/null || true
echo "JOB_DONE"
