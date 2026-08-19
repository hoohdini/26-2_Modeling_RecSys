#!/bin/bash
#SBATCH --job-name=grid_embed_beauty
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys/GRID
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/embed_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/embed_%j.err

# NOTE: 마스터 노드는 공유 저장소를 /data1/<user> 로, 계산 노드는 /mnt/data1/<user> 로 마운트한다.
# 배치 스크립트는 계산 노드에서 돌므로 반드시 /mnt/data1 경로를 써야 한다.
# conda 는 /data1 prefix 로 설치되어 activate 가 깨지므로 python 절대경로로 직접 호출한다.
set -euo pipefail
BASE=/mnt/data1/dsl05
PY=$BASE/miniconda3/envs/grid/bin/python

export HF_HOME=$BASE/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) ==="
nvidia-smi
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd $BASE/recsys/GRID
$PY -m src.inference \
  experiment=sem_embeds_inference_flat \
  data_dir=$BASE/recsys/grid_data/beauty_A \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  data_loading.datamodule.predict_dataloader_config.batch_size_per_device=64   data_loading.datamodule.predict_dataloader_config.num_workers=0   data_loading.datamodule.predict_dataloader_config.persistent_workers=false   data_loading.datamodule.predict_dataloader_config.timeout=0

echo "JOB_DONE"
