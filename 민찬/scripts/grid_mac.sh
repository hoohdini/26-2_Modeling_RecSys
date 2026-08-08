#!/usr/bin/env bash
# GRID를 맥(MPS)에서 돌리기 위한 공통 래퍼.
#
# GRID 기본 설정은 NVIDIA 다중 GPU를 가정한다. 맥에서 돌리려면 매번 아래를
# 덮어써야 해서, 그 부분만 여기에 모아 뒀다.
#
#   accelerator=gpu → mps          CUDA가 없으므로
#   strategy=ddp    → auto         분산 학습 불가
#   precision=bf16-mixed → 32      MPS는 bf16 혼합정밀도 지원이 불안정
#   num_workers     → 0            기본값. GRID_NUM_WORKERS 로 올릴 수 있다
#
# 주의 2가지
#   ① 프로젝트 경로에 공백과 한글이 있어서 Hydra 오버라이드가 파싱에 실패한다.
#      경로 값은 반드시 작은따옴표로 한 번 더 감싸야 한다.
#   ② num_workers=0 이면 PyTorch가 timeout==0 을 요구한다.
#      persistent_workers 도 같이 꺼야 한다. 셋을 함께 맞춘다.
#
# 사용:
#   scripts/grid_mac.sh train experiment=rkmeans_train_flat embedding_dim=768 ...
#   scripts/grid_mac.sh inference experiment=tiger_inference_flat ...
#
# 경로를 넘길 때는 ① 때문에 이렇게 감싼다:
#   "data_dir='/절대/경로/data/amazon_data/beauty'"

set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRID_DIR="$PROJ/code/grid"
PYTHON="${GRID_PYTHON:-/opt/anaconda3/envs/grid/bin/python}"

ENTRYPOINT="${1:?사용법: grid_mac.sh <train|inference> [hydra 오버라이드...]}"
shift

# ++ 를 쓰는 이유: train 설정에는 trainer.strategy 가 있지만 inference 설정에는
# 없다. ++ 는 있으면 덮어쓰고 없으면 추가하므로 양쪽에 같은 명령을 쓸 수 있다.
MAC_OVERRIDES=(
	++trainer.accelerator=mps
	++trainer.devices=1
	++trainer.strategy=auto
	++trainer.precision=32
	extras.print_config=false
)

# experiment 마다 train/val/test/predict 중 일부만 정의돼 있다. 없는 split을
# 덮어쓰면 Hydra가 반쪽짜리 노드를 만들어 instantiate 단계에서 터진다.
# 그래서 실제 정의된 split 만 골라낸다.
EXPERIMENT=""
for arg in "$@"; do
	case "$arg" in
		experiment=*) EXPERIMENT="${arg#experiment=}" ;;
	esac
done

EXP_FILE="$GRID_DIR/configs/experiment/${EXPERIMENT}.yaml"
if [[ -z "$EXPERIMENT" || ! -f "$EXP_FILE" ]]; then
	echo "experiment=<이름> 을 찾을 수 없습니다 (확인한 경로: $EXP_FILE)" >&2
	exit 1
fi

# 데이터 로딩 워커 수. 기본 0.
#
# 맥에서는 0이 가장 빠르다. 20스텝 실측(검증 제외):
#   워커 0개 27초/스텝 · 6개 34초 · 12개 36초
# 워커를 spawn 으로 띄우는 비용이 병렬화 이득보다 크기 때문이다.
# GPU 서버에서는 다를 수 있으니 거기서 다시 재고 GRID_NUM_WORKERS 로 조정한다.
#
# num_workers=0 일 때 PyTorch 는 timeout==0 을 요구하고, persistent_workers 도
# 켤 수 없다. 그래서 세 값을 함께 맞춘다.
WORKERS="${GRID_NUM_WORKERS:-0}"
if [[ ! "$WORKERS" =~ ^[0-9]+$ ]]; then
	echo "GRID_NUM_WORKERS 는 0 이상의 정수여야 합니다 (받은 값: $WORKERS)" >&2
	exit 1
fi
if [[ "$WORKERS" -gt 0 ]]; then
	WORKER_TIMEOUT=60
	PERSISTENT=true
else
	WORKER_TIMEOUT=0
	PERSISTENT=false
fi

# 설정 레이아웃이 두 종류다.
#   아이템 계열 (rkmeans/sem_embeds): data_loading.datamodule.<split>_dataloader_config
#   시퀀스 계열 (tiger):              data_loading.<split>_dataloader_config.dataloader
#                                     (datamodule 쪽은 이걸 참조만 한다)
for split in train val test predict; do
	if ! grep -qE "^[[:space:]]*${split}_dataloader_config:" "$EXP_FILE"; then
		continue
	fi

	# 해당 키 바로 다음 줄이 "dataloader:" 이면 시퀀스 계열 레이아웃이다.
	if awk -v key="${split}_dataloader_config:" '
		found == 1 { nested = ($0 ~ /^[[:space:]]*dataloader:[[:space:]]*$/); found = 2 }
		found == 0 && $1 == key { found = 1 }
		END { exit nested ? 0 : 1 }
	' "$EXP_FILE"; then
		BASE="data_loading.${split}_dataloader_config.dataloader"
	else
		BASE="data_loading.datamodule.${split}_dataloader_config"
	fi

	MAC_OVERRIDES+=(
		"${BASE}.num_workers=${WORKERS}"
		"${BASE}.timeout=${WORKER_TIMEOUT}"
		"${BASE}.persistent_workers=${PERSISTENT}"
	)
done

# rootutils 가 프로젝트 루트를 찾는 마커. GRID 저장소에 빠져 있어서
# 없으면 첫 실행이 FileNotFoundError 로 죽는다. 비어 있어도 되는 파일이다.
[[ -f "$GRID_DIR/.project-root" ]] || touch "$GRID_DIR/.project-root"

cd "$GRID_DIR"
exec env PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON" -m "src.$ENTRYPOINT" \
	"${MAC_OVERRIDES[@]}" "$@"
