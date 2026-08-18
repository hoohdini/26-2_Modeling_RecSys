#!/usr/bin/env bash
# =============================================================================
# 기본 / Graph / CRAB / Graph+CRAB, 4가지 SID를 GRID 원본 명령어로 생성.
# GRID 내부 코드는 수정하지 않고, hydra.run.dir 를 매번 명시적으로 지정해서
# 출력 경로(체크포인트, pickle 결과)를 우리가 예측 가능하게 고정한다.
# (GRID 기본값은 logs/<task>/runs/<날짜>/<시각> 처럼 실행마다 바뀌는 타임스탬프
#  경로라 스크립트에서 다음 단계 입력으로 재사용하기 어려움 -> 그래서 고정함)
#
# 사용법: GRID 레포 루트에서 실행
#   DATA_DIR=data/amazon_data/beauty \
#   RELATED_PATH=data/amazon_data/beauty/related.json \
#   POPULARITY_PATH=data/amazon_data/beauty/popularity.json \
#   bash sid_pipeline/run_all_variants.sh
# =============================================================================
set -euo pipefail

DATA_DIR=${DATA_DIR:-data/amazon_data/beauty}
RELATED_PATH=${RELATED_PATH:-$DATA_DIR/related.json}
POPULARITY_PATH=${POPULARITY_PATH:-$DATA_DIR/popularity.json}
WRAPPER_DIR=${WRAPPER_DIR:-sid_pipeline}
RUN_ROOT=${RUN_ROOT:-runs/sid_variants}   # 우리가 고정할 hydra 출력 루트
CODEBOOK_WIDTH=256
EMBEDDING_DIM=2048

latest_ckpt () {
    # model_checkpoint 콜백이 checkpoint_{epoch}_{step}.ckpt 형태로 저장하므로
    # (파일명이 고정돼있지 않음) 가장 최근 파일을 찾아서 반환
    ls -t "$1"/checkpoints/*.ckpt | head -n 1
}

echo "=== Step 1. 텍스트 -> 벡터 (GRID 원본) ==="
EMBED_RUN_DIR="$RUN_ROOT/text_embedding"
python -m src.inference \
    experiment=sem_embeds_inference_flat \
    data_dir="$DATA_DIR" \
    hydra.run.dir="$EMBED_RUN_DIR"
TEXT_EMBED="$EMBED_RUN_DIR/pickle/merged_predictions_tensor.pt"

echo "=== Step 2. G-SID 전처리 (우리 코드) ==="
GRAPH_EMBED="$DATA_DIR/graph_augmented_embedding.pt"
python "$WRAPPER_DIR/graph_sid_augment.py" \
    --embedding_path "$TEXT_EMBED" \
    --related_path "$RELATED_PATH" \
    --popularity_path "$POPULARITY_PATH" \
    --out_path "$GRAPH_EMBED"

run_rkmeans () {
    # $1 = 임베딩 경로  $2 = num_hierarchies  $3 = 태그(text/graph)
    local embed_path=$1 num_h=$2 tag=$3
    local train_dir="$RUN_ROOT/rkmeans_${tag}_h${num_h}/train"
    local infer_dir="$RUN_ROOT/rkmeans_${tag}_h${num_h}/infer"

    echo "  [rkmeans] ${tag} h=${num_h} 학습..." >&2
    python -m src.train \
        experiment=rkmeans_train_flat \
        data_dir="$DATA_DIR" \
        embedding_path="$embed_path" \
        embedding_dim=$EMBEDDING_DIM \
        num_hierarchies=$num_h \
        codebook_width=$CODEBOOK_WIDTH \
        hydra.run.dir="$train_dir" >&2

    local ckpt
    ckpt=$(latest_ckpt "$train_dir")

    echo "  [rkmeans] ${tag} h=${num_h} 추론... (ckpt=$ckpt)" >&2
    python -m src.inference \
        experiment=rkmeans_inference_flat \
        data_dir="$DATA_DIR" \
        embedding_path="$embed_path" \
        embedding_dim=$EMBEDDING_DIM \
        num_hierarchies=$num_h \
        codebook_width=$CODEBOOK_WIDTH \
        ckpt_path="$ckpt" \
        hydra.run.dir="$infer_dir" >&2

    echo "$infer_dir/pickle/merged_predictions_tensor.pt"
}

for H in 3 4; do
    echo ""
    echo "############ num_hierarchies = $H ############"

    echo "--- 1) 텍스트 SID ---"
    TEXT_CODES=$(run_rkmeans "$TEXT_EMBED" "$H" "text")
    cp "$TEXT_CODES" "$DATA_DIR/sid_text_h${H}.pt"

    echo "--- 2) Graph SID ---"
    GRAPH_CODES=$(run_rkmeans "$GRAPH_EMBED" "$H" "graph")
    cp "$GRAPH_CODES" "$DATA_DIR/sid_graph_h${H}.pt"

    echo "--- 3) CRAB SID (텍스트 SID 후처리) ---"
    python "$WRAPPER_DIR/crab_split.py" \
        --codes_path "$TEXT_CODES" \
        --embedding_path "$TEXT_EMBED" \
        --popularity_path "$POPULARITY_PATH" \
        --codebook_width $CODEBOOK_WIDTH \
        --out_path "$DATA_DIR/sid_crab_h${H}.pt"

    echo "--- 4) Graph + CRAB SID (Graph SID 후처리) ---"
    python "$WRAPPER_DIR/crab_split.py" \
        --codes_path "$GRAPH_CODES" \
        --embedding_path "$GRAPH_EMBED" \
        --popularity_path "$POPULARITY_PATH" \
        --codebook_width $CODEBOOK_WIDTH \
        --out_path "$DATA_DIR/sid_graph_crab_h${H}.pt"
done

echo ""
echo "=== 완료: $DATA_DIR 안에 sid_{text,graph,crab,graph_crab}_h{3,4}.pt 8개 생성됨 ==="
