"""GRID Step 2를 건너뛰기 위한 임베딩 파일 생성기 (A안).

GRID의 정식 Step 2는 Flan-T5-XL(약 3GB)을 내려받아 아이템 텍스트를 인코딩하고
2048차원 임베딩을 `merged_predictions_tensor.pt` 로 저장한다.

그런데 P5 전처리 데이터의 items/*.tfrecord.gz 안에는 이미 768차원 임베딩이
들어 있다. 이 스크립트는 그 임베딩을 읽어 Step 2 출력과 **완전히 같은 형식**으로
저장한다. 그러면 Step 3~6을 모델 다운로드 없이 그대로 돌릴 수 있다.

출력 형식 (src/utils/tensor_utils.py:merge_list_of_keyed_tensors_to_single_tensor)
    (N, d) float32 텐서. **행 번호가 곧 아이템 id** 이고, N은 아이템 개수.
    Step 3이 embedding_map[row["id"]] 로 바로 인덱싱하기 때문에 이 규약이 중요하다.

주의: 성능 수치는 GRID 논문값과 맞지 않는다. 임베딩 출처가 다르기 때문이다.
      이 경로의 목적은 "파이프라인이 끝까지 도는가" 확인이다.

사용:
    python scripts/build_step2_embeddings.py \
        --data-dir data/amazon_data/beauty \
        --out work/step2_768/beauty/merged_predictions_tensor.pt
"""

import argparse
import glob
import os

import numpy as np
import tensorflow as tf
import torch


def read_items(items_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """items/*.tfrecord.gz 전체에서 (id, embedding) 을 읽어 온다."""
    files = sorted(glob.glob(os.path.join(items_dir, "*.tfrecord.gz")))
    if not files:
        raise FileNotFoundError(f"tfrecord 파일이 없습니다: {items_dir}")

    ids: list[int] = []
    embeddings: list[np.ndarray] = []

    dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")
    for raw in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        feature = example.features.feature
        ids.append(feature["id"].int64_list.value[0])
        embeddings.append(np.asarray(feature["embedding"].float_list.value, dtype=np.float32))

    return np.asarray(ids, dtype=np.int64), np.stack(embeddings)


def build_tensor(ids: np.ndarray, embeddings: np.ndarray) -> torch.Tensor:
    """행 번호 = 아이템 id 인 (N, d) 텐서로 재배열한다."""
    n_rows = len(ids)
    unique_ids = np.unique(ids)

    if len(unique_ids) != n_rows:
        raise ValueError(f"아이템 id가 중복됩니다: 행 {n_rows}개, 고유 id {len(unique_ids)}개")
    # GRID의 병합 함수는 output_tensor 를 (행 개수, d) 로 잡고 id 로 인덱싱한다.
    # 따라서 id 가 행 개수를 넘으면 IndexError 가 난다. 같은 조건을 여기서 먼저 검사한다.
    if ids.max() >= n_rows:
        raise ValueError(f"아이템 id {ids.max()} 가 행 개수 {n_rows} 를 넘습니다")

    out = torch.zeros((n_rows, embeddings.shape[1]), dtype=torch.float32)
    out[torch.from_numpy(ids)] = torch.from_numpy(embeddings)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="beauty / sports / toys 폴더")
    parser.add_argument("--out", required=True, help="저장할 .pt 경로")
    args = parser.parse_args()

    ids, embeddings = read_items(os.path.join(args.data_dir, "items"))
    tensor = build_tensor(ids, embeddings)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(tensor, args.out)

    print(f"아이템 {tensor.shape[0]}개 · 임베딩 {tensor.shape[1]}차원")
    print(f"id 범위 {ids.min()}~{ids.max()} · L2 노름 평균 {tensor.norm(dim=1).mean():.4f}")
    print(f"저장 완료: {args.out}")
    print(f"→ Step 3 에 embedding_dim={tensor.shape[1]} 로 넘기세요")


if __name__ == "__main__":
    main()
