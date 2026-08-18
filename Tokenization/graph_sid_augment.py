"""
G-SID 전처리 스크립트

GRID의 sem_embeds_inference_flat 이 만든 임베딩 텐서를 그대로 읽어서
(이웃과 섞은) 새 텐서를 같은 포맷(N x embedding_dim, item_id로 직접 인덱싱)으로 저장한다.
이렇게 만든 파일을 그대로 GRID의 embedding_path 로 넘기면 rkmeans_train_flat /
rkmeans_inference_flat 은 전혀 손대지 않고 그대로 쓸 수 있다.

    python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty
        -> logs/.../merged_predictions_tensor.pt

    python graph_sid_augment.py \
        --embedding_path logs/.../merged_predictions_tensor.pt \
        --related_path   data/amazon_data/beauty/related.json \
        --popularity_path data/amazon_data/beauty/popularity.json \
        --out_path       data/amazon_data/beauty/graph_augmented_embedding.pt

    python -m src.train experiment=rkmeans_train_flat \
        data_dir=data/amazon_data/beauty \
        embedding_path=data/amazon_data/beauty/graph_augmented_embedding.pt \
        embedding_dim=2048 num_hierarchies=3 codebook_width=256
"""

import argparse
import json
from collections import defaultdict

import torch

ALPHA = 0.3
HEAD_RATIO = 0.2
EDGE_WEIGHT = {
    ("head", "head"): 0.5,
    ("head", "tail"): 1.5,
    ("tail", "head"): 1.5,
    ("tail", "tail"): 1.0,
}


def classify_head_tail(popularity: dict[int, int], n_items: int) -> list[str]:
    """iid -> 'head' | 'tail' | 'mid'. GRID의 아이템 id는 0..n_items-1 로 정렬돼있다는 전제."""
    counts = [popularity.get(i, 0) for i in range(n_items)]
    ranked = sorted(range(n_items), key=lambda i: -counts[i])
    head_cut = int(n_items * HEAD_RATIO)
    tail_cut = int(n_items * (1 - HEAD_RATIO))

    label = ["mid"] * n_items
    for rank, iid in enumerate(ranked):
        if rank < head_cut:
            label[iid] = "head"
        elif rank >= tail_cut:
            label[iid] = "tail"
    return label


def build_graph(related: dict[int, list[int]], n_items: int) -> list[list[int]]:
    graph = defaultdict(set)
    for iid, neighbors in related.items():
        iid = int(iid)
        for nb in neighbors:
            graph[iid].add(nb)
            graph[nb].add(iid)
    return [list(graph.get(i, [])) for i in range(n_items)]


def augment(embedding: torch.Tensor, graph: list[list[int]], labels: list[str], alpha: float) -> torch.Tensor:
    n = embedding.size(0)
    neighbor_avg = embedding.clone()

    for iid in range(n):
        neighbors = graph[iid]
        if not neighbors:
            continue
        my_label = labels[iid]
        weights = torch.tensor(
            [EDGE_WEIGHT.get((my_label, labels[nb]), 1.0) for nb in neighbors],
            dtype=embedding.dtype,
        )
        weights = weights / weights.sum()
        vecs = embedding[neighbors]  # (num_neighbors, d)
        neighbor_avg[iid] = (weights.unsqueeze(1) * vecs).sum(dim=0)

    return (1 - alpha) * embedding + alpha * neighbor_avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_path", required=True, help="GRID 임베딩 .pt (N x embedding_dim)")
    p.add_argument("--related_path", required=True, help='{"iid": [이웃 iid, ...]} 형태 json')
    p.add_argument("--popularity_path", required=True, help='{"iid": 등장횟수} 형태 json')
    p.add_argument("--out_path", required=True)
    p.add_argument("--alpha", type=float, default=ALPHA)
    args = p.parse_args()

    embedding = torch.load(args.embedding_path)  # (N, embedding_dim), GRID와 동일 포맷
    assert embedding.dim() == 2, "GRID 임베딩 텐서는 2D (N x embedding_dim) 이어야 함"
    n_items = embedding.size(0)

    with open(args.related_path, "r", encoding="utf-8") as f:
        related = json.load(f)
    with open(args.popularity_path, "r", encoding="utf-8") as f:
        popularity = {int(k): v for k, v in json.load(f).items()}

    labels = classify_head_tail(popularity, n_items)
    graph = build_graph(related, n_items)
    augmented = augment(embedding, graph, labels, args.alpha)

    torch.save(augmented, args.out_path)  # GRID가 그대로 torch.load 해서 쓸 수 있는 동일 포맷
    print(f"saved: {args.out_path} shape={tuple(augmented.shape)}")


if __name__ == "__main__":
    main()
