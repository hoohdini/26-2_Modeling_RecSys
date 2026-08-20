"""
그래프 기반 SID 임베딩 증강 (G-SID 전처리)

baseline 텍스트 임베딩(embeddings/beauty_A/merged_predictions_tensor.pt)에
also_bought / also_viewed / bought_together 그래프 이웃 정보를 가중 평균으로 섞어
새 임베딩을 만든다. 행 인덱스 = item_id 포맷을 그대로 유지하므로, 이 출력을 그대로
code/server/sid_beauty.sh 에 EMB=.../graph_augmented_embedding.pt DIM=2048 로 넘기면
baseline과 완전히 동일한 조건으로 SID를 생성할 수 있다 (docs/HANDOFF_graph_sid.md 4절).

입력은 전부 이 레포에 이미 있는 파일이며, 서버 접속 없이 로컬에서 바로 돌아간다:
    embeddings/beauty_A/merged_predictions_tensor.pt   (12101 x 2048)
    csv_export/dataset/relations_edges_long.csv         (src_item_id, dst_item_id, relation)
    csv_export/dataset/A_interactions_long.csv          (user_id, position, item_id, split)

    python Tokenization/graph_sid_augment.py
    (인자 없이 기본 경로로 바로 실행됨. 결과는 embeddings/graph_A/ 에 저장)
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent

ALPHA = 0.3        # 최종 임베딩 = (1-ALPHA)*원본 + ALPHA*이웃 가중평균
HEAD_RATIO = 0.2   # 인기도 상위/하위 20%를 head/tail로 분류, 나머지는 mid
EDGE_WEIGHT = {     # (내 label, 이웃 label) -> 가중치. head끼리는 신호가 흔해서 낮추고,
    ("head", "head"): 0.5,   # tail이 head/tail 이웃을 만나는 신호는 희소해서 높인다.
    ("head", "tail"): 1.5,
    ("tail", "head"): 1.5,
    ("tail", "tail"): 1.0,
}
RELATION_WEIGHT = {   # 관계 종류별 신뢰도: 실제 동시구매 > 함께구매이력 > 함께조회
    "bought_together": 2.0,
    "also_bought": 1.0,
    "also_viewed": 0.6,
}


def load_popularity(interactions_path: Path, split: str) -> dict:
    """item_id -> 등장 횟수. train split만 쓰는 게 기본값 (val/test 유출 방지)."""
    df = pd.read_csv(interactions_path)
    if split:
        df = df[df["split"] == split]
    return df["item_id"].value_counts().to_dict()


def classify_head_tail(popularity: dict, n_items: int) -> list:
    """item_id -> 'head' | 'tail' | 'mid'. item_id는 0..n_items-1 로 정렬돼있다는 전제."""
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


def build_graph(edges_path: Path, n_items: int):
    """item_id -> {neighbor_id: relation_weight} 무방향 그래프. 고립 노드 수도 함께 반환."""
    df = pd.read_csv(edges_path).drop_duplicates(subset=["src_item_id", "dst_item_id", "relation"])
    graph = [dict() for _ in range(n_items)]

    for src, dst, relation in df.itertuples(index=False):
        w = RELATION_WEIGHT.get(relation, 1.0)
        for a, b in ((src, dst), (dst, src)):  # 방향성 무시하고 상호 연결로 취급
            graph[a][b] = graph[a].get(b, 0.0) + w

    isolated = sum(1 for nb in graph if not nb)
    return graph, isolated


def augment(embedding: torch.Tensor, graph: list, labels: list, alpha: float) -> torch.Tensor:
    n = embedding.size(0)
    neighbor_avg = embedding.clone()

    for iid in range(n):
        neighbors = graph[iid]
        if not neighbors:
            continue  # 고립 노드: 원본 임베딩 그대로 유지하는 게 fallback.
            # 핸드오프 문서(1-2절)의 198개는 방향 그래프(src 기준) 수치이고,
            # 이 함수는 무방향으로 취급해서(위 build_graph) dst로만 등장하는
            # 56개가 추가로 연결돼 실제로는 142개만 고립 노드로 처리된다.
        my_label = labels[iid]
        nb_ids = list(neighbors.keys())
        weights = torch.tensor(
            [neighbors[nb] * EDGE_WEIGHT.get((my_label, labels[nb]), 1.0) for nb in nb_ids],
            dtype=embedding.dtype,
        )
        weights = weights / weights.sum()
        vecs = embedding[nb_ids]
        neighbor_avg[iid] = (weights.unsqueeze(1) * vecs).sum(dim=0)

    return (1 - alpha) * embedding + alpha * neighbor_avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_path", type=Path,
                    default=ROOT / "embeddings/beauty_A/merged_predictions_tensor.pt")
    p.add_argument("--edges_path", type=Path,
                    default=ROOT / "csv_export/dataset/relations_edges_long.csv")
    p.add_argument("--interactions_path", type=Path,
                    default=ROOT / "csv_export/dataset/A_interactions_long.csv")
    p.add_argument("--popularity_split", default="train",
                    help="인기도 집계에 쓸 split. 빈 문자열이면 전체 사용")
    p.add_argument("--out_path", type=Path,
                    default=ROOT / "embeddings/graph_A/graph_augmented_embedding.pt")
    p.add_argument("--alpha", type=float, default=ALPHA)
    args = p.parse_args()

    embedding = torch.load(args.embedding_path, map_location="cpu")
    assert embedding.dim() == 2, "임베딩 텐서는 2D (N x embedding_dim) 이어야 함"
    n_items = embedding.size(0)

    popularity = load_popularity(args.interactions_path, args.popularity_split)
    labels = classify_head_tail(popularity, n_items)
    graph, isolated = build_graph(args.edges_path, n_items)

    print(f"items={n_items}  isolated={isolated} ({isolated/n_items:.2%})  alpha={args.alpha}")

    augmented = augment(embedding, graph, labels, args.alpha)
    assert augmented.shape == embedding.shape

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(augmented, args.out_path)
    print(f"saved: {args.out_path} shape={tuple(augmented.shape)}")


if __name__ == "__main__":
    main()
