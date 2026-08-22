"""
그래프 기반 SID 임베딩 증강 (G-SID 전처리)

baseline 텍스트 임베딩(embeddings/beauty_A/merged_predictions_tensor.pt)에
also_bought / also_viewed / bought_together 그래프 이웃 정보를 가중 평균으로 섞어
새 임베딩을 만든다. 행 인덱스 = item_id 포맷을 그대로 유지하므로, 이 출력을 그대로
code/server/sid_beauty.sh 에 EMB=.../graph_augmented_embedding_*.pt DIM=2048 로 넘기면
baseline과 완전히 동일한 조건으로 SID를 생성할 수 있다 (docs/HANDOFF_graph_sid.md 4절).

입력은 전부 이 레포에 이미 있는 파일이며, 서버 접속 없이 로컬에서 바로 돌아간다:
    embeddings/beauty_A/merged_predictions_tensor.pt   (12101 x 2048)
    csv_export/dataset/relations_edges_long.csv         (src_item_id, dst_item_id, relation)
    csv_export/dataset/A_interactions_long.csv          (user_id, position, item_id, split)

    python Tokenization/graph_sid_augment.py
    (인자 없이 기본 경로로 바로 실행됨. 결과는 embeddings/graph_A/ 에 저장)

--- 중심화(centering) 관련 메모 (2026-08-21) ---
이웃 가중평균 블렌딩 `aug = (1-a)*e_i + a*neighbor_avg(e)` 자체는 평균벡터 μ의 이동에
불변이다 (e_i = μ + r_i 로 쪼개면 μ 항의 계수가 (1-a)+a = 1 로 정확히 보존되므로,
블렌딩 전/후 μ가 그대로 남아 raw 코사인 유사도로는 그 효과가 거의 안 보인다).
실제로 효과가 갈리는 지점은 그 다음 단계다: GRID의 RQ-KMeans는 레이어마다
`normalize_residuals=True`로 L2 정규화를 하는데(rkmeans_train_flat.yaml), 이 정규화는
이동에 불변이 아니다. μ가 지배적인 상태에서 정규화하면 방향이 전부 μ/|μ| 근처로
뭉치고, 이것이 HANDOFF 3-2절이 잡아낸 "1레벨에서 에너지 83.8% 소실" anisotropy의
원인이다. 그래서 μ를 먼저 빼고(중심화) GRID에 넘겨야, 정규화 이후에도 실제 방향
차이(그래프 이웃 신호 포함)가 살아남는다. 그래서 이 스크립트는 실행 한 번에 비중심화·
중심화 두 버전을 항상 같이 저장한다 (아래 main() 참고, 둘 다 비교해야 판단 가능하므로).

--- 다른 결합 방식(λ 재배정 등)을 시도하는 팀원에게 ---
그래프를 "어떻게 구성했는지"(관계별 가중치, head/tail 가중치, 무방향 처리, 인기도 기준)와
"어떻게 중심화하는지"는 실험마다 달라지면 안 된다 — 그래야 결과 차이가 결합 방식(단순
블렌딩 vs λ vs 기타) 때문이라고 말할 수 있다. 그래서 이 두 부분은 prepare_foundation()과
save_centered_and_uncentered() 함수로 고정해 뒀다. **이 두 함수는 그대로 가져다 쓰고,
augment() 자리만 본인의 결합 로직으로 바꿔서 쓸 것.** 직접 그래프를 다시 만들거나
중심화를 다시 구현하지 말 것 — 미묘하게 달라지면 비교가 무효화된다.

    from Tokenization.graph_sid_augment import prepare_foundation, save_centered_and_uncentered

    embedding, graph, labels, mu, isolated = prepare_foundation()
    result = my_own_combination(embedding, graph, labels)   # ← 여기만 새로 짜면 됨
    save_centered_and_uncentered(result, mu, out_dir=..., tag="lambda_v1")
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

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


def sample_pairwise_cosine(x: torch.Tensor, n_samples: int = 20000, seed: int = 0) -> float:
    """무작위 (i, j) 쌍의 평균 코사인 유사도. anisotropy 진단용 (HANDOFF 3-2절과 같은 지표)."""
    g = torch.Generator().manual_seed(seed)
    n = x.size(0)
    i = torch.randint(0, n, (n_samples,), generator=g)
    j = torch.randint(0, n, (n_samples,), generator=g)
    keep = i != j
    return F.cosine_similarity(x[i[keep]], x[j[keep]], dim=1).mean().item()


def prepare_foundation(
    embedding_path: Path = ROOT / "embeddings/beauty_A/merged_predictions_tensor.pt",
    edges_path: Path = ROOT / "csv_export/dataset/relations_edges_long.csv",
    interactions_path: Path = ROOT / "csv_export/dataset/A_interactions_long.csv",
    popularity_split: str = "train",
):
    """모든 그래프 SID 변형(단순 블렌딩·λ 재배정·기타)이 공통으로 써야 하는 기초 재료.

    ⚠️ 그래프 구성(관계별 가중치, head/tail 가중치, 무방향 처리)과 중심화 기준(mu)이
    실험마다 다르면 "결합 방식 때문에 결과가 달라졌다"고 말할 수 없게 된다. 새 결합
    방식을 시도할 때도 이 함수를 그대로 재사용할 것 — 다시 구현하지 말 것.

    Returns:
        embedding: (n_items, D) baseline 텍스트 임베딩 (원본, 안 건드림)
        graph: item_id -> {neighbor_id: relation_weight} 무방향 그래프 (build_graph 참고)
        labels: item_id -> 'head'|'tail'|'mid' (인기도 기준, train split만 사용)
        mu: (D,) baseline 임베딩의 population 평균벡터. 결합 결과를 중심화할 때 이 mu를 뺄 것
            (본인이 새로 mu를 계산하지 말 것 — 실험마다 기준이 달라짐)
        isolated: 무방향 그래프에서 이웃이 하나도 없는 아이템 수
    """
    embedding = torch.load(embedding_path, map_location="cpu")
    assert embedding.dim() == 2, "임베딩 텐서는 2D (N x embedding_dim) 이어야 함"
    n_items = embedding.size(0)

    popularity = load_popularity(interactions_path, popularity_split)
    labels = classify_head_tail(popularity, n_items)
    graph, isolated = build_graph(edges_path, n_items)
    mu = embedding.mean(dim=0)

    return embedding, graph, labels, mu, isolated


def save_centered_and_uncentered(
    result: torch.Tensor, mu: torch.Tensor, out_dir: Path, tag: str
) -> dict:
    """결합 결과 하나를 비중심화·중심화 두 버전으로 저장하고 진단 코사인을 출력한다.

    ⚠️ 중심화를 직접 구현하지 말고 이 함수를 쓸 것 — prepare_foundation()이 준 mu를
    그대로 빼야 baseline과 같은 기준으로 중심화된다 (모듈 docstring의 anisotropy 메모 참고).

    Returns: {"": 비중심화_경로, "_centered": 중심화_경로}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = {"": result, "_centered": result - mu}
    paths = {}
    for suffix, tensor in variants.items():
        out_path = out_dir / f"graph_augmented_embedding_{tag}{suffix}.pt"
        torch.save(tensor, out_path)
        cos = sample_pairwise_cosine(tensor)
        print(f"saved: {out_path} shape={tuple(tensor.shape)}  pairwise_cos={cos:.4f}")
        paths[suffix] = out_path
    return paths


def augment(embedding: torch.Tensor, graph: list, labels: list, alpha: float) -> torch.Tensor:
    """[결합 방식 예시 1: 단순 alpha 블렌딩] 다른 결합 방식을 시도하려면 이 함수 대신
    똑같은 시그니처(embedding, graph, labels 받아서 (n_items, D) 텐서 반환)로 본인 함수를
    새로 만들 것 — prepare_foundation()/save_centered_and_uncentered()는 그대로 재사용."""
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
    p.add_argument("--out_dir", type=Path, default=ROOT / "embeddings/graph_A",
                    help="비중심화·중심화 두 파일을 함께 저장할 디렉터리")
    p.add_argument("--alpha", type=float, default=ALPHA)
    args = p.parse_args()

    embedding, graph, labels, mu, isolated = prepare_foundation(
        args.embedding_path, args.edges_path, args.interactions_path, args.popularity_split
    )
    n_items = embedding.size(0)

    print(f"items={n_items}  isolated={isolated} ({isolated/n_items:.2%})  alpha={args.alpha}")

    baseline_centered_cos = sample_pairwise_cosine(embedding - mu)
    print(f"mu_norm={mu.norm().item():.4f}  "
          f"raw_pairwise_cos(baseline)={sample_pairwise_cosine(embedding):.4f}  "
          f"centered_pairwise_cos(baseline)={baseline_centered_cos:.4f}")

    augmented = augment(embedding, graph, labels, args.alpha)
    assert augmented.shape == embedding.shape

    alpha_tag = f"a{round(args.alpha * 10):02d}"
    paths = save_centered_and_uncentered(augmented, mu, args.out_dir, alpha_tag)
    augmented_centered_cos = sample_pairwise_cosine(torch.load(paths["_centered"], map_location="cpu"))

    # 의미 있는 비교는 "중심화 상태에서" baseline vs augmented 다. (비중심화 vs 중심화는
    # mu 크기 차이 때문에 코사인이 원래 다르게 나와서 비교 대상이 아님)
    print(f"판정 기준(중심화 공간): baseline={baseline_centered_cos:.4f} vs "
          f"augmented(alpha={args.alpha})={augmented_centered_cos:.4f}  "
          f"→ augmented 값이 더 크면 그래프 블렌딩이 아이템들을 더 뭉치게(anisotropy 악화) 만든다는 뜻")


if __name__ == "__main__":
    main()
