"""직교주입 튜닝식 임베딩 생성기 — β₀ 축 스윕용 (τ 없음 · κ 0)

csv_export/sid/gstar_noTau_k0_b05/README.md 의 수식을 그대로 구현한다:

    x̃ᵢ = xᵢ − μ
    mᵢ = Σ_{j∈N(i)} wᵢⱼ x̃ⱼ / Σ wᵢⱼ          (간선 전부 · 나혜님 가중치)
    zᵢ = x̃ᵢ + β₀‖x̃ᵢ‖ · unit( mᵢ − ⟨mᵢ, x̂ᵢ⟩ x̂ᵢ )

그래프 구성·head/tail 가중치·중심화 기준(μ)은 graph_sid_augment.prepare_foundation()
을 그대로 재사용한다 (모듈 docstring의 고정 규칙 — 직접 재구현 금지).
고립 노드(무방향 이웃 0개, 142개)는 x̃ᵢ 원본 유지.

검증 모드(--verify)는 기존 산출물과 비트 일치를 확인한다:
    β₀=0.5 → work/minchan/T1_noTau_k0_b05.pt
    β₀=1.0 → work/minchan/noTau_k0_b10.pt
일치가 확인된 뒤에만 새 β₀ 값을 생성할 것.

서버 사용 예 (마스터 노드, work/minchan 에서):
    PY=/data1/dsl05/miniconda3/envs/grid/bin/python
    $PY gen_gstar_beta.py --beta 0.5 --verify T1_noTau_k0_b05.pt
    $PY gen_gstar_beta.py --beta 1.0 --verify noTau_k0_b10.pt
    $PY gen_gstar_beta.py --beta 0.7 --out noTau_k0_b07.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_sid_augment import prepare_foundation  # noqa: E402


def orthogonal_inject(embedding: torch.Tensor, graph: list, mu: torch.Tensor,
                      beta0: float) -> torch.Tensor:
    centered = embedding - mu
    out = centered.clone()
    eps = 1e-12

    for iid in range(centered.size(0)):
        neighbors = graph[iid]
        if not neighbors:
            continue  # 고립 노드: x̃ᵢ 그대로
        nb_ids = list(neighbors.keys())
        weights = torch.tensor([neighbors[nb] for nb in nb_ids], dtype=centered.dtype)
        m = (weights.unsqueeze(1) * centered[nb_ids]).sum(dim=0) / weights.sum()

        x = centered[iid]
        x_norm = x.norm()
        if x_norm < eps:
            continue
        x_hat = x / x_norm
        perp = m - (m @ x_hat) * x_hat
        perp_norm = perp.norm()
        if perp_norm < eps:
            continue  # 이웃 평균이 자기 방향과 평행 → 주입할 새 정보 없음
        out[iid] = x + beta0 * x_norm * (perp / perp_norm)

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--embedding_path", type=Path,
                   default=Path("../../embeddings/beauty_A/merged_predictions_tensor.pt"))
    p.add_argument("--edges_path", type=Path,
                   default=Path("../../csv_export/dataset/relations_edges_long.csv"))
    p.add_argument("--interactions_path", type=Path,
                   default=Path("../../csv_export/dataset/A_interactions_long.csv"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--verify", type=Path, default=None,
                   help="기존 산출물 경로. 지정 시 저장 대신 일치 여부만 확인")
    args = p.parse_args()

    embedding, graph, labels, mu, isolated = prepare_foundation(
        args.embedding_path, args.edges_path, args.interactions_path, "train")
    print(f"items={embedding.size(0)}  isolated={isolated}  beta0={args.beta}")

    # 나혜님 가중치 = 관계 가중치(build_graph) × head/tail 곱 — augment()와 동일하게 적용
    from graph_sid_augment import EDGE_WEIGHT
    weighted_graph = []
    for iid, neighbors in enumerate(graph):
        my_label = labels[iid]
        weighted_graph.append({
            nb: w * EDGE_WEIGHT.get((my_label, labels[nb]), 1.0)
            for nb, w in neighbors.items()
        })

    z = orthogonal_inject(embedding, weighted_graph, mu, args.beta)

    if args.verify is not None:
        ref = torch.load(args.verify, map_location="cpu")
        exact = torch.equal(z, ref)
        close = torch.allclose(z, ref, atol=1e-5)
        max_diff = (z - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(z.flatten(), ref.flatten(), dim=0).item()
        print(f"verify vs {args.verify}: exact={exact} allclose={close} "
              f"max_diff={max_diff:.3e} flat_cos={cos:.6f}")
        sys.exit(0 if close else 1)

    out = args.out or Path(f"noTau_k0_b{int(round(args.beta * 10)):02d}.pt")
    torch.save(z, out)
    print(f"saved: {out} shape={tuple(z.shape)}")


if __name__ == "__main__":
    main()
