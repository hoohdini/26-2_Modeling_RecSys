"""
CRAB 후처리 스크립트 (v2 - 논문 원본 방법 기반)

원 논문: "CRAB: Codebook Rebalancing for Bias Mitigation in Generative
Recommendation" (Chen et al.)의 4.1절 "Rebalancing the Codebook" (Eq.4~8)을
그대로 구현하되, 논문은 "학습된 모델의 코드북"에 사후 적용하는 반면 우리는
"SID를 막 생성한 직후(증강 단계)"에 적용한다. 우리 쪽이 더 유리한 이유는
README 상단 주석 참고.

바뀐 점 (v1 대비):
  - 개별 아이템을 다시 K-means(k=2)하던 방식 -> "자식 토큰 단위"로 재분배
    (자식 토큰과 그 안의 아이템은 절대 쪼개지 않고 통째로 옮김: 논문의 하드 제약)
  - 분할 개수가 고정 2개 -> 토큰별 상대 인기도에 따라 가변 M (최대 3, 논문 그대로)
  - 단순 거리 기반 재군집화 -> 거리 + 균형 손실(balance loss)을 함께 최소화하는
    정규화 K-means (논문 Eq.6)
  - a단(첫 자리) 고정 -> 어느 층을 쪼갤지 선택 가능 (--split_level).
    논문은 중간 층(b단)을 쪼갤 때 효과가 가장 컸다고 보고함("Hourglass 현상")

주의 (근사 부분):
  - 논문의 residual은 GRID가 실제로 학습한 코드북 중심점 기준인데, 우리는 그
    중심점에 직접 접근하지 않고 "그 코드에 배정된 아이템들의 평균 벡터"로
    근사한다. K-means 수렴 시 중심점 = 배정된 점들의 평균이므로 근거 있는
    근사지만, GRID가 학습을 끝까지 수렴시키지 못했다면 차이가 날 수 있음.
  - 논문은 정규화 K-means를 Raymaekers & Zamar(2022)의 프레임워크로 정확히
    풀지만, 여기서는 좌표하강(assignment ↔ centroid update를 번갈아 반복)으로
    근사한다. 실무적으로 충분히 비슷한 결과를 주지만 전역 최적은 보장 못 함.

    python crab_split.py \
        --codes_path      GRID가_만든_SID.pt \
        --embedding_path  원본_또는_Graph_임베딩.pt \
        --popularity_path popularity.json \
        --split_level     1 \
        --out_path        crab_sid.pt
"""

import argparse
import json
from collections import defaultdict

import torch
import numpy as np

SPLIT_RATIO = 0.10      # 상위 몇 %의 토큰을 분할 대상으로 볼지 (논문 Fig.3 기본값)
MAX_NEW_TOKENS = 3      # 토큰 하나를 최대 몇 개로 쪼갤지 (논문 그대로: M <= 3)
LAMBDA_BALANCE = 0.2    # 균형 손실 가중치 (논문 Eq.10의 gamma와 동일 역할, 기본값도 논문값)
N_ITERS = 15            # 정규화 K-means 반복 횟수


def compute_token_popularity(level_codes: torch.Tensor, popularity: dict[int, float]) -> dict[int, float]:
    """레벨 하나(예: b단)에서, 토큰별로 그 토큰을 가진 아이템들의 인기도 합을 구함. 논문 Eq.3."""
    mass = defaultdict(float)
    for iid in range(level_codes.size(0)):
        mass[int(level_codes[iid].item())] += popularity.get(iid, 0)
    return dict(mass)


def approximate_residual(codes: torch.Tensor, embedding: torch.Tensor, split_level: int) -> torch.Tensor:
    """
    split_level 층에 들어가기 직전의 잔차를 근사한다.
    0층부터 split_level-1층까지, "그 코드를 가진 아이템들의 평균 벡터"를 그 층의
    중심점으로 근사하고 순서대로 빼나간다. (RQ의 정의 그대로, 중심점만 근사치)
    """
    residual = embedding.clone().float()
    for level in range(split_level):
        level_codes = codes[:, level]
        for code_val in level_codes.unique().tolist():
            member_idx = torch.where(level_codes == code_val)[0]
            centroid = residual[member_idx].mean(dim=0)
            residual[member_idx] -= centroid
    return residual


def determine_M(token_mass: float, avg_mass_at_level: float, max_m: int = MAX_NEW_TOKENS) -> int:
    """논문: "M은 대상 토큰의 빈도와 같은 층 평균 빈도의 비율로 정해지고, 상한은 3" """
    ratio = token_mass / avg_mass_at_level if avg_mass_at_level > 0 else 1.0
    m = max(2, round(ratio))
    return min(max_m, m)


def regularized_kmeans_split(
    child_residuals: np.ndarray,   # (num_children, d) - 자식 토큰별 평균 잔차 (r̄_j)
    child_weights: np.ndarray,     # (num_children,)   - 자식 토큰에 속한 아이템 수 (n_j)
    child_mass: np.ndarray,        # (num_children,)   - 자식 토큰의 인기 질량 (P(c_j))
    m: int,
    lam: float = LAMBDA_BALANCE,
    n_iters: int = N_ITERS,
    seed: int = 42,
) -> np.ndarray:
    """
    논문 Eq.6을 좌표하강으로 근사해서 푼다.
    목적함수: sum_j n_j * ||r̄_j - mu_{assign(j)}||^2  +  lam * sum_m (P(m) - P_avg)^2
    -> 거리 항(공간적으로 잘 뭉치게)과 균형 항(새 토큰들의 인기 질량이 비슷해지게)을
       동시에 고려해서 자식 토큰들을 M개의 새 부모에 배정한다.
    """
    rng = np.random.default_rng(seed)
    num_children = child_residuals.shape[0]
    if num_children <= m:
        # 자식 토큰 수가 나눌 개수보다 적으면 그냥 1개씩 배정
        return np.arange(num_children) % m

    # 초기화: 가중 k-means++ 스타일로 대충 흩뿌림 (거리 기준으로만 1차 배정)
    init_idx = rng.choice(num_children, size=m, replace=False)
    centroids = child_residuals[init_idx].copy()
    assign = np.zeros(num_children, dtype=int)

    p_avg = child_mass.sum() / m

    for _ in range(n_iters):
        # --- Assignment step: 거리 + 균형손실의 한계 변화량을 함께 고려 ---
        cluster_mass = np.zeros(m)
        for c in range(m):
            cluster_mass[c] = child_mass[assign == c].sum()

        new_assign = assign.copy()
        for j in range(num_children):
            cur_c = assign[j]
            best_c, best_cost = cur_c, np.inf
            for c in range(m):
                dist_term = child_weights[j] * np.sum((child_residuals[j] - centroids[c]) ** 2)
                # j를 c로 옮겼을 때(원래 c 소속이 아니라면) 균형손실이 얼마나 바뀌는지
                mass_c_after = cluster_mass[c] + (0 if c == cur_c else child_mass[j])
                mass_from_after = cluster_mass[cur_c] - (0 if c == cur_c else child_mass[j])
                bal_term = (mass_c_after - p_avg) ** 2 + (mass_from_after - p_avg) ** 2
                cost = dist_term + lam * bal_term
                if cost < best_cost:
                    best_cost, best_c = cost, c
            new_assign[j] = best_c

        if np.array_equal(new_assign, assign):
            break
        assign = new_assign

        # --- Update step: 각 클러스터 중심을 가중 평균으로 갱신 ---
        for c in range(m):
            members = assign == c
            if members.any():
                w = child_weights[members]
                centroids[c] = np.average(child_residuals[members], axis=0, weights=w)

    return assign


def apply_crab_rebalance(
    codes: torch.Tensor, embedding: torch.Tensor, popularity: dict[int, float],
    split_level: int, codebook_width: int,
    split_ratio: float = SPLIT_RATIO, lam: float = LAMBDA_BALANCE,
) -> torch.Tensor:
    """split_level 층의 과인기 토큰들을, 그 자식(split_level+1 층) 토큰 단위로 재분배한다."""
    codes = codes.clone()
    num_levels = codes.size(1)
    has_children = split_level + 1 < num_levels

    level_codes = codes[:, split_level]
    token_mass = compute_token_popularity(level_codes, popularity)
    ranked = sorted(token_mass.items(), key=lambda kv: -kv[1])
    n_targets = max(1, int(len(ranked) * split_ratio))
    targets = [t for t, _ in ranked[:n_targets]]
    avg_mass_at_level = sum(token_mass.values()) / max(len(token_mass), 1)

    residual = approximate_residual(codes, embedding, split_level)
    next_new_code = codebook_width

    for token in targets:
        member_idx = torch.where(level_codes == token)[0]
        if member_idx.numel() < 2:
            continue

        if has_children:
            # 자식 토큰(split_level+1) 단위로 그룹핑 -> 자식 통째로만 이동시킴
            child_vals = codes[member_idx, split_level + 1]
            groups: dict[int, torch.Tensor] = {}
            for cv in child_vals.unique().tolist():
                groups[cv] = member_idx[child_vals == cv]
        else:
            # 마지막 층이라 자식이 없으면 아이템 각각을 "자식 1개짜리 그룹"으로 취급
            groups = {int(i): torch.tensor([i]) for i in member_idx.tolist()}

        child_ids = list(groups.keys())
        child_residuals = np.stack([residual[groups[c]].mean(dim=0).numpy() for c in child_ids])
        child_weights = np.array([groups[c].numel() for c in child_ids], dtype=np.float32)
        child_mass = np.array([sum(popularity.get(int(i), 0) for i in groups[c]) for c in child_ids])

        m = determine_M(token_mass[token], avg_mass_at_level)
        assign = regularized_kmeans_split(child_residuals, child_weights, child_mass, m, lam)

        # 클러스터 0은 원래 토큰 번호를 유지, 나머지는 새 번호
        cluster_to_code = {0: token}
        for c in range(1, m):
            cluster_to_code[c] = next_new_code
            next_new_code += 1

        for local_i, c in enumerate(child_ids):
            new_code = cluster_to_code[assign[local_i]]
            codes[groups[c], split_level] = new_code

        total = sum(popularity.values()) or 1
        summary = ", ".join(
            f"{cluster_to_code[c]}:{sum(child_mass[assign == c]) / total:.1%}" for c in range(m)
        )
        print(f"  split level{split_level}={token} ({len(child_ids)}개 자식 토큰, M={m}) -> {summary}")

    return codes


def deduplicate_rows_in_tensor(data: torch.Tensor) -> torch.Tensor:
    """GRID의 src.utils.tensor_utils.deduplicate_rows_in_tensor 와 동일한 알고리즘.
    구분자는 0부터 시작 (팀 합의 규칙, docs/HANDOFF_graph_sid.md 5-1절)."""
    unique_rows, inverse_indices, counts = torch.unique(data, dim=0, return_inverse=True, return_counts=True)
    output_indices = torch.zeros_like(inverse_indices)
    duplicate_indices = torch.where(counts > 1)[0]
    for i in range(len(duplicate_indices)):
        num_of_collisions = counts[duplicate_indices[i]]
        indices_to_change = torch.where(inverse_indices == duplicate_indices[i])[0]
        range_to_add = torch.arange(0, num_of_collisions)
        output_indices = output_indices.scatter(0, indices_to_change, range_to_add)
    return torch.cat((data, output_indices.unsqueeze(1)), dim=1).long()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codes_path", required=True, help="GRID SID .pt (N x num_hierarchies+1, dedup열 포함)")
    p.add_argument("--embedding_path", required=True, help="원본(또는 G-SID) 임베딩 .pt")
    p.add_argument("--popularity_path", required=True)
    p.add_argument("--codebook_width", type=int, default=256)
    p.add_argument("--split_level", type=int, required=True,
                    help="어느 층을 쪼갤지 (0=a단, 1=b단...). 논문은 중간 층에서 효과가 가장 컸음")
    p.add_argument("--split_ratio", type=float, default=SPLIT_RATIO)
    p.add_argument("--lam", type=float, default=LAMBDA_BALANCE)
    p.add_argument("--out_path", required=True)
    args = p.parse_args()

    codes = torch.load(args.codes_path)
    embedding = torch.load(args.embedding_path)
    with open(args.popularity_path, "r", encoding="utf-8") as f:
        popularity = {int(k): v for k, v in json.load(f).items()}

    raw_codes = codes[:, :-1]  # dedup열 떼고 작업
    new_codes = apply_crab_rebalance(
        raw_codes, embedding, popularity,
        args.split_level, args.codebook_width, args.split_ratio, args.lam,
    )

    final = deduplicate_rows_in_tensor(new_codes)
    torch.save(final, args.out_path)
    print(f"saved: {args.out_path} shape={tuple(final.shape)}")


if __name__ == "__main__":
    main()
