"""
CRAB-SID: 생성형 추천 코드북 리밸런싱 (Stage 1)

CRAB 논문(arXiv 2604.05113) Section 4.1의 "Rebalancing the Codebook"을 재구현한다.
G-SID(또는 baseline text SID)로 이미 만들어진 SID 위에 후처리(post-hoc)로 적용해서,
과인기(over-popular) 토큰을 여러 개로 쪼개 롱테일 아이템의 노출 기회를 늘린다.

  [입력]  embeddings/*.pt  (n_items x D)   +  sid/*/sid_tensor.pt  (n_items x L)
  [출력]  새 SID 텐서 (n_items x L)  +  분할 이력 JSON

--------------------------------------------------------------------------
전제와 설계 결정 (팀 합의 사항 반영)
--------------------------------------------------------------------------
1. RQ-KMeans 전제 (트리 구조 유지)
   팀 실험은 GRID 권장 방식인 RK-Means를 쓰기로 확정됨. 따라서 자식 토큰이 부모를
   하나만 갖는 트리 구조가 성립하고, 논문 수식 5(단순 버전)를 쓴다.
   (RQ-VAE라면 자식이 여러 부모를 가질 수 있어 수식 8의 조건부 빈도가 필요하지만,
    우리 실험 매트릭스에서는 해당 없음)

2. 인기도 정의 = Tokenization/graph_sid_augment.py 의 load_popularity() 재사용
   train split 상호작용 횟수만 사용(val/test 유출 방지). G-SID와 완전히 동일한
   기준을 써야 두 처방의 비교가 유효하다. ★ 절대 여기서 새로 정의하지 말 것.

3. Head-only 스코프 (팀 충돌 조율 규칙)
   CRAB은 인기 토큰을 "분산"시키고 G-SID는 그래프 이웃을 "응집"시켜 방향이 반대다.
   G-SID가 head-tail 엣지를 1.5배로 증폭해 어렵게 만든 연결을 CRAB이 도로 깨면
   안 되므로, 분할 대상을 head 아이템 위주 토큰으로 제한한다 (--head-only).

4. 잔차(residual) 재구성
   GRID가 코드북/중심점을 저장해주지 않으므로, 임베딩 + SID 배정으로부터
   레벨별 중심점과 잔차를 되짚어 계산한다 (k-means 중심점 = 배정된 점들의 평균).
   ⚠️ GRID는 rkmeans_train_flat.yaml 에서 normalize_residuals=True 로 레이어마다
   L2 정규화를 한다. --normalize-residuals 플래그로 이를 맞출 수 있게 해뒀다.
   완전히 동일한 재현은 아니므로, 재구성 품질은 실행 시 출력되는 진단값으로 확인할 것.

--------------------------------------------------------------------------
사용 예
--------------------------------------------------------------------------
    python Tokenization/crab_sid.py \
        --embedding-path embeddings/graph_A/graph_augmented_embedding_a03_centered.pt \
        --sid-path sid/gsid_a03/L4/sid_tensor.pt \
        --out-dir sid/crab_gsid_a03/L4

    # 논문 방식(가변 M)으로 돌려서 비교하고 싶을 때
    python Tokenization/crab_sid.py ... --m-mode adaptive
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

# graph_sid_augment.py 의 인기도/head-tail 정의를 그대로 재사용한다.
# (경로 문제로 import가 안 되면 아래 fallback이 동작하지만, 가능하면 import 쪽을 쓸 것)
try:
    from Tokenization.graph_sid_augment import load_popularity, classify_head_tail
except ImportError:  # 단독 실행 대비 fallback
    import pandas as pd

    HEAD_RATIO = 0.2

    def load_popularity(interactions_path, split="train"):
        df = pd.read_csv(interactions_path)
        if split:
            df = df[df["split"] == split]
        return df["item_id"].value_counts().to_dict()

    def classify_head_tail(popularity, n_items):
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


# =====================================================================
# 하이퍼파라미터 (★ 팀에서 바꿔가며 실험할 값들)
# =====================================================================

SPLIT_RATIO = 0.10   # 각 레벨에서 인기 상위 몇 %를 분할 대상으로 삼을지.
                     # 논문 5.1절 구현값 = 10%. Figure 3에서 5~20% 스윕함.
                     # ★ 팀 합의: 10%로 시작

M_FIXED = 3          # ★ 팀 합의: 일단 M=3 고정으로 진행
                     # 논문 원문은 "대상 토큰 빈도 / 같은 레벨 평균 빈도" 비율로
                     # M을 정하고 상한만 3으로 뒀다(가변). --m-mode adaptive 로 전환 가능.
                     # 나중에 두 방식 비교 실험을 돌릴 수 있게 둘 다 구현해 둠.

M_CAP = 3            # M 상한 (논문 M <= 3). adaptive 모드에서만 의미 있음.

LAMBDA_BAL = 1.0     # 수식 6의 λ — 거리 보존 vs 인기도 균형의 트레이드오프 강도.
                     # ⚠️ 논문에 값이 명시돼 있지 않아 우리가 직접 스윕해야 하는 값.
                     # ⚠️ G-SID의 λ₁/λ₂(그래프 보너스)와는 완전히 다른 파라미터다.
                     #    이름을 lambda_bal 로 구분해 둔 이유.

TAIL_GUARD = 0.5     # 토큰 안에 tail 아이템이 이 비율 이상이면 분할 대상에서 제외.
                     # G-SID가 그래프로 일부러 묶어둔 head-tail 뭉침을 보호하는 장치.
                     # 0.0 으로 두면 가드 해제(= 논문 원본과 동일하게 전체 대상).

GU_DIAGNOSIS_RATIO = 0.05  # 논문 3절의 GU 진단용 상위 5% — SPLIT_RATIO와 혼동 금지.
                           # 이 스크립트에서는 진단 출력에만 쓴다.


# =====================================================================
# Step 0. 잔차 재구성 (GRID가 코드북을 저장해주지 않아 필요)
# =====================================================================

def reconstruct_centroids_and_residuals(embedding, sid, normalize_residuals=False):
    """임베딩 + SID 배정으로부터 레벨별 중심점과 잔차를 되짚어 계산한다.

    RQ-KMeans의 중심점은 정의상 "그 클러스터에 배정된 벡터들의 평균"이므로
    (CRAB 논문 2.1절), 배정 결과만 알면 역산할 수 있다.

    Returns:
        residuals: list of (n_items, D) — residuals[l] = 레벨 l 진입 시점의 잔차 r^l
        centroids: list of dict {token_id: (D,) 중심점}
    """
    n_items, n_levels = sid.shape
    residuals, centroids = [], []

    r = embedding.clone().float()          # r^1 = e  (논문 수식 1)
    for level in range(n_levels):
        if normalize_residuals:
            # GRID의 normalize_residuals=True 재현. 이동 불변이 아니므로
            # 중심화된 임베딩을 입력으로 넣는 게 전제다(graph_sid_augment 메모 참고).
            r = torch.nn.functional.normalize(r, dim=1)

        residuals.append(r.clone())

        # 이 레벨의 중심점 = 같은 토큰을 가진 아이템들의 잔차 평균
        cent = {}
        tokens = sid[:, level]
        for tok in torch.unique(tokens).tolist():
            mask = tokens == tok
            cent[tok] = r[mask].mean(dim=0)
        centroids.append(cent)

        # 다음 레벨 잔차: r^{l+1} = r^l - c^l   (논문 수식 1)
        cent_matrix = torch.stack([cent[int(t)] for t in tokens])
        r = r - cent_matrix

    return residuals, centroids


# =====================================================================
# Step A. 토큰 인기도 (논문 수식 3)
# =====================================================================

def token_popularity(sid, level, item_freq):
    """P(c_k^l) = sum of item_freq over items whose l-th token is c_k^l"""
    pop = defaultdict(float)
    for iid, tok in enumerate(sid[:, level].tolist()):
        pop[tok] += item_freq.get(iid, 0)
    return dict(pop)


def token_items(sid, level):
    """token_id -> [item_id, ...]"""
    members = defaultdict(list)
    for iid, tok in enumerate(sid[:, level].tolist()):
        members[tok].append(iid)
    return dict(members)


# =====================================================================
# Step B. 분할 대상 토큰 선정 (+ Head-only / Tail-guard 필터)
# =====================================================================

def select_tokens_to_split(pop, members, labels, split_ratio,
                           head_only=True, tail_guard=TAIL_GUARD):
    """인기도 상위 split_ratio 비율의 토큰을 고르되, 팀 규칙에 따라 필터링한다."""
    ranked = sorted(pop.items(), key=lambda kv: -kv[1])
    n_target = max(1, int(len(ranked) * split_ratio))

    selected, skipped = [], []
    for tok, _ in ranked:
        if len(selected) >= n_target:
            break
        items = members[tok]
        head_ratio = sum(labels[i] == "head" for i in items) / len(items)
        tail_ratio = sum(labels[i] == "tail" for i in items) / len(items)

        # G-SID가 그래프로 묶어둔 head-tail 혼합 토큰은 건드리지 않는다
        if tail_guard > 0 and tail_ratio >= tail_guard:
            skipped.append((tok, "tail_guard", round(tail_ratio, 3)))
            continue
        # head 아이템이 하나도 없는 토큰은 애초에 "과인기"로 볼 이유가 약하다
        if head_only and head_ratio == 0.0:
            skipped.append((tok, "no_head", 0.0))
            continue
        selected.append(tok)

    return selected, skipped


# =====================================================================
# Step C. 토큰별 분할 개수 M
# =====================================================================

def decide_m(token_pop, avg_pop, mode="fixed"):
    """★ 팀 합의: 기본은 fixed(M=3). adaptive는 논문 원문 방식(가변, 상한 3)."""
    if mode == "fixed":
        return M_FIXED
    if mode == "adaptive":
        return max(1, min(M_CAP, round(token_pop / avg_pop))) if avg_pop > 0 else 1
    raise ValueError(f"unknown m_mode: {mode}")


# =====================================================================
# Step D~E. 정규화 K-means로 자식 토큰을 M개 새 부모에 재배분 (논문 수식 5, 6)
# =====================================================================

def split_one_token(children_stats, M, lambda_bal, max_iter=20, seed=0):
    """과인기 토큰 하나를 M개로 쪼갠다.

    children_stats: {child_token: {"r_bar": (D,), "n": int, "pop": float}}
        r_bar = 그 자식에 속한 아이템들의 (부모 레벨에서의) 평균 잔차
        n     = 아이템 개수      → 수식 6의 거리항 가중치 n_j
        pop   = 인기도 총합      → 수식 5의 균형항 재료 P(c_j^{l+1})
        ⚠️ n 과 pop 은 서로 다른 값이다(개수 vs 소비 빈도). 헷갈리기 쉬운 지점.

    수식 6은 "거리항(자식별 독립) + 균형항(전체 배정에 의존)"이 섞인 결합 최적화라
    표준 Lloyd 알고리즘(각자 가장 가까운 중심에 배정)만으로는 균형항을 못 줄인다.
    그래서 할당 단계를 인기도 내림차순 그리디(bin-packing 스타일)로 구현했다.

    Returns: {child_token: new_slot_index}
    """
    kids = list(children_stats.keys())
    if len(kids) <= M:
        # 자식 수가 M 이하면 쪼갤 여지가 없다 (자식 하나당 새 토큰 하나)
        return {c: i for i, c in enumerate(kids)}

    R = torch.stack([children_stats[c]["r_bar"] for c in kids])          # (K, D)
    n = torch.tensor([children_stats[c]["n"] for c in kids], dtype=torch.float)
    pop = torch.tensor([children_stats[c]["pop"] for c in kids], dtype=torch.float)
    target = pop.sum().item() / M      # 수식 5의 P-bar (새 토큰 하나당 목표 인기도)

    # --- 초기화: k-means++ (인기도 큰 자식이 시드가 되도록 가중) ---
    g = torch.Generator().manual_seed(seed)
    first = int(torch.argmax(pop))
    centers = [R[first]]
    for _ in range(M - 1):
        d = torch.stack([((R - c) ** 2).sum(dim=1) for c in centers]).min(dim=0).values
        probs = d / (d.sum() + 1e-12)
        centers.append(R[int(torch.multinomial(probs, 1, generator=g))])
    centers = torch.stack(centers)                                        # (M, D)

    assign = {}
    for _ in range(max_iter):
        # --- 할당 단계: 거리 + λ*(균형 악화분) 최소화, 인기도 큰 자식부터 ---
        slot_pop = [0.0] * M
        new_assign = {}
        for idx in torch.argsort(pop, descending=True).tolist():
            best_m, best_cost = 0, float("inf")
            for m in range(M):
                dist = n[idx].item() * ((R[idx] - centers[m]) ** 2).sum().item()
                projected = slot_pop[m] + pop[idx].item()
                balance = lambda_bal * (projected - target) ** 2
                cost = dist + balance
                if cost < best_cost:
                    best_m, best_cost = m, cost
            new_assign[kids[idx]] = best_m
            slot_pop[best_m] += pop[idx].item()

        if new_assign == assign:
            break                       # 수렴
        assign = new_assign

        # --- 갱신 단계: 새 중심 = 배정된 자식들의 n-가중 평균 ---
        for m in range(M):
            idxs = [i for i, c in enumerate(kids) if assign[c] == m]
            if idxs:
                w = n[idxs].unsqueeze(1)
                centers[m] = (R[idxs] * w).sum(dim=0) / w.sum()

    return assign


# =====================================================================
# Step F~G. 전체 레벨에 적용하고 SID를 재라벨링
# =====================================================================

def run_crab(embedding, sid, item_freq, labels,
             split_ratio=SPLIT_RATIO, m_mode="fixed", lambda_bal=LAMBDA_BAL,
             head_only=True, tail_guard=TAIL_GUARD,
             levels=None, normalize_residuals=False, verbose=True):
    """CRAB Stage 1 전체 실행. 새 SID 텐서와 분할 이력을 반환한다."""
    n_items, n_levels = sid.shape
    new_sid = sid.clone()
    history = []

    residuals, _ = reconstruct_centroids_and_residuals(
        embedding, sid, normalize_residuals=normalize_residuals)

    # 마지막 레벨은 자식이 없어 쪼갤 수 없다(논문도 마지막 레벨 분할은 성능을 해친다고 보고).
    target_levels = levels if levels is not None else list(range(n_levels - 1))

    for level in target_levels:
        pop = token_popularity(sid, level, item_freq)
        members = token_items(sid, level)
        avg_pop = sum(pop.values()) / len(pop)

        selected, skipped = select_tokens_to_split(
            pop, members, labels, split_ratio, head_only, tail_guard)

        if verbose:
            top5 = sorted(pop.values(), reverse=True)[:max(1, int(len(pop) * GU_DIAGNOSIS_RATIO))]
            print(f"\n[level {level}] tokens={len(pop)}  avg_pop={avg_pop:.1f}  "
                  f"top{GU_DIAGNOSIS_RATIO:.0%}_mean_pop={sum(top5)/len(top5):.1f}")
            print(f"  분할 대상 {len(selected)}개 / 필터로 제외 {len(skipped)}개")

        next_free = int(new_sid[:, level].max()) + 1

        for tok in selected:
            # --- Step D: 자식 토큰별 통계 준비 ---
            children_stats = defaultdict(lambda: {"items": []})
            for iid in members[tok]:
                child = int(sid[iid, level + 1])
                children_stats[child]["items"].append(iid)

            stats = {}
            for child, d in children_stats.items():
                items = d["items"]
                stats[child] = {
                    "r_bar": residuals[level][items].mean(dim=0),
                    "n": len(items),
                    "pop": sum(item_freq.get(i, 0) for i in items),
                }

            M = decide_m(pop[tok], avg_pop, mode=m_mode)
            if M <= 1 or len(stats) <= 1:
                continue

            # --- Step E: 재배분 ---
            assign = split_one_token(stats, M, lambda_bal)

            # --- Step F: 재라벨링 ---
            # 하드 제약: 같은 (l+1)레벨 자식을 가진 아이템은 같은 새 토큰으로 간다.
            # 아래처럼 child 단위로 배정하면 이 제약이 자동으로 지켜진다.
            slot_to_token = {0: tok}          # slot 0은 원래 토큰 번호를 재사용
            for m in range(1, M):
                slot_to_token[m] = next_free
                next_free += 1

            new_pop = defaultdict(float)
            for child, slot in assign.items():
                for iid in children_stats[child]["items"]:
                    new_sid[iid, level] = slot_to_token[slot]
                    new_pop[slot_to_token[slot]] += item_freq.get(iid, 0)

            history.append({
                "level": level,
                "original_token": tok,
                "original_pop": pop[tok],
                "M": M,
                "n_children": len(stats),
                "new_tokens": {str(v): new_pop[v] for v in slot_to_token.values()},
            })

        if verbose and history:
            lv = [h for h in history if h["level"] == level]
            if lv:
                before = sum(h["original_pop"] for h in lv) / len(lv)
                after = sum(sum(h["new_tokens"].values()) / len(h["new_tokens"]) for h in lv) / len(lv)
                print(f"  분할 전 평균 인기도 {before:.1f} → 분할 후 새 토큰 평균 {after:.1f}")

    return new_sid, history


# =====================================================================
# 진단
# =====================================================================

def report(sid_before, sid_after, item_freq):
    """분할 전후 비교. 정확도 지표(Recall/NDCG)는 여기서 못 재므로 TIGER 학습 후 확인할 것."""
    print("\n" + "=" * 62)
    print("CRAB 적용 결과")
    print("=" * 62)
    for level in range(sid_before.shape[1]):
        b = len(torch.unique(sid_before[:, level]))
        a = len(torch.unique(sid_after[:, level]))
        print(f"  level {level}: 토큰 수 {b} → {a}  (+{a - b})")

    def gini(sid, level):
        pop = sorted(token_popularity(sid, level, item_freq).values())
        n, total = len(pop), sum(pop)
        if total == 0:
            return 0.0
        cum = sum((i + 1) * p for i, p in enumerate(pop))
        return (2 * cum) / (n * total) - (n + 1) / n

    print("\n  토큰 인기도 Gini (낮을수록 균등):")
    for level in range(sid_before.shape[1]):
        print(f"    level {level}: {gini(sid_before, level):.4f} → {gini(sid_after, level):.4f}")

    uniq_b = len({tuple(r) for r in sid_before.tolist()})
    uniq_a = len({tuple(r) for r in sid_after.tolist()})
    n = sid_before.shape[0]
    print(f"\n  고유 SID: {uniq_b}/{n} ({uniq_b/n:.2%}) → {uniq_a}/{n} ({uniq_a/n:.2%})")
    print("  ※ 충돌률은 대리 지표. 최종 판단은 TIGER 학습 후 Recall@k/NDCG@k로.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-path", type=Path, required=True,
                   help="SID를 만들 때 GRID에 넣었던 그 임베딩과 반드시 동일해야 함")
    p.add_argument("--sid-path", type=Path, required=True, help="sid_tensor.pt")
    p.add_argument("--interactions-path", type=Path,
                   default=Path("csv_export/dataset/A_interactions_long.csv"),
                   help="★ 지금은 분할 A(LOO). B로 갈 때 B_interactions_long.csv 로 교체")
    p.add_argument("--popularity-split", default="train")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split-ratio", type=float, default=SPLIT_RATIO)
    p.add_argument("--m-mode", choices=["fixed", "adaptive"], default="fixed",
                   help="fixed=M %d 고정(팀 합의) / adaptive=논문 원문 가변 방식" % M_FIXED)
    p.add_argument("--lambda-bal", type=float, default=LAMBDA_BAL)
    p.add_argument("--tail-guard", type=float, default=TAIL_GUARD)
    p.add_argument("--no-head-only", action="store_true", help="Head 필터 해제(논문 원본과 동일)")
    p.add_argument("--normalize-residuals", action="store_true",
                   help="GRID rkmeans_train_flat.yaml 의 normalize_residuals=True 를 재현")
    p.add_argument("--drop-dedup-col", action="store_true", default=True,
                   help="마지막 컬럼을 dedup 토큰으로 보고 분할 대상에서 제외 (L4=의미4+dedup1 구조)")
    p.add_argument("--keep-dedup-col", dest="drop_dedup_col", action="store_false",
                   help="마지막 컬럼도 의미 토큰으로 취급")
    args = p.parse_args()

    embedding = torch.load(args.embedding_path, map_location="cpu")
    sid = torch.load(args.sid_path, map_location="cpu")
    if sid.dim() != 2:
        raise ValueError(f"sid_tensor는 2D여야 함. 실제: {tuple(sid.shape)}")

    # --- GRID 출력은 (n_levels, n_items) 로 전치돼 있다. (레벨 수 << 아이템 수) ---
    n_emb_items = embedding.size(0)
    if sid.size(0) == n_emb_items:
        pass                                   # 이미 (n_items, n_levels)
    elif sid.size(1) == n_emb_items:
        print(f"[info] SID가 전치돼 있어 바로잡음: {tuple(sid.shape)} -> {tuple(sid.T.shape)}")
        sid = sid.T.contiguous()
    else:
        raise ValueError(f"아이템 수 불일치: emb={n_emb_items} vs sid={tuple(sid.shape)}")

    # --- dedup 토큰 컬럼 제거 ---
    # GRID는 SID 충돌(서로 다른 아이템이 같은 코드 조합)을 구분하려고 마지막에
    # dedup 토큰을 하나 붙인다. 이건 의미 토큰이 아니라 일련번호라서 CRAB의
    # 분할 대상이 되면 안 된다. 잘라내고 CRAB을 돌린 뒤 다시 붙여준다.
    dedup_col = None
    if args.drop_dedup_col:
        dedup_col = sid[:, -1].clone()
        sid = sid[:, :-1].contiguous()
        n_uniq = len(torch.unique(dedup_col))
        print(f"[info] 마지막 컬럼을 dedup 토큰으로 보고 분리 "
              f"(고유값 {n_uniq}개, 0이 아닌 항목 {(dedup_col != 0).sum().item()}개)")

    n_items = sid.size(0)
    print(f"items={n_items}  emb_dim={embedding.size(1)}  의미레벨={sid.size(1)}")

    popularity = load_popularity(args.interactions_path, args.popularity_split)
    labels = classify_head_tail(popularity, n_items)
    print(f"popularity split='{args.popularity_split}'  "
          f"head={labels.count('head')} / mid={labels.count('mid')} / tail={labels.count('tail')}")

    new_sid, history = run_crab(
        embedding, sid, popularity, labels,
        split_ratio=args.split_ratio,
        m_mode=args.m_mode,
        lambda_bal=args.lambda_bal,
        head_only=not args.no_head_only,
        tail_guard=args.tail_guard,
        normalize_residuals=args.normalize_residuals,
    )

    report(sid, new_sid, popularity)

    # dedup 토큰을 다시 붙이고, GRID가 읽던 원래 방향(n_levels, n_items)으로 되돌린다
    out_sid = new_sid
    if dedup_col is not None:
        out_sid = torch.cat([out_sid, dedup_col.unsqueeze(1)], dim=1)
        print("\n[주의] dedup 토큰을 그대로 복원했습니다. CRAB 분할로 SID 조합이 바뀌었으니, "
              "충돌이 실제로 어떻게 변했는지는 Tokenization/compare_sid_variants.py 로 "
              "다시 확인하고 필요하면 dedup을 재계산하세요.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out_sid.T.contiguous(), args.out_dir / "sid_tensor.pt")
    with open(args.out_dir / "crab_split_history.json", "w", encoding="utf-8") as f:
        json.dump({"config": vars(args) | {k: str(v) for k, v in vars(args).items()
                                           if isinstance(v, Path)},
                   "splits": history}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nsaved: {args.out_dir/'sid_tensor.pt'}")
    print(f"saved: {args.out_dir/'crab_split_history.json'}")


if __name__ == "__main__":
    main()
