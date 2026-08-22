#!/usr/bin/env python3
"""
고전 베이스라인 (전부 CPU) — W1 안전망
  MostPopular : 편향 상한. 다양성 지표의 최악 기준선
  ItemKNN     : 코사인 아이템 유사도
  EASE_R      : 선형 오토인코더. 확산 추천을 이긴 그 모델
  Random      : 하한

사용: python3 baselines.py build/Beauty.pkl [temporal|loo]
"""
import sys, json, pickle, time
from collections import Counter, defaultdict
import numpy as np
from scipy import sparse
from evaluate import Evaluator, load_split, paired_bootstrap

TOPK = 50


def to_csr(hist, n_items):
    users = sorted(hist)
    uidx = {u: k for k, u in enumerate(users)}
    r, c = [], []
    for u, items in hist.items():
        for i in set(items):
            r.append(uidx[u]); c.append(i)
    X = sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)),
                          shape=(len(users), n_items))
    return X, users, uidx


def rank_from_scores(S, users, hist, k=TOPK):
    """S: (n_user, n_items) 점수. 이미 본 아이템은 제외."""
    preds = {}
    for j, u in enumerate(users):
        s = S[j].copy()
        s[list(set(hist[u]))] = -np.inf
        idx = np.argpartition(-s, k)[:k]
        preds[u] = [int(x) for x in idx[np.argsort(-s[idx])]]
    return preds


def most_popular(hist, n_items, users, k=TOPK):
    c = Counter()
    for v in hist.values(): c.update(v)
    order = [i for i, _ in c.most_common()]
    order += [i for i in range(n_items) if i not in c]
    preds = {}
    for u in users:
        seen = set(hist[u])
        preds[u] = [i for i in order if i not in seen][:k]
    return preds


def random_rec(hist, n_items, users, k=TOPK, seed=0):
    rng = np.random.default_rng(seed); preds = {}
    for u in users:
        seen = set(hist[u])
        cand = rng.permutation(n_items)
        preds[u] = [int(i) for i in cand if int(i) not in seen][:k]
    return preds


def item_knn(X, users, hist, topk_sim=200, k=TOPK):
    Xc = X.T.tocsr().astype(np.float32)              # item x user
    norm = np.sqrt(Xc.multiply(Xc).sum(axis=1)).A.ravel(); norm[norm == 0] = 1
    Xn = sparse.diags(1.0 / norm) @ Xc
    S = (Xn @ Xn.T).toarray()
    np.fill_diagonal(S, 0)
    if topk_sim < S.shape[1]:                        # 이웃 상위 topk_sim만 유지
        thr = np.partition(S, -topk_sim, axis=1)[:, -topk_sim][:, None]
        S[S < thr] = 0
    return rank_from_scores((X @ S).astype(np.float32), users, hist, k)


def ease_r(X, users, hist, lam=250.0, k=TOPK):
    G = (X.T @ X).toarray().astype(np.float64)
    d = np.diag_indices(G.shape[0])
    G[d] += lam
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))
    B[d] = 0.0
    return rank_from_scores((X @ B).astype(np.float32), users, hist, k)


def run(pkl, split="temporal", window=0, K=(10, 20, 50)):
    tc, hist, tgt, n_items = load_split(pkl, split, window)
    ev = Evaluator(tc, n_items, tail_frac=0.5)
    users = [u for u in tgt if u in hist]
    hist = {u: hist[u] for u in users}
    X, ulist, _ = to_csr(hist, n_items)
    name = pkl.split("/")[-1].replace(".pkl", "")
    print(f"\n===== {name} / {split}{window if split=='temporal' else ''} "
          f"| 유저 {len(users):,} 아이템 {n_items:,} 타깃 {sum(len(v) for v in tgt.values()):,} =====")

    res, peruser = {}, {}
    def go(label, fn):
        t0 = time.time(); p = fn(); dt = time.time() - t0
        m, pu = ev.evaluate(p, tgt, K=K, per_user=True)
        m["label"] = label; m["sec"] = round(dt, 1)
        res[label] = m; peruser[label] = {u: d["ndcg"] for u, d in pu[max(K)].items()}
        print(f"  {label:12s} R@20 {m['recall@20']:.4f}  N@20 {m['ndcg@20']:.4f}  "
              f"COLD-R@50 {m['COLD_recall@50']:.4f}  Cov@20 {m['coverage@20']:.4f}  "
              f"TailExp@20 {m['tail_exposure@20']:.4f}  ({dt:.0f}s)")

    go("Random",      lambda: random_rec(hist, n_items, ulist))
    go("MostPopular", lambda: most_popular(hist, n_items, ulist))
    go("ItemKNN",     lambda: item_knn(X, ulist, hist))
    go("EASE_R",      lambda: ease_r(X, ulist, hist))

    # 통계 검정 예시: EASE_R vs ItemKNN
    bs = paired_bootstrap(peruser["EASE_R"], peruser["ItemKNN"])
    print(f"  [paired bootstrap] EASE_R - ItemKNN  NDCG@50 diff={bs['mean_diff']:+.5f} "
          f"CI95={bs['ci95']}  p={bs['p_value_gt0']:.4f}")
    res["_test_EASE_vs_KNN"] = bs

    # 버킷 분해 출력
    print("  --- 버킷별 Recall@50 (EASE_R) ---")
    for b, d in sorted(res["EASE_R"]["bucket_recall@50"].items(),
                       key=lambda x: ["zero-shot","few-shot","low","mid","head"].index(x[0])):
        print(f"      {b:10s} n={d['n']:6,d}  recall={d['recall']:.4f}")
    return res


if __name__ == "__main__":
    pkl = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "temporal"
    out = run(pkl, split)
    tag = pkl.split("/")[-1].replace(".pkl", "")
    json.dump({k: v for k, v in out.items()},
              open(f"/tmp/work/baseline_{tag}_{split}.json", "w"), ensure_ascii=False, indent=1)
