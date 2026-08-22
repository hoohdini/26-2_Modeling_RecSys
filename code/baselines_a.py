#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""고전 베이스라인을 Beauty_split_A.pkl 위에서 재계산 — 트랙 C 비교 기준선.

왜 다시 도는가:
  results/baseline_Beauty_loo.json 은 구 전처리 산출물(build/Beauty.pkl)로 만들어졌다.
  현행 Beauty_split_A.pkl 은 학습 이력을 max_seq_len=20 으로 잘라 두었기 때문에
  아이템별 학습 등장 횟수가 달라지고, 그 결과
    · 롱테일 집합(tail_frac 하위 50%)
    · 콜드 버킷 경계(zero/few/low/mid/head)
  가 서로 다른 아이템 위에 그어진다. 실제로 타깃 버킷 분포가 이렇게 어긋난다.

    구 baseline json : mid 8328 / head 7838 / low 4559 / few-shot 1500 / zero-shot 138
    Beauty_split_A   : mid 8229 / head 7537 / low 4692 / few-shot 1743 / zero-shot  162

  TIGER 는 Split A 로 학습·평가했으므로, 같은 정의 위에 베이스라인을 다시 올려야
  "TIGER 의 롱테일 노출이 EASE_R 보다 낮다" 같은 문장이 성립한다.

사용:
  python baselines_a.py                       # -> results/baseline_splitA_loo.json
  python baselines_a.py --models MostPopular,EASE_R
"""
import argparse
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import ease_r, item_knn, most_popular, random_rec, to_csr  # noqa: E402
from evaluate import Evaluator, load_split_a, load_split_b, paired_bootstrap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPK = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(ROOT, "Beauty_split_A.pkl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--window", default=None,
                    help="Temporal 트랙: Split B 의 윈도우 라벨 (예: W5). 주면 Split B 로 읽는다")
    ap.add_argument("--models", default="Random,MostPopular,ItemKNN,EASE_R")
    ap.add_argument("--K", default="10,20,50")
    a = ap.parse_args()
    K = tuple(int(x) for x in a.K.split(","))
    want = [m.strip() for m in a.models.split(",") if m.strip()]

    if a.window:
        tc, hist, tgt, n_items, _tiers = load_split_b(a.split, a.window)
        track = f"temporal/{a.window}"
    else:
        tc, hist, tgt, n_items = load_split_a(a.split)
        track = "leave-one-out"
    out_path = a.out or os.path.join(
        ROOT, "results",
        f"baseline_splitB_{a.window}.json" if a.window else "baseline_splitA_loo.json")

    n_tgt_all = len(tgt)
    users = [u for u in tgt if u in hist]
    # Temporal 에서는 그 윈도우에 학습 이력이 없는 테스트 유저가 많다.
    # 이력이 없으면 협업필터 계열은 애초에 예측을 못 하므로 채점에서 뺀다.
    # 몇 명을 뺐는지 반드시 남긴다 — 트랙 간 유저 수가 다르면 비교가 깨진다.
    dropped = n_tgt_all - len(users)
    tgt = {u: tgt[u] for u in users}
    hist = {u: hist[u] for u in users}
    ev = Evaluator(tc, n_items, tail_frac=0.5)
    X, ulist, _ = to_csr(hist, n_items)

    print(f"===== {os.path.basename(a.split)} / {track} "
          f"| 유저 {len(users):,} 아이템 {n_items:,} "
          f"타깃 {sum(len(v) for v in tgt.values()):,} =====")
    if a.window:
        print(f"  학습 이력이 없어 제외한 테스트 유저 {dropped:,}명 "
              f"(전체 {n_tgt_all:,}명 중 {100*dropped/max(1,n_tgt_all):.1f}%)")
    print(f"  롱테일(하위 50%) 아이템 {len(ev.tail):,}개 · 학습 상호작용 {sum(tc.values()):,}건")

    res, peruser = {}, {}

    def go(label, fn):
        if label not in want:
            return
        t0 = time.time()
        p = fn()
        dt = time.time() - t0
        m, pu = ev.evaluate(p, tgt, K=K, per_user=True)
        m["label"] = label
        m["sec"] = round(dt, 1)
        m["_source"] = {"split_file": os.path.basename(a.split), "tail_frac": 0.5,
                        "exclude_seen": True, "track": "temporal" if a.window else "loo",
                        "window": a.window, "n_user_scored": len(users),
                        "n_user_dropped_no_history": dropped}
        res[label] = m
        peruser[label] = {u: d["ndcg"] for u, d in pu[max(K)].items()}
        print(f"  {label:12s} R@20 {m['recall@20']:.4f}  N@20 {m['ndcg@20']:.4f}  "
              f"COLD-R@50 {m['COLD_recall@50']:.4f}  Cov@20 {m['coverage@20']:.4f}  "
              f"TailExp@20 {m['tail_exposure@20']:.4f}  ({dt:.0f}s)")

    go("Random", lambda: random_rec(hist, n_items, ulist, k=TOPK))
    go("MostPopular", lambda: most_popular(hist, n_items, ulist, k=TOPK))
    go("ItemKNN", lambda: item_knn(X, ulist, hist, k=TOPK))
    go("EASE_R", lambda: ease_r(X, ulist, hist, k=TOPK))

    if "EASE_R" in res and "ItemKNN" in res:
        bs = paired_bootstrap(peruser["EASE_R"], peruser["ItemKNN"])
        print(f"  [paired bootstrap] EASE_R - ItemKNN NDCG@{max(K)} "
              f"diff={bs['mean_diff']:+.5f} CI95={bs['ci95']} p={bs['p_value_gt0']:.4f}")
        res["_test_EASE_vs_KNN"] = bs

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
