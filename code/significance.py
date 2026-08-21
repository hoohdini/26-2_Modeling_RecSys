#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TIGER vs 고전 베이스라인 — 유저 단위 paired bootstrap.

"TIGER 가 EASE_R 보다 정확도 41% 높다" 를 신뢰구간과 함께 말하기 위한 것.
같은 유저 집합에서 유저별 NDCG 차이를 부트스트랩한다.

사용: python significance.py --pred tiger_runs/clip_L4/eval_dump_k50/test_predictions_rank0.pkl
"""
import argparse
import io
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import ease_r, item_knn, most_popular, to_csr  # noqa: E402
from evaluate import Evaluator, load_split_a, paired_bootstrap  # noqa: E402
from tiger_to_eval import build_sid2item, decode_predictions, remap_users  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--sid", default=os.path.join(ROOT, "sid", "L4", "sid_tensor.pt"))
    ap.add_argument("--split", default=os.path.join(ROOT, "Beauty_split_A.pkl"))
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--label", default="TIGER_clip_L4")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "significance_tiger.json"))
    a = ap.parse_args()

    tc, hist, tgt, n_items = load_split_a(a.split)
    users = [u for u in tgt if u in hist]
    hist = {u: hist[u] for u in users}
    ev = Evaluator(tc, n_items, tail_frac=0.5)

    sid2item, H, _ = build_sid2item(a.sid)
    pidx, stats, _ = decode_predictions(a.pred, sid2item, H)
    with open(a.split, "rb") as f:
        A = pickle.load(f)
    tiger, _ = remap_users(pidx, A["uid"])

    X, ulist, _ = to_csr(hist, n_items)
    preds = {a.label: tiger,
             "EASE_R": ease_r(X, ulist, hist, k=50),
             "ItemKNN": item_knn(X, ulist, hist, k=50),
             "MostPopular": most_popular(hist, n_items, ulist, k=50)}

    pu = {}
    for lab, p in preds.items():
        _, per = ev.evaluate(p, tgt, K=(a.k,), per_user=True)
        pu[lab] = {u: d["ndcg"] for u, d in per[a.k].items()}

    print(f"=== paired bootstrap · NDCG@{a.k} · {a.label} 기준 ===")
    out = {}
    for other in ("EASE_R", "ItemKNN", "MostPopular"):
        bs = paired_bootstrap(pu[a.label], pu[other])
        rel = bs["mean_diff"] / max(1e-12, sum(pu[other].values()) / len(pu[other]))
        out[f"{a.label}_vs_{other}"] = {**bs, "relative_gain": round(rel, 4)}
        sig = "유의" if bs["p_value_gt0"] < 0.05 else "유의하지 않음"
        print(f"  vs {other:12s} 차이 {bs['mean_diff']:+.5f} "
              f"(상대 {rel*100:+.1f}%)  CI95 {bs['ci95']}  p={bs['p_value_gt0']:.4f}  {sig}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"K": a.k, "n_user": len(pu[a.label]), "tests": out}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
