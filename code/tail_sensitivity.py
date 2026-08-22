#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""롱테일 경계(tail_frac) 민감도 — 트랙 C 프로토콜 설계 근거.

evaluate.Evaluator 는 롱테일을 "학습 인기 하위 tail_frac" 으로 정의한다.
현재 기본값은 0.5(하위 50%)인데, 논문 관행은 대체로 상위 20% 를 head 로 보는
80/20 쪽에 가깝다. 경계를 바꾸면 모델 순위가 뒤집히는가?를 확인한다.
순위가 안 바뀌면 경계 선택은 서술의 문제이고, 바뀌면 프로토콜에 못박아야 한다.

사용: python tail_sensitivity.py
"""
import io
import json
import os
import sys
from collections import Counter

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import ease_r, item_knn, most_popular, random_rec, to_csr  # noqa: E402
from evaluate import Evaluator, load_split_a  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRACS = [0.5, 0.667, 0.8, 0.9]        # 하위 50 / 66.7 / 80 / 90 %
# 컷오프는 정확도 파트(Recall@5/@10)와 맞춰 10. --k 로 바꿀 수 있다.
K = int(os.environ.get("TAIL_K", "10"))


def main():
    split = os.path.join(ROOT, "Beauty_split_A.pkl")
    tc, hist, tgt, n_items = load_split_a(split)
    users = [u for u in tgt if u in hist]
    hist = {u: hist[u] for u in users}

    # ── 인기 분포 자체를 먼저 본다 ────────────────────────────────
    cnt = np.array([tc.get(i, 0) for i in range(n_items)], float)
    order = np.sort(cnt)[::-1]
    tot = order.sum()
    cum = np.cumsum(order) / tot
    print(f"=== Beauty Split A 인기 분포 (아이템 {n_items:,} · 학습 상호작용 {int(tot):,}) ===")
    for p in (0.05, 0.1, 0.2, 0.3, 0.5):
        j = int(n_items * p)
        print(f"  상위 {p*100:4.1f}% 아이템({j:5,}개)이 상호작용의 {cum[j-1]*100:5.1f}% 를 차지")
    j80 = int(np.searchsorted(cum, 0.8)) + 1
    print(f"  상호작용의 80% 를 차지하는 데 필요한 아이템: 상위 {j80:,}개 "
          f"({100*j80/n_items:.1f}%)  <- 이 데이터의 '실제 80/20 지점'")
    zero = int((cnt == 0).sum())
    print(f"  학습 등장 0회 아이템 {zero:,}개 ({100*zero/n_items:.1f}%) · "
          f"중앙값 {int(np.median(cnt))}회 · 평균 {cnt.mean():.1f}회")

    for f in FRACS:
        ev = Evaluator(tc, n_items, tail_frac=f)
        thr = min(tc.get(i, 0) for i in ev.tail) if ev.tail else 0
        hi = max(tc.get(i, 0) for i in ev.tail) if ev.tail else 0
        share = sum(tc.get(i, 0) for i in ev.tail) / tot
        print(f"  tail_frac={f:<5} -> 롱테일 {len(ev.tail):5,}개 "
              f"(등장 {thr}~{hi}회) · 학습 상호작용의 {share*100:4.1f}%")

    # ── 모델별 tail_exposure 가 경계에 따라 어떻게 움직이나 ──────
    X, ulist, _ = to_csr(hist, n_items)
    preds = {
        "Random": random_rec(hist, n_items, ulist, k=50),
        "MostPopular": most_popular(hist, n_items, ulist, k=50),
        "ItemKNN": item_knn(X, ulist, hist, k=50),
        "EASE_R": ease_r(X, ulist, hist, k=50),
    }

    print(f"\n=== tail_exposure@{K} 가 경계에 따라 어떻게 변하나 ===")
    hdr = "  " + "모델".ljust(14) + "".join(f"tail={f:<8}" for f in FRACS)
    print(hdr)
    table = {}
    for label, p in preds.items():
        vals = []
        for f in FRACS:
            ev = Evaluator(tc, n_items, tail_frac=f)
            m = ev.evaluate(p, tgt, K=(K,))
            vals.append(m[f"tail_exposure@{K}"])
        table[label] = vals
        print("  " + label.ljust(14) + "".join(f"{v:<13.4f}" for v in vals))

    print("\n=== 경계별 다양성 순위 (높을수록 롱테일 노출 큼) ===")
    flipped = False
    base_rank = None
    for j, f in enumerate(FRACS):
        rank = [lab for lab, v in sorted(table.items(), key=lambda x: -x[1][j])]
        if base_rank is None:
            base_rank = rank
        elif rank != base_rank:
            flipped = True
        print(f"  tail_frac={f:<6} {' > '.join(rank)}")
    print(f"\n  순위 뒤집힘: {'있음 — 경계를 프로토콜에 못박아야 함' if flipped else '없음 — 경계 선택은 서술의 문제'}")

    out = os.path.join(ROOT, "results", "tail_sensitivity.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"fracs": FRACS, "K": K, "tail_exposure": table,
                   "n_items": n_items, "n_train_interaction": int(tot),
                   "items_for_80pct_interactions": j80}, fh, ensure_ascii=False, indent=1)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
