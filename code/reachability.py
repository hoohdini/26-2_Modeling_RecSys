#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""도달가능성 — TIGER 가 애초에 "닿을 수 있는" 상품이 카탈로그의 몇 %인가.

프로젝트는 콜드 문제를 도달가능성 / 도달확률 / 편향 세 축으로 분해한다.
Coverage@20 은 "실제로 추천된 비율"이라 재정렬로 올릴 수 있는 양이지만,
**후보 풀 전체의 합집합**은 재정렬로 절대 넘을 수 없는 천장이다. 그게 도달가능성이다.

내놓는 것
--------
  · 생성 후보 합집합이 덮는 상품 수 / 비율        (= 재정렬의 절대 천장)
  · 그중 롱테일·콜드 버킷이 차지하는 비율
  · 한 번도 후보에 오르지 못한 상품은 어떤 상품인가 (버킷별 분해)
  · 후보 순위 깊이별 누적 도달률 (top-10 / 50 / 200 각각)

사용:
  python reachability.py --pred tiger_runs/clip_L4/eval_dump_k200/test_predictions_rank0.pkl
"""
import argparse
import io
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import Evaluator, load_split_a  # noqa: E402
from tiger_to_eval import build_sid2item  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKET_ORDER = ["zero-shot", "few-shot", "low", "mid", "head"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--sid", default=os.path.join(ROOT, "sid", "L4", "sid_tensor.pt"))
    ap.add_argument("--split", default=os.path.join(ROOT, "Beauty_split_A.pkl"))
    ap.add_argument("--label", default="clip_L4")
    ap.add_argument("--depths", default="10,50,200")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    depths = [int(x) for x in a.depths.split(",")]
    out_path = a.out or os.path.join(ROOT, "results", f"reachability_{a.label}.json")

    import torch
    sid2item, H, _ = build_sid2item(a.sid)
    tc, hist, tgt, n_items = load_split_a(a.split)
    ev = Evaluator(tc, n_items, tail_frac=0.5)

    with open(a.pred, "rb") as f:
        recs = pickle.load(f)

    maxd = max(depths)
    hit_at = {d: Counter() for d in depths}     # 깊이 d 안에 등장한 상품 -> 등장 유저 수
    for rec in recs:
        sids = rec["semantic_ids"]
        sids = sids.view(-1, H) if torch.is_tensor(sids) else sids
        seen, rank = set(), 0
        for row in sids[:maxd]:
            it = sid2item.get(tuple(int(x) for x in row))
            rank += 1
            if it is None or it in seen:
                continue
            seen.add(it)
            for d in depths:
                if rank <= d:
                    hit_at[d][it] += 1

    print(f"=== 도달가능성 ({a.label}) · 유저 {len(recs):,} · 카탈로그 {n_items:,} ===")
    print(f"  {'후보 깊이':>10} {'도달 상품':>10} {'도달률':>8}   버킷별 도달률")
    payload = {"label": a.label, "n_items": n_items, "n_user": len(recs),
               "pred_file": os.path.relpath(a.pred, ROOT).replace("\\", "/"), "depths": {}}

    for d in depths:
        reach = set(hit_at[d])
        by_b = {b: [0, 0] for b in BUCKET_ORDER}       # [도달, 전체]
        for i in range(n_items):
            b = ev.bucket_of.get(i, "head")
            by_b[b][1] += 1
            if i in reach:
                by_b[b][0] += 1
        tail_reach = len(reach & ev.tail)
        line = "  ".join(f"{b} {by_b[b][0]/max(1,by_b[b][1])*100:.0f}%" for b in BUCKET_ORDER)
        print(f"  {'top-'+str(d):>10} {len(reach):>10,} {len(reach)/n_items*100:7.1f}%   {line}")
        payload["depths"][str(d)] = {
            "n_reached": len(reach),
            "reach_rate": round(len(reach) / n_items, 6),
            "tail_reached": tail_reach,
            "tail_reach_rate": round(tail_reach / max(1, len(ev.tail)), 6),
            "by_bucket": {b: {"reached": by_b[b][0], "total": by_b[b][1],
                              "rate": round(by_b[b][0] / max(1, by_b[b][1]), 6)}
                          for b in BUCKET_ORDER},
        }

    # ---- 아무 유저에게도 안 뜨는 상품 ----
    deepest = max(depths)
    never = [i for i in range(n_items) if i not in hit_at[deepest]]
    print(f"\n  top-{deepest} 안에 단 한 번도 안 뜨는 상품: {len(never):,}개 "
          f"({len(never)/n_items*100:.1f}%)")
    nb = Counter(ev.bucket_of.get(i, "head") for i in never)
    for b in BUCKET_ORDER:
        tot = sum(1 for i in range(n_items) if ev.bucket_of.get(i, "head") == b)
        print(f"      {b:10s} {nb.get(b,0):5,} / {tot:5,}  ({nb.get(b,0)/max(1,tot)*100:4.1f}%)")

    # ---- 노출 편중: 상위 몇 개가 후보의 몇 %를 먹나 ----
    cnt = np.array(sorted(hit_at[deepest].values(), reverse=True), float)
    if len(cnt):
        cum = np.cumsum(cnt) / cnt.sum()
        j = int(np.searchsorted(cum, 0.5)) + 1
        print(f"\n  후보 등장 횟수의 절반을 차지하는 상품: 상위 {j:,}개 "
              f"({j/n_items*100:.1f}%)  <- 후보 단계의 편중")
        payload["items_for_half_candidate_slots"] = j

    payload["never_reached"] = {
        "n": len(never),
        "rate": round(len(never) / n_items, 6),
        "by_bucket": {b: nb.get(b, 0) for b in BUCKET_ORDER},
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
