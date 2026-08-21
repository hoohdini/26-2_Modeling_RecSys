#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TIGER 에 다이얼을 달아 정확도-롱테일 파레토 곡선을 그린다 — 트랙 C 본 목표.

문제
----
파레토 곡선은 "정확도를 조금 포기하면 다양성을 얼마나 얻는가"의 지도다.
곡선을 그리려면 조절 손잡이가 있어야 하는데, TIGER 는 빔서치라 손잡이가 없다.
그냥 돌리면 점 하나만 찍힌다.

이 스크립트의 접근 — 생성 후 재정렬(post-hoc re-ranking)
-------------------------------------------------------
GPU 를 한 번만 쓴다. eval_step 덤프에는 유저당 후보 C개와 그 생성확률이 들어 있으므로,
그 후보 목록을 인기도 페널티로 다시 정렬한다.

    점수(i) = log p(i) - lam * log(1 + 학습등장횟수(i))

lam=0 이면 원래 빔서치 순서 그대로다. lam 을 키우면 인기 상품이 아래로 밀린다.
lam 을 훑으면 덤프 한 번으로 곡선 전체가 나온다.

한계 (반드시 같이 보고할 것)
---------------------------
· 이건 생성 자체의 다이얼이 아니라 **재정렬 다이얼**이다. 후보 풀(C개) 밖의 상품은
  아무리 lam 을 키워도 추천될 수 없다. 즉 다양성에 천장이 있다.
  곡선이 오른쪽에서 평평해지면 top_k_for_generation 을 키워 풀을 넓혀야 한다.
· MaskGR 의 확산 다이얼과 같은 종류의 기여가 아니다. TIGER 쪽 곡선은
  "AR 모델에서도 이만큼은 된다"는 참조선으로 쓰는 것이 정직하다.

사용:
  python pareto_dial.py --pred tiger_runs/clip_L4/eval_dump_k50/test_predictions_rank0.pkl
"""
import argparse
import io
import json
import os
import pickle
import sys

import numpy as np

if __name__ == "__main__":      # import 될 때 남의 stdout 을 갈아끼우면 안 된다
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import Evaluator, load_split_a, pareto_table  # noqa: E402
from tiger_to_eval import build_sid2item  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def load_candidates(pred_path, sid2item, H):
    """덤프 -> [(user_idx, [item...], [logp...])] · 무효 SID 제거."""
    import torch
    with open(pred_path, "rb") as f:
        recs = pickle.load(f)
    if "scores" not in recs[0]:
        raise SystemExit("이 덤프에는 scores 가 없습니다. "
                         "prediction_dumper 의 save_scores=True 로 다시 덤프하세요.")
    out, n_invalid, n_slot = [], 0, 0
    for rec in recs:
        u = rec["user_id"]
        u = int(u.item()) if torch.is_tensor(u) else int(u)
        sids = rec["semantic_ids"]
        sids = sids.view(-1, H) if torch.is_tensor(sids) else sids
        sc = rec["scores"].reshape(-1).tolist()
        items, logps = [], []
        for j, row in enumerate(sids):
            n_slot += 1
            it = sid2item.get(tuple(int(x) for x in row))
            if it is None:
                n_invalid += 1
                continue
            items.append(it)
            # marginal_probs 는 확률. 0 이 섞일 수 있어 바닥을 깐다.
            logps.append(float(np.log(max(sc[j], 1e-30))))
        out.append((u, items, logps))
    return out, {"n_slot": n_slot, "n_invalid": n_invalid,
                 "invalid_rate": round(n_invalid / max(1, n_slot), 6)}


def rerank(cands, idx2raw, pop_log, lam):
    """lam 으로 재정렬한 {raw_user: [item...]} 을 만든다 (전역 인기도 페널티)."""
    preds = {}
    for u, items, logps in cands:
        raw = idx2raw.get(u)
        if raw is None or not items:
            continue
        if lam == 0.0:
            preds[raw] = list(items)          # 원래 빔서치 순서 보존
            continue
        s = np.asarray(logps) - lam * pop_log[np.asarray(items)]
        order = np.argsort(-s, kind="stable")
        preds[raw] = [items[j] for j in order]
    return preds


def rerank_exposure(cands, idx2raw, pop_log, mu, n_items, k_slot=20, lam=0.0):
    """이미 이번 라운드에서 많이 노출된 상품을 눌러 준다 (노출 인지 재정렬).

    전역 인기도 페널티(lam)의 약점은 4-4절에서 드러났다 — 값을 키우면
    **모든 유저에게 같은 희귀 상품**을 밀게 되어 Coverage 가 오히려 떨어진다.
    여기서는 유저를 순서대로 처리하면서 지금까지 추천된 횟수를 세고,
    그 횟수에 비례해 점수를 깎는다. 손잡이는 mu 하나다.

      점수(i) = log p(i) - lam·log(1+학습등장) - mu·log(1+이번_라운드_노출횟수(i))

    주의: 유저 처리 순서에 결과가 의존한다(온라인 알고리즘). 순서를 고정해
    재현성은 확보하지만, "유저별로 독립인 추천"이라는 가정은 깨진다.
    발표에서는 이 점을 반드시 밝힐 것.
    """
    exposed = np.zeros(n_items, dtype=np.float64)
    preds = {}
    for u, items, logps in sorted(cands, key=lambda c: c[0]):   # 순서 고정
        raw = idx2raw.get(u)
        if raw is None or not items:
            continue
        idx = np.asarray(items)
        s = np.asarray(logps) - lam * pop_log[idx] - mu * np.log1p(exposed[idx])
        order = np.argsort(-s, kind="stable")
        ranked = [items[j] for j in order]
        preds[raw] = ranked
        np.add.at(exposed, np.asarray(ranked[:k_slot]), 1.0)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="eval_step 덤프 pkl (scores 포함)")
    ap.add_argument("--sid", default=os.path.join(ROOT, "sid", "L4", "sid_tensor.pt"))
    ap.add_argument("--split", default=os.path.join(ROOT, "Beauty_split_A.pkl"))
    ap.add_argument("--label", default="tiger")
    ap.add_argument("--k", type=int, default=20, help="파레토 축 컷오프")
    ap.add_argument("--lams", default=",".join(str(x) for x in DEFAULT_LAMBDAS))
    ap.add_argument("--mode", choices=("pop", "exposure"), default="pop",
                    help="pop=전역 인기도 페널티 · exposure=노출 인지(온라인) 재정렬")
    ap.add_argument("--base-lam", type=float, default=0.0,
                    help="exposure 모드에서 함께 걸 인기도 페널티")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lams = [float(x) for x in a.lams.split(",")]
    out_path = a.out or os.path.join(ROOT, "results", f"pareto_{a.label}_k{a.k}.json")

    sid2item, H, n_items_sid = build_sid2item(a.sid)
    cands, cstat = load_candidates(a.pred, sid2item, H)
    mode_label = {"pop": "전역 인기도 페널티", "exposure": "노출 인지(온라인)"}[a.mode]
    print(f"=== 파레토 다이얼 ({a.label}) · {mode_label} ===")
    print(f"  덤프 {os.path.basename(a.pred)} · 유저 {len(cands):,} · "
          f"유저당 후보 {cstat['n_slot'] / max(1, len(cands)):.1f}개 "
          f"(무효 SID {100 * cstat['invalid_rate']:.3f}% 제거)")

    tc, hist, tgt, n_items = load_split_a(a.split)
    with open(a.split, "rb") as f:
        A = pickle.load(f)
    idx2raw = {v: k for k, v in A["uid"].items()}
    pop_log = np.log1p(np.array([tc.get(i, 0) for i in range(n_items)], float))
    ev = Evaluator(tc, n_items, tail_frac=0.5)

    acc_key, div_key = f"ndcg@{a.k}", f"tail_exposure@{a.k}"
    points = []
    knob = "mu" if a.mode == "exposure" else "lam"
    print(f"\n  {knob:>6}  {'Recall@'+str(a.k):>10} {'NDCG@'+str(a.k):>10} "
          f"{'TailExp':>9} {'Cov':>8} {'Gini':>8} {'COLD-R@50':>10}")
    for lam in lams:
        if a.mode == "exposure":
            preds = rerank_exposure(cands, idx2raw, pop_log, lam, n_items,
                                    k_slot=a.k, lam=a.base_lam)
        else:
            preds = rerank(cands, idx2raw, pop_log, lam)
        # @10 은 정확도 파트(Recall@5/@10)와 컷오프를 맞추기 위해 항상 함께 계산한다.
        # 재정렬 자체는 a.k 에만 의존하므로(exposure 모드의 k_slot), 이 줄은 지표를
        # 더 뽑을 뿐 다이얼 동작을 바꾸지 않는다.
        m = ev.evaluate(preds, tgt, K=tuple(sorted({10, a.k, 50})))
        m["label"] = f"{a.label}_{knob}{lam:g}"
        m["w"] = lam
        points.append(m)
        print(f"  {lam:6g}  {m[f'recall@{a.k}']:10.4f} {m[acc_key]:10.4f} "
              f"{m[div_key]:9.4f} {m[f'coverage@{a.k}']:8.4f} "
              f"{m[f'exposure_gini@{a.k}']:8.4f} {m.get('COLD_recall@50', float('nan')):10.4f}")

    table = pareto_table(points, acc=acc_key, div=div_key)
    front = [r["label"] for r in table if r["on_frontier"]]
    print(f"\n  파레토 프론티어 위 지점 {len(front)}/{len(table)}개: {', '.join(front)}")

    # 곡선이 천장을 쳤는지: 마지막 두 지점의 다양성 증가폭
    if len(points) >= 2:
        d1, d2 = points[-2][div_key], points[-1][div_key]
        if abs(d2 - d1) < 1e-4:
            print("  [주의] lam 을 더 키워도 다양성이 안 오른다 = 후보 풀이 천장이다. "
                  "top_k_for_generation 을 키워 다시 덤프할 것.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"axes": {"acc": acc_key, "div": div_key},
                   "mode": a.mode,
                   "base_lam": a.base_lam,
                   "dial": ("log p(i) - lam * log(1 + train_count(i))" if a.mode == "pop"
                            else f"log p(i) - {a.base_lam} * log(1 + train_count(i))"
                                 " - mu * log(1 + round_exposure(i))"),
                   "source": os.path.basename(a.pred),
                   "candidate_stats": cstat,
                   "pareto": table,
                   "points": {p["label"]: p for p in points}}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
