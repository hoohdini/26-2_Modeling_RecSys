#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""여러 체크포인트의 파레토 곡선을 겹쳐 비교한다 — "곡선 차이가 유의한가" 판정용.

왜 필요한가
----------
곡선 하나만 그려 놓으면 두 곡선의 간격이 진짜 차이인지 학습 노이즈인지 알 수 없다.
같은 레시피에 시드만 다른 두 실행(clip_L4 / clip_L4_seed7)의 간격이 곧 **노이즈 바닥**이고,
다른 조건(전수 검증본, 60k 스텝, 접두사 제약 디코딩)의 간격이 그보다 커야 의미가 있다.

무엇을 내놓나
------------
  1) λ 별로 각 실행의 NDCG / 롱테일 노출을 나란히 놓은 표
  2) 시드 간 간격(노이즈 바닥)과 각 조건의 간격을 같은 축에서 비교
  3) 기준 실행 대비 유저 단위 paired bootstrap → 차이의 95% 신뢰구간

사용:
  python pareto_compare.py --ref clip_L4=<덤프.pkl> \
                           --run seed7=<덤프.pkl> --run fullval=<덤프.pkl> ...
"""
import argparse
import io
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import Evaluator, load_split_a, paired_bootstrap  # noqa: E402
from pareto_dial import load_candidates, rerank  # noqa: E402
from tiger_to_eval import build_sid2item  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_kv(items):
    out = []
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--run/--ref 는 이름=경로 형식이어야 합니다: {it}")
        name, path = it.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="기준 실행. 이름=덤프.pkl")
    ap.add_argument("--run", action="append", default=[], help="비교 실행. 이름=덤프.pkl (반복 가능)")
    ap.add_argument("--sid", default=os.path.join(ROOT, "sid", "L4", "sid_tensor.pt"))
    ap.add_argument("--split", default=os.path.join(ROOT, "Beauty_split_A.pkl"))
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--lams", default="0,0.2,0.5,1,2,5")
    ap.add_argument("--noise-pair", default=None,
                    help="노이즈 바닥으로 볼 두 실행 이름, 예: clip_L4,seed7")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pareto_compare.json"))
    a = ap.parse_args()

    lams = [float(x) for x in a.lams.split(",")]
    runs = parse_kv([a.ref]) + parse_kv(a.run)
    ref_name = runs[0][0]

    tc, hist, tgt, n_items = load_split_a(a.split)
    with open(a.split, "rb") as f:
        A = pickle.load(f)
    idx2raw = {v: k for k, v in A["uid"].items()}
    pop_log = np.log1p(np.array([tc.get(i, 0) for i in range(n_items)], float))
    ev = Evaluator(tc, n_items, tail_frac=0.5)
    sid2item, H, _ = build_sid2item(a.sid)

    acc_key, div_key = f"ndcg@{a.k}", f"tail_exposure@{a.k}"

    print(f"=== 파레토 곡선 비교 (기준 = {ref_name}) ===")
    res = {}       # name -> lam -> metrics
    peruser = {}   # name -> lam -> {user: ndcg}
    meta = {}
    for name, path in runs:
        if not os.path.exists(path):
            print(f"  [건너뜀] {name}: 파일 없음 {path}")
            continue
        cands, cstat = load_candidates(path, sid2item, H)
        meta[name] = {"pred_file": os.path.relpath(path, ROOT).replace("\\", "/"),
                      "n_user": len(cands),
                      "cand_per_user": round(cstat["n_slot"] / max(1, len(cands)), 1),
                      "invalid_sid_rate": cstat["invalid_rate"]}
        res[name], peruser[name] = {}, {}
        for lam in lams:
            preds = rerank(cands, idx2raw, pop_log, lam)
            m, pu = ev.evaluate(preds, tgt, K=(a.k, 50) if a.k != 50 else (50,), per_user=True)
            res[name][lam] = m
            peruser[name][lam] = {u: d["ndcg"] for u, d in pu[a.k].items()}
        print(f"  {name:12s} 유저 {meta[name]['n_user']:,} · 후보 {meta[name]['cand_per_user']}개 "
              f"· 무효 SID {100*meta[name]['invalid_sid_rate']:.3f}%")

    names = list(res)

    # ---- 표 1: λ 별 정확도 / 다양성 ----
    for key, label in ((acc_key, f"NDCG@{a.k}"), (div_key, f"롱테일 노출@{a.k}")):
        print(f"\n=== {label} ===")
        print("  " + "λ".ljust(7) + "".join(n.ljust(14) for n in names) + "최대차")
        for lam in lams:
            vals = [res[n][lam][key] for n in names]
            spread = max(vals) - min(vals)
            print("  " + f"{lam:g}".ljust(7) + "".join(f"{v:<14.4f}" for v in vals) + f"{spread:.4f}")

    # ---- 노이즈 바닥 ----
    noise = None
    if a.noise_pair:
        p = [x.strip() for x in a.noise_pair.split(",")]
        if all(x in res for x in p):
            noise = {}
            for lam in lams:
                noise[lam] = {
                    "acc": abs(res[p[0]][lam][acc_key] - res[p[1]][lam][acc_key]),
                    "div": abs(res[p[0]][lam][div_key] - res[p[1]][lam][div_key]),
                }
            print(f"\n=== 노이즈 바닥: {p[0]} vs {p[1]} (레시피 동일, 시드만 다름) ===")
            print("  " + "λ".ljust(7) + "NDCG 차".ljust(14) + "롱테일 차")
            for lam in lams:
                print("  " + f"{lam:g}".ljust(7) +
                      f"{noise[lam]['acc']:<14.4f}{noise[lam]['div']:.4f}")
            mx_a = max(v["acc"] for v in noise.values())
            mx_d = max(v["div"] for v in noise.values())
            print(f"  -> 노이즈 바닥(최대): NDCG {mx_a:.4f} · 롱테일 노출 {mx_d:.4f}")
            print("     이보다 작은 차이는 시드 하나 바꾼 것과 구별되지 않는다.")

    # ---- paired bootstrap (기준 대비) ----
    print(f"\n=== {ref_name} 대비 유저 단위 paired bootstrap (NDCG@{a.k}) ===")
    print("  " + "실행".ljust(12) + "λ".ljust(7) + "차이".ljust(12) +
          "CI95".ljust(26) + "판정")
    tests = {}
    for n in names:
        if n == ref_name:
            continue
        tests[n] = {}
        for lam in lams:
            bs = paired_bootstrap(peruser[n][lam], peruser[ref_name][lam])
            lo, hi = bs["ci95"]
            sig = "유의" if (lo > 0 or hi < 0) else "구별 안 됨"
            tests[n][lam] = {**bs, "significant": bool(lo > 0 or hi < 0)}
            print("  " + n.ljust(12) + f"{lam:g}".ljust(7) +
                  f"{bs['mean_diff']:+.5f}".ljust(12) +
                  f"[{lo:+.5f}, {hi:+.5f}]".ljust(26) + sig)

    payload = {
        "K": a.k, "lams": lams, "ref": ref_name, "axes": {"acc": acc_key, "div": div_key},
        "meta": meta,
        "curves": {n: {str(l): {k2: res[n][l][k2] for k2 in
                                (acc_key, div_key, f"recall@{a.k}", f"coverage@{a.k}",
                                 f"exposure_gini@{a.k}", "COLD_recall@50")}
                       for l in lams} for n in names},
        "noise_floor": {str(l): v for l, v in (noise or {}).items()},
        "bootstrap_vs_ref": {n: {str(l): v for l, v in d.items()} for n, d in tests.items()},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
