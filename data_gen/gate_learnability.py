# -*- coding: utf-8 -*-
"""G-L 게이트 — **이 데이터셋에 배울 것이 있는가**를 GPU 를 쓰기 전에 CPU 로 판정한다.

왜 필요한가
-----------
2026-08-31, 이 게이트가 없어서 학습 불가능한 데이터셋에 GPU 13시간을 썼다.
TIGER 5셀이 전부 정상 종료했지만 train loss 가 내려가지 않았고(9.79 vs Beauty 5.75),
18,738 명 전원에게 사실상 같은 11개 아이템을 추천했다(coverage@10 0.00092).
원인은 시뮬레이터의 TEMP=6.0 — 적합도 항이 Gumbel 잡음에 묻혀 상호작용이
인기도+잡음만으로 결정됐다. 담당자별 직업 일관성이 무작위 대비 +0.0015 였다.

**G0 은 그래프 품질 검사지 상호작용 품질 검사가 아니다.** 이 게이트가 그 자리를 메운다.

무엇을 재는가
-------------
같은 LOO 프로토콜 위에서 세 추천기의 recall@K 를 비교한다.

    MostPopular   개인화 없음. 이 데이터의 "인기도만으로 얻는 점수" = 바닥선
    ContentKNN    이력 아이템의 스킬 벡터 중심 ↔ 후보 코사인.
                  **TIGER 의 대리자** — TIGER 의 SID 도 콘텐츠에서 나온다
    ItemKNN       아이템-아이템 코사인 공기(共起). 순수 협업 신호

판정
----
    ContentKNN / MostPopular >= 1.5   콘텐츠 신호가 인기도를 유의미하게 이긴다
    ItemKNN    / MostPopular >= 1.5   협업 신호도 존재한다

참고 기준 (Amazon Beauty Split A, results/baseline_splitA_loo.json)
    MostPopular 0.01207 · ItemKNN 0.04253 (3.52x) · EASE_R 0.05089 (4.21x)

둘 다 1.0 을 밑돌면 **어떤 모델도 인기도를 못 이긴다.** 학습을 제출하지 말 것.

사용
----
    python gate_learnability.py                                   # out/Jobs_split_A.pkl
    python gate_learnability.py --split path.pkl --skill path.npy
"""
import argparse
import collections
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data_gen", "out")
IN = os.environ.get("JOBS_IN_DIR", OUT)
OUT = os.environ.get("JOBS_OUT_DIR", OUT)

BEAUTY_REF = {"MostPopular": 0.01207, "ItemKNN": 0.04253, "EASE_R": 0.05089}
PASS_RATIO = 1.5


def load(split_path, skill_path):
    with open(split_path, "rb") as f:
        D = pickle.load(f)
    V = np.load(skill_path).astype(np.float32)
    n_items = len(D["iid"])
    if V.shape[0] != n_items:
        sys.exit(f"스킬 행렬 {V.shape[0]} != 아이템 {n_items}")
    hist = {u: list(v) for u, v in D["loo_train"].items() if len(v) >= 1}
    tgt = {u: D["loo_test"][u] for u in hist}
    return hist, tgt, n_items, V


def evaluate(hist, tgt, n_items, V, K):
    from scipy import sparse

    users = sorted(hist)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    cnt = np.zeros(n_items, dtype=np.float64)
    r_, c_ = [], []
    for j, u in enumerate(users):
        for i in hist[u]:
            cnt[i] += 1
        for i in set(hist[u]):
            r_.append(j)
            c_.append(i)
    pop_order = np.argsort(-cnt)

    X = sparse.csr_matrix((np.ones(len(r_), np.float32), (r_, c_)), shape=(len(users), n_items))
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel()) + 1e-9
    S = (X.T @ X).toarray().astype(np.float32)
    S /= nrm[:, None]
    S /= nrm[None, :]
    np.fill_diagonal(S, 0.0)
    Sc = np.asarray(X @ S)

    hit = {"MostPopular": 0, "ContentKNN": 0, "ItemKNN": 0}
    for j, u in enumerate(users):
        seen = set(hist[u])
        g = tgt[u]
        rec = [i for i in pop_order if i not in seen][:K]
        if g in rec:
            hit["MostPopular"] += 1

        pr = Vn[hist[u]].mean(0)
        pr /= np.linalg.norm(pr) + 1e-9
        s = Vn @ pr
        s[list(seen)] = -np.inf
        if g in set(np.argpartition(-s, K)[:K].tolist()):
            hit["ContentKNN"] += 1

        s2 = Sc[j].copy()
        s2[X[j].indices] = -np.inf
        if g in set(np.argpartition(-s2, K)[:K].tolist()):
            hit["ItemKNN"] += 1
    n = len(users)
    return {k: v / n for k, v in hit.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(OUT, "Jobs_split_A.pkl"))
    ap.add_argument("--skill", default=os.path.join(IN, "profile_skill.npy"))
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--json", default=None, help="결과를 JSON 으로도 저장")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    hist, tgt, n_items, V = load(a.split, a.skill)
    res, n = evaluate(hist, tgt, n_items, V, a.K)
    pop = res["MostPopular"]
    if a.json:
        import json
        json.dump({"K": a.K, "n_users": n, "n_items": n_items, "recall": res,
                   "ratio": {k: (res[k] / pop if pop else 0.0) for k in res}},
                  open(a.json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"G-L 학습가능성 게이트 · {os.path.basename(a.split)} · 유저 {n:,} · 아이템 {n_items:,}")
    print()
    print(f"  {'추천기':<14}{'recall@' + str(a.K):>11}{'/ MostPopular':>16}")
    print("  " + "-" * 41)
    for k in ("MostPopular", "ContentKNN", "ItemKNN"):
        ratio = res[k] / pop if pop > 0 else 0.0
        print(f"  {k:<14}{res[k]:>11.5f}{ratio:>15.2f}x")
    print()
    print("  참고 · Amazon Beauty Split A")
    print(f"  {'MostPopular':<14}{BEAUTY_REF['MostPopular']:>11.5f}{1.00:>15.2f}x")
    print(f"  {'ItemKNN':<14}{BEAUTY_REF['ItemKNN']:>11.5f}"
          f"{BEAUTY_REF['ItemKNN'] / BEAUTY_REF['MostPopular']:>15.2f}x")
    print(f"  {'EASE_R':<14}{BEAUTY_REF['EASE_R']:>11.5f}"
          f"{BEAUTY_REF['EASE_R'] / BEAUTY_REF['MostPopular']:>15.2f}x")
    print()

    rc = res["ContentKNN"] / pop if pop > 0 else 0.0
    rk = res["ItemKNN"] / pop if pop > 0 else 0.0
    ok_c, ok_k = rc >= PASS_RATIO, rk >= PASS_RATIO
    print(f"  ContentKNN/Pop {rc:.2f}x  {'통과' if ok_c else '미달'} (기준 {PASS_RATIO}x)")
    print(f"  ItemKNN/Pop    {rk:.2f}x  {'통과' if ok_k else '미달'} (기준 {PASS_RATIO}x)")
    print()
    if ok_c and ok_k:
        print("  ✅ G-L 통과 — 학습 제출 가능")
        return 0
    if rc <= 1.0 and rk <= 1.0:
        print("  ❌ G-L 실패 — 어떤 모델도 인기도를 못 이긴다. 제출하지 말 것.")
        print("     시뮬레이터 TEMP 를 낮춰 적합도 항의 비중을 올릴 것.")
        return 1
    print("  ⚠️  G-L 경계 — 신호는 있으나 약하다. 제출 전 판단 필요.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
