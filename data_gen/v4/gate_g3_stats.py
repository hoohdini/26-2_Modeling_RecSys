# -*- coding: utf-8 -*-
"""G3 통계 정합성 게이트 — 생성 데이터의 분포를 목표 분포와 대조한다.

    python data_gen/v4/gate_g3_stats.py --interactions data_gen/out/interactions.csv \
        --profiles data_gen/out/profiles.csv --target data_gen/spec/kaggle_target_dist.json \
        --out results/g3_v4.json --md docs/G3_통계정합성.md

    python data_gen/v4/gate_g3_stats.py --beauty Beauty_split_A.pkl --out data_gen/spec/beauty_ref_dist.json
        (Beauty 참고선을 같은 형식으로 만든다)

허용 구간은 docs/G3_허용구간.md 에 **생성 전에** 고정하고, 같은 값을 data_gen/spec/g3_tolerance.json 에 둔다.
판정은 항목마다 PASS / WARN / FAIL. 게이트 통과 = FAIL 0개. 종료 코드 0 = 통과.
"""
import argparse
import csv
import io
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Q = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def qdict(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {}
    d = {"p%d" % int(q * 100): float(np.quantile(x, q)) for q in Q}
    d.update(mean=float(x.mean()), n=int(len(x)))
    return d


def stats_from_rows(rows, n_items, career_years=None, edu=None):
    """rows: (user, item, t) 리스트. t 는 일 단위 실수(없으면 None)."""
    per_u = defaultdict(int)
    per_i = np.zeros(n_items)
    ts = defaultdict(list)
    for u, i, t in rows:
        per_u[u] += 1
        per_i[i] += 1
        if t is not None:
            ts[u].append(t)
    pu = np.array(list(per_u.values()), dtype=float)
    pi = np.sort(per_i)[::-1]
    pi_pos = pi[pi > 0]
    out = {"n_users": int(len(per_u)), "n_items": int(n_items), "n_interactions": int(len(rows)),
           "density": float(len(rows) / max(len(per_u) * n_items, 1)),
           "apps_per_user": qdict(pu),
           "apps_per_job_applied_only": qdict(pi_pos)}
    out["apps_per_user"].update(share_ge3=float((pu >= 3).mean()), share_ge5=float((pu >= 5).mean()))
    out["apps_per_job_applied_only"].update(
        gini=gini(pi_pos), top50_share=float(pi_pos[: len(pi_pos) // 2].sum() / pi_pos.sum()),
        top20_share=float(pi_pos[: max(1, len(pi_pos) // 5)].sum() / pi_pos.sum()),
        share_le3=float((pi_pos <= 3).mean()), share_le5=float((pi_pos <= 5).mean()))
    out["share_jobs_with_zero_apps"] = float((per_i == 0).mean())
    out["gini_all_items"] = gini(pi)
    out["top50_share_all_items"] = float(pi[: len(pi) // 2].sum() / max(pi.sum(), 1))
    if ts:
        spans = [max(v) - min(v) for v in ts.values() if len(v) >= 2]
        gaps = (np.concatenate([np.diff(np.sort(v)) for v in ts.values() if len(v) >= 2])
                if spans else np.array([]))
        out["user_active_span_days"] = qdict(spans)
        out["gap_between_apps_days"] = qdict(gaps)
        if len(gaps):
            out["gap_between_apps_days"]["share_same_day"] = float((gaps < 1.0).mean())
    if career_years is not None:
        out["total_years_experience"] = qdict(career_years)
    if edu is not None:
        tot = max(len(edu), 1)
        c = defaultdict(int)
        for e in edu:
            c[e] += 1
        out["edu_share"] = {k: v / tot for k, v in c.items()}
    return out


def from_jobs(interactions, profiles):
    rows = []
    with io.open(interactions, encoding="utf-8") as f:
        next(f)
        for line in f:
            u, _, i, t = line.rstrip("\n").split(",")
            rows.append((int(u), int(i), float(t)))
    prof = list(csv.DictReader(io.open(profiles, encoding="utf-8")))
    yrs = [float(r["career_years"]) for r in prof]
    edu = [r["edu_level"] for r in prof]
    return stats_from_rows(rows, len(prof), yrs, edu)


def from_beauty(pkl):
    D = pickle.load(open(pkl, "rb"))
    rows = [(u, i, None) for u, seq in D["user_seq"].items() for i in seq]
    return stats_from_rows(rows, len(D["iid"]))


# 대조 항목: (이름, 경로, 종류). rel = 상대 오차, abs = 절대 오차(비율 지표)
ITEMS = [
    ("사용자당 건수 중앙", ("apps_per_user", "p50"), "rel"),
    ("사용자당 건수 평균", ("apps_per_user", "mean"), "rel"),
    ("사용자당 건수 p90", ("apps_per_user", "p90"), "rel"),
    ("3건 이상 사용자 비율", ("apps_per_user", "share_ge3"), "abs"),
    ("아이템 Gini (1건 이상)", ("apps_per_job_applied_only", "gini"), "abs"),
    ("상위 50% 아이템 점유율", ("apps_per_job_applied_only", "top50_share"), "abs"),
    ("3건 이하 아이템 비율", ("apps_per_job_applied_only", "share_le3"), "abs"),
    ("0건 아이템 비율", ("share_jobs_with_zero_apps",), "abs"),
    ("활동 기간 중앙(일)", ("user_active_span_days", "p50"), "rel"),
    ("경력 연수 중앙", ("total_years_experience", "p50"), "rel"),
]


def get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def compare(gen, target, tol):
    rows = []
    for name, path, kind in ITEMS:
        g, t = get(gen, path), get(target, path)
        key = "/".join(path)
        tl = tol.get(key, tol.get("default_" + kind))
        if g is None or t is None or tl is None:
            rows.append((name, key, g, t, None, "SKIP"))
            continue
        if tl == "report":   # 구조적으로 비교 불가한 항목: 값만 보고, 판정·점수 제외
            err = abs(g - t) / max(abs(t), 1e-9) if kind == "rel" else abs(g - t)
            rows.append((name, key, g, t, err, "REPORT"))
            continue
        err = abs(g - t) / max(abs(t), 1e-9) if kind == "rel" else abs(g - t)
        warn, fail = tl if isinstance(tl, list) else (tl, tl * 2)
        v = "PASS" if err <= warn else ("WARN" if err <= fail else "FAIL")
        rows.append((name, key, g, t, err, v))
    return rows


def score(rows):
    """캘리브레이션용 단일 점수: 항목별 오차/WARN한계 의 합. 작을수록 좋다."""
    return None


def fmt(x):
    if x is None:
        return "-"
    return "%.4f" % x if abs(x) < 10 else "%.1f" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions")
    ap.add_argument("--profiles")
    ap.add_argument("--beauty", help="Beauty split pkl → 참고선 JSON 만 만든다")
    ap.add_argument("--target")
    ap.add_argument("--tolerance", default=os.path.join(ROOT, "data_gen", "spec", "g3_tolerance.json"))
    ap.add_argument("--out")
    ap.add_argument("--md")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if a.beauty:
        st = from_beauty(a.beauty)
        st["source"] = os.path.basename(a.beauty)
        if a.out:
            json.dump(st, io.open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(json.dumps({k: v for k, v in st.items() if not isinstance(v, dict)}, indent=1, ensure_ascii=False))
        print("→ %s" % a.out)
        return 0

    if not (a.interactions and a.profiles):
        ap.error("--interactions 와 --profiles 가 필요합니다 (또는 --beauty)")
    st = from_jobs(a.interactions, a.profiles)
    result = {"generated": st}
    rc = 0
    if a.target:
        target = json.load(io.open(a.target, encoding="utf-8"))
        tol = json.load(io.open(a.tolerance, encoding="utf-8")) if os.path.exists(a.tolerance) else {}
        rows = compare(st, target, tol)
        result["target"] = target.get("source", a.target)
        result["verdict"] = [dict(zip(("name", "key", "gen", "target", "err", "verdict"), r)) for r in rows]
        n_fail = sum(1 for r in rows if r[5] == "FAIL")
        n_warn = sum(1 for r in rows if r[5] == "WARN")
        result["pass"] = n_fail == 0
        rc = 0 if n_fail == 0 else 1
        L = ["| 항목 | 생성 | 목표 | 오차 | 판정 |", "|---|---|---|---|---|"]
        for name, key, g, t, err, v in rows:
            L.append("| %s | %s | %s | %s | %s |" % (name, fmt(g), fmt(t), fmt(err), v))
        print("\n".join(L))
        print("\nG3 %s · FAIL %d · WARN %d" % ("통과" if n_fail == 0 else "실패", n_fail, n_warn))
        if a.md:
            io.open(a.md, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(result, io.open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("→ %s" % a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
