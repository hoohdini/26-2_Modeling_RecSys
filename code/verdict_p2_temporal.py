# -*- coding: utf-8 -*-
"""Beauty Temporal 예산 통일(P2) 판정표 — 노션 실험 계획 5절의 규칙 C 를 그대로 적용한다.

    C. 세 트랙 3시드 재현: W4, W5 각각 (튜닝식 b10 − 텍스트) 가 3/3 부호 일치이고 평균 차이가 바닥보다 크다.
       통과 못 하면 해당 윈도우는 방향만 보고한다.

바닥 = 같은 레시피에서 시드만 바꾼 실행들의 최대 간격 (조건별 max−min 의 최대). 윈도우마다 따로 잰다.
입력: results_p2_temporal/tp_{text,b10}_W{4,5}_s{42,7,13}.json   (tiger_to_eval.py --window 출력)
    RESULTS_DIR · SEEDS 환경변수로 바꿀 수 있다. LOO 참고선은 results_gsid_week/ns_{base,b10}_s*.json 이 있으면 같이 찍는다.
"""
import glob
import json
import math
import os
import statistics as S
import sys

ROOT = next((p for p in ("/mnt/data1/dsl05/recsys", "/data1/dsl05/recsys", os.path.expanduser("~/recsys"))
             if os.path.isdir(p)), ".")
RES = os.environ.get("RESULTS_DIR", "results_p2_temporal")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,7,13").split(",")]
WINDOWS = ["W4", "W5"]
KS = ["recall@10", "ndcg@10", "recall@50", "ndcg@50", "aplt@50", "coverage@10", "COLD_recall@50"]
COND = {"text": "텍스트", "b10": "튜닝식 β₀1.0"}
TCRIT = {2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943}


def load(res):
    R = {}
    for f in glob.glob(os.path.join(ROOT, res, "*.json")):
        try:
            for k, v in json.load(open(f)).items():
                if isinstance(v, dict) and "ndcg@10" in v:
                    R[k] = v
        except Exception:
            pass
    return R


def paired(R, name, c1, c2, have, k):
    e = [R[name(c1, s)][k] - R[name(c2, s)][k] for s in have]
    fl = max(max(R[name(c, s)][k] for s in have) - min(R[name(c, s)][k] for s in have) for c in (c1, c2))
    same = all(x > 0 for x in e) or all(x < 0 for x in e)
    m = S.mean(e)
    sd = S.stdev(e) if len(e) > 1 else 0.0
    t = m / (sd / math.sqrt(len(e))) if sd else float("inf")
    return e, fl, same, m, t


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    R = load(RES)
    summary = {}
    for W in WINDOWS:
        name = lambda c, s, W=W: "tp_%s_%s_s%d" % (c, W, s)
        have = [s for s in SEEDS if all(name(c, s) in R for c in COND)]
        miss = [name(c, s) for s in SEEDS for c in COND if name(c, s) not in R]
        print("=" * 100)
        print("Temporal %s · 결과 폴더 %s · 완성 시드 %s" % (W, RES, have))
        if miss:
            print("  아직 없는 셀: " + ", ".join(miss))
        if len(have) < 2:
            print("  판정 불가 — 완성 시드가 2개 미만")
            continue
        print()
        print("[원자료]  (채점 유저 %s명)" % R[name("text", have[0])].get("n_user_eval", "?"))
        print("%-16s" % "cell" + "".join("%14s" % k for k in KS))
        for c in COND:
            for s in have:
                n = name(c, s)
                print("%-16s" % n + "".join("%14.5f" % R[n][k] for k in KS))
        print()
        print("[시드 평균 ± sd · n=%d]" % len(have))
        print("%-16s" % "metric" + "".join("%22s" % COND[c] for c in COND))
        for k in KS:
            row = "%-16s" % k
            for c in COND:
                x = [R[name(c, s)][k] for s in have]
                row += "%13.5f±%.5f" % (S.mean(x), S.stdev(x) if len(x) > 1 else 0.0)
            print(row)
        print()
        print("[%s · 그래프(튜닝식) − 텍스트]  규칙 C 통과 = 부호 %d/%d 일치 AND 평균차 > 바닥" % (W, len(have), len(have)))
        print("%-16s %s %10s %10s %8s %6s  %s" % ("metric", "".join("%10s" % ("s%d" % s) for s in have), "평균", "바닥", "t", "부호", "판정"))
        for k in KS:
            e, fl, same, m, t = paired(R, name, "b10", "text", have, k)
            v = "PASS" if (same and abs(m) > fl) else ("부호반전" if not same else "바닥 아래")
            print("%-16s %s %+10.5f %10.5f %8.2f %6s  %s" % (k, "".join("%+10.5f" % x for x in e), m, fl, t,
                                                            "%d/%d" % (sum(1 for x in e if (x > 0) == (m > 0)), len(e)), v))
            if k == "ndcg@10":
                summary[W] = (len(have), same, m, fl, v)
        print("  짝지은 t 임계값(단측 α=.05, df=%d): %.3f" % (len(have) - 1, TCRIT.get(len(have) - 1, float("nan"))))

    print()
    print("=" * 100)
    print("[규칙 C 요약 · NDCG@10 · 그래프 − 텍스트]")
    L = load("results_gsid_week")
    loo = [s for s in (42, 7, 13, 3, 21) if "ns_b10_s%d" % s in L and "ns_base_s%d" % s in L]
    if len(loo) >= 2:
        e, fl, same, m, t = paired(L, lambda c, s: "ns_%s_s%d" % (c, s), "b10", "base", loo, "ndcg@10")
        print("  LOO (참고, n=%d): 평균차 %+.5f · 바닥 %.5f · %s" % (len(loo), m, fl, "PASS" if (same and m > fl) else "미달"))
    for W in WINDOWS:
        if W in summary:
            n, same, m, fl, v = summary[W]
            print("  %s (n=%d): 평균차 %+.5f · 바닥 %.5f · %s%s" % (W, n, m, fl, v, "" if n >= 3 else " (3시드 미완)"))
        else:
            print("  %s: 판정 불가" % W)
    ok = all(W in summary and summary[W][0] >= 3 and summary[W][4] == "PASS" for W in WINDOWS)
    print("  → 규칙 C %s" % ("통과 — 세 트랙(LOO·W4·W5) 3시드 재현 주장 가능" if ok else "미통과 — 통과 못 한 윈도우는 방향만 보고한다"))
    print()
    print("주의: Temporal 은 내부 검증 정답이 placeholder 라 patience 8 조기종료가 이르게 걸린다(6k~9k 최적). 이는 LOO 와 같은 레시피의")
    print("      일부이며 바꾸지 않았다. 멈춘 스텝은 보고서 끝의 '셀별 소요' 절에 있다. 1시드 수치는 인용하지 않는다.")


if __name__ == "__main__":
    main()
