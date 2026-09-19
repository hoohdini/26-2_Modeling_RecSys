# -*- coding: utf-8 -*-
"""Beauty LOO 시드 확장(P1) 판정표 — 노션 실험 계획 5절의 규칙 A·E 를 그대로 적용한다.

    A. 튜닝식(b10)이 고전 블렌딩(a07)보다 낫다: n=5 에서 NDCG@10 짝지은 차의 평균이 바닥(0.0016)보다 크고 5/5 부호 일치
    E. 다양성 개선: APLT@50 5시드 평균 차이(b10 − base)가 5시드 바닥보다 크다
    참고: 그래프 > 텍스트 (b10 − base) 도 같은 방식으로 다시 잰다 (기존 3시드 결론의 n=5 확인)

바닥 = 같은 레시피에서 시드만 바꾼 실행들의 최대 간격 (조건별 max−min 의 최대).
입력: results_gsid_week/ns_{base,b10,a07}_s{42,7,13,3,21}.json   (tiger_to_eval.py 출력)
    RESULTS_DIR · SEEDS 환경변수로 바꿀 수 있다.
"""
import glob
import json
import math
import os
import statistics as S
import sys

ROOT = next((p for p in ("/mnt/data1/dsl05/recsys", "/data1/dsl05/recsys", os.path.expanduser("~/recsys"))
             if os.path.isdir(p)), ".")
RES = os.environ.get("RESULTS_DIR", "results_gsid_week")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,7,13,3,21").split(",")]
KS = ["recall@10", "ndcg@10", "recall@50", "ndcg@50", "aplt@50", "coverage@10", "COLD_recall@50"]
FLOOR_NDCG_PREREG = 0.0016     # 노션 5절 A 의 고정 바닥 (3시드 실측)
COND = {"base": "텍스트", "b10": "튜닝식 β₀1.0", "a07": "고전 α0.7"}
TCRIT = {2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943}


def load():
    R = {}
    for f in glob.glob(os.path.join(ROOT, RES, "*.json")):
        try:
            for k, v in json.load(open(f)).items():
                if isinstance(v, dict) and "ndcg@10" in v:
                    R[k] = v
        except Exception:
            pass
    return R


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    R = load()
    have = [s for s in SEEDS if all("ns_%s_s%d" % (c, s) in R for c in COND)]
    miss = [("ns_%s_s%d" % (c, s)) for s in SEEDS for c in COND if "ns_%s_s%d" % (c, s) not in R]
    print("Beauty LOO 시드 판정 · 결과 폴더 %s · 완성 시드 %s" % (RES, have))
    if miss:
        print("  아직 없는 셀: " + ", ".join(miss))
    if len(have) < 2:
        print("판정 불가 — 완성 시드가 2개 미만")
        return
    print()
    print("[원자료]")
    print("%-14s" % "cell" + "".join("%14s" % k for k in KS))
    for c in COND:
        for s in have:
            n = "ns_%s_s%d" % (c, s)
            print("%-14s" % n + "".join("%14.5f" % R[n][k] for k in KS))
    print()
    print("[시드 평균 ± sd · n=%d]" % len(have))
    print("%-16s" % "metric" + "".join("%22s" % COND[c] for c in COND))
    for k in KS:
        row = "%-16s" % k
        for c in COND:
            x = [R["ns_%s_s%d" % (c, s)][k] for s in have]
            row += "%13.5f±%.5f" % (S.mean(x), S.stdev(x) if len(x) > 1 else 0.0)
        print(row)

    def paired(c1, c2, k):
        e = [R["ns_%s_s%d" % (c1, s)][k] - R["ns_%s_s%d" % (c2, s)][k] for s in have]
        fl = max(max(R["ns_%s_s%d" % (c, s)][k] for s in have) - min(R["ns_%s_s%d" % (c, s)][k] for s in have)
                 for c in (c1, c2))
        same = all(x > 0 for x in e) or all(x < 0 for x in e)
        m = S.mean(e)
        sd = S.stdev(e) if len(e) > 1 else 0.0
        t = m / (sd / math.sqrt(len(e))) if sd else float("inf")
        return e, fl, same, m, t

    for c1, c2, title in (("b10", "base", "그래프(튜닝식) − 텍스트"), ("b10", "a07", "튜닝식 − 고전 블렌딩 (규칙 A)"),
                          ("a07", "base", "고전 블렌딩 − 텍스트")):
        print()
        print("[%s]  통과 = 부호 %d/%d 일치 AND 평균차 > 바닥" % (title, len(have), len(have)))
        print("%-16s %s %10s %10s %8s %6s  %s" % ("metric", "".join("%10s" % ("s%d" % s) for s in have), "평균", "바닥", "t", "부호", "판정"))
        for k in KS:
            e, fl, same, m, t = paired(c1, c2, k)
            floor = FLOOR_NDCG_PREREG if (k == "ndcg@10" and c2 == "a07") else fl
            v = "PASS" if (same and abs(m) > floor) else ("부호반전" if not same else "바닥 아래")
            if k == "ndcg@10" and c2 == "a07":
                v += " (바닥 = 사전 고정 0.0016)"
            print("%-16s %s %+10.5f %10.5f %8.2f %6s  %s" % (k, "".join("%+10.5f" % x for x in e), m, floor, t,
                                                            "%d/%d" % (sum(1 for x in e if (x > 0) == (m > 0)), len(e)), v))
        print("  짝지은 t 임계값(단측 α=.05, df=%d): %.3f" % (len(have) - 1, TCRIT.get(len(have) - 1, float("nan"))))

    print()
    print("[규칙 E · 다양성] APLT@50: b10 − base 평균 vs 5시드 바닥")
    e, fl, same, m, t = paired("b10", "base", "aplt@50")
    print("  평균차 %+.5f · 바닥 %.5f · %s" % (m, fl, "PASS — 개선" if (same and m > fl) else "측정 불가 유지"))
    if len(have) < 5:
        print()
        print("⚠️ 시드 %d개. 규칙 A·E 는 n=5 가 다 나와야 확정이다." % len(have))


if __name__ == "__main__":
    main()
