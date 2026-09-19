# -*- coding: utf-8 -*-
"""P4 학습 예산 민감도(규칙 D) + P5 접두사 제약 덤프 표.

    D. 예산 무관: 15k, 45k 에서 (그래프 b10 − 텍스트) NDCG@10 이 2/2 부호 일치 (시드 42, 7).
       못 하면 "30k 에서만 확인됐다" 고 한정해서 쓴다. 30k 참고선은 P1 n=5 (results_gsid_week).
    P5. 접두사 제약 ON/OFF · 텍스트/그래프 × 3시드: 정확도·롱테일·bucket_recall@50(zero-shot 포함)·무효 SID 비율.
       구인구직 §8 표와 같은 형식. 임계값 없음, 수치 그대로 보고. 1시드 수치는 인용하지 않는다.

입력: results_p4/p4_{base,b10}_{15k,45k}_s{42,7}.json · results_p5/nsP_{base,b10}_s{42,7,13}.json
      results_gsid_week/ns_{base,b10}_s*.json (OFF 원본 · 30k 참고선)
"""
import glob
import json
import math
import os
import statistics as S
import sys

ROOT = next((p for p in ("/mnt/data1/dsl05/recsys", "/data1/dsl05/recsys", os.path.expanduser("~/recsys"))
             if os.path.isdir(p)), ".")
KS = ["recall@10", "ndcg@10", "recall@50", "ndcg@50", "aplt@50", "coverage@10", "COLD_recall@50"]
BUCKETS = ["zero-shot", "few-shot", "low", "mid", "head"]
TCRIT = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015}


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


def mean_sd(x):
    return S.mean(x), (S.stdev(x) if len(x) > 1 else 0.0)


def paired_t(e):
    m = S.mean(e)
    sd = S.stdev(e) if len(e) > 1 else 0.0
    return m, (m / (sd / math.sqrt(len(e))) if sd else float("inf"))


def part_p4():
    R = load("results_p4")
    W = load("results_gsid_week")
    print("=" * 100)
    print("[P4] 학습 예산 민감도 · 규칙 D = 15k, 45k 각각 (b10 − base) NDCG@10 2/2 부호 일치")
    seeds30 = [s for s in (42, 7, 13, 3, 21) if "ns_b10_s%d" % s in W and "ns_base_s%d" % s in W]
    rows = {}
    verdict = {}
    for st, seeds, src, pre in (("15k", (42, 7), R, "p4_%s_15k_s%d"), ("30k", seeds30, W, "ns_%s_s%d"), ("45k", (42, 7), R, "p4_%s_45k_s%d")):
        have = [s for s in seeds if pre % ("base", s) in src and pre % ("b10", s) in src]
        miss = [pre % (c, s) for s in seeds for c in ("base", "b10") if pre % (c, s) not in src]
        print()
        print("--- %s 스텝 · 완성 시드 %s%s" % (st, have, ("  아직 없는 셀: " + ", ".join(miss)) if miss else ""))
        if not have:
            continue
        print("%-18s" % "cell" + "".join("%14s" % k for k in KS))
        for c in ("base", "b10"):
            for s in have:
                n = pre % (c, s)
                print("%-18s" % n + "".join("%14.5f" % src[n][k] for k in KS))
        print("%-18s" % "b10 − base" + "".join("%14s" % "" for _ in KS))
        line = {}
        for k in KS:
            e = [src[pre % ("b10", s)][k] - src[pre % ("base", s)][k] for s in have]
            same = all(x > 0 for x in e) or all(x < 0 for x in e)
            m, t = paired_t(e)
            line[k] = (m, same, len(have))
            rows.setdefault(k, {})[st] = (m, same, len(have))
        for k in KS:
            m, same, n = line[k]
            print("  %-16s 평균차 %+.5f · 부호 %s (%d/%d)" % (k, m, "일치" if same else "불일치",
                                                            sum(1 for s in have if (src[pre % ("b10", s)][k] - src[pre % ("base", s)][k] > 0) == (m > 0)), n))
        if st != "30k":
            m, same, n = line["ndcg@10"]
            verdict[st] = (n >= 2 and same and m > 0)
    print()
    print("[규칙 D 요약 · NDCG@10 b10 − base]")
    for st in ("15k", "30k", "45k"):
        if st in rows.get("ndcg@10", {}):
            m, same, n = rows["ndcg@10"][st]
            print("  %s: 평균차 %+.5f · 부호 %s · n=%d%s" % (st, m, "일치" if same else "불일치", n, "" if st == "30k" else ("  → " + ("PASS" if verdict.get(st) else "미달"))))
        else:
            print("  %s: 결과 없음" % st)
    ok = verdict.get("15k") and verdict.get("45k")
    print("  → 규칙 D %s" % ("통과 — 그래프 이득이 예산(15k·30k·45k)과 무관" if ok else "미통과 — 30k 에서만 확인됐다고 한정해서 쓴다"))
    print("  주의: 코사인 스케줄이 총 스텝에 맞춰 늘어나므로 45k 의 30k 시점은 30k 실행과 같지 않다. patience 8 조기종료로 45k 도 일찍 멈출 수 있다(멈춘 스텝은 보고서 끝).")


def part_p5():
    P = load("results_p5")
    W = load("results_gsid_week")
    print()
    print("=" * 100)
    print("[P5] 접두사 제약 디코딩 ON/OFF · Beauty LOO · 3시드 평균 (구인구직 §8 과 같은 표)")
    seeds = (42, 7, 13)
    cond = {"base": ("텍스트", "ns_base_s%d", "nsP_base_s%d"), "b10": ("그래프 b10", "ns_b10_s%d", "nsP_b10_s%d")}
    have = {c: [s for s in seeds if cond[c][1] % s in W and cond[c][2] % s in P] for c in cond}
    miss = [cond[c][2] % s for c in cond for s in seeds if cond[c][2] % s not in P]
    if miss:
        print("  아직 없는 셀: " + ", ".join(miss))
    for c in cond:
        if not have[c]:
            print("  %s: 결과 없음" % cond[c][0])
            return
    print("  완성 시드: " + ", ".join("%s %s" % (cond[c][0], have[c]) for c in cond))

    def col(c, on, k):
        src, pat = (P, cond[c][2]) if on else (W, cond[c][1])
        return [src[pat % s][k] for s in have[c]]

    print()
    print("%-18s %14s %14s %14s %14s" % ("metric", "text OFF", "text ON", "gsid OFF", "gsid ON"))
    for k in KS:
        print("%-18s" % k + "".join("%14.5f" % S.mean(col(c, on, k)) for c in ("base", "b10") for on in (False, True)))
    print("%-18s" % "무효 SID 비율" + "".join("%14.5f" % S.mean([(P if on else W)[(cond[c][2] if on else cond[c][1]) % s]["_source"]["invalid_sid_rate"] for s in have[c]]) for c in ("base", "b10") for on in (False, True)))
    print()
    print("bucket_recall@50 · 3시드 평균 (n = 정답 수)")
    print("%-12s %7s %14s %14s %14s %14s" % ("bucket", "n", "text OFF", "text ON", "gsid OFF", "gsid ON"))
    for b in BUCKETS:
        vals = []
        n = None
        for c in ("base", "b10"):
            for on in (False, True):
                src, pat = (P, cond[c][2]) if on else (W, cond[c][1])
                xs = [src[pat % s]["bucket_recall@50"].get(b, {}).get("recall", float("nan")) for s in have[c]]
                n = n or src[pat % have[c][0]]["bucket_recall@50"].get(b, {}).get("n")
                vals.append(S.mean(xs))
        print("%-12s %7s" % (b, n) + "".join("%14.5f" % v for v in vals))
    print()
    print("[제약 ON 에서 그래프 − 텍스트 · 짝지은 t]  (같은 시드끼리)")
    common = [s for s in seeds if s in have["base"] and s in have["b10"]]
    if len(common) >= 2:
        for k in ("recall@10", "ndcg@10", "recall@50", "ndcg@50", "COLD_recall@50"):
            e = [P[cond["b10"][2] % s][k] - P[cond["base"][2] % s][k] for s in common]
            m, t = paired_t(e)
            print("  %-16s 평균차 %+.5f · t %.2f (임계 %.3f, df=%d) · %s" % (k, m, t, TCRIT.get(len(common) - 1, float("nan")), len(common) - 1,
                                                                        "유의" if t > TCRIT.get(len(common) - 1, 99) else "no"))
    print()
    print("[제약 ON − OFF · 조건별 순효과 · 3시드 평균]")
    for c in ("base", "b10"):
        for k in ("recall@10", "ndcg@10", "recall@50", "COLD_recall@50"):
            e = [P[cond[c][2] % s][k] - W[cond[c][1] % s][k] for s in have[c]]
            m, t = paired_t(e)
            base = S.mean(col(c, False, k))
            print("  %-10s %-16s %+.5f (%+.1f%%)" % (cond[c][0], k, m, 100 * m / base if base else float("nan")))
    z = {c: {on: S.mean(col(c, on, "COLD_recall@50")) for on in (False, True)} for c in cond}
    zs = {c: {on: S.mean([(P if on else W)[(cond[c][2] if on else cond[c][1]) % s]["bucket_recall@50"].get("zero-shot", {}).get("recall", 0.0) for s in have[c]]) for on in (False, True)} for c in cond}
    print()
    print("[요약] zero-shot recall@50: text OFF %.5f · ON %.5f · gsid OFF %.5f · ON %.5f" % (zs["base"][False], zs["base"][True], zs["b10"][False], zs["b10"][True]))
    print("       구인구직에서는 그래프 + 제약 ON 에서만 0 을 벗어났다(0.00041). Beauty 의 zero-shot 정답은 138개뿐이라 한두 개 차이로 값이 흔들린다. 과장 금지.")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    part_p4()
    part_p5()


if __name__ == "__main__":
    main()
