# -*- coding: utf-8 -*-
"""v3 3시드 판정표를 만든다. 서버에서 채점 직후 자동 실행된다 (finish_v3.sh).

판정 규칙 (사전등록)
    노이즈 바닥 = 같은 레시피에 시드만 바꾼 실행들의 최대 간격
    통과       = 세 시드 부호가 모두 같고, 그중 가장 작은 효과도 바닥을 넘을 것
"""
import glob
import json
import os
import statistics as S
import sys
from datetime import datetime

ROOT = next((p for p in ("/mnt/data1/dsl05/recsys", "/data1/dsl05/recsys",
                         os.path.expanduser("~/recsys")) if os.path.isdir(p)), ".")
# v4 부터는 환경변수로 버전·결과 폴더·시드를 바꿔 같은 판정 규칙을 재사용한다.
#   JOBS_VER=v4 SEEDS=42,7,13,3,21,99 python verdict_v3.py
VER = os.environ.get("JOBS_VER", "v3")
RES_DIR = os.environ.get("RESULTS_DIR", "results_jobs_" + VER)
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,7,13").split(",")]
KS = ["recall@10", "ndcg@10", "recall@50", "ndcg@50", "aplt@50", "coverage@10", "COLD_recall@50"]


def load(d):
    R = {}
    for f in glob.glob(os.path.join(ROOT, d, "*.json")):
        try:
            for k, v in json.load(open(f)).items():
                if isinstance(v, dict) and "ndcg@10" in v:
                    R[k] = v
        except Exception:
            pass
    return R


def main():
    R = load(RES_DIR)
    out = []
    P = out.append
    P("=" * 78)
    P("%s %d시드 판정 — 생성 %s" % (VER, len(SEEDS), datetime.now().strftime("%Y-%m-%d %H:%M")))
    P("MAX_STEPS=30000 · PATIENCE=1000(비활성) · LIMIT_VAL=150(검증 51%)")
    P("=" * 78)

    need = ["jobs_%s_%s_s%d" % (r, VER, s) for r in ("text", "gsid_b10") for s in SEEDS]
    miss = [n for n in need if n not in R]
    if miss:
        P("")
        P("⚠️ 아직 없는 셀: " + ", ".join(miss))
        P("   덤프가 끝나면 `bash ~/recsys/score_jobs_v3.sh` 후 이 스크립트를 다시 실행하세요.")
    have = [s for s in SEEDS if all("jobs_%s_%s_s%d" % (r, VER, s) in R for r in ("text", "gsid_b10"))]
    if not have:
        P("")
        P("판정 불가 — 완성된 시드 쌍이 없습니다.")
        return "\n".join(out)

    P("")
    P("[원자료]  (완성된 시드: %s)" % ", ".join(str(s) for s in have))
    P("%-24s" % "cell" + "".join("%13s" % k for k in KS))
    for r in ("text", "gsid_b10"):
        for s in have:
            n = "jobs_%s_%s_s%d" % (r, VER, s)
            P("%-24s" % n + "".join("%13.5f" % R[n][k] for k in KS))

    if len(have) >= 2:
        P("")
        P("[시드 평균 ± sd]")
        P("%-16s %19s %19s %13s" % ("metric", "text", "G-SID", "효과"))
        for k in KS:
            t = [R["jobs_text_%s_s%d" % (VER, s)][k] for s in have]
            g = [R["jobs_gsid_b10_%s_s%d" % (VER, s)][k] for s in have]
            sd = (lambda x: S.stdev(x) if len(x) > 1 else 0.0)
            P("%-16s %10.5f±%.5f %10.5f±%.5f %+13.5f"
              % (k, S.mean(t), sd(t), S.mean(g), sd(g), S.mean(g) - S.mean(t)))

        P("")
        P("[판정] 효과 vs 노이즈 바닥 — 통과 조건: 부호 일치 AND 최소효과 > 바닥")
        P("%-16s %s %10s %8s %8s  %s"
          % ("metric", "".join("%11s" % ("eff s%d" % s) for s in have), "바닥", "평균/바닥", "최소/바닥", "판정"))
        verdict = {}
        for k in KS:
            e = [R["jobs_gsid_b10_%s_s%d" % (VER, s)][k] - R["jobs_text_%s_s%d" % (VER, s)][k] for s in have]
            t = [R["jobs_text_%s_s%d" % (VER, s)][k] for s in have]
            g = [R["jobs_gsid_b10_%s_s%d" % (VER, s)][k] for s in have]
            fl = max(max(t) - min(t), max(g) - min(g))
            same = all(x > 0 for x in e) or all(x < 0 for x in e)
            mn, av = min(abs(x) for x in e), abs(sum(e) / len(e))
            v = "PASS" if (same and fl > 0 and mn > fl) else ("부호반전" if not same else "FAIL")
            verdict[k] = v
            P("%-16s %s %10.5f %8.2f %8.2f  %s"
              % (k, "".join("%+11.5f" % x for x in e), fl,
                 (av / fl if fl else 0), (mn / fl if fl else 0), v))

        P("")
        P("[짝지은 검정] 같은 시드 안의 차이를 평균 — 시드 주효과가 상쇄된다")
        P("  ※ 사전등록은 위의 바닥 방식이다. 이 절은 사후 추가한 표준 분석이며,")
        P("     짝지음을 버린 바닥 방식이 설계상 보수적이라는 근거로 함께 싣는다.")
        import math
        # one-tailed alpha .05, df = n-1. n=3 → 2.920, n=5 → 2.132, n=6 → 2.015 (JOBS_결과정리 §4-3 과 동일)
        TCRIT_TABLE = {2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}
        TCRIT = TCRIT_TABLE.get(len(have) - 1, 2.920)
        P("%-16s %11s %10s %9s %8s  %s" % ("metric", "평균효과", "sd", "sem", "t(df%d)" % (len(have) - 1), "유의(t>%.3f)" % TCRIT))
        for k in KS:
            e = [R["jobs_gsid_b10_%s_s%d" % (VER, s)][k] - R["jobs_text_%s_s%d" % (VER, s)][k] for s in have]
            if len(e) < 3:
                continue
            m = S.mean(e); sd = S.stdev(e); sem = sd / math.sqrt(len(e))
            t = m / sem if sem else 0.0
            P("%-16s %+11.5f %10.5f %9.5f %8.2f  %s"
              % (k, m, sd, sem, t, "YES" if t > TCRIT else "no"))

        # ---- G1 : rewired 대조군 ----
        gr = [s for s in SEEDS if "jobs_rewired_%s_s%d" % (VER, s) in R]
        P("")
        if not gr:
            P("[G1] rewired 셀 없음 — 아직 판정 불가.")
        else:
            P("[G1] 무작위 그래프 대조 (시드 %s) — 통과 조건: 이득이 사라진다"
              % ", ".join(str(s) for s in gr))
            P("%-16s %13s %14s  %s" % ("metric", "gsid-text", "rewired-text", "판정"))
            for k in KS:
                ge = S.mean([R["jobs_gsid_b10_%s_s%d" % (VER, s)][k] - R["jobs_text_%s_s%d" % (VER, s)][k] for s in have])
                re_ = S.mean([R["jobs_rewired_%s_s%d" % (VER, s)][k] - R["jobs_text_%s_s%d" % (VER, s)][k]
                              for s in gr if "jobs_text_%s_s%d" % (VER, s) in R])
                if ge > 0 and re_ <= 0:
                    v = "통과 — 이득 소멸"
                elif ge > 0 and re_ < ge * 0.5:
                    v = "부분통과 — 이득 축소"
                elif ge > 0:
                    v = "실패 — 이득 잔존"
                else:
                    v = "판정불가 — 이득 없음"
                P("%-16s %+13.5f %+14.5f  %s" % (k, ge, re_, v))

        P("")
        P("[요약]")
        pas = [k for k in KS if verdict[k] == "PASS"]
        P("  통과: " + (", ".join(pas) if pas else "없음"))
        P("  주 지표 ndcg@10: " + verdict["ndcg@10"])
        if len(have) < 3:
            P("  ⚠️ 시드 %d개로 잰 바닥입니다. 3개가 다 나와야 확정입니다." % len(have))

    P("")
    P("[참고] v2 (조기종료 미고정) 3시드 판정은 전 지표 FAIL 이었습니다.")
    P("       text 기준선 best step 이 5,000 / 3,000 / 17,000 으로 흩어진 것이 원인입니다.")
    return "\n".join(out)


if __name__ == "__main__":
    txt = main()
    sys.stdout.write(txt + "\n")
    with open(os.path.join(ROOT, "V3_아침보고.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
