# -*- coding: utf-8 -*-
"""G4 인간 블라인드 검수 시트 — 생성 프로필 50건 + 캐글 실제 프로필 요약 50건.

무엇을 재나
-----------
사람이 **속성 조합**(직무·경력·학력·전공)만 보고 생성/실제를 구분하는지 본다.
두 쪽을 **같은 문장 템플릿**으로 렌더링한다. 그래야 문체가 아니라 조합의 그럴듯함이 검사된다.
출처를 드러내는 필드(지역: 서울 vs 미국 주, 스킬·자격증: 캐글에 없음, 전공: 캐글은 자유 텍스트)는 양쪽 다 뺀다.
캐글 직무명은 title_map.csv 로 우리 영문 직업명으로 바꿔 어휘를 맞춘다 (auto 구간만).

    python data_gen/v4/make_g4_sheet.py            # 시트 생성
    python data_gen/v4/make_g4_sheet.py --score data_gen/v4/out/g4/answers_A.csv data_gen/v4/out/g4/answers_B.csv

산출
    data_gen/v4/out/g4/sheet.csv       id, text, your_answer(빈칸)  — 팀원에게 준다
    <external>/g4_key.csv              id, source                    — 답안. 저장소 밖
    docs/G4_인간검수.md                --score 후 정확도와 평가자 간 일치도(Cohen kappa)
검수자는 sheet.csv 의 your_answer 에 gen 또는 real 을 적어 answers_<이름>.csv 로 저장한다.
"""
import argparse
import csv
import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(ROOT, "data_gen")
DEFAULT_SRC = "D:/DSL/_external/kaggle_job_recommendation"
OUTD = os.path.join(GEN, "v4", "out", "g4")

sys.path.insert(0, GEN)
from write_profile_text import EDU_EN, FIELD_EN, load_map, seniority  # noqa: E402

DEGREE_EN = {"none": "secondary education", "high school": "a high school diploma",
             "vocational": "a vocational certificate", "associate's": "an associate degree",
             "bachelor's": "a bachelor's degree", "master's": "a master's degree", "phd": "a doctoral degree"}


def render(rng, role, yrs, edu, field):
    if yrs < 1:
        span = "under a year"
    else:
        n = max(1, int(round(yrs)))
        span = "%d year%s" % (n, "" if n == 1 else "s")
    sen = seniority(yrs)
    leads = ["%s %s with %s of experience." % (sen.capitalize(), role, span),
             "Working as a %s for %s." % (role, span),
             "%s of hands-on work as a %s." % (span.capitalize(), role),
             "I have spent %s working as a %s." % (span, role)]
    bg = ("Holds %s in %s." % (edu, field)) if field else ("Holds %s." % edu)
    return "%s %s" % (leads[int(rng.integers(len(leads)))], bg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score", nargs="*", default=None)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(OUTD, exist_ok=True)
    key_path = os.path.join(a.src, "g4_key.csv")

    if a.score is not None:
        key = {r["id"]: r["source"] for r in csv.DictReader(io.open(key_path, encoding="utf-8"))}
        raters = {}
        for p in a.score:
            ans = {r["id"]: (r.get("your_answer") or "").strip().lower() for r in csv.DictReader(io.open(p, encoding="utf-8"))}
            raters[os.path.basename(p)] = ans
        L = ["# G4 인간 블라인드 검수", "", "| 평가자 | 답한 수 | 정확도 |", "|---|---|---|"]
        for nm, ans in raters.items():
            done = [i for i in key if ans.get(i) in ("gen", "real")]
            acc = sum(1 for i in done if ans[i] == key[i]) / max(len(done), 1)
            L.append("| %s | %d | %.1f%% |" % (nm, len(done), 100 * acc))
        names = list(raters)
        if len(names) >= 2:
            L += ["", "평가자 간 일치도 (Cohen kappa)", "", "| 쌍 | kappa |", "|---|---|"]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    A, B = raters[names[i]], raters[names[j]]
                    ids = [k for k in key if A.get(k) in ("gen", "real") and B.get(k) in ("gen", "real")]
                    if not ids:
                        continue
                    po = sum(1 for k in ids if A[k] == B[k]) / len(ids)
                    pa = sum(1 for k in ids if A[k] == "gen") / len(ids)
                    pb = sum(1 for k in ids if B[k] == "gen") / len(ids)
                    pe = pa * pb + (1 - pa) * (1 - pb)
                    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
                    L.append("| %s vs %s | %.3f |" % (names[i], names[j], kappa))
        L += ["", "우연 수준은 50%다. 정확도가 50%에 가까울수록 사람이 생성과 실제를 구분하지 못한다는 뜻이다.",
              "임계값을 두지 않고 수치를 그대로 보고한다 (사전등록 G4)."]
        md = os.path.join(ROOT, "docs", "G4_인간검수.md")
        io.open(md, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
        print("\n".join(L))
        print("→ " + md)
        return

    rng = np.random.default_rng(a.seed)
    jt = load_map(os.path.join(GEN, "job_titles_en.tsv"))
    prof = list(csv.DictReader(io.open(os.path.join(GEN, "out", "profiles.csv"), encoding="utf-8")))
    gen_rows = [prof[i] for i in rng.choice(len(prof), size=a.n, replace=False)]
    # 전공은 양쪽 다 뺀다. 캐글 Major 는 자유 텍스트(예: german, international business)라 어휘만으로
    # 출처가 드러난다. 학위 수준만 남긴다.
    gen_txt = [render(rng, jt.get(r["job_cd"], "Professional"), float(r["career_years"]),
                      EDU_EN.get(r["edu_level"], "a bachelor's degree"), "")
               for r in gen_rows]

    users = pd.read_csv(os.path.join(a.src, "users.tsv"), sep="\t", quoting=3, on_bad_lines="skip",
                        encoding="utf-8", encoding_errors="replace", low_memory=False)
    hist = pd.read_csv(os.path.join(a.src, "user_history.tsv"), sep="\t", quoting=3, on_bad_lines="skip",
                       encoding="utf-8", encoding_errors="replace", low_memory=False)
    tmap = {}
    mp = os.path.join(a.src, "title_map.csv")
    if os.path.exists(mp):
        for r in csv.DictReader(io.open(mp, encoding="utf-8")):
            if r["band"] == "auto":
                tmap[r["title"]] = r["job_title_en"]
    cur = hist[hist["Sequence"] == 1].dropna(subset=["JobTitle"])
    cur = cur.assign(t=cur["JobTitle"].astype(str).str.strip().str.lower())
    cur = cur[cur["t"].isin(tmap)] if tmap else cur
    m = cur.merge(users, on="UserID", how="inner")
    m = m.dropna(subset=["TotalYearsExperience"])
    m = m[pd.to_numeric(m["TotalYearsExperience"], errors="coerce").notna()]
    pick = m.iloc[rng.choice(len(m), size=a.n, replace=False)]
    real_txt = []
    for _, r in pick.iterrows():
        role = tmap.get(r["t"], str(r["JobTitle"]).strip().title())
        deg = DEGREE_EN.get(str(r.get("DegreeType", "")).strip().lower(), "a bachelor's degree")
        real_txt.append(render(rng, role, float(r["TotalYearsExperience"]), deg, ""))

    items = [("gen", t) for t in gen_txt] + [("real", t) for t in real_txt]
    order = rng.permutation(len(items))
    with io.open(os.path.join(OUTD, "sheet.csv"), "w", encoding="utf-8", newline="") as f, \
            io.open(key_path, "w", encoding="utf-8", newline="") as fk:
        w = csv.writer(f)
        wk = csv.writer(fk)
        w.writerow(["id", "text", "your_answer"])
        wk.writerow(["id", "source"])
        for n, k in enumerate(order):
            src, t = items[k]
            w.writerow(["P%03d" % (n + 1), t, ""])
            wk.writerow(["P%03d" % (n + 1), src])
    print("시트 %d건 → %s\n답안 → %s (저장소 밖)" % (len(items), os.path.join(OUTD, "sheet.csv"), key_path))
    if not tmap:
        print("주의: title_map.csv 가 없어 캐글 직무명을 원문 그대로 썼다. 어휘 차이로 구분이 쉬워질 수 있다.")


if __name__ == "__main__":
    main()
