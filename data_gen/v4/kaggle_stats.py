# -*- coding: utf-8 -*-
"""캐글 CareerBuilder 2012 (Job Recommendation Challenge) → G3 목표 분포.

무엇을 내놓나
-------------
원본 파일은 저장소 밖(D:/DSL/_external/kaggle_job_recommendation)에 두고,
여기서는 **집계 통계만** 뽑아 저장소에 넣는다 (캐글 규칙: 재배포 금지, 학술·비상업 사용 가능).

    data_gen/spec/kaggle_target_dist.json   G3 목표 분포 (집계값만)
    docs/G3_목표분포_캐글.md                 같은 내용의 표
    <external>/title_freq.tsv               직무명 빈도표 — 매핑 단계 입력. 저장소 밖에만 둔다

입력 파일 (캐글 규칙 동의 후 다운로드, 압축 해제)
    users.tsv         UserID WindowID Split City State Country ZipCode DegreeType Major
                      GraduationDate WorkHistoryCount TotalYearsExperience CurrentlyEmployed
                      ManagedOthers ManagedHowMany
    apps.tsv          UserID WindowID Split ApplicationDate JobID
    user_history.tsv  UserID WindowID Split Sequence JobTitle
    jobs.tsv          JobID WindowID Title Description Requirements City State Country Zip5
                      StartDate EndDate          (약 1.5GB · 본문은 읽지 않고 Title·WindowID 만)
    window_dates.tsv  Window "Train Start" "Train End" "Test Start" "Test End"

우리 사상과의 대응
    캐글 user(구직자) → 우리 담당자(user)   : 사용자당 지원 수 = 담당자당 컨택 수
    캐글 job(공고)   → 우리 프로필(item)    : 공고당 지원 수 = 프로필당 컨택 수
    방향이 반대이지만 **분포의 모양**(사용자당 건수, 아이템 쏠림, 밀도, 활동 기간)만 목표로 쓴다.

사용
    python data_gen/v4/kaggle_stats.py                       # 기본 외부 경로
    python data_gen/v4/kaggle_stats.py --src D:/path/to/dir  # 다른 위치
    python data_gen/v4/kaggle_stats.py --skip-jobs           # jobs.tsv(1.5GB) 생략
"""
import argparse
import io
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC = os.path.join(ROOT, "data_gen", "spec")
DOCS = os.path.join(ROOT, "docs")
DEFAULT_SRC = "D:/DSL/_external/kaggle_job_recommendation"

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


def find(src, name):
    for cand in (name, name + ".gz", name.replace(".tsv", ".zip")):
        p = os.path.join(src, cand)
        if os.path.exists(p):
            return p
    return None


def read_tsv(path, usecols=None, **kw):
    # 캐글 원본은 탭 구분이며 본문에 따옴표·개행이 섞여 있어 quoting 을 끈다.
    return pd.read_csv(path, sep="\t", usecols=usecols, quoting=3, on_bad_lines="skip",
                       encoding="utf-8", encoding_errors="replace", low_memory=False, **kw)


def is_train(s):
    return s.astype(str).str.strip().str.lower().str.startswith("train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out-json", default=os.path.join(SPEC, "kaggle_target_dist.json"))
    ap.add_argument("--out-md", default=os.path.join(DOCS, "G3_목표분포_캐글.md"))
    ap.add_argument("--skip-jobs", action="store_true", help="jobs.tsv(1.5GB) 를 읽지 않는다")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    need = {n: find(a.src, n) for n in ("users.tsv", "apps.tsv", "user_history.tsv")}
    miss = [n for n, p in need.items() if p is None]
    if miss:
        sys.exit("캐글 파일 없음: %s\n  %s 에 압축 해제한 tsv 를 두세요. "
                 "(kaggle.com/c/job-recommendation/data 에서 규칙 동의 후 다운로드)" % (miss, a.src))

    print("읽는 중: " + a.src)
    users = read_tsv(need["users.tsv"])
    apps = read_tsv(need["apps.tsv"])
    hist = read_tsv(need["user_history.tsv"])
    for df in (users, apps, hist):
        df.columns = [c.strip() for c in df.columns]
    for df, cols, nm in ((users, ["UserID", "WindowID", "Split"], "users"),
                         (apps, ["UserID", "WindowID", "Split", "ApplicationDate", "JobID"], "apps"),
                         (hist, ["UserID", "WindowID", "Sequence", "JobTitle"], "user_history")):
        lack = [c for c in cols if c not in df.columns]
        if lack:
            sys.exit("%s.tsv 에 열이 없음: %s · 실제 열: %s" % (nm, lack, list(df.columns)))

    apps["ApplicationDate"] = pd.to_datetime(apps["ApplicationDate"], errors="coerce")
    out = {"source": "Kaggle Job Recommendation Challenge (CareerBuilder 2012)",
           "note": "집계 통계만 저장. 원본·파생 파일은 저장소에 올리지 않는다.",
           "n_users": int(users["UserID"].nunique()), "n_apps": int(len(apps)),
           "n_windows": int(apps["WindowID"].nunique()), "windows": {}}

    # ---- 사용자당 지원 수 (train split, 지원 1건 이상인 사용자) ----
    tr = apps[is_train(apps["Split"])]
    per_user = tr.groupby("UserID").size()
    users_tr = users[is_train(users["Split"])]
    n_users_tr = int(users_tr["UserID"].nunique())
    out["apps_per_user"] = qdict(per_user.values)
    out["apps_per_user"].update(
        share_users_with_any_app=float(len(per_user) / max(n_users_tr, 1)),
        share_ge3=float((per_user >= 3).mean()), share_ge5=float((per_user >= 5).mean()))

    # ---- 공고당 지원 수 (train) ----
    per_job = tr.groupby("JobID").size()
    pj = np.sort(per_job.values)[::-1]
    out["apps_per_job_applied_only"] = qdict(pj)
    out["apps_per_job_applied_only"].update(
        gini=gini(pj), top50_share=float(pj[: len(pj) // 2].sum() / pj.sum()),
        top20_share=float(pj[: max(1, len(pj) // 5)].sum() / pj.sum()),
        share_le3=float((pj <= 3).mean()), share_le5=float((pj <= 5).mean()))

    # ---- 윈도우별 밀도·기간 ----
    for w, g in tr.groupby("WindowID"):
        nu = int(g["UserID"].nunique())
        nj = int(g["JobID"].nunique())
        dt = g["ApplicationDate"].dropna()
        span = int((dt.max() - dt.min()).days) if len(dt) else None
        out["windows"][str(int(w))] = {
            "n_apps": int(len(g)), "n_users_applied": nu, "n_jobs_applied": nj,
            "density_applied": float(len(g) / max(nu * nj, 1)),
            "span_days": span, "apps_per_day": float(len(g) / max(span or 1, 1))}

    # ---- 사용자 활동 기간(첫 지원~마지막 지원, 일)과 지원 간격 ----
    gb = tr.dropna(subset=["ApplicationDate"]).sort_values(["UserID", "ApplicationDate"])
    span_u = gb.groupby("UserID")["ApplicationDate"].agg(lambda s: (s.max() - s.min()).days)
    multi = span_u[per_user.reindex(span_u.index).fillna(0) >= 2]
    out["user_active_span_days"] = qdict(multi.values)
    gaps = gb.groupby("UserID")["ApplicationDate"].diff().dt.total_seconds().dropna() / 86400.0
    out["gap_between_apps_days"] = qdict(gaps.values)
    out["gap_between_apps_days"]["share_same_day"] = float((gaps < 1.0).mean()) if len(gaps) else None

    # ---- 사용자 속성 ----
    if "TotalYearsExperience" in users_tr.columns:
        ue = pd.to_numeric(users_tr["TotalYearsExperience"], errors="coerce").dropna()
        out["total_years_experience"] = qdict(ue.values)
    if "DegreeType" in users_tr.columns:
        d = users_tr["DegreeType"].fillna("None").astype(str).str.strip().value_counts(normalize=True)
        out["degree_type_share"] = {k: float(v) for k, v in d.items()}
    if "WorkHistoryCount" in users_tr.columns:
        out["work_history_count"] = qdict(
            pd.to_numeric(users_tr["WorkHistoryCount"], errors="coerce").dropna().values)
    if "ManagedOthers" in users_tr.columns:
        m = users_tr["ManagedOthers"].astype(str).str.lower().isin(["yes", "true", "1"])
        out["managed_others_share"] = float(m.mean())
    if "State" in users_tr.columns:
        st = users_tr["State"].fillna("").astype(str).value_counts(normalize=True)
        out["user_state_top10_share"] = float(st.head(10).sum())

    # ---- 직무명 빈도 (저장소 밖에만 저장) ----
    titles = Counter()
    ht = hist["JobTitle"].dropna().astype(str).str.strip()
    titles.update(ht[ht != ""].str.lower().tolist())
    n_job_titles = 0
    if not a.skip_jobs:
        jp = find(a.src, "jobs.tsv")
        if jp:
            try:
                jt = read_tsv(jp, usecols=["JobID", "WindowID", "Title"])
                t = jt["Title"].dropna().astype(str).str.strip()
                titles.update(t[t != ""].str.lower().tolist())
                n_job_titles = int(len(t))
                out["n_jobs_total"] = int(jt["JobID"].nunique())
                applied = set(tr["JobID"].unique())
                out["share_jobs_with_zero_apps"] = float((~jt["JobID"].isin(applied)).mean())
            except Exception as e:  # 본문 따옴표 문제로 실패해도 나머지는 살린다
                print("  jobs.tsv 읽기 실패 (건너뜀): %s" % e)
    tf_path = os.path.join(a.src, "title_freq.tsv")
    with io.open(tf_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("title\tcount\n")
        for t, c in titles.most_common():
            f.write("%s\t%d\n" % (t.replace("\t", " "), c))
    tot = max(sum(titles.values()), 1)
    out["title_vocab"] = {"n_history_rows": int(len(ht)), "n_job_title_rows": n_job_titles,
                          "n_distinct_titles": int(len(titles)),
                          "top500_coverage": float(sum(c for _, c in titles.most_common(500)) / tot)}

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(out, io.open(a.out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---- 표 ----
    apu = out["apps_per_user"]
    apj = out["apps_per_job_applied_only"]
    dens = [v["density_applied"] for v in out["windows"].values()]
    g = lambda d, k: d.get(k, float("nan"))
    L = ["# G3 목표 분포 — 캐글 CareerBuilder 2012 실측", "",
         "생성 스크립트 `data_gen/v4/kaggle_stats.py`. 원본은 저장소 밖에 두고 집계값만 기록한다.", "",
         "| 항목 | 값 |", "|---|---|",
         "| 사용자 | %s |" % format(out["n_users"], ","),
         "| 지원(train) | %s |" % format(len(tr), ","),
         "| 윈도우 | %d |" % out["n_windows"],
         "| 사용자당 지원 수 중앙/평균 | %.1f / %.2f |" % (apu["p50"], apu["mean"]),
         "| 사용자당 지원 p90/p99 | %.0f / %.0f |" % (apu["p90"], apu["p99"]),
         "| 지원 3건 이상 사용자 비율 (LOO 채점 가능) | %.1f%% |" % (100 * apu["share_ge3"]),
         "| 공고당 지원 Gini (지원 1건 이상 공고) | %.4f |" % apj["gini"],
         "| 상위 50%% 공고의 지원 점유율 | %.1f%% |" % (100 * apj["top50_share"]),
         "| 지원 3건 이하 공고 비율 | %.1f%% |" % (100 * apj["share_le3"])]
    if "share_jobs_with_zero_apps" in out:
        L.append("| 지원 0건 공고 비율 (전체 공고 중) | %.1f%% |" % (100 * out["share_jobs_with_zero_apps"]))
    L += ["| 윈도우 밀도 (지원/사용자×공고) 중앙 | %.2e |" % np.median(dens),
          "| 사용자 활동 기간 중앙 (일, 2건 이상) | %.1f |" % g(out["user_active_span_days"], "p50"),
          "| 지원 간격 중앙 (일) | %.2f |" % g(out["gap_between_apps_days"], "p50"),
          "| 경력 연수 중앙/평균 | %.1f / %.1f |" % (g(out.get("total_years_experience", {}), "p50"),
                                                  g(out.get("total_years_experience", {}), "mean")),
          "| 직무명 종류 / 상위 500 커버리지 | %s / %.1f%% |" % (
              format(out["title_vocab"]["n_distinct_titles"], ","), 100 * out["title_vocab"]["top500_coverage"])]
    if "degree_type_share" in out:
        L += ["", "학력 분포", "", "| DegreeType | 비율 |", "|---|---|"]
        L += ["| %s | %.1f%% |" % (k, 100 * v)
              for k, v in sorted(out["degree_type_share"].items(), key=lambda kv: -kv[1])]
    io.open(a.out_md, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\n→ %s\n→ %s\n→ %s (저장소 밖)" % (a.out_json, a.out_md, tf_path))


if __name__ == "__main__":
    main()
