"""후보자 프로필 생성 — 1단계: 구조 속성 (텍스트는 다음 단계).

고용24 실데이터(직업 492 · 지식33+능력44 점수 · 관련직업 · 학과 · KECO)를 씨앗으로
후보자 프로필 약 12,000개의 **속성**을 만든다. 텍스트는 `write_profile_text.py` 가 맡는다.

설계 원칙 (사전등록 docs/PREREG_생성데이터셋.md §3)
------------------------------------------------
1. 직업의 지식·능력 점수는 **실데이터 그대로** 쓴다. 지어내지 않는다.
2. 프로필마다 **개인 편차**를 준다. 안 주면 같은 직업 24명이 동일인이 되고,
   그러면 그래프가 "같은 직업 = 완전연결"이 되어 Beauty 에서 강등된 속성 그래프와
   같은 모양이 된다(G0 탈락). 편차는 실험을 성립시키는 조건이지 장식이 아니다.
3. 여기서 만드는 것은 `z`(잠재 적합도)가 **아니다**. z 는 상호작용 생성기에서 따로
   만든다. 이 파일이 내놓는 것은 **표면 속성**이고, 그래프는 표면 속성에서만 나온다.
   (§3-1 순환 금지)

가정 (실데이터가 없어 우리가 정한 것 — 전부 여기 명시하고 고정한다)
    · 직업별 후보 수    Zipf 유사 (일부 직업에 지원자가 몰리는 현실)
    · 경력연수          로그정규, 0~25년
    · 지역              서울·경기 편중 (공개 통계의 알려진 경향)
    · 개인 편차 σ       12점 (0~100 척도)
난수는 SEED 로 고정. 같은 SEED 면 같은 데이터가 나온다.
"""
import argparse
import csv
import glob
import io
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_gen", "raw")
OUT = os.path.join(ROOT, "data_gen", "out")

SEED = 42
N_PROFILES = 12000
SIGMA = 12.0          # 개인 편차 (0~100 척도)
ZIPF_A = 0.6          # 직업별 후보 수 쏠림
REGION = [("서울", .34), ("경기", .24), ("인천", .06), ("부산", .07), ("대구", .05),
          ("대전", .04), ("광주", .04), ("울산", .03), ("세종", .01), ("강원", .03),
          ("충북", .03), ("충남", .03), ("전북", .02), ("전남", .02), ("경북", .02), ("경남", .02)]

# <schDpt> 안의 7개 태그 = 전공 계열별 **비율(%)**. 계열 이름이 아니다.
FIELD_TAGS = [("cultLangDpt", "인문"), ("socDpt", "사회"), ("eduDpt", "교육"),
              ("engnrDpt", "공학"), ("natrlDpt", "자연"), ("mediDpt", "의약"),
              ("artphyDpt", "예체능")]
# <edubg> 안의 6개 태그 = 학력 분포(%)
EDU_TAGS = [("edubgMgraduUndr", "중졸 이하"), ("edubgHgradu", "고졸"),
            ("edubgCgraduUndr", "전문대졸"), ("edubgUgradu", "대졸"),
            ("edubgGgradu", "석사"), ("edubgDgradu", "박사")]


def rd(p):
    return io.open(p, encoding="utf-8").read()


def blocks(s, tag):
    return re.findall(rf"<{tag}>(.*?)</{tag}>", s, re.S)


def one(s, tag, default=""):
    m = re.search(rf"<{tag}>([^<]*)</{tag}>", s)
    return m.group(1).strip() if m else default


def pick_weighted(rng, dist, default=""):
    """{이름: 비율(%)} 에서 하나 뽑는다. 전부 0이거나 없으면 default."""
    if not dist:
        return default
    ks = [k for k, v in dist.items() if v > 0]
    if not ks:
        return default
    w = np.array([dist[k] for k in ks], dtype=float)
    return str(rng.choice(ks, p=w / w.sum()))


def major_dictionary():
    """학과정보 923건에서 학과명 사전. 긴 이름부터 봐야 부분일치 오염이 없다."""
    s = rd(os.path.join(RAW, "majorcd.xml"))
    names = set()
    for b in re.findall(r"<majorList>(.*?)</majorList>", s, re.S):
        for t in ("knowDtlSchDptNm", "knowSchDptNm"):
            v = one(b, t)
            if len(v) >= 3:
                names.add(v)
    return sorted(names, key=len, reverse=True)


def load_jobs():
    """직업 492개의 실데이터를 모은다."""
    jobs = {}
    for m in re.finditer(r"<jobList>(.*?)</jobList>", rd(os.path.join(RAW, "jobcd.xml")), re.S):
        b = m.group(1)
        code = one(b, "jobCd")
        if code:
            jobs[code] = {"jobCd": code, "jobNm": one(b, "jobNm"),
                          "clcd": one(b, "jobClcd"), "clcdNm": one(b, "jobClcdNM")}

    kn_names, ab_names = set(), set()
    for p in glob.glob(os.path.join(RAW, "jobdtl", "*_d5.xml")):
        code = os.path.basename(p).split("_")[0]
        if code not in jobs:
            continue
        s = rd(p)
        kn = {one(b, "knwldgNm"): float(one(b, "knwldgStatus", "0") or 0) for b in blocks(s, "Knwldg")}
        ab = {one(b, "jobAblNm"): float(one(b, "jobAblStatus", "0") or 0) for b in blocks(s, "jobAbil")}
        kn.pop("", None)
        ab.pop("", None)
        jobs[code]["kn"], jobs[code]["ab"] = kn, ab
        kn_names |= set(kn)
        ab_names |= set(ab)

    dept_dict = major_dictionary()
    for p in glob.glob(os.path.join(RAW, "jobdtl", "*_d3.xml")):
        code = os.path.basename(p).split("_")[0]
        if code not in jobs:
            continue
        s = rd(p)
        jobs[code]["keco"] = one(s, "kecoCd")
        jobs[code]["kecoNm"] = one(s, "kecoNm")
        jobs[code]["certs"] = [c for c in re.findall(r"<certNm>([^<]*)</certNm>", s) if c.strip()]

        # 전공 계열 분포 (%) — 실데이터
        sch = re.search(r"<schDpt>(.*?)</schDpt>", s, re.S)
        fld = {}
        if sch:
            for tag, nm in FIELD_TAGS:
                try:
                    fld[nm] = float(one(sch.group(1), tag, "0") or 0)
                except ValueError:
                    fld[nm] = 0.0
        jobs[code]["field_dist"] = fld

        # 학력 분포 (%) — 실데이터
        eb = re.search(r"<edubg>(.*?)</edubg>", s, re.S)
        edu = {}
        if eb:
            for tag, nm in EDU_TAGS:
                try:
                    edu[nm] = float(one(eb.group(1), tag, "0") or 0)
                except ValueError:
                    edu[nm] = 0.0
        jobs[code]["edu_dist"] = edu

        # 구체 학과명 — technKnow 서술문에서 923개 학과 사전과 대조해 추출
        tk = re.search(r"<technKnow>(.*?)</technKnow>", s, re.S)
        found = []
        if tk:
            txt = tk.group(1)
            for nm in dept_dict:
                if nm in txt and not any(nm in f for f in found):
                    found.append(nm)
                if len(found) >= 6:
                    break
        jobs[code]["depts"] = found


    # 관련직업 — 그래프 1순위 간선
    rel = defaultdict(set)
    for p in glob.glob(os.path.join(RAW, "jobdtl", "*_d1.xml")) + \
             glob.glob(os.path.join(RAW, "jobdtl", "*_d2.xml")):
        code = os.path.basename(p).split("_")[0]
        for b in blocks(rd(p), "relJobList"):
            for c in re.findall(r"<jobCd>([^<]+)</jobCd>", b):
                if c != code and c in jobs:
                    rel[code].add(c)
    for c in jobs:
        jobs[c]["rel"] = sorted(rel.get(c, ()))

    return jobs, sorted(kn_names), sorted(ab_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=N_PROFILES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    args = ap.parse_args()
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rng = np.random.default_rng(args.seed)
    os.makedirs(OUT, exist_ok=True)

    jobs, K, A = load_jobs()
    codes = [c for c in jobs if jobs[c].get("kn") and jobs[c].get("ab")]
    print(f"직업 {len(jobs)} · 지식·능력 완비 {len(codes)} · 차원 {len(K)}+{len(A)}={len(K)+len(A)}")
    edges = sum(len(jobs[c]["rel"]) for c in codes)
    print(f"관련직업 간선 {edges} · 학과 보유 직업 {sum(1 for c in codes if jobs[c].get('depts'))}")

    # 직업별 후보 수 — Zipf 유사 쏠림
    w = 1.0 / np.power(np.arange(1, len(codes) + 1), ZIPF_A)
    rng.shuffle(w)
    w /= w.sum()
    assign = rng.choice(len(codes), size=args.n, p=w)

    # 직업 기준 벡터
    base = np.array([[jobs[c]["kn"].get(k, 0.0) for k in K] + [jobs[c]["ab"].get(a, 0.0) for a in A]
                     for c in codes], dtype=np.float32)

    regions = [r for r, _ in REGION]
    rprob = np.array([p for _, p in REGION], dtype=float)
    rprob /= rprob.sum()

    rows = []
    vecs = np.empty((args.n, base.shape[1]), dtype=np.float32)
    for i in range(args.n):
        j = assign[i]
        code = codes[j]
        job = jobs[code]

        # 경력 — 로그정규, 0~25년
        yrs = float(np.clip(rng.lognormal(1.35, 0.75), 0, 25))
        # 개인 편차 + 경력에 따른 완만한 상승 (경력이 길수록 전반적으로 높다)
        v = base[j] + rng.normal(0, args.sigma, base.shape[1]).astype(np.float32)
        v = np.clip(v * (0.88 + 0.012 * yrs), 0, 100)
        vecs[i] = v

        # 학력·전공계열은 그 직업의 **실제 분포**에서 뽑는다 (고용24 dtlGb=3)
        edu = pick_weighted(rng, job.get("edu_dist"), "대졸")
        field = pick_weighted(rng, job.get("field_dist"), "")
        dept = str(rng.choice(job["depts"])) if job.get("depts") else ""
        certs = job.get("certs") or []
        cert = str(rng.choice(certs)) if certs and rng.random() < 0.55 else ""

        top = np.argsort(-v)[:3]
        names = K + A
        rows.append({
            "profile_id": i,
            "job_cd": code,
            "job_nm": job["jobNm"],
            "job_clcd": job["clcd"],
            "job_clcd_nm": job["clcdNm"],
            "keco_cd": job.get("keco", ""),
            "career_years": round(yrs, 1),
            "edu_level": edu,
            "major_field": field,
            "major": dept,
            "cert": cert,
            "region": str(rng.choice(regions, p=rprob)),
            "top_skill_1": f"{names[top[0]]} {v[top[0]]:.0f}",
            "top_skill_2": f"{names[top[1]]} {v[top[1]]:.0f}",
            "top_skill_3": f"{names[top[2]]} {v[top[2]]:.0f}",
        })

    with io.open(os.path.join(OUT, "profiles.csv"), "w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    np.save(os.path.join(OUT, "profile_skill.npy"), vecs)
    with io.open(os.path.join(OUT, "skill_dims.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(K + A))

    used = len(set(assign))
    per = np.bincount(assign, minlength=len(codes))
    print(f"\n프로필 {args.n:,} 생성")
    print(f"  직업 사용 {used}/{len(codes)} · 직업당 후보 최소 {per[per > 0].min()} · 중앙 {int(np.median(per[per > 0]))} · 최대 {per.max()}")
    print(f"  경력 중앙 {np.median([r['career_years'] for r in rows]):.1f}년")
    print(f"  스킬 벡터 {vecs.shape} · 값 {vecs.min():.0f}~{vecs.max():.0f}")
    print(f"→ {OUT}/profiles.csv · profile_skill.npy")


if __name__ == "__main__":
    main()
