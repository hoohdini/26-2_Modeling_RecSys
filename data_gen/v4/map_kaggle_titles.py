# -*- coding: utf-8 -*-
"""캐글 직무명 → 고용24 직업(492) · KECO 소분류 매핑 (규칙 + 사전 + 문자열 유사도).

입력
    <external>/title_freq.tsv        kaggle_stats.py 가 만든 직무명 빈도표 (저장소 밖)
    data_gen/job_titles_en.tsv       직업코드 → 영문 직업명 (492)
    data_gen/out/profiles.csv        직업코드 → KECO 코드·직업 중분류 (실데이터에서 온 것)

출력 (캐글 문자열이 들어가므로 저장소 밖에 둔다)
    <external>/title_map.csv         title, count, job_cd, job_title_en, keco_cd, score, band
    <external>/title_review_200.csv  사람이 검수할 200건 (빈도 상위 100 + 검토 구간 무작위 100)
저장소에는 집계만 남긴다
    data_gen/spec/title_map_summary.json   커버리지(빈도 가중), 구간별 건수, 임계값

판정 구간 (실행 전에 고정)
    auto    score >= 0.60   자동 채택
    review  0.35 <= score < 0.60   사람이 본다
    none    score < 0.35   기타(unmapped). 분포 목표는 직무명과 무관한 통계를 우선 쓰므로 치명적이지 않다

검수 시트를 채운 뒤:
    python data_gen/v4/map_kaggle_titles.py --score-review <external>/title_review_200_filled.csv
    → 검수 일치율(자동 매핑이 사람 판정과 같은 비율)을 summary 에 기록한다
"""
import argparse
import csv
import io
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(ROOT, "data_gen")
DEFAULT_SRC = "D:/DSL/_external/kaggle_job_recommendation"
AUTO, REVIEW = 0.60, 0.35

# 직급·고용형태·잡음 토큰. 직무 자체를 바꾸지 않으므로 지운다.
STOP = set("""jr sr ii iii iv senior junior lead principal staff assistant associate chief head
part time full parttime fulltime temp temporary contract seasonal intern internship trainee
entry level experienced needed wanted hiring immediate urgent now the a an of and for in at to
with or new local remote""".split())

# 미국 자유 텍스트 직무명 → 우리 492 영문 직업명 (job_titles_en.tsv 의 표기 그대로, 소문자 비교).
# 캐글 빈도 상위 직무명을 보고 2026-09-16 손으로 작성. 정확한 대응이 없으면 가장 가까운 상위 직업으로 보낸다.
SYNONYM = {
    "customer service representative": "Customer Service Representative",
    "customer service rep": "Customer Service Representative",
    "customer service": "Customer Service Representative",
    "call center": "Customer Service Representative",
    "csr": "Customer Service Representative",
    "administrative assistant": "General Affairs Clerk",
    "admin assistant": "General Affairs Clerk",
    "office assistant": "General Affairs Clerk",
    "office clerk": "General Affairs Clerk",
    "office manager": "General Administrative Officer",
    "receptionist": "Receptionist",
    "executive assistant": "Executive Secretary",
    "secretary": "Executive Secretary",
    "data entry": "Data Entry and Office Support Clerk",
    "cashier": "Retail Cashier",
    "sales associate": "Retail Salesperson",
    "retail sales": "Retail Salesperson",
    "retail wireless sales consultant": "Mobile Device and Telecom Sales Representative",
    "sales representative": "Product and Advertising Sales Representative",
    "sales consultant": "Product and Advertising Sales Representative",
    "account executive": "Product and Advertising Sales Representative",
    "account manager": "Product and Advertising Sales Representative",
    "account representative": "Product and Advertising Sales Representative",
    "sales manager": "Sales and Retail Manager",
    "store manager": "Sales and Retail Manager",
    "assistant store manager": "Sales and Retail Manager",
    "assistant manager": "Sales and Retail Manager",
    "branch manager": "Sales and Retail Manager",
    "general manager": "Corporate Support Manager",
    "operations manager": "Corporate Support Manager",
    "project manager": "Corporate Support Manager",
    "manager": "Corporate Support Manager",
    "supervisor": "Corporate Support Manager",
    "medical assistant": "Nursing Assistant",
    "certified nursing assistant": "Care Worker and Nursing Assistant",
    "cna": "Care Worker and Nursing Assistant",
    "lpn": "Nursing Assistant",
    "registered nurse": "Registered Nurse",
    "rn": "Registered Nurse",
    "caregiver": "Care Worker and Nursing Assistant",
    "home health aide": "Care Worker and Nursing Assistant",
    "pharmacy technician": "Pharmacist",
    "pharmacist": "Pharmacist",
    "physical therapist": "Physiotherapist",
    "dental assistant": "Dental Technician",
    "dental hygienist": "Dental Hygienist",
    "physician": "General Practitioner",
    "staff accountant": "Accounting Clerk",
    "senior accountant": "Certified Public Accountant",
    "accountant": "Certified Public Accountant",
    "accounting clerk": "Accounting Clerk",
    "accounts payable": "Accounting Clerk",
    "accounts receivable": "Accounting Clerk",
    "bookkeeper": "Bookkeeping Clerk",
    "controller": "Finance Manager",
    "financial analyst": "Investment Analyst",
    "business analyst": "Management Consultant",
    "auditor": "Internal Audit Clerk",
    "payroll": "HR and Training Clerk",
    "human resources": "Human Resources Specialist",
    "hr generalist": "Human Resources Specialist",
    "recruiter": "Human Resources Specialist",
    "teller": "Bank Teller",
    "loan officer": "Banking Clerk",
    "insurance agent": "Insurance Agent and Broker",
    "real estate agent": "Real Estate Consultant and Broker",
    "paralegal": "Legal Clerk",
    "attorney": "Lawyer",
    "lawyer": "Lawyer",
    "marketing manager": "Marketing and Communications Manager",
    "marketing coordinator": "Advertising and Marketing Clerk",
    "marketing": "Advertising and Marketing Clerk",
    "graphic designer": "Visual Designer",
    "software engineer": "Application Software Developer",
    "software developer": "Application Software Developer",
    "java developer": "Application Software Developer",
    ".net developer": "Application Software Developer",
    "programmer": "Application Software Developer",
    "web developer": "Web Developer",
    "systems administrator": "Network and Cloud Administrator",
    "network engineer": "Network Systems Engineer",
    "it support": "IT Support Specialist",
    "help desk": "IT Support Specialist",
    "data analyst": "Data Analyst",
    "driver": "Truck and Special Vehicle Driver",
    "truck driver": "Truck and Special Vehicle Driver",
    "cdl driver": "Truck and Special Vehicle Driver",
    "delivery driver": "Parcel Delivery Worker",
    "bus driver": "Bus Driver",
    "dispatcher": "Road and Rail Transport Clerk",
    "forklift operator": "Forklift Operator",
    "warehouse": "Logistics and Materials Clerk",
    "shipping and receiving": "Logistics and Materials Clerk",
    "machine operator": "Metalworking Machine Operator",
    "cnc machinist": "Metal Machine Tool Operator",
    "assembler": "General Machinery Assembler",
    "production worker": "General Machinery Assembler",
    "production supervisor": "Production Manager",
    "quality": "Production and Quality Control Clerk",
    "packer": "Filling, Packaging and Labelling Machine Operator",
    "maintenance technician": "Industrial Machinery Installation and Maintenance Technician",
    "welder": "Welder",
    "electrician": "Building Electrician",
    "mechanic": "Automotive Service Technician",
    "hvac": "HVAC and Refrigeration Technician",
    "plumber": "Construction Plumber",
    "carpenter": "Architectural Carpenter",
    "painter": "Building Painter",
    "laborer": "Construction and Mining Labourer",
    "general labor": "Construction and Mining Labourer",
    "construction": "Construction and Mining Labourer",
    "security officer": "Building Security Guard",
    "security guard": "Building Security Guard",
    "janitor": "Cleaner",
    "custodian": "Cleaner",
    "housekeeper": "Accommodation Service Attendant",
    "server": "Restaurant Server",
    "waitress": "Restaurant Server",
    "waiter": "Restaurant Server",
    "bartender": "Bartender",
    "barista": "Barista and Beverage Preparer",
    "cook": "Institutional Catering Cook",
    "chef": "Head Chef and Culinary Researcher",
    "restaurant manager": "Food Service Manager",
    "teacher": "Secondary School Teacher",
    "substitute teacher": "Secondary School Teacher",
    "tutor": "Home-Visit Learning Tutor",
    "teaching assistant": "Teaching Assistant",
    "social worker": "Social Worker",
    "counselor": "Psychological Counsellor",
    "librarian": "Librarian",
    "photographer": "Photojournalist and Photographer",
    "editor": "Publishing Editor",
    "writer": "Broadcast Writer",
    "translator": "Translator",
    "interpreter": "Interpreter",
    "veterinarian": "Veterinarian",
    "vet tech": "Veterinary Assistant",
    "hair stylist": "Hairdresser",
    "hairstylist": "Hairdresser",
    "cosmetologist": "Skin and Body Care Specialist",
    "nail technician": "Nail Artist",
    "massage therapist": "Massage Therapist",
    "personal trainer": "Sports Instructor and Trainer",
    "fitness": "Sports Instructor and Trainer",
    "firefighter": "Firefighter",
    "police officer": "Police Officer",
    "military": "Non-Commissioned Officer",
    "flight attendant": "Flight Attendant",
    "pilot": "Airline Pilot",
    "civil engineer": "Civil Structural Design Engineer",
    "mechanical engineer": "Plant Machinery Engineer",
    "electrical engineer": "Electrical Equipment Development Engineer",
    "telemarketer": "Telemarketer",
}

# 직무가 아닌 것. 매핑하지 않는다 (none).
NOT_A_JOB = {"intern", "internship", "volunteer", "owner", "own your own franchise!", "sales / franchise",
             "mobile tool sales / franchise distributor", "student", "self employed", "self-employed",
             "unemployed", "consultant", "various", "other", "none", "n/a", "na", "freelance", "contractor"}


def norm(s):
    s = s.lower()
    s = re.sub(r"[\(\)\[\]\{\}/\\,;:!?\"'#*+&$%@|~`^<>=_\-–—.]", " ", s)
    s = re.sub(r"\b\d+\b", " ", s)
    toks = [t for t in s.split() if t not in STOP]
    return " ".join(toks)


def load_map(path):
    d = {}
    for line in io.open(path, encoding="utf-8"):
        if line.strip():
            a, b = line.rstrip("\n").split("\t")[:2]
            d[a] = b.strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--top", type=int, default=5000, help="빈도 상위 N 직무명만 매핑")
    ap.add_argument("--score-review", default=None, help="검수 완료 시트 → 일치율 계산")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    summ_path = os.path.join(GEN, "spec", "title_map_summary.json")

    if a.score_review:
        rows = list(csv.DictReader(io.open(a.score_review, encoding="utf-8")))
        done = [r for r in rows if (r.get("human_ok") or "").strip() != ""]
        ok = sum(1 for r in done if (r["human_ok"] or "").strip().lower() in ("y", "yes", "1", "o"))
        summ = json.load(io.open(summ_path, encoding="utf-8")) if os.path.exists(summ_path) else {}
        summ["review"] = {"n_reviewed": len(done), "n_agree": ok,
                          "agree_rate": ok / max(len(done), 1)}
        json.dump(summ, io.open(summ_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("검수 %d건 · 일치 %d · 일치율 %.1f%%" % (len(done), ok, 100 * ok / max(len(done), 1)))
        return

    tf = os.path.join(a.src, "title_freq.tsv")
    if not os.path.exists(tf):
        sys.exit("없음: %s — 먼저 kaggle_stats.py 를 실행하세요" % tf)
    titles = []
    for i, line in enumerate(io.open(tf, encoding="utf-8")):
        if i == 0:
            continue
        t, c = line.rstrip("\n").split("\t")
        titles.append((t, int(c)))
    total = sum(c for _, c in titles)
    titles = titles[: a.top]

    jt = load_map(os.path.join(GEN, "job_titles_en.tsv"))          # job_cd → en
    prof = list(csv.DictReader(io.open(os.path.join(GEN, "out", "profiles.csv"), encoding="utf-8")))
    keco = {}
    for r in prof:
        keco.setdefault(r["job_cd"], (r["keco_cd"], r["job_clcd"], r["job_clcd_nm"]))
    codes = [c for c in jt if c in keco]
    names = [jt[c] for c in codes]
    names_n = [norm(n) for n in names]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(names_n)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit(names_n)
    Mw, Mc = vw.transform(names_n), vc.transform(names_n)

    q_raw = [t for t, _ in titles]
    q = [norm(t) for t in q_raw]
    Sw = cosine_similarity(vw.transform(q), Mw)
    Sc = cosine_similarity(vc.transform(q), Mc)
    S = 0.5 * Sw + 0.5 * Sc

    # 사전 히트는 유사도에 가산: 사전 값이 직업명에 부분 일치하는 후보들에 +0.5
    lower_names = [n.lower() for n in names]
    syn_keys = sorted(SYNONYM, key=len, reverse=True)
    for qi, (qr, qn) in enumerate(zip(q_raw, q)):
        for k in syn_keys:
            if k in qr or (qn and k in qn):
                v = SYNONYM[k].lower()
                hit = [j for j, n in enumerate(lower_names) if n == v] or [j for j, n in enumerate(lower_names) if v in n]
                if hit:
                    if k == qr.strip() or k == qn:      # 사전 키와 완전히 같으면 사전이 결정한다
                        S[qi, hit] = 1.0
                    else:                              # 부분 일치는 가산만
                        S[qi, hit] += 0.5
                break

    # 질의에 없는 토큰이 많은 후보(더 구체적인 직업명)에 작은 감점 → 일반 직업명을 우선한다
    for qi, qn in enumerate(q):
        qt = set(qn.split())
        extra = np.array([len(set(n.split()) - qt) for n in names_n], dtype=float)
        S[qi] -= 0.03 * extra
    for qi, qr in enumerate(q_raw):
        if qr.strip() in NOT_A_JOB:
            S[qi] = -1.0
    out_rows, band_cnt, band_w = [], {"auto": 0, "review": 0, "none": 0}, {"auto": 0, "review": 0, "none": 0}
    for qi, (t, c) in enumerate(titles):
        j = int(np.argmax(S[qi]))
        sc = float(min(S[qi, j], 1.0))
        band = "auto" if sc >= AUTO else ("review" if sc >= REVIEW else "none")
        band_cnt[band] += 1
        band_w[band] += c
        top3 = np.argsort(-S[qi])[:3]
        out_rows.append({"title": t, "count": c, "job_cd": codes[j] if band != "none" else "",
                         "job_title_en": names[j] if band != "none" else "",
                         "keco_cd": keco[codes[j]][0] if band != "none" else "",
                         "job_clcd": keco[codes[j]][1] if band != "none" else "",
                         "score": round(sc, 4), "band": band,
                         "alt2": names[top3[1]], "alt3": names[top3[2]]})

    mp = os.path.join(a.src, "title_map.csv")
    with io.open(mp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    rng = np.random.default_rng(42)
    top100 = out_rows[:100]
    pool = [r for r in out_rows[100:] if r["band"] == "review"] or out_rows[100:]
    pick = list(rng.choice(len(pool), size=min(100, len(pool)), replace=False))
    review = top100 + [pool[i] for i in pick]
    rp = os.path.join(a.src, "title_review_200.csv")
    with io.open(rp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]) + ["human_ok", "human_job_title_en"])
        w.writeheader()
        for r in review:
            w.writerow(dict(r, human_ok="", human_job_title_en=""))

    covered = sum(c for _, c in titles)
    summ = {"n_titles_mapped": len(titles), "freq_share_of_topN": covered / max(total, 1),
            "thresholds": {"auto": AUTO, "review": REVIEW},
            "band_count": band_cnt,
            "band_freq_share": {k: v / max(covered, 1) for k, v in band_w.items()},
            "n_jobs_hit": len({r["job_cd"] for r in out_rows if r["job_cd"]}),
            "n_keco_hit": len({r["keco_cd"] for r in out_rows if r["keco_cd"]})}
    os.makedirs(os.path.dirname(summ_path), exist_ok=True)
    json.dump(summ, io.open(summ_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    print("→ %s\n→ %s (검수 시트)\n→ %s" % (mp, rp, summ_path))


if __name__ == "__main__":
    main()
