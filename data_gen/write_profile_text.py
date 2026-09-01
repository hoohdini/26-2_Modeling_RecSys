"""프로필 텍스트 생성 (영문).

왜 영문인가
----------
임베딩 인코더 `google/flan-t5-xl` 의 토크나이저는 한글 어절을 통째로 `<unk>` 로 만든다.
실측: "공인회계사로 6년간 ..." → 14토큰 중 6개가 `<unk>`, 남는 건 숫자와 마침표뿐.
한글로는 임베딩이 의미를 담지 못하므로 SID 도 무의미해진다.
영문으로 쓰면 파이프라인을 하나도 바꾸지 않고 Amazon Beauty 와 **동일한 인코더**를
쓸 수 있어, "조건을 바꿔서 좋아진 것 아니냐"는 반론도 차단된다.
링크드인 시나리오와도 맞는다.

무엇이 실데이터인가
------------------
· 직업명 492종      고용24 직업정보 → 영문 대응 (data_gen/job_titles_en.tsv)
· 스킬 77종         고용24 지식33+능력44 = O*NET 체계 → 원명 복원 (skill_names_en.tsv)
· 스킬 점수 0~100   고용24 실데이터 + 개인 편차 (build_profiles.py)
· 학력·전공 계열    고용24 직업별 실제 분포에서 샘플
· 자격증            고용24 관련자격 목록 → 등급 규칙으로 영문화

다양성이 왜 중요한가
------------------
같은 직업 사람들이 비슷한 글을 쓰면 텍스트 이웃 = 같은 직업이 되고,
그러면 그래프와 겹쳐 G0 에서 떨어진다. 다양성은 미관이 아니라 실험 성립 조건이다.
그래서 문장 템플릿·순서·언급 스킬 수를 프로필마다 흔든다.
"""
import argparse
import csv
import io
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "data_gen")
OUT = os.path.join(GEN, "out")

EDU_EN = {"중졸 이하": "secondary education", "고졸": "a high school diploma",
          "전문대졸": "an associate degree", "대졸": "a bachelor's degree",
          "석사": "a master's degree", "박사": "a doctoral degree"}
FIELD_EN = {"인문": "humanities", "사회": "social sciences", "교육": "education",
            "공학": "engineering", "자연": "natural sciences", "의약": "health sciences",
            "예체능": "arts and physical education"}
REGION_EN = {"서울": "Seoul", "경기": "Gyeonggi", "인천": "Incheon", "부산": "Busan",
             "대구": "Daegu", "대전": "Daejeon", "광주": "Gwangju", "울산": "Ulsan",
             "세종": "Sejong", "강원": "Gangwon", "충북": "North Chungcheong",
             "충남": "South Chungcheong", "전북": "North Jeolla", "전남": "South Jeolla",
             "경북": "North Gyeongsang", "경남": "South Gyeongsang"}

# 자격증은 516종이라 개별 번역 대신 한국 국가자격 등급 규칙으로 영문화한다.
CERT_RULES = [
    ("기술사", "a Professional Engineer licence"),
    ("기능장", "a Master Craftsman licence"),
    ("산업기사", "an Industrial Engineer certificate"),
    ("기능사", "a Craftsman certificate"),
    ("기사", "an Engineer-grade national certificate"),
    ("국가공인 민간", "an accredited professional certification"),
    ("국가전문", "a national professional licence"),
    ("면허", "the relevant national licence"),
]


def cert_en(name):
    for k, v in CERT_RULES:
        if k in name:
            return v
    return "a relevant national certificate"


def seniority(y):
    if y < 1:
        return "entry-level"
    if y < 3:
        return "early-career"
    if y < 6:
        return "mid-level"
    if y < 11:
        return "experienced"
    return "senior"


def load_map(path, sep="\t"):
    d = {}
    for line in io.open(path, encoding="utf-8"):
        if line.strip():
            a, b = line.rstrip("\n").split(sep)[:2]
            d[a] = b.strip()
    return d


def join(xs):
    xs = list(xs)
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def compose(rng, role, yrs, sen, skills, edu, field, cert, region):
    """문장 조각을 뽑아 섞는다. 프로필마다 다른 조합이 나오게 한다."""
    if yrs < 1:
        span = "under a year"        # "under a year of experience" 처럼 자연스럽게 붙는 형태
    else:
        n_y = max(1, int(round(yrs)))
        span = f"{n_y} year" + ("s" if n_y != 1 else "")

    leads = [
        f"{sen.capitalize()} {role} with {span} of experience.",
        f"{role} based in {region}, {span} in the field.",
        f"Working as a {role} for {span}.",
        f"{span.capitalize()} of hands-on work as a {role}.",
        f"{role}. {sen.capitalize()} practitioner, {span} in role.",
        f"I have spent {span} working as a {role}.",
    ]
    li = int(rng.integers(len(leads)))
    lead = leads[li]
    region_in_lead = (li == 1)

    n = int(rng.integers(2, 5))
    sk = join(skills[:n])
    strength = rng.choice([
        f"Strongest areas are {sk}.",
        f"Day-to-day work draws on {sk}.",
        f"Colleagues point to {sk} as my strengths.",
        f"Core competencies: {sk}.",
        f"Most of the role rests on {sk}.",
        f"Comfortable with {sk}.",
    ])

    if field:
        bg = rng.choice([
            f"Holds {edu} in {field}.",
            f"Educational background: {edu} in {field}.",
            f"Studied {field}, {edu}.",
            f"{edu.capitalize()} in {field}.",
        ])
    else:
        bg = rng.choice([f"Holds {edu}.", f"Educational background: {edu}."])

    parts = [lead, strength, bg]

    if cert:
        parts.append(rng.choice([
            f"Also holds {cert}.",
            f"Certified with {cert}.",
            f"{cert.capitalize()} on record.",
        ]))

    if not region_in_lead and rng.random() < 0.65:
        parts.append(rng.choice([
            f"Based in {region}.",
            f"Currently working in {region}.",
            f"Located in {region} and open to nearby roles.",
        ]))

    # 배경 문장 위치를 가끔 앞으로 보내 문단 구조를 흔든다
    if rng.random() < 0.25 and len(parts) >= 3:
        parts.insert(0, parts.pop(2))
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(OUT, "profile_text.tsv"))
    args = ap.parse_args()
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rng = np.random.default_rng(args.seed)
    jt = load_map(os.path.join(GEN, "job_titles_en.tsv"))
    sk_en = load_map(os.path.join(GEN, "skill_names_en.tsv"))
    dims = [x for x in io.open(os.path.join(OUT, "skill_dims.txt"), encoding="utf-8").read().split("\n") if x]
    dims_en = [sk_en[d] for d in dims]
    vec = np.load(os.path.join(OUT, "profile_skill.npy"))
    rows = list(csv.DictReader(io.open(os.path.join(OUT, "profiles.csv"), encoding="utf-8")))

    texts = []
    for r in rows:
        i = int(r["profile_id"])
        v = vec[i]
        # 상위 스킬은 **개인 벡터** 기준이라 같은 직업이어도 사람마다 다르다
        top = [dims_en[j] for j in np.argsort(-v)[:6]]
        rng.shuffle(top[:4])
        yrs = float(r["career_years"])
        cert = cert_en(r["cert"]) if r["cert"] else ""
        txt = compose(rng, jt.get(r["job_cd"], "Professional"), yrs, seniority(yrs),
                      top, EDU_EN.get(r["edu_level"], "a bachelor's degree"),
                      FIELD_EN.get(r["major_field"], ""), cert,
                      REGION_EN.get(r["region"], "Seoul"))
        texts.append((i, re.sub(r"\s+", " ", txt).strip()))

    with io.open(args.out, "w", encoding="utf-8", newline="") as f:
        for i, t in texts:
            f.write(f"{i}\t{t}\n")

    wc = [len(t.split()) for _, t in texts]
    uniq = len({t for _, t in texts})
    print(f"프로필 텍스트 {len(texts):,} 생성 · 고유 {uniq:,} ({uniq/len(texts)*100:.1f}%)")
    print(f"  단어 수 최소 {min(wc)} · 중앙 {int(np.median(wc))} · 최대 {max(wc)}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
