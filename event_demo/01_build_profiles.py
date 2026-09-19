# -*- coding: utf-8 -*-
"""연명부 엑셀 → 익명 프로필 표.

입력  D:/DSL/DataScience Lab 연명부.xlsx (시트 1 만. 시트 2 좌석표는 쓰지 않는다)
출력  D:/DSL/_event_data/profiles.csv     pid, cohort, role(alumni/active), status, college, dept, level, career, text
      D:/DSL/_event_data/id_map.csv       pid → 이름 (로컬 전용. 저장소·서버에 올리지 않는다)

전화번호·학번·영문명은 읽지 않는다. 'text' 가 임베딩 입력 문장이다.
"""
import os, re, sys
import pandas as pd

SRC = r"D:/DSL/DataScience Lab 연명부.xlsx"
OUT = r"D:/DSL/_event_data"
os.makedirs(OUT, exist_ok=True)

df = pd.read_excel(SRC, sheet_name=0, header=0, usecols=[0, 1, 2, 4, 5, 7, 8])
df.columns = ["name", "cohort", "level", "college", "dept", "status", "career"]
df = df[df["cohort"].notna() & df["name"].notna() & ~df["name"].astype(str).str.endswith("기")].copy()
df["cohort"] = df["cohort"].astype(int)
for c in ("level", "college", "dept", "status", "career"):
    df[c] = df[c].astype(str).str.strip().replace({"nan": "", "-": ""})

STOP = {"졸업", "졸업예정", "재학", "수료", "인턴", "예정", "현재", "및", "석사", "박사", "학사", "과정", "석박", "통합"}

def clean_career(t):
    t = re.sub(r"\d+\.\s*", " / ", t)        # "1. xxx 2. yyy" → " / xxx / yyy"
    t = re.sub(r"\(\s*\d{4}[^)]*\)", "", t)  # 연도 괄호 제거
    t = re.sub(r"\s+", " ", t).strip(" /")
    return t

def career_tokens(t):
    toks = re.split(r"[\s/,()·\-–]+", t)
    return {x for x in toks if len(x) >= 2 and x not in STOP and not re.fullmatch(r"[\d.]+", x)}

df["career"] = df["career"].map(clean_career)
df["role"] = df.apply(lambda r: "active" if (r["status"] == "재학" and r["cohort"] >= 12) else "alumni", axis=1)

def text(r):
    lvl = {"학부": "학부생", "대학원": "대학원생"}.get(r["level"], "")
    st = {"졸업": "졸업", "재학": "재학 중"}.get(r["status"], "")
    s = f"연세대학교 {r['college']} {r['dept']} {lvl} {st}. 데이터사이언스랩 {r['cohort']}기."
    if r["career"]:
        s += f" 경력: {r['career']}."
    return re.sub(r"\s+", " ", s)

df["text"] = df.apply(text, axis=1)
df["pid"] = ["p%03d" % i for i in range(len(df))]
df["career_tokens"] = df["career"].map(lambda t: "|".join(sorted(career_tokens(t))))

cols = ["pid", "cohort", "role", "status", "level", "college", "dept", "career", "career_tokens", "text"]
df[cols].to_csv(os.path.join(OUT, "profiles.csv"), index=False, encoding="utf-8-sig")
df[["pid", "name"]].to_csv(os.path.join(OUT, "id_map.csv"), index=False, encoding="utf-8-sig")

sys.stdout.reconfigure(encoding="utf-8")
print("사람", len(df), "· role:", df["role"].value_counts().to_dict())
print("경력 있음", (df["career"] != "").sum(), "· 경력 토큰 종류", len(set("|".join(df["career_tokens"]).split("|")) - {""}))
print("예시 text:", df["text"].iloc[3][:120])
print("저장:", OUT)
