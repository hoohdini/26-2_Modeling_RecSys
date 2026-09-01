"""프로필-프로필 그래프 구축 — 간선 후보 4종.

출력은 `Tokenization/graph_sid_augment.py:95` 가 요구하는 3열 CSV 하나뿐이다.
    src_item_id, dst_item_id, relation

간선 후보 (사전등록 docs/PREREG_생성데이터셋.md §3-1, 우선순위 순)
    1 related_occupation  직업 A 의 관련직업으로 등재된 B  (relJobList, 실데이터)
    2 skill_vector        77차원 지식·능력 벡터 코사인 >= theta
    3 career_path         같은 전공에서 서로 다른 직업으로 갈린 경우
    4 same_occupation     같은 KECO 코드

⚠️ 1·3·4 는 **직업 수준** 관계라 프로필로 펼치면 블록(클리크)이 된다.
   직업당 후보가 최대 443명이라 그대로 펼치면 간선이 폭발하고 텍스트와 겹친다.
   그래서 직업 수준 관계는 **프로필별로 표본을 뽑아** 붙인다(--fanout).
   2 는 프로필 수준 연속값이라 이 문제가 없다.

밀도 목표: 평균 차수 20~30 (Amazon Beauty 11,903노드/301,103간선 = 약 25).
**밀도로 맞추는 것이지 결과로 맞추는 것이 아니다** (사전등록 §3-1).
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


def rd(p):
    return io.open(p, encoding="utf-8").read()


def load_profiles():
    with io.open(os.path.join(OUT, "profiles.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    vec = np.load(os.path.join(OUT, "profile_skill.npy"))
    return rows, vec


def job_relations():
    """직업 수준 관련직업 (실데이터)."""
    rel = defaultdict(set)
    for p in glob.glob(os.path.join(RAW, "jobdtl", "*_d1.xml")) + \
             glob.glob(os.path.join(RAW, "jobdtl", "*_d2.xml")):
        code = os.path.basename(p).split("_")[0]
        for b in re.findall(r"<relJobList>(.*?)</relJobList>", rd(p), re.S):
            for c in re.findall(r"<jobCd>([^<]+)</jobCd>", b):
                if c != code:
                    rel[code].add(c)
    return rel


def topk_cosine(vec, k, chunk=512):
    """중심화 후 코사인 상위 k. 중심화하는 이유는 graph_sid_augment.py:18-28 참고."""
    x = vec.astype(np.float32) - vec.mean(0)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    n = len(x)
    idx = np.empty((n, k), dtype=np.int32)
    sim = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        S = x[s:e] @ x.T
        S[np.arange(e - s), np.arange(s, e)] = -9
        part = np.argpartition(-S, k, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        order = np.argsort(-S[rows, part], axis=1)
        idx[s:e] = part[rows, order]
        sim[s:e] = S[rows, part[rows, order]]
    return idx, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fanout", type=int, default=8,
                    help="직업 수준 관계 하나당 프로필이 붙일 이웃 수")
    ap.add_argument("--skill-k", type=int, default=12, help="skill_vector 이웃 수")
    ap.add_argument("--skill-theta", type=float, default=0.0,
                    help="코사인 하한 (0이면 상위 k 무조건)")
    ap.add_argument("--skill-cross-job", action="store_true", default=True,
                    help="skill_vector 이웃에서 같은 직업 제외 (기본 켬)")
    ap.add_argument("--skill-same-job-ok", dest="skill_cross_job", action="store_false",
                    help="대조군: 같은 직업도 허용")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(OUT, "jobs_edges_long.csv"))
    args = ap.parse_args()
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rng = np.random.default_rng(args.seed)
    rows, vec = load_profiles()
    n = len(rows)
    by_job = defaultdict(list)
    by_keco = defaultdict(list)
    by_major = defaultdict(list)
    for r in rows:
        i = int(r["profile_id"])
        by_job[r["job_cd"]].append(i)
        if r["keco_cd"]:
            by_keco[r["keco_cd"]].append(i)
        if r["major"]:
            by_major[r["major"]].append(i)
    job_of = {int(r["profile_id"]): r["job_cd"] for r in rows}

    edges = set()

    def add(a, b, rel):
        if a != b:
            edges.add((a, b, rel))

    # ---- 1. related_occupation ----
    rel = job_relations()
    cnt = 0
    for i in range(n):
        pool = [p for c in rel.get(job_of[i], ()) for p in by_job.get(c, ())]
        if not pool:
            continue
        for p in rng.choice(pool, size=min(args.fanout, len(pool)), replace=False):
            add(i, int(p), "related_occupation")
            cnt += 1
    print(f"1. related_occupation  {cnt:,}")

    # ---- 2. skill_vector ----
    # ⚠️ 같은 직업은 제외한다. 스킬 벡터가 직업 기준 벡터에서 파생되므로,
    #    그냥 최근접을 뽑으면 이웃이 전부 같은 직업 사람이 되어
    #    `same_occupation` 을 다르게 부른 것이 된다(실측 Jaccard 0.265 — 최악 대리 텍스트 대비).
    #    "직업은 다른데 역량이 겹치는 사람"을 이어야 텍스트가 모르는 정보가 된다.
    pool_k = args.skill_k * 12 if args.skill_cross_job else args.skill_k
    idx, sim = topk_cosine(vec, min(pool_k, n - 1))
    cnt = 0
    kept_sim = []
    for i in range(n):
        taken = 0
        for j, s in zip(idx[i], sim[i]):
            if taken >= args.skill_k:
                break
            if s < args.skill_theta:
                break
            if args.skill_cross_job and job_of[int(j)] == job_of[i]:
                continue
            add(i, int(j), "skill_vector")
            kept_sim.append(s)
            taken += 1
            cnt += 1
    tag = "다른 직업만" if args.skill_cross_job else "직업 무관"
    print(f"2. skill_vector        {cnt:,}  ({tag} · 코사인 중앙 {np.median(kept_sim):.3f})")

    # ---- 3. career_path : 같은 전공, 다른 직업 ----
    cnt = 0
    for major, members in by_major.items():
        if len(members) < 2:
            continue
        arr = np.array(members)
        for i in members:
            other = arr[[job_of[int(x)] != job_of[i] for x in arr]]
            if len(other) == 0:
                continue
            for p in rng.choice(other, size=min(args.fanout // 2, len(other)), replace=False):
                add(i, int(p), "career_path")
                cnt += 1
    print(f"3. career_path         {cnt:,}")

    # ---- 4. same_occupation ----
    cnt = 0
    for keco, members in by_keco.items():
        if len(members) < 2:
            continue
        arr = np.array(members)
        for i in members:
            other = arr[arr != i]
            for p in rng.choice(other, size=min(args.fanout // 2, len(other)), replace=False):
                add(i, int(p), "same_occupation")
                cnt += 1
    print(f"4. same_occupation     {cnt:,}")

    with io.open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_item_id", "dst_item_id", "relation"])
        for a, b, r in sorted(edges):
            w.writerow([a, b, r])

    und = defaultdict(set)
    for a, b, _ in edges:
        und[a].add(b)
        und[b].add(a)
    deg = np.array([len(und.get(i, ())) for i in range(n)])
    print(f"\n총 간선 {len(edges):,} (라벨 포함) · 고유 방향 {len({(a, b) for a, b, _ in edges}):,}")
    print(f"무방향 평균 차수 {deg.mean():.1f} · 중앙 {np.median(deg):.0f} · 고립 {(deg == 0).sum()}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
