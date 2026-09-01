"""G1 대조군 — 차수를 보존한 채 간선만 무작위로 다시 잇는다.

왜 필요한가
----------
G-SID 가 좋아졌다면 그것이 **그래프의 의미** 때문인지, 아니면 단지
"이웃 임베딩을 평균내 매끄럽게 만든 효과" 때문인지 구분해야 한다.
차수 분포·간선 수·관계별 비율을 그대로 두고 **누가 누구와 이어지는지만** 부순 그래프로
같은 파이프라인을 돌린다.

    통과 = 이득이 사라진다      → 이득의 출처가 그래프의 의미다
    실패 = 이득이 남는다        → 그래프 의미가 아닌 다른 것이 원인. 결과 폐기

방법: 관계별 configuration model.
    출발점 목록(출차수만큼 반복)과 도착점 목록(입차수만큼 반복)을 만들고
    도착점만 섞어 다시 짝짓는다. 자기 자신·중복은 버린다.
    → 출차수·입차수 분포가 (버려진 소수를 빼면) 그대로 유지된다.
"""
import argparse
import csv
import io
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data_gen", "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(OUT, "jobs_edges_final.csv"))
    ap.add_argument("--out", default=os.path.join(OUT, "jobs_edges_rewired.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rng = np.random.default_rng(args.seed)
    by_rel = defaultdict(list)
    with io.open(args.src, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, r = line.rstrip("\n").split(",")
            by_rel[r].append((int(a), int(b)))

    new_edges = []
    for r, edges in by_rel.items():
        src = np.array([a for a, _ in edges])
        dst = np.array([b for _, b in edges])
        kept = 0
        for _ in range(6):                       # 몇 번 섞어 손실을 줄인다
            perm = rng.permutation(len(dst))
            cand = list(zip(src.tolist(), dst[perm].tolist()))
            seen = set()
            out = []
            for a, b in cand:
                if a != b and (a, b) not in seen:
                    seen.add((a, b))
                    out.append((a, b))
            if len(out) > kept:
                kept, best = len(out), out
        new_edges += [(a, b, r) for a, b in best]
        print(f"  {r:<22} {len(edges):>8,} → {kept:>8,}  ({kept/len(edges):.1%} 보존)")

    with io.open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_item_id", "dst_item_id", "relation"])
        w.writerows(sorted(new_edges))

    def degs(path):
        d = defaultdict(set)
        with io.open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                a, b, _ = line.rstrip("\n").split(",")
                d[int(a)].add(int(b)); d[int(b)].add(int(a))
        return np.array([len(d.get(i, ())) for i in range(12000)])

    d0, d1 = degs(args.src), degs(args.out)
    print(f"\n원본   간선 {sum(len(v) for v in by_rel.values()):,} · 평균차수 {d0.mean():.1f} · 고립 {(d0==0).sum()}")
    print(f"무작위 간선 {len(new_edges):,} · 평균차수 {d1.mean():.1f} · 고립 {(d1==0).sum()}")
    print(f"차수 상관(원본 vs 무작위) {np.corrcoef(d0, d1)[0,1]:.4f}  ← 1에 가까울수록 차수가 잘 보존됨")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
