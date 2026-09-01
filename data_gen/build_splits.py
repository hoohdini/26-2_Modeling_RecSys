"""상호작용 로그 → Split A(LOO) · Split B(Temporal) 피클.

`preprocessing/repreprocess.py:318-345` 가 만드는 것과 **똑같은 스키마**로 낸다.
그래야 `code/to_grid_v2.py` · `to_grid_b.py` · `code/evaluate.py` 가 수정 없이 읽는다.

Split A 페이로드
    uid, iid, user_seq, item_text, item_graph, item_salesrank,
    loo_train, loo_val, loo_test          (len(seq) < 3 인 유저는 제외)

Split B 페이로드
    위 + windows{W3,W4,W5}, cold_tiers, item_counts
    윈도우는 Beauty 재전처리와 같은 방식 — **상호작용 건수 분위수**로 자르고
    타임스탬프 경계로 스냅한다 (달력 균등 분할은 Beauty 에서 train 이 말라 실패했다).

콜드 등급도 Beauty 와 동일한 경계를 쓴다: 0 unseen / <=3 very_rare / <=5 rare / else normal
"""
import argparse
import csv
import io
import os
import pickle
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data_gen", "out")
N_ITEMS = 12000


def rows_to_seq(rows):
    d = defaultdict(list)
    for u, _, i, t in rows:
        d[u].append((t, i))
    return {u: [i for _, i in sorted(v)] for u, v in d.items()}


def rows_to_targets(rows):
    d = defaultdict(set)
    for u, _, i, _ in rows:
        d[u].add(i)
    return {u: sorted(v) for u, v in d.items()}


def tiers_of(counts):
    t = {}
    for i in range(N_ITEMS):
        c = counts.get(i, 0)
        t[i] = "unseen" if c == 0 else ("very_rare" if c <= 3 else ("rare" if c <= 5 else "normal"))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=5)
    args = ap.parse_args()
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # ---- 적재 ----
    rows = []
    with io.open(os.path.join(OUT, "interactions.csv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            u, p, i, t = line.rstrip("\n").split(",")
            rows.append((int(u), int(p), int(i), float(t)))
    rows.sort(key=lambda r: (r[3], r[0], r[1]))

    text = {}
    with io.open(os.path.join(OUT, "profile_text.tsv"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                i, t = line.rstrip("\n").split("\t", 1)
                text[int(i)] = t

    graph = defaultdict(set)
    with io.open(os.path.join(OUT, "jobs_edges_final.csv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, _ = line.rstrip("\n").split(",")
            graph[int(a)].add(int(b))

    prof = list(csv.DictReader(io.open(os.path.join(OUT, "profiles.csv"), encoding="utf-8")))
    # item_salesrank 자리에는 그 직업의 KECO 코드를 넣는다 (Beauty 의 salesRank 대응 슬롯)
    salesrank = {int(r["profile_id"]): {"keco": r["keco_cd"]} for r in prof}

    uid = {u: u for u in sorted({r[0] for r in rows})}
    iid = {i: i for i in range(N_ITEMS)}          # 이미 조밀한 0..N-1
    user_seq = rows_to_seq(rows)

    # ---- Split A : LOO ----
    tr, va, te = {}, {}, {}
    for u, seq in user_seq.items():
        if len(seq) < 3:
            continue
        tr[u], va[u], te[u] = seq[:-2], seq[-2], seq[-1]

    split_a = {"uid": uid, "iid": iid, "user_seq": user_seq, "item_text": text,
               "item_graph": {k: sorted(v) for k, v in graph.items()},
               "item_salesrank": salesrank,
               "loo_train": tr, "loo_val": va, "loo_test": te}
    with open(os.path.join(OUT, "Jobs_split_A.pkl"), "wb") as f:
        pickle.dump(split_a, f)
    print(f"Split A · 유저 {len(user_seq):,} · 채점 가능 {len(te):,} · 아이템 {N_ITEMS:,}")

    # ---- Split B : 상호작용 건수 분위수로 자르고 타임스탬프로 스냅 ----
    k = args.windows
    q = np.quantile([r[3] for r in rows], np.linspace(0, 1, k + 1)[1:-1])
    bounds = [-np.inf] + list(q) + [np.inf]
    chunks = [[r for r in rows if bounds[j] <= r[3] < bounds[j + 1]] for j in range(k)]
    print(f"윈도우 경계(일) {[round(float(x), 1) for x in q]}")
    print(f"청크 크기 {[len(c) for c in chunks]}")

    windows, cold_tiers, item_counts = {}, {}, {}
    for w in (3, 4, 5):
        train_rows = [r for c in chunks[:w - 2] for r in c]
        val_rows, test_rows = chunks[w - 2], chunks[w - 1]
        cnt = defaultdict(int)
        for _, _, i, _ in train_rows:
            cnt[i] += 1
        label = f"W{w}"
        windows[label] = {
            "train_seq": rows_to_seq(train_rows),
            "val_targets": rows_to_targets(val_rows),
            "test_targets": rows_to_targets(test_rows),
            "n_train_interactions": len(train_rows),
            "n_val_interactions": len(val_rows),
            "n_test_interactions": len(test_rows),
        }
        cold_tiers[label] = tiers_of(cnt)
        item_counts[label] = dict(cnt)
        tt = windows[label]["test_targets"]
        scorable = sum(1 for u in tt if u in windows[label]["train_seq"])
        n_unseen = sum(1 for v in cold_tiers[label].values() if v == "unseen")
        print(f"  {label}: train {len(train_rows):,} · 학습유저 {len(windows[label]['train_seq']):,} · "
              f"테스트유저 {len(tt):,} (채점가능 {scorable:,}) · unseen 아이템 {n_unseen:,}")

    split_b = dict(split_a)
    split_b.pop("loo_train"); split_b.pop("loo_val"); split_b.pop("loo_test")
    split_b.update({"windows": windows, "cold_tiers": cold_tiers, "item_counts": item_counts})
    with open(os.path.join(OUT, "Jobs_split_B.pkl"), "wb") as f:
        pickle.dump(split_b, f)
    print(f"→ {OUT}/Jobs_split_A.pkl · Jobs_split_B.pkl")


if __name__ == "__main__":
    main()
