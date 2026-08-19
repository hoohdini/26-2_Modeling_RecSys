"""RQ-KMeans SID 결과(.pt)를 엑셀에서 열 수 있는 CSV로 내보낸다.

입력: sid/L3/sid_tensor.pt, sid/L4/sid_tensor.pt
      (GRID 후처리에서 transpose 되어 (레벨+1, 아이템) 형태로 저장돼 있음)
출력: csv_export/sid/
"""
import io
import os
import pickle
import sys
from collections import Counter

import pandas as pd
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"D:\DSL\RecSys"
OUT = os.path.join(ROOT, "csv_export", "sid")
os.makedirs(OUT, exist_ok=True)
ENC = "utf-8-sig"

with open(os.path.join(ROOT, "Beauty_split_A.pkl"), "rb") as f:
    A = pickle.load(f)
item_text = A["item_text"]
iid2asin = {v: k for k, v in A["iid"].items()}
n_items = len(item_text)

overview = []

for L in (3, 4):
    t = torch.load(os.path.join(ROOT, "sid", f"L{L}", "sid_tensor.pt"), map_location="cpu")
    sid = t.T if t.shape[0] == L + 1 else t          # (아이템, 레벨+1) 로 되돌림
    assert sid.shape == (n_items, L + 1)

    codes = sid[:, :L]
    dedup = sid[:, L]
    tuples = [tuple(int(x) for x in row) for row in codes]
    cnt = Counter(tuples)

    # ── 아이템별 SID 표 ──────────────────────────────────────────
    df = pd.DataFrame({"item_id": range(n_items),
                       "asin": [iid2asin.get(i) for i in range(n_items)]})
    for lv in range(L):
        df[f"code_{lv+1}"] = codes[:, lv].numpy()
    df["dedup_last"] = dedup.numpy()
    df["sid"] = ["-".join(str(int(x)) for x in row) for row in sid]
    df["sid_prefix"] = ["-".join(str(int(x)) for x in row) for row in codes]
    df["collision_group_size"] = [cnt[tp] for tp in tuples]
    df["is_collided"] = [cnt[tp] > 1 for tp in tuples]
    df["item_text"] = [item_text.get(i, "") for i in range(n_items)]

    p = os.path.join(OUT, f"sid_L{L}_items.csv")
    df.to_csv(p, index=False, encoding=ENC)
    print(f"[{os.path.getsize(p)/1e6:6.2f} MB] sid_L{L}_items.csv  ({len(df):,} rows)")

    # ── 충돌 그룹 표 (충돌 난 것만) ──────────────────────────────
    col = df[df.is_collided].sort_values(["collision_group_size", "sid_prefix", "dedup_last"],
                                         ascending=[False, True, True])
    cols = ["sid_prefix", "collision_group_size", "dedup_last", "item_id", "asin", "item_text"]
    p = os.path.join(OUT, f"sid_L{L}_collisions.csv")
    col[cols].to_csv(p, index=False, encoding=ENC)
    print(f"[{os.path.getsize(p)/1e6:6.2f} MB] sid_L{L}_collisions.csv  ({len(col):,} rows)")

    # ── 레벨별 코드 사용 분포 ────────────────────────────────────
    rows = []
    for lv in range(L):
        vc = pd.Series(codes[:, lv].numpy()).value_counts()
        for code, c in vc.items():
            rows.append({"level": lv + 1, "code": int(code), "item_count": int(c)})
    p = os.path.join(OUT, f"sid_L{L}_code_usage.csv")
    pd.DataFrame(rows).sort_values(["level", "code"]).to_csv(p, index=False, encoding=ENC)
    print(f"[{os.path.getsize(p)/1e6:6.2f} MB] sid_L{L}_code_usage.csv")

    # ── 요약 ────────────────────────────────────────────────────
    used = [int(codes[:, lv].unique().numel()) for lv in range(L)]
    overview.append({
        "설정": f"{L}단계",
        "코드북 레벨 수": L,
        "레벨당 코드북 크기": 256,
        "표현 가능 조합 수": 256 ** L,
        "실제 고유 코드 조합": len(cnt),
        "코드만으로 유일한 비율_%": round(100 * len(cnt) / n_items, 2),
        "충돌 아이템 수": int(df.is_collided.sum()),
        "충돌 비율_%": round(100 * float(df.is_collided.mean()), 2),
        "최대 충돌 그룹 크기": int(max(cnt.values())),
        "마지막 자리 최댓값": int(dedup.max()),
        "레벨별 사용 코드 수": "/".join(map(str, used)),
        "전체 SID 유일성": "OK" if len(set(map(tuple, sid.tolist()))) == n_items else "중복 있음",
    })

ov = pd.DataFrame(overview)
p = os.path.join(OUT, "sid_summary.csv")
ov.to_csv(p, index=False, encoding=ENC)
print(f"\n[{os.path.getsize(p)/1e6:6.2f} MB] sid_summary.csv")
print(ov.to_string(index=False))
