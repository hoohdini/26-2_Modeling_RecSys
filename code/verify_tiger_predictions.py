"""TIGER 예측 육안 검증 — 지표 코드를 거치지 않는 독립 확인.

tiger_infer.sh 가 남긴 merged_predictions.pkl (user_id -> 생성 SID top-10) 을 읽어
  - 정답 아이템과 대조해 Recall@5/@10 을 직접 재계산하고
  - 유저 몇 명의 히스토리·생성결과·정답을 상품명으로 출력한다.
학습 로그의 test 지표와 값이 일치하면 지표 코드가 옳다는 뜻이다.
"""
import io
import os
import pickle
import sys

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"D:\DSL\RecSys"
RUN = sys.argv[1] if len(sys.argv) > 1 else "sched_L4"
N_SHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 4

PRED = os.path.join(ROOT, "tiger_runs", RUN, "infer", "merged_predictions.pkl")
SID = os.path.join(ROOT, "sid", "L4", "sid_tensor.pt")

with open(os.path.join(ROOT, "Beauty_split_A.pkl"), "rb") as f:
    A = pickle.load(f)
item_text, loo_test = A["item_text"], A["loo_test"]
uid = A["uid"]
u2raw = {v: k for k, v in uid.items()}
name = lambda i: (item_text.get(i, "") or "")[:64]

sid_t = torch.load(SID, map_location="cpu").t()            # (12101, 5)
sid2item = {tuple(int(x) for x in sid_t[i]): i for i in range(sid_t.shape[0])}

with open(PRED, "rb") as f:
    preds = pickle.load(f)
print(f"[{RUN}] 예측 {len(preds):,}건 로드")
sample = preds[0]
print(f"  레코드 형태: {type(sample).__name__}, 키: {list(sample) if isinstance(sample, dict) else '?'}")

# ── 직접 재계산 ─────────────────────────────────────────────────
hit5 = hit10 = n = 0
invalid = 0
rows = []
for rec in preds:
    u = int(rec["user_id"]) if not torch.is_tensor(rec["user_id"]) else int(rec["user_id"].item())
    sids = rec["semantic_ids"]
    if torch.is_tensor(sids):
        sids = sids.view(-1, sid_t.shape[1])
    cands = [tuple(int(x) for x in row) for row in sids]
    raw = u2raw.get(u)
    if raw is None or raw not in loo_test:
        continue
    tgt = loo_test[raw]
    tgt_sid = tuple(int(x) for x in sid_t[tgt])
    n += 1
    if tgt_sid in cands[:5]:
        hit5 += 1
    if tgt_sid in cands[:10]:
        hit10 += 1
    invalid += sum(1 for c in cands[:10] if c not in sid2item)
    if len(rows) < N_SHOW:
        rows.append((u, tgt, tgt_sid, cands))

print(f"\n=== 직접 재계산 (유저 {n:,}명) ===")
print(f"  Recall@5  = {hit5/n:.4f}   ({hit5:,}/{n:,})")
print(f"  Recall@10 = {hit10/n:.4f}   ({hit10:,}/{n:,})")
print(f"  생성 후보 중 실제 아이템에 매핑 안 되는 SID: {invalid:,}개 "
      f"({100*invalid/(n*10):.2f}%)")

print(f"\n=== 예시 {len(rows)}명 ===")
for u, tgt, tgt_sid, cands in rows:
    print("-" * 74)
    print(f"user {u}")
    print(f"  정답 [{tgt:5d}] {'-'.join(map(str,tgt_sid))}  {name(tgt)}")
    for r, c in enumerate(cands[:5], 1):
        it = sid2item.get(c)
        mark = "  <== 정답" if c == tgt_sid else ""
        label = name(it) if it is not None else "(존재하지 않는 SID)"
        print(f"   {r}. {'-'.join(map(str,c)):16s} {label}{mark}")
