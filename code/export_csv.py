"""전처리 pkl 3종 + 임베딩 .pt 를 엑셀에서 열 수 있는 CSV로 내보낸다.

출력: D:\\DSL\\RecSys\\csv_export\\
인코딩은 전부 utf-8-sig (엑셀에서 한글/특수문자 깨짐 방지).
"""
import os
import pickle
from collections import Counter

import pandas as pd
import torch

ROOT = r"D:\DSL\RecSys"
OUT = os.path.join(ROOT, "csv_export")
OUT_EMB = os.path.join(OUT, "embeddings")
OUT_DS = os.path.join(OUT, "dataset")
for d in (OUT, OUT_EMB, OUT_DS):
    os.makedirs(d, exist_ok=True)

ENC = "utf-8-sig"
manifest = []


def save(df, path, note):
    df.to_csv(path, index=False, encoding=ENC)
    size_mb = os.path.getsize(path) / 1e6
    rel = os.path.relpath(path, OUT)
    manifest.append({"파일": rel, "행 수": len(df), "열 수": df.shape[1],
                     "크기(MB)": round(size_mb, 2), "내용": note})
    print(f"  [{size_mb:8.2f} MB] {rel}  ({len(df):,} rows x {df.shape[1]} cols)")


print("== pkl 로드 ==")
with open(os.path.join(ROOT, "Beauty_split_A.pkl"), "rb") as f:
    A = pickle.load(f)
with open(os.path.join(ROOT, "Beauty_split_B.pkl"), "rb") as f:
    B = pickle.load(f)
with open(os.path.join(ROOT, "Beauty_related_separate.pkl"), "rb") as f:
    R = pickle.load(f)

iid2asin = {v: k for k, v in A["iid"].items()}          # item_id -> ASIN
uid2raw = {v: k for k, v in A["uid"].items()}           # user_id -> 원본 유저 문자열
item_text = A["item_text"]
n_items = len(item_text)


def salesrank_of(i):
    d = A["item_salesrank"].get(i) or {}
    if not d:
        return None, None
    cat, rank = next(iter(d.items()))
    return cat, rank


# ────────────────────────────────────────────────────────────────
# 1. 아이템 마스터
# ────────────────────────────────────────────────────────────────
print("\n== 1. 아이템 마스터 ==")
rows = []
for i in range(n_items):
    cat, rank = salesrank_of(i)
    txt = item_text.get(i, "")
    rows.append({
        "item_id": i,
        "asin": iid2asin.get(i),
        "salesrank_category": cat,
        "salesrank": rank,
        "text_length": len(txt),
        "item_text": txt,
    })
items_df = pd.DataFrame(rows)
save(items_df, os.path.join(OUT_DS, "A_items.csv"),
     "아이템 12,101개 마스터 (ASIN·판매순위·GRID 입력 텍스트). 임베딩 행 순서와 동일")

# ────────────────────────────────────────────────────────────────
# 2. Split A — 유저별 시퀀스 요약 / 롱포맷
# ────────────────────────────────────────────────────────────────
print("\n== 2. Split A (Leave-One-Out) ==")
user_rows, long_rows = [], []
for uraw, seq in A["loo_train"].items():
    u = A["uid"][uraw]
    val_i = A["loo_val"][uraw]
    test_i = A["loo_test"][uraw]
    user_rows.append({
        "user_id": u,
        "user_raw_id": uraw,
        "train_seq_len": len(seq),
        "train_sequence": " ".join(map(str, seq)),
        "val_item_id": val_i,
        "test_item_id": test_i,
    })
    for pos, it in enumerate(seq):
        long_rows.append({"user_id": u, "position": pos, "item_id": it, "split": "train"})
    long_rows.append({"user_id": u, "position": len(seq), "item_id": val_i, "split": "val"})
    long_rows.append({"user_id": u, "position": len(seq) + 1, "item_id": test_i, "split": "test"})

users_df = pd.DataFrame(user_rows).sort_values("user_id")
save(users_df, os.path.join(OUT_DS, "A_users.csv"),
     "유저 22,363명: train 시퀀스(공백 구분)·val/test 정답 아이템")

inter_df = pd.DataFrame(long_rows).sort_values(["user_id", "position"])
save(inter_df, os.path.join(OUT_DS, "A_interactions_long.csv"),
     "Split A 상호작용 롱포맷 (유저·순번·아이템·split). 피벗/집계용")

# ────────────────────────────────────────────────────────────────
# 3. Split B — 윈도우별
# ────────────────────────────────────────────────────────────────
print("\n== 3. Split B (절대시간 분할) ==")
win_summary, b_seq_rows, b_target_rows, b_cold_rows = [], [], [], []
for w in B["windows"]:
    lab = w["window_label"]
    tr, va, te = w["train_user_seq"], w["val_targets"], w["test_targets"]
    n_tr_i = sum(len(v) for v in tr.values())
    n_va_i = sum(len(v) for v in va.values())
    n_te_i = sum(len(v) for v in te.values())
    tier_cnt = Counter(w["cold_tiers"].values())
    win_summary.append({
        "window": lab,
        "train_users": len(tr), "train_interactions": n_tr_i,
        "val_users": len(va), "val_interactions": n_va_i,
        "test_users": len(te), "test_interactions": n_te_i,
        "train_ratio_%": round(100 * n_tr_i / max(1, n_tr_i + n_va_i + n_te_i), 2),
        **{f"cold_{k}": v for k, v in sorted(tier_cnt.items())},
    })
    for uraw, seq in tr.items():
        for pos, it in enumerate(seq):
            b_seq_rows.append({"window": lab, "user_raw_id": uraw,
                               "user_id": A["uid"].get(uraw), "position": pos, "item_id": it})
    for split, tgt in (("val", va), ("test", te)):
        for uraw, items in tgt.items():
            for it in items:
                b_target_rows.append({"window": lab, "split": split, "user_raw_id": uraw,
                                      "user_id": A["uid"].get(uraw), "item_id": it})
    for i, tier in w["cold_tiers"].items():
        b_cold_rows.append({"window": lab, "item_id": i, "cold_tier": tier,
                            "item_count": w["item_counts"].get(i, 0)})

save(pd.DataFrame(win_summary), os.path.join(OUT_DS, "B_windows_summary.csv"),
     "W3/W4/W5 윈도우별 규모 요약 ★ 분할 비율 이상(train 0.04~2.6%)이 여기서 바로 보임")
save(pd.DataFrame(b_seq_rows), os.path.join(OUT_DS, "B_train_sequences_long.csv"),
     "Split B 윈도우별 train 시퀀스 롱포맷")
save(pd.DataFrame(b_target_rows), os.path.join(OUT_DS, "B_targets_long.csv"),
     "Split B 윈도우별 val/test 정답 아이템 롱포맷")
save(pd.DataFrame(b_cold_rows), os.path.join(OUT_DS, "B_cold_tiers.csv"),
     "윈도우별 아이템 콜드 등급(unseen/very_rare/...)과 등장 횟수")

# ────────────────────────────────────────────────────────────────
# 4. 관계 그래프
# ────────────────────────────────────────────────────────────────
print("\n== 4. 관계 그래프 ==")
edge_rows = []
for src, nbrs in R["item_graph_detailed"].items():
    for dst, rels in nbrs.items():
        for rel in rels:
            edge_rows.append({"src_item_id": src, "dst_item_id": dst, "relation": rel})
edges_df = pd.DataFrame(edge_rows)
save(edges_df, os.path.join(OUT_DS, "relations_edges_long.csv"),
     "아이템 관계 간선 (also_bought / also_viewed / bought_together) 롱포맷")

rel_summary = edges_df["relation"].value_counts().rename_axis("relation").reset_index(name="edge_count")
rel_summary["ratio_%"] = (100 * rel_summary["edge_count"] / len(edges_df)).round(2)
distinct_edges = edges_df[["src_item_id", "dst_item_id"]].drop_duplicates()
rel_summary.loc[len(rel_summary)] = ["(중복 제거한 방향 간선 수)", len(distinct_edges), 100.0]
save(rel_summary, os.path.join(OUT_DS, "relations_summary.csv"),
     "관계 종류별 간선 수 (한 간선이 여러 관계를 동시에 가질 수 있어 합계 > 고유 간선 수)")

# ────────────────────────────────────────────────────────────────
# 5. 데이터셋 전체 요약
# ────────────────────────────────────────────────────────────────
print("\n== 5. 데이터셋 요약 ==")
n_A_inter = len(inter_df)
summary = [
    ("공통", "유저 수", len(A["uid"])),
    ("공통", "아이템 수", len(A["iid"])),
    ("공통", "max_seq_len", A["max_seq_len"]),
    ("공통", "관계 그래프에 등장하는 아이템 수", len(R["item_graph_detailed"])),
    ("Split A", "train 시퀀스 총 상호작용", int(users_df["train_seq_len"].sum())),
    ("Split A", "val 정답 수", len(A["loo_val"])),
    ("Split A", "test 정답 수", len(A["loo_test"])),
    ("Split A", "총 상호작용(롱포맷 행 수)", n_A_inter),
    ("Split A", "유저당 train 길이 평균", round(float(users_df["train_seq_len"].mean()), 2)),
    ("Split A", "유저당 train 길이 최대", int(users_df["train_seq_len"].max())),
    ("Split B", "윈도우 수", len(B["windows"])),
    ("관계", "관계 라벨 붙은 간선 수", len(edges_df)),
    ("관계", "고유 방향 간선 수", len(distinct_edges)),
    ("임베딩", "차원", 2048),
    ("임베딩", "모델", "google/flan-t5-xl (encoder)"),
]
save(pd.DataFrame(summary, columns=["구분", "항목", "값"]),
     os.path.join(OUT_DS, "dataset_summary.csv"), "전체 통계 한 장 요약 (발표자료용)")

# ────────────────────────────────────────────────────────────────
# 6. 임베딩
# ────────────────────────────────────────────────────────────────
print("\n== 6. 임베딩 ==")
emb = torch.load(os.path.join(ROOT, "embeddings", "beauty_A", "merged_predictions_tensor.pt"),
                 map_location="cpu")
assert emb.shape[0] == n_items, (emb.shape, n_items)
dim = emb.shape[1]

# 6-1. 미리보기 (엑셀에서 가볍게 열림)
prev_n, prev_d = 300, 16
prev = pd.DataFrame(emb[:prev_n, :prev_d].numpy(),
                    columns=[f"dim_{j}" for j in range(prev_d)]).round(6)
prev.insert(0, "item_text", [item_text.get(i, "")[:120] for i in range(prev_n)])
prev.insert(0, "asin", [iid2asin.get(i) for i in range(prev_n)])
prev.insert(0, "item_id", range(prev_n))
save(prev, os.path.join(OUT_EMB, "item_embeddings_preview.csv"),
     f"임베딩 미리보기: 앞쪽 {prev_n}개 아이템 x 앞쪽 {prev_d}차원 + 상품명 (엑셀에서 바로 열림)")

# 6-2. 아이템별 통계
norms = emb.norm(dim=1)
stats = pd.DataFrame({
    "item_id": range(n_items),
    "asin": [iid2asin.get(i) for i in range(n_items)],
    "l2_norm": norms.numpy().round(6),
    "mean": emb.mean(dim=1).numpy().round(6),
    "std": emb.std(dim=1).numpy().round(6),
    "min": emb.min(dim=1).values.numpy().round(6),
    "max": emb.max(dim=1).values.numpy().round(6),
    "text_length": [len(item_text.get(i, "")) for i in range(n_items)],
})
save(stats, os.path.join(OUT_EMB, "item_embedding_stats.csv"),
     "아이템별 임베딩 통계(노름·평균·표준편차 등). 이상치 점검용")

# 6-3. PCA 2D / 50D  (torch SVD, sklearn 불필요)
X = emb - emb.mean(dim=0, keepdim=True)
U, S, V = torch.pca_lowrank(X, q=50, center=False)
proj = X @ V                                   # (n_items, 50)
var = (S ** 2) / (S ** 2).sum()

pca2 = pd.DataFrame({
    "item_id": range(n_items),
    "asin": [iid2asin.get(i) for i in range(n_items)],
    "pc1": proj[:, 0].numpy().round(5),
    "pc2": proj[:, 1].numpy().round(5),
    "salesrank_category": items_df["salesrank_category"],
    "salesrank": items_df["salesrank"],
    "item_text_short": [item_text.get(i, "")[:80] for i in range(n_items)],
})
save(pca2, os.path.join(OUT_EMB, "item_embeddings_pca2d.csv"),
     "PCA 2차원 좌표 ★ 발표자료 산점도용 (엑셀 분산형 차트로 바로 그릴 수 있음)")

pca50 = pd.DataFrame(proj.numpy().round(5), columns=[f"pc_{j+1}" for j in range(50)])
pca50.insert(0, "asin", [iid2asin.get(i) for i in range(n_items)])
pca50.insert(0, "item_id", range(n_items))
save(pca50, os.path.join(OUT_EMB, "item_embeddings_pca50.csv"),
     "PCA 50차원 축약 임베딩 (2048차원 원본 대신 가볍게 다룰 때)")

save(pd.DataFrame({"component": [f"pc_{j+1}" for j in range(50)],
                   "explained_variance_ratio": var.numpy().round(6),
                   "cumulative": var.cumsum(0).numpy().round(6)}),
     os.path.join(OUT_EMB, "pca_explained_variance.csv"),
     "PCA 주성분별 설명 분산 비율 (몇 차원이면 충분한지 근거)")

# 6-4. 최근접 이웃 top-5
print("  최근접 이웃 계산 중...")
Xn = torch.nn.functional.normalize(emb, dim=1)
nn_rows = []
CH = 512
for s in range(0, n_items, CH):
    sims = Xn[s:s + CH] @ Xn.T
    for r in range(sims.shape[0]):
        i = s + r
        sims[r, i] = -2
    top = sims.topk(5, dim=1)
    for r in range(sims.shape[0]):
        i = s + r
        for rank, (sc, j) in enumerate(zip(top.values[r].tolist(), top.indices[r].tolist()), 1):
            nn_rows.append({
                "item_id": i,
                "item_text": item_text.get(i, "")[:100],
                "rank": rank,
                "neighbor_item_id": j,
                "cosine_similarity": round(sc, 5),
                "neighbor_text": item_text.get(j, "")[:100],
            })
save(pd.DataFrame(nn_rows), os.path.join(OUT_EMB, "item_nearest_neighbors_top5.csv"),
     "아이템별 코사인 유사도 top-5 이웃 ★ 임베딩 품질 근거 (발표자료 예시표로 사용)")

# 6-5. 전체 임베딩 (용량 큼)
print("  전체 임베딩 CSV 작성 중 (수백 MB, 수 분 소요)...")
full_path = os.path.join(OUT_EMB, "item_embeddings_full_2048d.csv")
full = pd.DataFrame(emb.numpy(), columns=[f"dim_{j}" for j in range(dim)])
full.insert(0, "asin", [iid2asin.get(i) for i in range(n_items)])
full.insert(0, "item_id", range(n_items))
full.to_csv(full_path, index=False, encoding=ENC, float_format="%.6f")
sz = os.path.getsize(full_path) / 1e6
manifest.append({"파일": os.path.relpath(full_path, OUT), "행 수": n_items, "열 수": dim + 2,
                 "크기(MB)": round(sz, 2),
                 "내용": "임베딩 원본 전체 12,101 x 2048 ⚠️ 용량 큼 — 엑셀보다 pandas/R 권장"})
print(f"  [{sz:8.2f} MB] {os.path.basename(full_path)}")

# ────────────────────────────────────────────────────────────────
# 7. 목록 파일
# ────────────────────────────────────────────────────────────────
mf = pd.DataFrame(manifest)
mf.to_csv(os.path.join(OUT, "00_파일목록.csv"), index=False, encoding=ENC)
print("\n== 완료 ==")
print(mf.to_string(index=False))
