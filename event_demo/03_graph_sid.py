# -*- coding: utf-8 -*-
"""그래프 + 튜닝식 G-SID (β₀=1.0) + RQ-KMeans 주소 발급, 텍스트 SID 와 비교.

그래프 (사람 = 노드)
  같은 학과 1.0 · 같은 기수 1.0 · 같은 단과대 0.3 · 경력 토큰 공유 1.0/토큰 (회사·대학원·직무명)
튜닝식 직교 주입 (Beauty 승자 레시피, docs/RESULTS_실험표.md ⑤)
  x̃ = x − μ,  m = Σ w x̃_j / Σ w (이웃 평균),  z = x̃ + β₀‖x̃‖·unit(m − ⟨m, x̂⟩x̂),  β₀ = 1.0
RQ-KMeans
  L 단계 × K 코드. 243명이라 K=16, L=3 (16³ = 4,096 주소). 충돌은 4번째 자리(구분자)로 푼다.

출력  D:/DSL/_event_data/sid_text.csv · sid_gsid.csv (pid, c1, c2, c3, dis) · z_gsid.npy · report_sid.txt
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

D = r"D:/DSL/_event_data"
K, L, BETA0, SEED = int(os.environ.get("K", 16)), int(os.environ.get("L", 3)), float(os.environ.get("BETA0", 1.0)), 42
sys.stdout.reconfigure(encoding="utf-8")
out = []
P = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))

df = pd.read_csv(os.path.join(D, "profiles.csv")).fillna("")
X = np.load(os.path.join(D, "emb.npy")).astype(np.float64)
N = len(df)

# ---------- 그래프 ----------
W = np.zeros((N, N))
def link(mask_key, w):
    keys = df[mask_key].values
    for i in range(N):
        same = (keys == keys[i]) & (keys != "") & (np.arange(N) != i)
        W[i, same] += w
link("dept", 1.0); link("cohort", 1.0); link("college", 0.3)
tok = [set(t.split("|")) - {""} for t in df["career_tokens"]]
for i in range(N):
    for j in range(i + 1, N):
        c = len(tok[i] & tok[j])
        if c: W[i, j] += c; W[j, i] += c
deg = (W > 0).sum(1)
P(f"그래프: 노드 {N} · 간선 {int((W > 0).sum() / 2)} · 차수 중앙 {int(np.median(deg))} · 고립 {(deg == 0).sum()}")

# ---------- G0 비중첩 (그래프 이웃이 텍스트 최근접과 얼마나 겹치나) ----------
S = X @ X.T; np.fill_diagonal(S, -1)
top10 = np.argsort(-S, 1)[:, :10]
ov = [len(set(top10[i]) & set(np.where(W[i] > 0)[0])) / 10 for i in range(N) if deg[i] > 0]
P(f"G0: 텍스트 top-10 이웃 중 그래프 이웃 비율 평균 {np.mean(ov):.2f}  (1.0 이면 그래프가 텍스트와 중복, 0 이면 새 정보)")

# ---------- 튜닝식 직교 주입 ----------
mu = X.mean(0); Xt = X - mu
Z = Xt.copy()
for i in range(N):
    if deg[i] == 0: continue
    w = W[i]; m = (w[:, None] * Xt).sum(0) / w.sum()
    xh = Xt[i] / (np.linalg.norm(Xt[i]) + 1e-12)
    r = m - (m @ xh) * xh
    nr = np.linalg.norm(r)
    if nr > 1e-12:
        Z[i] = Xt[i] + BETA0 * np.linalg.norm(Xt[i]) * r / nr
np.save(os.path.join(D, "z_gsid.npy"), Z.astype(np.float32))

# ---------- RQ-KMeans ----------
def rq_kmeans(V, K, L, seed):
    codes = np.zeros((len(V), L), int); R = V.copy(); cbs = []
    for l in range(L):
        km = KMeans(n_clusters=K, n_init=10, random_state=seed + l).fit(R)
        codes[:, l] = km.labels_; R = R - km.cluster_centers_[km.labels_]; cbs.append(km.cluster_centers_)
    dis = np.zeros(len(V), int); seen = {}
    for i, c in enumerate(map(tuple, codes)):
        dis[i] = seen.get(c, 0); seen[c] = dis[i] + 1
    return codes, dis

def usage_bits(codes):
    return [float(-(p := np.bincount(codes[:, l], minlength=K) / len(codes))[p > 0] @ np.log2(p[p > 0])) for l in range(L)]

res = {}
for name, V in (("text", Xt), ("gsid", Z)):
    codes, dis = rq_kmeans(V, K, L, SEED)
    sid = pd.DataFrame({"pid": df["pid"], **{f"c{l+1}": codes[:, l] for l in range(L)}, "dis": dis})
    sid.to_csv(os.path.join(D, f"sid_{name}.csv"), index=False)
    res[name] = (codes, dis)
    # 품질
    n_unique = len(set(map(tuple, codes))); coll = (dis > 0).mean()
    pre2 = pd.Series([f"{a}-{b}" for a, b in codes[:, :2]])
    grp = df.assign(pre=pre2.values).groupby("pre")
    def purity(col):
        return float(np.mean([g[col].value_counts(normalize=True).iloc[0] for _, g in grp if len(g) > 1]))
    sizes = grp.size()
    P(f"\n[{name}] K={K} L={L} · 고유 주소 {n_unique}/{N} · 충돌(구분자>0) {coll:.1%} · 코드 사용 비트 {[round(b,2) for b in usage_bits(codes)]} (최대 {np.log2(K):.0f})")
    P(f"  2계층 접두사 동네 {len(sizes)}개 · 크기 중앙 {int(sizes.median())} 최대 {int(sizes.max())} · 동네 안 같은 학과 비율 {purity('dept'):.2f} · 같은 기수 {purity('cohort'):.2f} · 같은 역할(alumni/active) {purity('role'):.2f}")

# ---------- 두 주소가 얼마나 다른가 ----------
ct, cg = res["text"][0], res["gsid"][0]
P(f"\n텍스트 vs G-SID: 1계층 코드 일치율(라벨 재배열 무시, 같은 동네에 함께 있는 쌍 기준) —")
def pair_agree(c1, c2, l):
    a = c1[:, :l]; b = c2[:, :l]
    same_a = (a[:, None, :] == a[None, :, :]).all(2); same_b = (b[:, None, :] == b[None, :, :]).all(2)
    iu = np.triu_indices(N, 1)
    return (same_a[iu] & same_b[iu]).sum() / max(1, same_a[iu].sum())
P(f"  텍스트 1계층 동네 쌍이 G-SID 에서도 같은 1계층인 비율 {pair_agree(ct, cg, 1):.2f} · 2계층 {pair_agree(ct, cg, 2):.2f}")

# 무작위 기대치
exp_dept = float((df["dept"].value_counts(normalize=True) ** 2).sum()); exp_coh = float((df["cohort"].value_counts(normalize=True) ** 2).sum())
P(f"  참고 — 무작위로 묶었을 때 같은 학과 {exp_dept:.2f} · 같은 기수 {exp_coh:.2f}")

with open(os.path.join(D, "report_sid.txt"), "w", encoding="utf-8") as f: f.write("\n".join(out))
