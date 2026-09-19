# -*- coding: utf-8 -*-
"""테이블 배치 시뮬레이션 — 알럼나이 20 + 활동기수 30 = 50명, 테이블 9개.

목표   같은 테이블 안의 친화도 합을 최대화한다. 친화도 = G-SID 주입 벡터의 코사인 + 2계층 접두사(동네) 공유 보너스.
제약   테이블 크기 5~6 (6명 5개 + 5명 4개), 테이블마다 알럼나이 2~3명 (20명 → 2명×7 + 3명×2).
방법   역할별 무작위 초기 배치 → 같은 역할끼리 자리를 바꾸는 담금질(simulated annealing).
비교   같은 제약을 지키는 무작위 배치 300회와 대조 → 알고리즘 배치가 우연보다 얼마나 뭉치는지.
       텍스트 SID 로 같은 절차를 돌려 G-SID 와 비교.

출력  D:/DSL/_event_data/tables_<addr>.csv (이름 포함, 로컬 전용) · tables_<addr>_anon.csv · report_tables.txt
"""
import os, sys, math, random
import numpy as np, pandas as pd

D = r"D:/DSL/_event_data"
N_ALUM, N_ACT, N_TAB, SEED = 20, 30, 9, int(os.environ.get("SEED", 7))
BONUS = 0.15         # 같은 2계층 동네 보너스 (코사인 척도)
sys.stdout.reconfigure(encoding="utf-8")
out = []
P = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))
rng = random.Random(SEED); nrng = np.random.default_rng(SEED)

df = pd.read_csv(os.path.join(D, "profiles.csv")).fillna("")
names = pd.read_csv(os.path.join(D, "id_map.csv")).set_index("pid")["name"]
X = np.load(os.path.join(D, "emb.npy")).astype(np.float64); Z = np.load(os.path.join(D, "z_gsid.npy")).astype(np.float64)
X = X - X.mean(0)   # e5 임베딩은 모두 비슷한 방향이라(코사인 0.9 이상) 중심화해야 친화도 척도가 G-SID 쪽(중심화된 Z)과 비교된다

# ---------- 참석자 50명 표본 ----------
alum = df[df["role"] == "alumni"].sample(N_ALUM, random_state=SEED)
act = df[df["role"] == "active"].sample(N_ACT, random_state=SEED)
att = pd.concat([alum, act]).reset_index(drop=True)
idx = att.index.values; pid2row = {p: i for i, p in enumerate(df["pid"])}
rows = np.array([pid2row[p] for p in att["pid"]])
P(f"참석자 {len(att)}명 = alumni {N_ALUM} (기수 {sorted(alum['cohort'].unique().tolist())}) + active {N_ACT} (기수 {sorted(act['cohort'].unique().tolist())})")
P(f"학과 종류 {att['dept'].nunique()} · 최다 {att['dept'].value_counts().iloc[0]}명 ({att['dept'].value_counts().index[0]})")

sizes = [6] * 5 + [5] * 4                       # 50명
alum_quota = [3, 3] + [2] * 7                    # 20명
assert sum(sizes) == N_ALUM + N_ACT and sum(alum_quota) == N_ALUM

def affinity(V, sid):
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    A = Vn[rows] @ Vn[rows].T
    pre = sid.set_index("pid").loc[att["pid"], ["c1", "c2"]].values
    same = (pre[:, None, :] == pre[None, :, :]).all(2)
    A = A + BONUS * same
    np.fill_diagonal(A, 0)
    return A

def score(assign, A):
    s = 0.0
    for t in range(N_TAB):
        m = np.where(assign == t)[0]
        s += A[np.ix_(m, m)].sum() / 2
    return s

def random_assign():
    a = np.full(len(att), -1)
    al = [i for i in range(len(att)) if att["role"][i] == "alumni"]; ac = [i for i in range(len(att)) if att["role"][i] == "active"]
    rng.shuffle(al); rng.shuffle(ac)
    for t in range(N_TAB):
        for _ in range(alum_quota[t]): a[al.pop()] = t
        for _ in range(sizes[t] - alum_quota[t]): a[ac.pop()] = t
    return a

def anneal(A, iters=40000, T0=0.5, T1=0.005):
    a = random_assign(); best = a.copy(); s = score(a, A); bs = s
    role = att["role"].values
    for k in range(iters):
        T = T0 * (T1 / T0) ** (k / iters)
        i, j = rng.randrange(len(a)), rng.randrange(len(a))
        if a[i] == a[j] or role[i] != role[j]: continue
        ti, tj = a[i], a[j]
        mi = np.where(a == ti)[0]; mj = np.where(a == tj)[0]
        d = (A[j, mi].sum() - A[i, j] - A[i, mi].sum()) + (A[i, mj].sum() - A[i, j] - A[j, mj].sum())
        if d > 0 or rng.random() < math.exp(d / T):
            a[i], a[j] = tj, ti; s += d
            if s > bs: bs, best = s, a.copy()
    return best, bs

def describe(assign, tag, A):
    within = []; dept_div = []; coh_div = []
    for t in range(N_TAB):
        m = np.where(assign == t)[0]
        within.append(A[np.ix_(m, m)].sum() / (len(m) * (len(m) - 1)))
        dept_div.append(att["dept"].iloc[m].nunique()); coh_div.append(att["cohort"].iloc[m].nunique())
    return np.mean(within), np.mean(dept_div), np.mean(coh_div)

results = {}
for addr, V in (("gsid", Z), ("text", X)):
    sid = pd.read_csv(os.path.join(D, f"sid_{addr}.csv"))
    A = affinity(V, sid)
    rand_scores = np.array([score(random_assign(), A) for _ in range(300)])
    best, bs = anneal(A)
    w, dd, cd = describe(best, addr, A); rw = np.mean([describe(random_assign(), addr, A)[0] for _ in range(50)])
    zs = (bs - rand_scores.mean()) / rand_scores.std()
    P(f"\n[{addr}] 배치 점수 {bs:.2f} · 무작위 300회 평균 {rand_scores.mean():.2f}±{rand_scores.std():.2f} · z {zs:.1f} · 무작위 최고 {rand_scores.max():.2f}")
    P(f"  테이블 안 평균 친화도 {w:.3f} (무작위 {rw:.3f}) · 테이블당 학과 종류 {dd:.1f} · 기수 종류 {cd:.1f}")
    results[addr] = (best, A)
    tab = att.assign(table=best + 1).sort_values(["table", "role", "cohort"])
    tab["name"] = tab["pid"].map(names)
    tab[["table", "name", "role", "cohort", "dept", "career"]].to_csv(os.path.join(D, f"tables_{addr}.csv"), index=False, encoding="utf-8-sig")
    tab[["table", "pid", "role", "cohort", "dept"]].to_csv(os.path.join(D, f"tables_{addr}_anon.csv"), index=False, encoding="utf-8-sig")

# 두 배치가 얼마나 다른가
bg, bt = results["gsid"][0], results["text"][0]
same_pair = lambda a: (a[:, None] == a[None, :])
iu = np.triu_indices(len(att), 1)
agree = (same_pair(bg)[iu] & same_pair(bt)[iu]).sum() / same_pair(bg)[iu].sum()
P(f"\nG-SID 배치에서 같은 테이블인 쌍 중 텍스트 배치에서도 같은 테이블인 비율 {agree:.2f}")

# G-SID 배치 표 (익명)
P("\n[G-SID 배치 · 테이블별 구성]")
tab = pd.read_csv(os.path.join(D, "tables_gsid_anon.csv"))
for t, g in tab.groupby("table"):
    P(f"  테이블 {t}: {len(g)}명 · alumni {(g['role']=='alumni').sum()} · 기수 {sorted(g['cohort'].tolist())} · 학과 {g['dept'].value_counts().to_dict()}")

with open(os.path.join(D, "report_tables.txt"), "w", encoding="utf-8") as f: f.write("\n".join(out))
