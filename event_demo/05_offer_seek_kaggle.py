# -*- coding: utf-8 -*-
"""Offer·Seek 두 SID 아이디어 검증 — 캐글 CareerBuilder 2012 (나혜 9/17 방안 1).

사람 A 의 Offer = 프로필(학위·전공·경력·과거 직함) 텍스트,  Seek = A 가 앞서 지원한 공고들의 텍스트 평균.
상대 B 의 Offer = 공고 텍스트(제목 + 요구사항). 질문 둘:
  1) Seek(A) 가 다음 지원 공고를 맞히는가, Offer(A) 보다 나은가?
  2) Offer 와 Seek 를 한 코드북에 섞으면 1계층이 내용이 아니라 문체(사람/공고)로 갈리는가? 갈린다면 어떻게 막나?

절차
  1. 윈도우 1 Train 유저 중 지원 4건 이상인 사람 표본 → 마지막 지원 = 정답, 앞선 지원 = Seek 재료
  2. 공고 풀 = 표본이 지원한 모든 공고. 임베딩 multilingual-e5-small
     공고·Offer = "passage: " 접두사. Seek = (a) 공고 passage 임베딩 평균  (b) 같은 텍스트를 "query: " 로 임베딩한 평균
  3. 검색: 정답 공고가 풀 안에서 몇 등인가 → recall@10/@50. Offer 단독 · Seek(a) · Seek(b) · 결합
  4. 코드북 (RQ-KMeans K=16 L=3) 네 가지:
     (a) 세 출처 합쳐 학습, 전부 passage   (b) Seek 만 query 접두사
     (c) 출처별 평균을 뺀 뒤 합쳐 학습     (d) 공고로만 학습하고 Seek·Offer 는 배정만
     각각 1계층의 출처 순도(문체 분리), Seek-SID 와 정답 공고 SID 의 접두사 일치율(무작위 대비)
  5. (e) 후보를 Seek 2계층 동네로 한정했을 때 검색이 유지되는가

출력  D:/DSL/_event_data/kaggle_offer_seek_report.txt
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, re, csv, time, html
import numpy as np, pandas as pd, torch
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

K = r"D:/DSL/_external/kaggle_job_recommendation"
OUT = r"D:/DSL/_event_data"
N_USERS, SEED, KC, LV = int(os.environ.get("N_USERS", 2500)), 42, 16, 3
sys.stdout.reconfigure(encoding="utf-8")
rep = []
P = lambda *a: (print(*a), rep.append(" ".join(str(x) for x in a)))
rng = np.random.default_rng(SEED)
t0 = time.time()

# ---------- 1. 표본 ----------
apps = pd.read_csv(f"{K}/apps.tsv", sep="\t", usecols=["UserID", "WindowID", "Split", "ApplicationDate", "JobID"])
apps = apps[(apps.WindowID == 1) & (apps.Split == "Train")].sort_values(["UserID", "ApplicationDate"])
cnt = apps.groupby("UserID").size()
elig = cnt[cnt >= 4].index.values
users = rng.choice(elig, size=min(N_USERS, len(elig)), replace=False)
a = apps[apps.UserID.isin(users)]
last = a.groupby("UserID").tail(1).set_index("UserID")["JobID"]
prev = a[~a.index.isin(a.groupby("UserID").tail(1).index)].groupby("UserID")["JobID"].apply(list)
pool = sorted(set(a.JobID))
P(f"표본 유저 {len(users)} (지원 4건 이상 {len(elig)}명 중) · 공고 풀 {len(pool)} · 유저당 앞선 지원 중앙 {int(prev.map(len).median())}")

# ---------- 2. 텍스트 ----------
def strip(t):
    t = html.unescape(str(t)); t = re.sub(r"\\r|\\n|<[^>]+>", " ", t); return re.sub(r"\s+", " ", t).strip()
jobs = {}
need = {str(x) for x in pool}
for ch in pd.read_csv(f"{K}/jobs.tsv", sep="\t", usecols=["JobID", "Title", "Description", "Requirements"], quoting=csv.QUOTE_NONE,
                      on_bad_lines="skip", chunksize=200000, dtype=str, encoding_errors="ignore"):
    ch = ch[ch.JobID.isin(need)]
    for r in ch.itertuples(index=False):
        req = strip(r.Requirements); des = strip(r.Description)
        body = req if len(req) > 40 else des
        jobs[int(r.JobID)] = f"{strip(r.Title)}. {body[:500]}"
    if len(jobs) >= len(need): break
pool = [j for j in pool if j in jobs]
jidx = {j: i for i, j in enumerate(pool)}
users = [u for u in users if last[u] in jidx and all(j in jidx for j in prev.get(u, [])) and len(prev.get(u, [])) >= 1]
P(f"공고 텍스트 확보 {len(pool)} · 평가 유저 {len(users)} · {time.time()-t0:.0f}초")

U = pd.read_csv(f"{K}/users.tsv", sep="\t").set_index("UserID")
H = pd.read_csv(f"{K}/user_history.tsv", sep="\t", usecols=["UserID", "Sequence", "JobTitle"]).dropna()
H = H[H.UserID.isin(users)].sort_values(["UserID", "Sequence"]).groupby("UserID")["JobTitle"].apply(lambda s: "; ".join(map(str, s[-5:])))
def offer_text(u):
    r = U.loc[u]
    deg = str(r.DegreeType) if pd.notna(r.DegreeType) else "no degree listed"
    maj = f" in {r.Major}" if pd.notna(r.Major) and str(r.Major).strip() else ""
    yrs = int(r.TotalYearsExperience) if pd.notna(r.TotalYearsExperience) else 0
    mng = "manages others" if str(r.ManagedOthers) == "Yes" else "individual contributor"
    hist = H.get(u, "")
    return f"{deg}{maj}, {yrs} years of experience, {mng}. Past job titles: {hist}." if hist else f"{deg}{maj}, {yrs} years of experience, {mng}."

# ---------- 3. 임베딩 ----------
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = SentenceTransformer("intfloat/multilingual-e5-small", device=dev)
enc = lambda texts, pre: np.asarray(m.encode([pre + t for t in texts], batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
J_p = enc([jobs[j] for j in pool], "passage: ")
J_q = enc([jobs[j] for j in pool], "query: ")
O = enc([offer_text(u) for u in users], "passage: ")
def mean_norm(M, idx_lists):
    X = np.stack([M[i].mean(0) for i in idx_lists]); return X / np.linalg.norm(X, axis=1, keepdims=True)
prev_idx = [[jidx[j] for j in prev[u]] for u in users]
S_a = mean_norm(J_p, prev_idx); S_b = mean_norm(J_q, prev_idx)
tgt = np.array([jidx[last[u]] for u in users])
P(f"임베딩 완료 · 공고 {len(pool)} · 유저 {len(users)} · {time.time()-t0:.0f}초 · 장치 {dev}")

# ---------- 4. 검색 평가 ----------
def ranks(Q):
    S = Q @ J_p.T
    for i in range(len(users)): S[i, prev_idx[i]] = -9      # 이미 지원한 공고는 후보에서 제외
    return (S > S[np.arange(len(users)), tgt][:, None]).sum(1)
def evaluate(Q, name):
    rank = ranks(Q); r10, r50, mrr = (rank < 10).mean(), (rank < 50).mean(), (1 / (rank + 1)).mean()
    P(f"  {name:<30} recall@10 {r10:.3f} · recall@50 {r50:.3f} · MRR {mrr:.3f}")
P("")
P("[검색] 정답 = 마지막 지원 공고 · 후보 = 공고 풀 전체(이미 지원한 것 제외) · 무작위 recall@10 = %.4f" % (10 / len(pool)))
evaluate(O, "Offer(A) 프로필만")
evaluate(S_a, "Seek(a) 지원공고 평균(passage)")
evaluate(S_b, "Seek(b) 지원공고 평균(query)")
for w in (0.5, 0.3):
    C = w * O + (1 - w) * S_a; C /= np.linalg.norm(C, axis=1, keepdims=True); evaluate(C, f"{w:.1f} Offer + {1-w:.1f} Seek(a)")

# ---------- 5. 코드북 ----------
CB = {}
def rq(V, K_, L_, seed=SEED, name=None):
    codes = np.zeros((len(V), L_), int); R = V.copy(); cbs = []
    for l in range(L_):
        km = KMeans(n_clusters=K_, n_init=4, random_state=seed + l).fit(R)
        codes[:, l] = km.labels_; R = R - km.cluster_centers_[km.labels_]; cbs.append(km.cluster_centers_)
    if name: CB[name] = cbs
    return codes
def assign(V, cbs):
    codes = np.zeros((len(V), len(cbs)), int); R = V.copy()
    for l, c in enumerate(cbs):
        d = ((R[:, None, :] - c[None, :, :]) ** 2).sum(2); codes[:, l] = d.argmin(1); R = R - c[codes[:, l]]
    return codes
def prefix_stats(codesJ, codesQ, name):
    out = []
    for L_ in (1, 2):
        keyJ = [tuple(c[:L_]) for c in codesJ]; keyQ = [tuple(c[:L_]) for c in codesQ]
        hit = np.mean([keyQ[i] == keyJ[tgt[i]] for i in range(len(users))])
        sizes = pd.Series(keyJ).value_counts(normalize=True)
        rand = np.mean([sizes.get(keyQ[i], 0.0) for i in range(len(users))])
        out.append(f"{L_}계층 일치 {hit:.3f} (무작위 {rand:.3f}, ×{hit/max(rand,1e-9):.1f})")
    P(f"  {name:<24} " + " · ".join(out))
def shared(tag, J, O_, S_):
    V = np.vstack([J, O_, S_]); src = np.array(["job"] * len(J) + ["offer"] * len(O_) + ["seek"] * len(S_))
    codes = rq(V, KC, LV)
    cJ, cO, cS = codes[:len(J)], codes[len(J):len(J) + len(O_)], codes[len(J) + len(O_):]
    ct = pd.crosstab(codes[:, 0], src); pur = ct.div(ct.sum(1), axis=0).max(1).mean(); mixed = int((ct > 0).sum(1).eq(3).sum())
    P("")
    P(f"[코드북 {tag}] 1계층 16칸 중 세 출처가 모두 든 칸 {mixed} · 칸별 최다 출처 비율 평균 {pur:.2f} (1.0 이면 문체로 완전 분리)")
    P(f"  출처별 1계층 코드 종류 — 공고 {len(set(cJ[:,0]))} · Offer {len(set(cO[:,0]))} · Seek {len(set(cS[:,0]))}")
    prefix_stats(cJ, cS, "Seek-SID → 다음 공고"); prefix_stats(cJ, cO, "Offer-SID → 다음 공고")
    return cJ, cS
shared("(a) 세 출처 합쳐 학습 · 전부 passage", J_p, O, S_a)
shared("(b) Seek 만 query 접두사", J_p, O, S_b)
shared("(c) 출처별 평균 제거 후 합쳐 학습", J_p - J_p.mean(0), O - O.mean(0), S_a - S_a.mean(0))
muJ = J_p.mean(0)
cJ = rq(J_p - muJ, KC, LV, name="job"); cS = assign(S_a - muJ, CB["job"]); cO = assign(O - muJ, CB["job"])
P("")
P(f"[코드북 (d) 공고(상대 Offer)로만 학습 · Seek·Offer 는 배정만] Seek 가 쓰는 1계층 코드 {len(set(cS[:,0]))} · Offer {len(set(cO[:,0]))} / 16")
prefix_stats(cJ, cS, "Seek-SID → 다음 공고"); prefix_stats(cJ, cO, "Offer-SID → 다음 공고")

# ---------- 6. (e) 동네 한정 검색 ----------
S = S_a @ J_p.T
for i in range(len(users)): S[i, prev_idx[i]] = -9
full10 = ((S > S[np.arange(len(users)), tgt][:, None]).sum(1) < 10).mean()
key2J = np.array([c[0] * 100 + c[1] for c in cJ]); key2S = np.array([c[0] * 100 + c[1] for c in cS])
same = (key2S[:, None] == key2J[None, :]); S2 = np.where(same, S, -9)
rank = (S2 > S2[np.arange(len(users)), tgt][:, None]).sum(1); inb = same[np.arange(len(users)), tgt]
P(f"  (e) 후보를 Seek 2계층 동네(코드북 d)로 한정: 정답이 동네 안 {inb.mean():.3f} · 동네 크기 중앙 {int(np.median(same.sum(1)))} / {len(pool)} · 동네 안 recall@10 {((rank < 10) & inb).mean():.3f} (전수 검색 {full10:.3f})")
P("")
P(f"총 {time.time()-t0:.0f}초")
with open(os.path.join(OUT, "kaggle_offer_seek_report.txt"), "w", encoding="utf-8") as f: f.write("\n".join(rep))
