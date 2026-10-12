# -*- coding: utf-8 -*-
"""행사 한 바퀴 시뮬레이션 — 등록 → 체크인 → SID 발급 → 테이블토크 배정 → 피드백·명함 교환 → G-SID 재발급 → 커피챗 재배정 → 개인 추천.

목적은 성능 측정이 아니라 **파이프라인이 실제 운영 조건(노쇼·워크인·무응답·빈 Seek·워커 장애)에서 끝까지 도는지**와
어디서 깨지는지를 보는 것이다. 사람·Offer 텍스트는 연명부 243명(익명)이고, Seek 텍스트와 피드백은 합성이다(아래 주의).

단계 (회의록 12 타임라인 그대로)
  T0  등록          연명부에서 REG 명 표본. Offer = 프로필 문장, Seek = 합성 문장(20% 는 빈칸)
  T1  체크인        노쇼 NOSHOW 명 빠지고 워크인 WALKIN 명 추가(등록 안 한 사람 → 즉시 발급)
  T2  SID 발급      e5-small 임베딩 → Offer 코드북(K=8, L=3) 학습, Seek 는 같은 코드북에 배정 (캐글 검증 결과 d 방식)
  T3  테이블토크    Offer 끼리 코사인 → 테이블 크기 자동(5~6), 호스트(알럼나이) 라운드로빈, 같은 기수 ≤2, 담금질
  T4  피드백        "더 얘기하고 싶은 사람" 1~2명 (응답률 RESP), 자유 네트워킹 명함 교환 (인당 약 1.5회)
  T5  G-SID 재발급  상호작용 간선으로 Offer 를 직교 주입(β₀ 자동 조정: 동네 상한 넘으면 반으로)
  T6  커피챗        상호 점수 min(Seek_i·Offer_j, Seek_j·Offer_i), 빈 Seek 는 Offer+교환상대 평균으로 대체,
                    같은 테이블토크 쌍 재회 금지. 응답률 < 50% 면 텍스트 SID + 재회 금지만 (대체 경로)
  T7  개인 추천     정확 10 + 탐색 2, 이미 만난 사람 제외, 노출 상한 순차 갱신, 이유 칩
  지연 도착 LATE 명은 T6 뒤에 와서 가장 작은 테이블에 붙는다. 워커 장애 1회를 주입해 폴백(이전 버전 유지)이 작동하는지 본다.

주의  피드백은 "진짜 선호 = Seek·Offer 코사인 + 잡음" 으로 만들었으므로 커피챗 점수가 오르는 것은 당연하다.
      여기서 볼 것은 (1) 끝까지 도는가 (2) 제약이 지켜지는가 (3) 대체 경로가 작동하는가 (4) 노출 쏠림이 통제되는가 이다.

출력  D:/DSL/_event_data/event_loop.json (익명, 시각화용) · event_loop_report.txt · tables_r1.csv / tables_r2.csv (이름 포함, 로컬 전용)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, json, math, random, time
import numpy as np, pandas as pd, torch
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

D = r"D:/DSL/_event_data"
REG, NOSHOW, WALKIN, LATE = 75, 8, 4, 2
RESP = float(os.environ.get("RESP", 0.6))          # 피드백 응답률 (0.35 로 주면 대체 경로가 발동한다)
SEED, KC, LV, BETA0, NBHD_CAP = int(os.environ.get("SEED", 11)), 8, 3, 0.5, 0.35
sys.stdout.reconfigure(encoding="utf-8")
rng = random.Random(SEED); nrng = np.random.default_rng(SEED)
rep, log = [], []
P = lambda *a: (print(*a), rep.append(" ".join(str(x) for x in a)))
def EV(t, kind, msg): log.append(dict(t=t, kind=kind, msg=msg)); P(f"[{t}] {kind}: {msg}")
T = time.time()

# ---------- T0 등록 ----------
df = pd.read_csv(os.path.join(D, "profiles.csv")).fillna("")
names = pd.read_csv(os.path.join(D, "id_map.csv")).set_index("pid")["name"]
def grp(d):
    if d in ("응용통계학과", "통계데이터사이언스학과", "통계학과", "계량위험관리학과"): return "통계·데이터"
    if d in ("경제학과", "경영학과", "언더우드 경제학과", "경영학"): return "경제·경영"
    if any(k in d for k in ("공학", "컴퓨터", "전기전자", "IT", "소프트웨어", "기계", "화공", "건축", "도시", "산업", "전산", "수학", "물리", "화학", "생명", "의예", "바이오")): return "공학·자연"
    return "기타"
df["g"] = df["dept"].map(grp)
TOPIC = {"통계·데이터": ["추천시스템", "실험설계·인과추론", "시계열 예측", "데이터 분석 커리어"], "경제·경영": ["핀테크·금융 데이터", "마케팅 분석", "컨설팅·전략", "창업"],
         "공학·자연": ["MLOps·서빙", "컴퓨터비전", "LLM 응용", "대학원 진학"], "기타": ["데이터 저널리즘", "공공 데이터·정책", "UX 리서치", "해외 커리어"]}
def seek_text(r):
    if rng.random() < 0.2: return ""                                   # 빈 Seek
    g = r["g"] if rng.random() < 0.6 else rng.choice(list(TOPIC))
    who = "현직 선배" if r["role"] == "active" else (rng.choice(["재학생 후배", "같은 분야 동문"]))
    return f"만나고 싶은 사람: {who} 중 {g} 분야에서 {rng.choice(TOPIC[g])} 이야기를 나눌 수 있는 사람."
N_ACT = int(os.environ.get("N_ACT", 30))                                 # 등록자 중 활동 기수 수 (연명부는 알럼나이가 많아 층화 표본)
act = df[df.role == "active"].sample(N_ACT, random_state=SEED); alm = df[df.role == "alumni"].sample(REG + WALKIN + LATE - N_ACT, random_state=SEED)
pool = pd.concat([act, alm]).sample(frac=1, random_state=SEED).reset_index(drop=True)
pool["seek"] = pool.apply(seek_text, axis=1)
reg = pool.iloc[:REG].copy(); walk = pool.iloc[REG:REG + WALKIN].copy(); late = pool.iloc[REG + WALKIN:].copy()
EV("T0", "등록", f"사전 등록 {len(reg)}명 (alumni {(reg.role=='alumni').sum()} · active {(reg.role=='active').sum()}) · Seek 빈칸 {(reg.seek=='').sum()}명")

# ---------- T1 체크인 ----------
noshow = reg.sample(NOSHOW, random_state=SEED)
att = pd.concat([reg.drop(noshow.index), walk]).reset_index(drop=True)
att["src"] = ["reg"] * (REG - NOSHOW) + ["walkin"] * WALKIN
EV("T1", "체크인", f"노쇼 {NOSHOW}명 제외, 워크인 {WALKIN}명 추가 → 참석 {len(att)}명. 이후 모든 계산은 체크인 명단만 쓴다")

# ---------- T2 SID 발급 ----------
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = SentenceTransformer("intfloat/multilingual-e5-small", device=dev)
enc = lambda ts: np.asarray(m.encode(["passage: " + t for t in ts], batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
def issue(att):
    O = enc(att["text"].tolist()); S = enc([t if t else "" for t in att["seek"]])
    blank = (att["seek"] == "").values; S[blank] = O[blank]           # 빈 Seek 1차 대체 = 자기 Offer
    mu = O.mean(0); Oc, Sc = O - mu, S - mu
    codes, R, cbs = np.zeros((len(O), LV), int), Oc.copy(), []
    for l in range(LV):
        km = KMeans(n_clusters=KC, n_init=10, random_state=SEED + l).fit(R); codes[:, l] = km.labels_; R = R - km.cluster_centers_[km.labels_]; cbs.append(km.cluster_centers_)
    def assign(V):
        c, R = np.zeros((len(V), LV), int), V.copy()
        for l, cb in enumerate(cbs): c[:, l] = ((R[:, None, :] - cb[None]) ** 2).sum(2).argmin(1); R = R - cb[c[:, l]]
        return c
    return O, S, blank, mu, codes, assign(Sc), cbs, assign
t = time.time(); O, S, blank, mu, sidO, sidS, cbs, assign = issue(att)
EV("T2", "SID 발급", f"{len(att)}명 Offer·Seek 임베딩 + 코드북 K={KC} L={LV} · {time.time()-t:.1f}초 · Offer 고유주소 {len(set(map(tuple,sidO)))} · 빈 Seek {blank.sum()}명은 Offer 로 대체")

# ---------- 배정 엔진 ----------
def table_sizes(n, lo=5, hi=6):
    k = math.ceil(n / hi); sizes = [hi] * k; over = hi * k - n
    for i in range(over): sizes[i] -= 1
    assert all(lo <= s <= hi for s in sizes) or n < lo * k, sizes
    return sizes
def assign_tables(A, roles, cohorts, forbid=None, iters=30000, tag=""):
    """A: 쌍 점수 (대칭). 목적 = 개인 만족(테이블 동료 평균)의 평균 + 2×하위 10% 평균 − 제약 페널티."""
    n = len(A); sizes = table_sizes(n); k = len(sizes)
    hosts = [i for i in range(n) if roles[i] == "alumni"]; rng.shuffle(hosts)
    others = [i for i in range(n) if roles[i] != "alumni"]; rng.shuffle(others)
    assign = np.full(n, -1); cap = list(sizes)
    for t_, h in zip([i % k for i in range(len(hosts))], hosts): assign[h] = t_; cap[t_] -= 1
    for o in others:                                                     # 남은 자리는 한 자리씩, 지금까지 배정된 사람과의 점수로
        best, bs = None, -1e9
        for t_ in range(k):
            if cap[t_] <= 0: continue
            mem = np.where(assign == t_)[0]; s = A[o, mem].mean() if len(mem) else 0
            if bs < s: bs, best = s, t_
        assign[o] = best; cap[best] -= 1
    forbid = forbid if forbid is not None else np.zeros((n, n), bool)
    def objective(a):
        sat = np.zeros(n); pen = 0.0
        for t_ in range(k):
            mem = np.where(a == t_)[0]
            sub = A[np.ix_(mem, mem)]; sat[mem] = sub.sum(1) / max(1, len(mem) - 1)
            pen += forbid[np.ix_(mem, mem)].sum() / 2 * 5.0
            cc = pd.Series(cohorts[mem]).value_counts(); pen += ((cc[cc > 2] - 2).sum()) * 0.5
        srt = np.sort(sat); return sat.mean() + 2 * srt[:max(1, n // 10)].mean() - pen, sat
    cur, _ = objective(assign); best_a, best_v = assign.copy(), cur
    for it in range(iters):
        Tm = 0.3 * (0.01 / 0.3) ** (it / iters)
        i, j = rng.randrange(n), rng.randrange(n)
        if assign[i] == assign[j] or roles[i] != roles[j]: continue
        assign[i], assign[j] = assign[j], assign[i]; v, _ = objective(assign)
        if v > cur or rng.random() < math.exp((v - cur) / Tm): cur = v
        else: assign[i], assign[j] = assign[j], assign[i]
        if cur > best_v: best_v, best_a = cur, assign.copy()
    _, sat = objective(best_a)
    return best_a, sat, sizes
def describe(a, sat, tag):
    k = a.max() + 1
    sizes = [int((a == t_).sum()) for t_ in range(k)]; hosts = [int(((a == t_) & (att.role.values == "alumni")).sum()) for t_ in range(k)]
    coh_viol = sum(int((pd.Series(att.cohort.values[a == t_]).value_counts() > 2).sum()) for t_ in range(k))
    P(f"  {tag}: 테이블 {k}개 크기 {sorted(sizes)} · 호스트/테이블 {min(hosts)}~{max(hosts)} · 같은 기수 3명+ 위반 {coh_viol} · 만족 평균 {sat.mean():.3f} · 하위10% {np.sort(sat)[:max(1,len(sat)//10)].mean():.3f} · 최저 {sat.min():.3f}")
    return dict(sizes=sizes, hosts=hosts, coh_viol=coh_viol, mean=float(sat.mean()), bottom10=float(np.sort(sat)[:max(1,len(sat)//10)].mean()), min=float(sat.min()))

# ---------- T3 테이블토크 ----------
Oc = O - mu; On = Oc / np.linalg.norm(Oc, axis=1, keepdims=True)
A1 = On @ On.T; np.fill_diagonal(A1, 0)
t = time.time(); a1, sat1, sizes1 = assign_tables(A1, att.role.values, att.cohort.values, tag="r1")
m1 = describe(a1, sat1, "테이블토크(Offer 유사도)"); EV("T3", "테이블토크 배정", f"{time.time()-t:.1f}초 · 참가자 화면에는 테이블 번호 + 테이블 공통 동네 라벨")
pairs1 = np.zeros((len(att), len(att)), bool)
for t_ in range(a1.max() + 1):
    mem = np.where(a1 == t_)[0]; pairs1[np.ix_(mem, mem)] = True
np.fill_diagonal(pairs1, False)

# ---------- T4 피드백 · 명함 교환 (합성) ----------
Sc = S - mu; Sn = Sc / np.linalg.norm(Sc, axis=1, keepdims=True)
TRUE = Sn @ On.T                                                     # "진짜 선호" (합성): 내 Seek 가 상대 Offer 를 얼마나 원하나
E = np.zeros((len(att), len(att)))                                   # 상호작용 간선 (가중치)
responded = nrng.random(len(att)) < RESP
picks = 0
for i in range(len(att)):
    if not responded[i]: continue
    mates = np.where(pairs1[i])[0]
    pref = TRUE[i, mates] + nrng.normal(0, 0.05, len(mates))
    for j in mates[np.argsort(-pref)[:rng.choice([1, 2])]]: E[i, j] += 1.0; picks += 1
exch = 0
for i in range(len(att)):
    for _ in range(rng.choice([0, 1, 2, 3])):
        cand = np.argsort(-(TRUE[i] + nrng.normal(0, 0.08, len(att))))[:8]; j = rng.choice(list(cand))
        if j != i and not pairs1[i, j]: E[i, j] += 1.0; E[j, i] += 1.0; exch += 1
EV("T4", "피드백·명함 교환", f"피드백 응답 {responded.sum()}/{len(att)} ({responded.mean():.0%}) · 선택 {picks}건 · 명함 교환 {exch}건 · 간선 있는 사람 {(E.sum(1)>0).sum()}명")

# 워커 장애 주입: 재발급 1차 시도 실패 → 이전 버전(sidO) 유지, 2차 시도 성공
EV("T5", "워커 장애 주입", "재발급 1차 시도 실패(타임아웃 가정) → heartbeat 끊김 감지 → 이전 SID 버전 v1 유지, 배정 발송 보류. 2분 뒤 재시도")

# ---------- T5 G-SID 재발급 ----------
def gsid(Oc, E, beta):
    Z = Oc.copy(); W = E + E.T
    for i in range(len(Oc)):
        if W[i].sum() == 0: continue
        mvec = (W[i][:, None] * Oc).sum(0) / W[i].sum(); xh = Oc[i] / (np.linalg.norm(Oc[i]) + 1e-12)
        r = mvec - (mvec @ xh) * xh; nr = np.linalg.norm(r)
        if nr > 1e-12: Z[i] = Oc[i] + beta * np.linalg.norm(Oc[i]) * r / nr
    return Z
beta = BETA0
for attempt in range(4):
    Z = gsid(Oc, E, beta); sidG = assign(Z)
    top = pd.Series([tuple(c[:2]) for c in sidG]).value_counts(normalize=True).iloc[0]
    if top <= NBHD_CAP: break
    EV("T5", "β₀ 자동 조정", f"2계층 최대 동네 {top:.0%} > 상한 {NBHD_CAP:.0%} → β₀ {beta} → {beta/2}"); beta /= 2
moved = int((sidG[:, 0] != sidO[:, 0]).sum())
EV("T5", "G-SID 재발급", f"β₀ {beta} · 1계층 코드가 바뀐 사람 {moved}/{len(att)} · 간선 없는 사람 {(E.sum(1)==0).sum()}명은 텍스트 주소 유지")

# ---------- T6 커피챗 ----------
Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
S2 = Sn.copy()
for i in np.where(blank)[0]:                                          # 빈 Seek 2차 대체: 교환 상대 Offer 평균
    nb = np.where(E[i] + E[:, i] > 0)[0]
    if len(nb): v = On[nb].mean(0) + On[i]; S2[i] = v / np.linalg.norm(v)
fallback = responded.mean() < 0.5
if fallback:
    A2 = A1.copy(); EV("T6", "대체 경로", f"응답률 {responded.mean():.0%} < 50% → 텍스트 유사도 + 재회 금지만으로 커피챗 배정")
else:
    F = S2 @ Zn.T; A2 = np.minimum(F, F.T); np.fill_diagonal(A2, 0)
t = time.time(); a2, sat2, sizes2 = assign_tables(A2, att.role.values, att.cohort.values, forbid=pairs1, tag="r2")
repeat = int(sum(pairs1[np.ix_(np.where(a2 == t_)[0], np.where(a2 == t_)[0])].sum() // 2 for t_ in range(a2.max() + 1)))
m2 = describe(a2, sat2, "커피챗(상호 min 점수)"); EV("T6", "커피챗 배정", f"{time.time()-t:.1f}초 · 테이블토크 쌍 재회 {repeat}건 · 운영 콘솔 확인 후 발송(자동 발송 아님)")
# 지연 도착
late_rows = []
for _, r in late.iterrows():
    o = enc([r.text])[0]; oc = o - mu; c = assign(oc[None])[0]
    tsz = pd.Series(a2).value_counts(); t_ = int(tsz.idxmin()); s_ = (On @ (oc / np.linalg.norm(oc)))[a2 == t_].mean()
    late_rows.append(dict(pid=r.pid, table=t_ + 1, sid=[int(x) for x in c])); EV("T6", "지연 도착", f"{r.pid} 즉시 발급(SID {list(map(int,c))}) → 가장 작은 테이블 {t_+1} 에 추가 (동료 평균 유사도 {s_:.2f})")
# 비교: 같은 제약에서 텍스트만으로 배정했다면
a2t, sat2t, _ = assign_tables(A1, att.role.values, att.cohort.values, forbid=pairs1, iters=15000, tag="r2text")
true_sat = lambda a: np.array([TRUE[i, np.where(a == a[i])[0]].mean() for i in range(len(att))])
P(f"  [참고] 커피챗 배정을 '진짜 선호'(합성)로 채점: 상호작용 G-SID 배정 {true_sat(a2).mean():.3f} (하위10% {np.sort(true_sat(a2))[:7].mean():.3f}) vs 텍스트만 {true_sat(a2t).mean():.3f} (하위10% {np.sort(true_sat(a2t))[:7].mean():.3f}) vs 테이블토크 {true_sat(a1).mean():.3f}")

# ---------- T7 개인 추천 ----------
met = pairs1.copy()
for t_ in range(a2.max() + 1):
    mem = np.where(a2 == t_)[0]; met[np.ix_(mem, mem)] = True
met |= (E + E.T) > 0; np.fill_diagonal(met, True)
Rsc = np.minimum(S2 @ Zn.T, (S2 @ Zn.T).T)
exposure = np.zeros(len(att), int); CAP = 14; recs = {}
order = list(range(len(att))); rng.shuffle(order)
for i in order:
    cand = [j for j in np.argsort(-Rsc[i]) if not met[i, j] and exposure[j] < CAP]
    exact = cand[:10]
    rest = [j for j in cand[10:]]; explore = sorted(rest, key=lambda j: exposure[j])[:2]
    lst = exact + explore
    for j in lst: exposure[j] += 1
    def why(i, j):
        w = []
        if sidG[i][0] == sidG[j][0]: w.append("같은 궤도")
        if att.g[i] == att.g[j]: w.append("같은 분야")
        elif att.g[i] != att.g[j]: w.append("다른 분야")
        if att.role[i] != att.role[j]: w.append("기수 교차")
        return w
    recs[int(i)] = [dict(j=int(j), s=round(float(Rsc[i, j]), 3), why=why(i, j), explore=bool(j in explore)) for j in lst]
g = np.sort(exposure); gini = (np.cumsum(g).sum() / (g.sum() * len(g)) if g.sum() else 0); gini = 1 + 1 / len(g) - 2 * gini
EV("T7", "개인 추천", f"인당 12칸(정확 10 + 탐색 2) · 노출 상한 {CAP} · 노출 최소 {exposure.min()} 최대 {exposure.max()} · Gini {gini:.3f} · 한 번도 안 뜬 사람 {(exposure==0).sum()}")
P(f"\n총 {time.time()-T:.0f}초 · 참석 {len(att)} · 테이블토크 {a1.max()+1}개 · 커피챗 {a2.max()+1}개")

# ---------- 저장 ----------
people = [dict(pid=r.pid, g=r.g, dept=r.dept, cohort=int(r.cohort), role=r.role, src=r.src, blank=bool(blank[i]), responded=bool(responded[i]),
               sidO=[int(x) for x in sidO[i]], sidS=[int(x) for x in sidS[i]], sidG=[int(x) for x in sidG[i]], t1=int(a1[i]) + 1, t2=int(a2[i]) + 1,
               sat1=round(float(sat1[i]), 3), sat2=round(float(sat2[i]), 3), exposure=int(exposure[i]), deg=int((E[i] + E[:, i] > 0).sum()))
          for i, r in att.iterrows()]
edges = [dict(a=int(i), b=int(j), w=float(E[i, j] + E[j, i])) for i in range(len(att)) for j in range(i + 1, len(att)) if E[i, j] + E[j, i] > 0]
out = dict(people=people, edges=edges, late=late_rows, recs=recs, log=log, metrics=dict(r1=m1, r2=m2, resp=float(responded.mean()), fallback=bool(fallback), beta=float(beta), repeat=int(repeat),
           true=dict(r1=float(true_sat(a1).mean()), r2=float(true_sat(a2).mean()), r2text=float(true_sat(a2t).mean())), gini=float(gini), n=len(att), reg=REG, noshow=NOSHOW, walkin=WALKIN))
json.dump(out, open(os.path.join(D, "event_loop.json"), "w", encoding="utf-8"), ensure_ascii=False)
for rnd, a in (("r1", a1), ("r2", a2)):
    tb = att.assign(table=a + 1, name=att.pid.map(names)).sort_values(["table", "role"])[["table", "name", "role", "cohort", "dept"]]
    tb.to_csv(os.path.join(D, f"tables_{rnd}.csv"), index=False, encoding="utf-8-sig")
open(os.path.join(D, "event_loop_report.txt"), "w", encoding="utf-8").write("\n".join(rep))
