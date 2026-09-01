"""상호작용 시뮬레이터 — 채용담당자가 후보 프로필을 열람·컨택한 로그를 만든다.

LLM 에게 한 건씩 판단시키지 않는다. **명시적 수식**으로 만든다.
수식이라야 재현 가능하고, 남이 따져볼 수 있고, 왜 그렇게 나왔는지 답할 수 있다.

점수 (사전등록 docs/PREREG_생성데이터셋.md §3)
--------------------------------------------
    s(r, i) = <q_r, v_i~> / T  +  gamma * log(1 + pop_i)  +  delta * recency_i  +  Gumbel

    q_r    담당자 r 의 요구 프로필 — 목표 직업의 기준 벡터 + 개인 편차
    v_i~   후보 i 의 스킬 벡터를 **담당자가 보는 방식으로 가린 것** (아래 참고)
    pop_i  그 시점까지 누적 컨택 수 → 인기 쏠림(부익부)을 만든다
    recency_i  최근 등록일수록 눈에 띈다
    Gumbel  Gumbel-top-k 로 비복원 표본을 뽑기 위한 잡음

⚠️ 순환을 어떻게 줄였나
----------------------
그래프(`skill_vector`)도 스킬 벡터에서 나오므로, 담당자가 **같은 벡터로 같은 방식으로**
점수를 매기면 그래프가 정답지를 그대로 보는 셈이 된다. 그래서 담당자는 전체 77차원이
아니라 **목표 직업의 요구 차원 m 개만** 보고, 거기에 관측 잡음을 얹은 `v_i~` 로 점수를 낸다.
그래프는 전체 벡터를 쓴다. 즉 그래프가 담당자의 점수 함수를 복사하지 않는다.

그래도 결합이 완전히 사라지지는 않는다. 실제 채용도 스킬로 고르고 스킬은 관측 가능하니
어느 정도 결합은 현실 자체다. **남은 결합의 크기는 G1(차수 보존 무작위화 대조)이 잰다.**
G1 에서 가짜 그래프로도 이득이 남으면 이 데이터는 폐기한다.

시간
----
프로필마다 등록 시각이 있고 **등록 전에는 컨택될 수 없다.** 이것이 신규 가입자(콜드)를
자연스럽게 만든다 — Beauty 의 zero-shot 아이템에 해당한다.
"""
import argparse
import csv
import io
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data_gen", "out")

N_RECRUITER = 20000
N_INTERACT = 200000
T_DAYS = 730          # 2년
SIGMA_R = 10.0        # 담당자 요구 프로필의 개인 편차
OBS_NOISE = 14.0      # 담당자가 후보를 볼 때의 관측 잡음 (순환 완화의 핵심)
TOP_M = 20            # 담당자가 실제로 보는 요구 차원 수 (77 중)
GAMMA = 0.55          # 인기 쏠림
DELTA = 0.9           # 최신성
TEMP = 6.0            # 적합도 온도 (작을수록 적합도 지배)
CAND_POOL = 3000      # 한 담당자가 훑는 후보 풀 (전수 스캔은 비현실적)


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recruiters", type=int, default=N_RECRUITER)
    ap.add_argument("--interactions", type=int, default=N_INTERACT)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--obs-noise", type=float, default=OBS_NOISE)
    ap.add_argument("--cand-pool", type=int, default=CAND_POOL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(OUT, "interactions.csv"))
    args = ap.parse_args()
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rng = np.random.default_rng(args.seed)
    rows = list(csv.DictReader(io.open(os.path.join(OUT, "profiles.csv"), encoding="utf-8")))
    V = np.load(os.path.join(OUT, "profile_skill.npy")).astype(np.float32)
    n_p, dim = V.shape
    job = np.array([r["job_cd"] for r in rows])
    jobs = sorted(set(job))
    jidx = {c: k for k, c in enumerate(jobs)}
    job_i = np.array([jidx[j] for j in job])

    # 직업별 기준 벡터 = 그 직업 프로필들의 평균 (담당자의 "요구 프로필" 씨앗)
    base = np.zeros((len(jobs), dim), dtype=np.float32)
    for k in range(len(jobs)):
        base[k] = V[job_i == k].mean(0)

    # 프로필 등록 시각 — 균등하게 흩되 뒤로 갈수록 조금 더 많이 (플랫폼 성장)
    reg = np.sort(T_DAYS * rng.beta(1.25, 1.0, n_p)).astype(np.float32)
    order = rng.permutation(n_p)          # 등록 순서와 profile_id 를 분리
    reg_of = np.empty(n_p, dtype=np.float32)
    reg_of[order] = reg

    # 담당자 — 목표 직업은 그 직업의 후보 수에 비례해 뽑는다 (수요가 몰리는 직업이 있다)
    cnt = np.bincount(job_i, minlength=len(jobs)).astype(float)
    tgt = rng.choice(len(jobs), size=args.recruiters, p=cnt / cnt.sum())
    r_time = np.sort(T_DAYS * rng.uniform(0.0, 0.6, args.recruiters)).astype(np.float32)
    # 담당자는 한 시점에만 활동하지 않는다. 여러 달에 걸쳐 반복 채용한다.
    # 이게 없으면 담당자의 컨택이 전부 한 윈도우에 몰려 Temporal 트랙이 성립하지 않는다
    # (학습 이력이 있는 테스트 유저가 0명이 된다 — 실제로 겪은 실패).
    span = np.clip(rng.lognormal(5.75, 0.7, args.recruiters), 30, T_DAYS).astype(np.float32)
    span = np.minimum(span, T_DAYS - r_time)

    # 담당자당 컨택 수 — 로그정규, 평균이 목표 총량이 되도록
    k_r = np.clip(rng.lognormal(1.55, 0.75, args.recruiters).round(), 1, 60).astype(int)
    k_r = np.maximum(1, (k_r * (args.interactions / k_r.sum())).round().astype(int))

    pop = np.zeros(n_p, dtype=np.float32)
    out = []
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    for r in range(args.recruiters):
        t = r_time[r]
        t_end = t + span[r]
        # 담당자는 활동 기간 내내 후보를 본다. 자격은 **각 컨택 시각** 기준이므로
        # 풀은 기간 종료 시점까지 등록된 프로필로 잡고, 뒤에서 시각을 맞춰준다.
        # (풀을 시작 시점으로 잡으면 늦게 등록한 프로필이 영영 후보에 못 든다 — 실제로 겪은 버그)
        live = np.flatnonzero(reg_of <= t_end)
        if len(live) < 50:
            continue
        if len(live) > args.cand_pool:
            live = rng.choice(live, args.cand_pool, replace=False)

        # 담당자의 요구 프로필: 목표 직업 기준 + 개인 편차
        q = base[tgt[r]] + rng.normal(0, SIGMA_R, dim).astype(np.float32)
        # 요구 차원 m 개만 본다 — 담당자는 전체 스킬을 다 보지 않는다
        keep = np.argsort(-q)[:TOP_M]
        mask = np.zeros(dim, dtype=np.float32)
        mask[keep] = 1.0
        qm = q * mask
        qm /= np.linalg.norm(qm) + 1e-9

        # 후보를 **관측 잡음을 통해** 본다 (그래프가 쓰는 원본 벡터와 다르게 만든다)
        Vi = V[live] * mask + rng.normal(0, args.obs_noise, (len(live), dim)).astype(np.float32) * mask
        Vi /= np.linalg.norm(Vi, axis=1, keepdims=True) + 1e-9
        fit = Vi @ qm

        age = np.clip(t_end - reg_of[live], 0, None)
        recency = np.exp(-age / 180.0)
        s = fit / (args.temp / 100.0) + args.gamma * np.log1p(pop[live]) + args.delta * recency
        s = s + rng.gumbel(0, 1, len(live))          # Gumbel-top-k = 비복원 표본

        k = min(k_r[r], len(live))
        pick = live[np.argpartition(-s, k - 1)[:k]]
        pick = pick[np.argsort(-s[np.searchsorted(live, pick)])] if False else pick
        rng.shuffle(pick)                            # 컨택 순서는 점수 순이 아니다
        pop[pick] += 1
        # 컨택 시각을 활동 기간에 흩는다 → 한 담당자가 여러 윈도우에 걸친다
        ts = np.sort(t + span[r] * rng.random(len(pick)))
        rec = []
        for p, tt in zip(pick, ts):
            tt = max(float(tt), float(reg_of[p]) + 0.5)   # 등록 전에는 컨택될 수 없다
            if tt <= t_end:
                rec.append((tt, int(p)))
        rec.sort()
        for pos, (tt, p) in enumerate(rec):
            out.append((r, pos, p, tt))

    with io.open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "position", "item_id", "timestamp"])
        w.writerows(out)
    np.save(os.path.join(OUT, "profile_regtime.npy"), reg_of)

    users = len({r for r, _, _, _ in out})
    per_u = np.bincount([r for r, _, _, _ in out])
    per_u = per_u[per_u > 0]
    print(f"상호작용 {len(out):,} · 담당자 {users:,} · 프로필 {n_p:,}")
    print(f"  담당자당 컨택  중앙 {np.median(per_u):.0f} · 평균 {per_u.mean():.1f} · 최대 {per_u.max()}")
    srt = np.sort(pop)[::-1]
    print(f"  프로필 컨택수  중앙 {np.median(pop):.0f} · 평균 {pop.mean():.1f} · 최대 {pop.max():.0f}")
    print(f"  컨택 0회(신규·미노출) {int((pop == 0).sum()):,} ({(pop == 0).mean():.1%})")
    print(f"  인기도 Gini {gini(pop):.4f}   (Amazon Beauty 참고: 상위 50%가 상호작용의 82.4%)")
    top50 = srt[:n_p // 2].sum() / max(srt.sum(), 1)
    print(f"  상위 50% 프로필이 가진 컨택 비중 {top50:.1%}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
