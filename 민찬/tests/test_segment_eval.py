"""segment_eval 의 지표 계산 검증 (손으로 답을 아는 케이스)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from segment_eval import (  # noqa: E402
    bootstrap_ci,
    compare,
    evaluate,
    gini,
    paired_bootstrap,
)

# 아이템 4개, SID 자릿수 2
sid_map = np.array([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=np.int64)
segments = {0: "HEAD", 1: "HEAD", 2: "TAIL", 3: "TAIL"}

# 유저 A: 정답 0(HEAD). 1순위로 맞힘        → HR=1, NDCG=1/log2(2)=1.0
# 유저 B: 정답 2(TAIL). 2순위로 맞힘        → HR=1, NDCG=1/log2(3)=0.6309
# 유저 C: 정답 3(TAIL). 못 맞힘 + 환각 1건  → HR=0
predictions = {
    10: np.array([[1, 1], [1, 2]]),          # 아이템 0, 1
    11: np.array([[1, 2], [2, 1]]),          # 아이템 1, 2
    12: np.array([[9, 9], [1, 1]]),          # 환각, 아이템 0
}
targets = {10: 0, 11: 2, 12: 3}

result = evaluate(predictions, targets, sid_map, segments, ks=[1, 2])
rows = {r["segment"]: r for r in result["rows"]}
d = result["diagnostics"]

def check(label, got, want, tol=1e-4):
    ok = abs(got - want) < tol
    print(f"{'OK ' if ok else 'FAIL'} {label}: got {got:.4f}, want {want:.4f}")
    return ok

ok = True
ok &= check("HEAD n_users", rows["HEAD"]["n_users"], 1)
ok &= check("TAIL n_users", rows["TAIL"]["n_users"], 2)
ok &= check("HEAD HR@1", rows["HEAD"]["HR@1"], 1.0)
ok &= check("HEAD NDCG@2", rows["HEAD"]["NDCG@2"], 1.0)
ok &= check("TAIL HR@1", rows["TAIL"]["HR@1"], 0.0)        # B는 2순위라 @1에서는 미스
ok &= check("TAIL HR@2", rows["TAIL"]["HR@2"], 0.5)        # B만 맞힘 / 2명
ok &= check("TAIL NDCG@2", rows["TAIL"]["NDCG@2"], (1/np.log2(3))/2)
ok &= check("ALL HR@2", rows["ALL"]["HR@2"], 2/3)

# 노출: 아이템0 2회(A 1순위, C 2순위), 아이템1 2회, 아이템2 1회, 아이템3 0회
ok &= check("HEAD 노출수", rows["HEAD"]["exposure"], 4)
ok &= check("TAIL 노출수", rows["TAIL"]["exposure"], 1)
ok &= check("TAIL coverage", rows["TAIL"]["coverage"], 0.5)  # 아이템2만 추천됨
ok &= check("환각률", d["hallucination_rate"], 1/6)
ok &= check("SID 충돌률", d["sid_collision_rate"], 0.5)      # 앞 1자리만 보면 2종류/4개

# Gini 경계 확인
ok &= check("Gini 균등", gini(np.array([5, 5, 5, 5])), 0.0)

# ── 노출 공정성 ──
# 노출 5회 중 HEAD 4회 / TAIL 1회. 아이템은 2개씩이라 각 구간의 아이템 점유율은 0.5.
ok &= check("HEAD 노출점유", rows["HEAD"]["exposure_share"], 4 / 5)
ok &= check("HEAD 노출배율", rows["HEAD"]["exposure_lift"], (4 / 5) / 0.5)   # 제 몫의 1.6배
ok &= check("TAIL 노출배율", rows["TAIL"]["exposure_lift"], (1 / 5) / 0.5)   # 제 몫의 0.4배
ok &= check("HEAD 개당노출", rows["HEAD"]["exposure_per_item"], 2.0)
ok &= check("HEAD:TAIL 노출비", d["head_tail_exposure_ratio"], 2.0 / 0.5)
ok &= check("ALL 배율은 항상 1", rows["ALL"]["exposure_lift"], 1.0)

# ── 신뢰구간 ──
# 전부 같은 값이면 재표본해도 그 값뿐이라 구간 폭이 0이다.
low, high = bootstrap_ci(np.ones(50))
ok &= check("모두 1이면 신뢰구간 [1,1] 하한", low, 1.0)
ok &= check("모두 1이면 신뢰구간 [1,1] 상한", high, 1.0)

low, high = bootstrap_ci(np.array([0.0] * 50 + [1.0] * 50))
ok &= bool(low < 0.5 < high)
print(f"OK   신뢰구간이 평균을 감싼다: 0/1 반반 100명 → [{low:.3f}, {high:.3f}]")

# 표본이 작을수록 구간이 넓어야 한다 — TAIL/COLD 를 조심해야 하는 이유
narrow = bootstrap_ci(np.array([0.0, 1.0] * 500))
wide = bootstrap_ci(np.array([0.0, 1.0] * 5))
ok &= bool((wide[1] - wide[0]) > (narrow[1] - narrow[0]))
print(f"OK   표본이 작으면 신뢰구간이 넓다: n=10 폭 {wide[1]-wide[0]:.3f} "
      f"> n=1000 폭 {narrow[1]-narrow[0]:.3f}")

# ── 짝지은 검정 ──
# 모든 유저에서 b가 a보다 정확히 0.2 높으면 차이는 0.2, 흔들림이 없으니 p가 최소여야 한다.
same = np.linspace(0, 1, 200)
paired = paired_bootstrap(same, same + 0.2)
ok &= check("짝지은 차이", paired["delta"], 0.2)
ok &= bool(paired["p_value"] <= 0.001)
print(f"OK   차이가 일정하면 p 가 바닥 ({paired['p_value']:.4f})")

# 차이가 없으면 유의하지 않아야 한다
rng = np.random.default_rng(0)
a_rand = rng.random(500)
null = paired_bootstrap(a_rand, a_rand.copy())
ok &= check("같은 값끼리는 차이 0", null["delta"], 0.0)
ok &= check("p 는 1을 넘지 않는다", null["p_value"], 1.0)
print(f"OK   차이가 없으면 유의하지 않다 (p={null['p_value']:.3f})")

# 진짜로 미세한 차이만 있을 때도 유의하다고 말하면 안 된다
noise = rng.normal(0, 1, 400)
subtle = paired_bootstrap(noise, noise + rng.normal(0, 1, 400) * 0.01)
ok &= bool(subtle["p_value"] > 0.05)
print(f"OK   잡음뿐이면 유의하지 않다 (p={subtle['p_value']:.3f})")

# ── 두 실행 비교 ──
# 같은 예측을 스스로와 비교하면 모든 구간에서 차이가 0이어야 한다.
result_b = evaluate(predictions, targets, sid_map, segments, ks=[1, 2])
cmp = compare(result, result_b)
ok &= check("자기 자신과 비교하면 공통 유저 = 전체", cmp["n_common_users"], 3)
zero = all(r["HR@2"]["delta"] == 0.0 for r in cmp["rows"])
ok &= bool(zero)
print(f"OK   자기 자신과 비교하면 모든 구간 차이가 0")

# 한쪽 유저를 빼면 교집합만 비교하고, 뺀 수를 보고해야 한다
trimmed = {
    "ks": result["ks"],
    "per_user": {
        "user_ids": result["per_user"]["user_ids"][:2],
        "segments": result["per_user"]["segments"][:2],
        "hit": {k: v[:2] for k, v in result["per_user"]["hit"].items()},
        "ndcg": {k: v[:2] for k, v in result["per_user"]["ndcg"].items()},
    },
}
cmp2 = compare(result, trimmed)
ok &= check("교집합만 비교한다", cmp2["n_common_users"], 2)
ok &= check("한쪽에만 있는 유저를 센다", cmp2["n_base_only"], 1)

print("\n합성 데이터 검증:", "통과" if ok else "실패")
sys.exit(0 if ok else 1)
