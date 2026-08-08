"""segment_eval.evaluate 의 지표 계산 검증 (손으로 답을 아는 케이스)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from segment_eval import evaluate, gini  # noqa: E402

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
print("\n합성 데이터 검증:", "통과" if ok else "실패")
sys.exit(0 if ok else 1)
