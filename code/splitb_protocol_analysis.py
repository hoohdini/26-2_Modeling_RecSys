"""Split B 프로토콜 비교 — preprocess.py 의 실제 로직을 그대로 재현해 비교한다.

stratify_cold() 임계값(0=unseen / 1~3=very_rare / 4~5=rare / 6+=normal)과
build_split_B_paper() 구조(누적 train, 직전 윈도우 val)를 동일하게 적용하고,
윈도우 등분 기준만 '달력 시간'(현재) vs '리뷰 건수'(제안)로 바꿔 비교한다.

실행: python code/splitb_protocol_analysis.py
"""
import gzip
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REVIEWS = r"D:\DSL\_teamrepo\Data\reviews_Beauty_5.json.gz"

rows = []
with gzip.open(REVIEWS, "rt", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        rows.append((d["asin"], d["reviewerID"], int(d["unixReviewTime"])))
rows.sort(key=lambda r: r[2])
N = len(rows)
T0, T1 = rows[0][2], rows[-1][2]
ALL_ITEMS = sorted({a for a, _, _ in rows})
fmt = lambda t: datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")

print(f"리뷰 {N:,}건 · 아이템 {len(ALL_ITEMS):,} · 기간 {fmt(T0)} ~ {fmt(T1)}\n")


def stratify_cold(train_rows):
    """preprocess.py 의 stratify_cold 와 동일한 임계값."""
    cnt = Counter(a for a, _, _ in train_rows)
    tiers = {}
    for a in ALL_ITEMS:
        c = cnt.get(a, 0)
        if c == 0:
            tiers[a] = "unseen"
        elif c <= 3:
            tiers[a] = "very_rare"
        elif c <= 5:
            tiers[a] = "rare"
        else:
            tiers[a] = "normal"
    return tiers


def windows_by_time(k=5):
    span = T1 - T0
    b = [T0 + span * i / k for i in range(k + 1)]
    W = [[r for r in rows if b[i] <= r[2] < b[i + 1]] for i in range(k - 1)]
    W.append([r for r in rows if b[k - 1] <= r[2] <= b[k]])
    return W, b


def windows_by_count(k=5):
    q = [int(N * i / k) for i in range(k + 1)]
    W = [rows[q[i]:q[i + 1]] for i in range(k)]
    b = [T0] + [rows[q[i]][2] for i in range(1, k)] + [T1]
    return W, b


ORDER = ["normal", "rare", "very_rare", "unseen"]


def report(title, W, bnd):
    print("=" * 78)
    print(f"### {title}")
    print(f"    윈도우 경계: {' / '.join(fmt(x) for x in bnd[1:-1])}")
    print(f"    윈도우별 리뷰 수: {' / '.join(f'{len(w):,}' for w in W)}")
    print()
    hdr = (f"    {'라벨':<5} {'train':>8} {'val':>8} {'test':>8} | "
           f"{'normal':>7} {'rare':>6} {'v_rare':>7} {'unseen':>7} | "
           f"{'test아이템중 normal':>18}")
    print(hdr)
    print("    " + "-" * 92)
    for ti in range(2, len(W)):
        vi = ti - 1
        train = sum(W[:vi], [])
        val, test = W[vi], W[ti]
        tiers = stratify_cold(train)
        c = Counter(tiers.values())
        test_items = {a for a, _, _ in test}
        c_test = Counter(tiers[a] for a in test_items)
        flag = "OK" if c_test.get("normal", 0) >= 500 else "부족"
        print(f"    W{ti+1:<4} {len(train):>8,} {len(val):>8,} {len(test):>8,} | "
              f"{c.get('normal',0):>7,} {c.get('rare',0):>6,} "
              f"{c.get('very_rare',0):>7,} {c.get('unseen',0):>7,} | "
              f"{c_test.get('normal',0):>8,} ({flag})")
    print()


Wt, bt = windows_by_time(5)
report("현재 방식 — 전체 기간을 달력 기준 균등 5등분", Wt, bt)

Wc, bc = windows_by_count(5)
report("제안(P3) — 리뷰 건수 기준 균등 5등분 (구조는 동일)", Wc, bc)

print("=" * 78)
print("판단 기준: 콜드스타트 실험은 '콜드가 normal 대비 얼마나 나쁜가'를 재는 것이므로")
print("           비교 기준선이 되는 normal 아이템이 test 에 충분히 있어야 한다.")
print("           (참고: Split A(LOO) 는 head 7,838 / mid 8,328 / low 4,559 로 대조가 충분)")
