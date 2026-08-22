"""
Amazon 2014 Beauty 전처리 파이프라인 (수정판 — 원본은 preprocess.py)
0단계: 파싱 -> ID매핑 -> 유저별 시간순 정렬 -> 그래프 추출 -> 텍스트 구성 -> 분할 A/B -> 콜드 계층화

preprocess.py 대비 무엇이/왜 바뀌었는지는 preprocessing/전처리_수정문서.md 참고.

입력은 이 레포의 Data/ 에 있는 원본 파일만 사용한다 (이미 정제된 pkl은 참조하지 않음):
    Data/reviews_Beauty_5.json.gz
    Data/meta_Beauty.json.gz

Split B 는 docs/REQUEST_전처리_splitB.md 의 제안(리뷰 건수 분위수로 위치를 잡되,
실제 경계는 날짜 단위로 스냅)을 그대로 구현한다. 자세한 이유는 그 문서 참고.

실행:
    python preprocessing/repreprocess.py

출력 (preprocessing/output/, .gitignore 의 *.pkl 규칙에 걸려 커밋되지 않음):
    Beauty_split_A.pkl        Leave-One-Out 분할 (TIGER 논문 재현/sanity check 전용)
    Beauty_split_B.pkl        절대시간 분할, W3/W4/W5 (콜드스타트 공식 결과용)
    Beauty_related_separate.pkl   관계종류(also_bought/also_viewed/bought_together) 보존 그래프
"""
import gzip
import json
import ast
import pickle
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
OUT_DIR = ROOT / "preprocessing" / "output"
CATEGORY = "Beauty"

RELATION_TYPES = ("also_bought", "also_viewed", "bought_together")


# ── 0단계: 파싱 ──────────────────────────────────────────────────────────

def load_reviews(path):
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            rows.append({
                'reviewerID': d['reviewerID'],
                'asin': d['asin'],
                'unixReviewTime': d['unixReviewTime'],
                'orig_order': idx,  # 동일 타임스탬프 tie-break용 (날짜 단위라 39% 가 동점)
            })
    return rows


def load_meta(path):
    # meta_*.json.gz는 유효한 JSON이 아니라 파이썬 literal 형식 -> ast.literal_eval 필수
    meta = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            d = ast.literal_eval(line)
            meta[d['asin']] = {
                'title': d.get('title', ''),
                'brand': d.get('brand', ''),
                'description': d.get('description', ''),
                'categories': d.get('categories', []),
                'related': d.get('related', {}),
                'salesRank': d.get('salesRank', {}),
            }
    return meta


# ── ID 매핑 / 유저 시퀀스 ────────────────────────────────────────────────

def build_id_maps(reviews):
    users = sorted({r['reviewerID'] for r in reviews})
    asins = sorted({r['asin'] for r in reviews})
    uid = {u: k for k, u in enumerate(users)}
    iid = {a: k for k, a in enumerate(asins)}
    return uid, iid, users, asins


def build_user_sequences(reviews, iid):
    """유저 -> 시간순 item_id 리스트. 전체 기간을 관통하는 하나의 시퀀스."""
    user_rows = defaultdict(list)
    for r in reviews:
        user_rows[r['reviewerID']].append(r)

    user_seq = {}
    for u, rows in user_rows.items():
        rows.sort(key=lambda x: (x['unixReviewTime'], x['orig_order']))
        user_seq[u] = [iid[r['asin']] for r in rows]
    return user_seq


# ── 그래프 ───────────────────────────────────────────────────────────────

def build_item_graph(asins, meta, iid):
    """관계 3종을 합친, 관계종류를 구분하지 않는 단순 연결 그래프 (방향: src->dst)."""
    item_graph = defaultdict(set)
    for a in asins:
        related = meta.get(a, {}).get('related', {})
        neighbors = (related.get('also_bought', []) +
                     related.get('also_viewed', []) +
                     related.get('bought_together', []))
        for n in neighbors:
            if n in iid:
                item_graph[iid[a]].add(iid[n])
    return dict(item_graph)


def build_item_graph_detailed(asins, meta, iid):
    """{src_item_id: {dst_item_id: {관계종류, ...}}} — 관계종류를 보존한 그래프.
    G-SID 에서 관계별 가중치를 다르게 주고 싶을 때 사용 (Beauty_related_separate.pkl 전용)."""
    graph = defaultdict(lambda: defaultdict(set))
    for a in asins:
        related = meta.get(a, {}).get('related', {})
        src = iid[a]
        for rel in RELATION_TYPES:
            for n in related.get(rel, []):
                if n in iid:
                    graph[src][iid[n]].add(rel)
    return {src: dict(nbrs) for src, nbrs in graph.items()}


# ── 텍스트 / 인기도 ──────────────────────────────────────────────────────

def flatten_categories(cats):
    # 정렬해서 join: set 순회 순서에 기대면 실행마다 item_text 가 달라진다
    # (Split A/B 텍스트가 11,909개 아이템에서 어긋났던 재현성 버그의 원인).
    flat = set()
    for path in cats:
        flat.update(path)
    return ' '.join(sorted(flat))


def build_item_text(asins, meta, iid):
    item_text = {}
    for a in asins:
        m = meta.get(a, {})
        parts = [m.get('title', '')]
        if m.get('brand'):
            parts.append(m['brand'])  # 51% 결측 -> 빈 문자열이면 "None" 오염 방지 위해 조건부
        if m.get('description'):
            parts.append(m['description'])
        parts.append(flatten_categories(m.get('categories', [])))
        item_text[iid[a]] = ' '.join(p for p in parts if p).strip()
    return item_text


def build_item_salesrank(asins, meta, iid):
    # 원본 salesRank({카테고리: 순위})만 그대로 노출. 가중치 변환은 다음 단계(G-SID)에서.
    return {iid[a]: meta.get(a, {}).get('salesRank', {}) for a in asins}


# ── 분할 A: Leave-One-Out ────────────────────────────────────────────────

def split_leave_one_out(user_seq):
    train, val, test = {}, {}, {}
    for u, seq in user_seq.items():
        if len(seq) < 3:
            continue
        train[u] = seq[:-2]
        val[u] = seq[-2]
        test[u] = seq[-1]
    return train, val, test


# ── 분할 B: 절대시간 (리뷰수 분위수 + 날짜 경계 스냅) ────────────────────

def split_chronological_windows(reviews, n_windows=5, by='count'):
    """
    by='time'  : 전체 기간을 달력 기준 균등 n등분 (Can GR Reach Cold Items? 논문 원안).
                 이 데이터는 리뷰가 2013년에 몰려있어 초기 윈도우가 사실상 비어 실험 불가.
    by='count' : 리뷰 건수 기준 균등 n등분 (기본값, docs/REQUEST_전처리_splitB.md 제안).

    ★ 건수 분위수로 '위치'를 찾되, 실제로 자르는 경계는 그 위치의 '타임스탬프 값'으로
      스냅한다. 이 데이터는 타임스탬프가 일 단위(고유 2,927개, 하루 평균 15건, 최대
      634건)라, 인덱스로 그냥 자르면 같은 날 리뷰가 train/test 양쪽으로 쪼개져
      누수가 된다 (LOO 가 갖고 있던 "전역 시간 커트라인 없음"과 같은 종류의 결함).
      날짜로 스냅하면 하루 전체가 항상 한쪽으로만 가서 이 누수가 사라진다.
    """
    # 동점 시각의 2차 정렬 키를 고정해 재현성 확보
    rows = sorted(reviews, key=lambda r: (r['unixReviewTime'], r['asin'], r['reviewerID']))
    n = len(rows)

    if by == 'time':
        times = [r['unixReviewTime'] for r in rows]
        t_min, t_max = min(times), max(times)
        span = t_max - t_min
        bounds_ts = [t_min + span * i / n_windows for i in range(n_windows + 1)]
    else:
        bounds_ts = [rows[0]['unixReviewTime']]
        for i in range(1, n_windows):
            t = rows[int(n * i / n_windows)]['unixReviewTime']
            bounds_ts.append(t)
        bounds_ts.append(rows[-1]['unixReviewTime'])

    windows = []
    for i in range(n_windows):
        lo, hi = bounds_ts[i], bounds_ts[i + 1]
        if i == n_windows - 1:
            w = [r for r in rows if lo <= r['unixReviewTime'] <= hi]
        else:
            w = [r for r in rows if lo <= r['unixReviewTime'] < hi]
        windows.append(w)
    return windows  # [W1, W2, ..., Wn], 시간순 · 건수 거의 균등 · 날짜는 안 쪼개짐


def _rows_to_user_sequences(rows, iid):
    """리뷰 행 리스트 -> {user_id(원본): [item_id, ...]} (시간순). B_train_sequences_long 용."""
    by_user = defaultdict(list)
    for r in rows:
        by_user[r['reviewerID']].append(r)
    seqs = {}
    for u, urows in by_user.items():
        urows.sort(key=lambda x: (x['unixReviewTime'], x['orig_order']))
        seqs[u] = [iid[r['asin']] for r in urows]
    return seqs


def _rows_to_user_targets(rows, iid):
    """리뷰 행 리스트 -> {user_id(원본): [item_id, ...]}. val/test 는 순서 없이 그 구간에
    등장한 아이템 전부가 각각 하나의 예측 대상(target)이 된다. B_targets_long 용."""
    targets = defaultdict(list)
    for r in rows:
        targets[r['reviewerID']].append(iid[r['asin']])
    return dict(targets)


def build_split_B_windows(chunks, asins, iid, uid):
    """5개의 건수-균등 청크(C1..C5, chunks[0]=W1 ... chunks[4]=W5)로부터,
    누적 train / 직전 청크 val / 현재 청크 test 구조의 W3/W4/W5 스냅샷을 만든다.
    (docs/REQUEST_전처리_splitB.md §3 구조 그대로: train 은 뒤로 갈수록 누적된다.)
    """
    windows = {}
    cold_tiers = {}
    item_counts = {}

    for k in (3, 4, 5):  # 스냅샷 라벨 Wk
        train_rows = [r for chunk in chunks[:k - 2] for r in chunk]  # C1..C(k-2)
        val_rows = chunks[k - 2]                                     # C(k-1)
        test_rows = chunks[k - 1]                                    # Ck

        windows[f"W{k}"] = {
            'train_seq': {uid[u]: seq for u, seq in _rows_to_user_sequences(train_rows, iid).items()},
            'val_targets': {uid[u]: items for u, items in _rows_to_user_targets(val_rows, iid).items()},
            'test_targets': {uid[u]: items for u, items in _rows_to_user_targets(test_rows, iid).items()},
            'n_train_interactions': len(train_rows),
            'n_val_interactions': len(val_rows),
            'n_test_interactions': len(test_rows),
        }

        tiers, pop = stratify_cold(train_rows, asins, iid)
        cold_tiers[f"W{k}"] = tiers
        item_counts[f"W{k}"] = pop

    return windows, cold_tiers, item_counts


def stratify_cold(train_rows, asins, iid):
    """윈도우의 train 구간에서 아이템별 등장 횟수를 세고 콜드 등급을 매긴다.
    등급 라벨과 원본 횟수(item_pop)를 함께 반환한다 — CRAB 처럼 등급이 아니라
    실제 횟수가 필요한 처방을 위해 횟수를 버리지 않는다."""
    counts = defaultdict(int)
    for r in train_rows:
        counts[iid[r['asin']]] += 1

    tiers = {}
    item_pop = {}
    for a in asins:
        c = counts[iid[a]]
        item_pop[iid[a]] = c
        if c == 0:
            tiers[iid[a]] = 'unseen'
        elif c <= 3:
            tiers[iid[a]] = 'very_rare'
        elif c <= 5:
            tiers[iid[a]] = 'rare'
        else:
            tiers[iid[a]] = 'normal'
    return tiers, item_pop


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    reviews_path = DATA_DIR / f"reviews_{CATEGORY}_5.json.gz"
    meta_path = DATA_DIR / f"meta_{CATEGORY}.json.gz"

    print(f"원본 로딩: {reviews_path.name}, {meta_path.name}")
    reviews = load_reviews(reviews_path)
    meta = load_meta(meta_path)

    uid, iid, users, asins = build_id_maps(reviews)
    print(f"유저 {len(users)}명, 아이템 {len(asins)}개, 상호작용 {len(reviews)}건")

    user_seq_raw = build_user_sequences(reviews, iid)  # reviewerID(string) -> [item_id, ...]
    user_seq = {uid[u]: seq for u, seq in user_seq_raw.items()}  # 저장용: user_id(int) 키
    item_graph = build_item_graph(asins, meta, iid)
    item_graph_detailed = build_item_graph_detailed(asins, meta, iid)
    n_edge_labels = sum(len(rels) for nbrs in item_graph_detailed.values() for rels in nbrs.values())
    print(f"그래프 있는 아이템: {len(item_graph)} / {len(asins)}  (관계 라벨 {n_edge_labels}개)")

    item_text = build_item_text(asins, meta, iid)
    item_salesrank = build_item_salesrank(asins, meta, iid)

    # 분할 A: LOO (user_seq 가 이미 user_id(int) 키라서 결과도 그대로 int 키)
    loo_train, loo_val, loo_test = split_leave_one_out(user_seq)
    print(f"Split A(LOO): 평가 가능 유저 {len(loo_train)} / {len(users)}")

    # 분할 B: 절대시간 (리뷰수 분위수 + 날짜 경계 스냅, n_windows=5)
    chunks = split_chronological_windows(reviews, n_windows=5, by='count')
    print("청크별 리뷰 수 (W1..W5):", [len(c) for c in chunks])
    windows, cold_tiers, item_counts = build_split_B_windows(chunks, asins, iid, uid)
    for label, w in windows.items():
        n_unseen = sum(1 for t in cold_tiers[label].values() if t == 'unseen')
        n_normal = sum(1 for t in cold_tiers[label].values() if t == 'normal')
        print(f"  {label}: train {w['n_train_interactions']} / val {w['n_val_interactions']} "
              f"/ test {w['n_test_interactions']}  (unseen {n_unseen}, normal {n_normal})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_a = {
        'uid': uid, 'iid': iid,
        'user_seq': user_seq,
        'item_text': item_text,
        'item_graph': item_graph,
        'item_salesrank': item_salesrank,
        'loo_train': loo_train, 'loo_val': loo_val, 'loo_test': loo_test,
    }
    with open(OUT_DIR / f"{CATEGORY}_split_A.pkl", 'wb') as f:
        pickle.dump(split_a, f)

    split_b = {
        'uid': uid, 'iid': iid,
        'user_seq': user_seq,
        'item_text': item_text,
        'item_graph': item_graph,
        'item_salesrank': item_salesrank,
        'windows': windows,               # W3/W4/W5: train_seq / val_targets / test_targets
        'cold_tiers': cold_tiers,          # 윈도우별 {item_id: 등급}
        'item_counts': item_counts,        # 윈도우별 {item_id: train 등장 횟수}
    }
    with open(OUT_DIR / f"{CATEGORY}_split_B.pkl", 'wb') as f:
        pickle.dump(split_b, f)

    related_separate = {
        'iid': iid,
        'item_graph_detailed': item_graph_detailed,
    }
    with open(OUT_DIR / f"{CATEGORY}_related_separate.pkl", 'wb') as f:
        pickle.dump(related_separate, f)

    print(f"저장 완료: {OUT_DIR}/")


if __name__ == '__main__':
    main()
