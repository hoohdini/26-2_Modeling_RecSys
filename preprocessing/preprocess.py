"""
Amazon 2014 Beauty 전처리 파이프라인
0단계: 파싱 -> ID매핑 -> 유저별 시간순 정렬 -> 그래프 추출 -> 텍스트 구성 -> 분할 A/B -> 콜드 계층화
"""
import gzip
import json
import ast
import pickle
from collections import defaultdict

DATA_DIR = r"C:\Users\User\Desktop\DSL 26-2\Modeling\data\amazon2014"
CATEGORY = "Beauty"


def load_reviews(path):
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            rows.append({
                'reviewerID': d['reviewerID'],
                'asin': d['asin'],
                'unixReviewTime': d['unixReviewTime'],
                'orig_order': idx,  # 동일 타임스탬프 tie-break용
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


def build_id_maps(reviews):
    users = sorted({r['reviewerID'] for r in reviews})
    asins = sorted({r['asin'] for r in reviews})
    uid = {u: k for k, u in enumerate(users)}
    iid = {a: k for k, a in enumerate(asins)}
    return uid, iid, users, asins


def build_user_sequences(reviews, iid):
    user_rows = defaultdict(list)
    for r in reviews:
        user_rows[r['reviewerID']].append(r)

    user_seq = {}
    for u, rows in user_rows.items():
        rows.sort(key=lambda x: (x['unixReviewTime'], x['orig_order']))
        user_seq[u] = [iid[r['asin']] for r in rows]
    return user_seq


def build_item_graph(asins, meta, iid):
    # 가중치 없는 순수 연결 구조만 추출. 그래프 가중치 적용 여부/방식은
    # G-SID 알고리즘(2단계) 설계 사항이라 여기서 결정하지 않음.
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


def build_item_salesrank(asins, meta, iid):
    # 원본 salesRank({카테고리: 순위})만 그대로 노출. 가중치 변환은 다음 단계에서.
    return {iid[a]: meta.get(a, {}).get('salesRank', {}) for a in asins}


def flatten_categories(cats):
    flat = set()
    for path in cats:
        flat.update(path)
    return ' '.join(flat)


def build_item_text(asins, meta, iid):
    item_text = {}
    for a in asins:
        m = meta.get(a, {})
        parts = [m.get('title', '')]
        if m.get('brand'):
            parts.append(m['brand'])
        if m.get('description'):
            parts.append(m['description'])
        parts.append(flatten_categories(m.get('categories', [])))
        item_text[iid[a]] = ' '.join(p for p in parts if p).strip()
    return item_text


def split_leave_one_out(user_seq):
    train, val, test = {}, {}, {}
    for u, seq in user_seq.items():
        if len(seq) < 3:
            continue
        train[u] = seq[:-2]
        val[u] = seq[-2]
        test[u] = seq[-1]
    return train, val, test


def split_temporal(reviews):
    all_times = sorted(r['unixReviewTime'] for r in reviews)
    n = len(all_times)

    def q_time(q):
        return all_times[int(n * q)]

    windows = [
        (q_time(0.80), q_time(0.85)),
        (q_time(0.85), q_time(0.90)),
        (q_time(0.90), q_time(0.95)),
    ]

    def build_window(train_cutoff, test_end):
        train_rows = [r for r in reviews if r['unixReviewTime'] < train_cutoff]
        test_rows = [r for r in reviews if train_cutoff <= r['unixReviewTime'] < test_end]
        return train_rows, test_rows

    return [build_window(a, b) for a, b in windows]


def stratify_cold(train_rows, asins, iid):
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


def main():
    reviews = load_reviews(f"{DATA_DIR}\\reviews_{CATEGORY}_5.json.gz")
    meta = load_meta(f"{DATA_DIR}\\meta_{CATEGORY}.json.gz")

    uid, iid, users, asins = build_id_maps(reviews)
    print(f"유저 {len(users)}명, 아이템 {len(asins)}개")

    user_seq = build_user_sequences(reviews, iid)
    item_graph = build_item_graph(asins, meta, iid)
    print(f"그래프 있는 아이템: {len(item_graph)} / {len(asins)}")

    item_text = build_item_text(asins, meta, iid)
    item_salesrank = build_item_salesrank(asins, meta, iid)

    loo_train, loo_val, loo_test = split_leave_one_out(user_seq)

    # TODO(팀 결정 필요): 시퀀스 최대길이 20을 여기서 자를지, 모델 학습 단계에서 자를지
    temporal_splits = split_temporal(reviews)
    cold_stratify_per_window = [
        stratify_cold(train, asins, iid) for train, test in temporal_splits
    ]
    cold_tiers_per_window = [tiers for tiers, pop in cold_stratify_per_window]
    item_pop_per_window = [pop for tiers, pop in cold_stratify_per_window]

    out_path = f"{DATA_DIR}\\{CATEGORY}_preprocessed.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump({
            'uid': uid, 'iid': iid,
            'user_seq': user_seq, 'item_graph': item_graph,
            'item_text': item_text, 'item_salesrank': item_salesrank,
            'loo_train': loo_train, 'loo_val': loo_val, 'loo_test': loo_test,
            'temporal_splits': temporal_splits,
            'cold_tiers_per_window': cold_tiers_per_window,
            'item_pop_per_window': item_pop_per_window,
        }, f)
    print(f"저장 완료: {out_path}")


if __name__ == '__main__':
    main()
