"""아이템을 인기도 구간(HEAD / BODY / TAIL)으로 나눈다.

팀 공통 평가의 1단계. "비인기 추천이 좋아졌다"를 재려면 먼저 무엇이 비인기인지
합의된 정의가 있어야 한다. 이 스크립트가 그 정의를 파일로 고정한다.

구간 정의 (VarLenRec 방식, docs/03_실험_계획.md)
    학습 데이터에서 아이템별 등장 횟수를 세고 많은 순으로 정렬한 뒤
    HEAD  상위 20%   BODY  중간 60%   TAIL  하위 20%

    Ghost 는 누적 상호작용 80% 기준을 쓴다. 기준이 다르면 숫자가 안 맞으므로
    비교할 때 어느 기준인지 반드시 밝혀야 한다.

COLD 와 UNSEEN 은 다르다
    UNSEEN  우연히 학습에 안 나온 아이템. 원본 데이터에는 33개(0.3%)뿐이라
            통계를 낼 수 없다. leave-one-out 이 시퀀스 끝 2개를 떼면서 생긴 부산물이다.
    COLD    일부러 학습에서 빼낸 아이템. scripts/cold_split.py 가 만든다.
            --cold-items 로 그 목록을 주면 여기서 COLD 로 표시한다.
    둘을 섞으면 "신규 아이템 성능"이 33개짜리 잡음이 된다. 반드시 구분한다.

사용:
    python scripts/item_segments.py \
        --data-dir data/amazon_data/beauty \
        --out work/segments/beauty.json

    # COLD 데이터셋에 대해
    python scripts/item_segments.py \
        --data-dir work/cold/beauty_strat10 \
        --cold-items work/cold/beauty_strat10/cold_items.json \
        --out work/segments/beauty_strat10.json
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import tensorflow as tf

SEGMENT_ORDER = ["HEAD", "BODY", "TAIL", "COLD", "UNSEEN"]
# HEAD 상위 20%, BODY 중간 60%, TAIL 하위 20%
HEAD_FRACTION = 0.2
TAIL_FRACTION = 0.2


def count_interactions(split_dir: str, field: str = "sequence_data") -> Counter:
    """한 split 의 모든 시퀀스에서 아이템 등장 횟수를 센다."""
    files = sorted(glob.glob(os.path.join(split_dir, "*.tfrecord.gz")))
    if not files:
        raise FileNotFoundError(f"tfrecord 파일이 없습니다: {split_dir}")

    counts: Counter = Counter()
    dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")
    for raw in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        counts.update(example.features.feature[field].int64_list.value)
    return counts


def count_items(items_dir: str) -> int:
    """카탈로그 전체 아이템 수를 센다."""
    files = sorted(glob.glob(os.path.join(items_dir, "*.tfrecord.gz")))
    total = 0
    dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")
    for _ in dataset:
        total += 1
    return total


def assign_segments(counts: Counter, n_items: int, cold: set[int] | None = None) -> dict:
    """아이템 id → 구간 이름.

    cold 로 준 아이템은 학습 등장 횟수와 무관하게 COLD 로 표시한다.
    (cold_split.py 를 거쳤다면 등장 횟수는 이미 0이다. 0이 아니면 데이터가
    잘못 만들어진 것이므로 아래에서 걸러 알려 준다.)
    """
    cold = cold or set()
    # 학습에 한 번도 안 나온 아이템은 인기 순위를 매길 수 없으므로 먼저 뺀다.
    seen = np.array([i for i in range(n_items) if counts.get(i, 0) > 0], dtype=np.int64)
    unseen = [i for i in range(n_items) if counts.get(i, 0) == 0]

    popularity = np.array([counts[i] for i in seen], dtype=np.int64)
    # 인기 많은 순. 동점은 아이템 id 오름차순으로 깨서 실행마다 결과가 같게 한다.
    order = np.lexsort((seen, -popularity))
    ranked = seen[order]

    n_seen = len(ranked)
    head_end = int(round(n_seen * HEAD_FRACTION))
    tail_start = n_seen - int(round(n_seen * TAIL_FRACTION))

    segments = {}
    for rank, item_id in enumerate(ranked):
        if rank < head_end:
            segments[int(item_id)] = "HEAD"
        elif rank < tail_start:
            segments[int(item_id)] = "BODY"
        else:
            segments[int(item_id)] = "TAIL"
    for item_id in unseen:
        segments[int(item_id)] = "UNSEEN"
    for item_id in cold:
        segments[int(item_id)] = "COLD"
    return segments


def summarize(segments: dict, counts: Counter) -> list[dict]:
    """구간별 아이템 수와 상호작용 점유율."""
    total_interactions = sum(counts.values())
    rows = []
    for name in SEGMENT_ORDER:
        ids = [i for i, s in segments.items() if s == name]
        interactions = sum(counts.get(i, 0) for i in ids)
        rows.append(
            {
                "segment": name,
                "n_items": len(ids),
                "item_share": len(ids) / len(segments) if segments else 0.0,
                "interactions": interactions,
                "interaction_share": interactions / total_interactions
                if total_interactions
                else 0.0,
                "min_count": min((counts.get(i, 0) for i in ids), default=0),
                "max_count": max((counts.get(i, 0) for i in ids), default=0),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="beauty / sports / toys 폴더")
    parser.add_argument("--out", required=True, help="저장할 .json 경로")
    parser.add_argument(
        "--popularity-split",
        default="training",
        help="인기도를 셀 split. 기본 training (평가 대상 정보가 새면 안 되므로)",
    )
    parser.add_argument(
        "--cold-items",
        default=None,
        help="cold_split.py 가 만든 cold_items.json. 주면 그 아이템들을 COLD 로 표시한다",
    )
    args = parser.parse_args()

    cold = set()
    if args.cold_items:
        with open(args.cold_items) as f:
            cold = {int(i) for i in json.load(f)["cold_items"]}

    counts = count_interactions(os.path.join(args.data_dir, args.popularity_split))
    n_items = count_items(os.path.join(args.data_dir, "items"))

    # COLD 인데 학습에 등장한다면 데이터셋이 잘못 만들어진 것이다. 조용히 넘기면 안 된다.
    leaked = sorted(i for i in cold if counts.get(i, 0) > 0)
    if leaked:
        raise SystemExit(
            f"❌ COLD 로 지정된 아이템 {len(leaked)}개가 {args.popularity_split} 에 등장합니다.\n"
            f"   예: {leaked[:10]}\n"
            f"   --data-dir 가 cold_split.py 의 출력이 맞는지 확인하세요."
        )

    segments = assign_segments(counts, n_items, cold)
    rows = summarize(segments, counts)

    print(f"카탈로그 아이템 {n_items}개 · 학습 상호작용 {sum(counts.values())}건")
    print(f"{'구간':<8}{'아이템수':>9}{'아이템비율':>11}{'상호작용':>11}{'점유율':>9}{'등장횟수':>13}")
    for row in rows:
        span = f"{row['min_count']}~{row['max_count']}"
        print(
            f"{row['segment']:<8}{row['n_items']:>9,}{row['item_share']:>10.1%}"
            f"{row['interactions']:>11,}{row['interaction_share']:>9.1%}{span:>13}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "data_dir": args.data_dir,
                "popularity_split": args.popularity_split,
                "head_fraction": HEAD_FRACTION,
                "tail_fraction": TAIL_FRACTION,
                "n_items": n_items,
                "cold_items_file": args.cold_items,
                "n_cold_items": len(cold),
                "summary": rows,
                # json 키는 문자열이어야 해서 아이템 id 를 문자열로 저장한다.
                "segments": {str(k): v for k, v in sorted(segments.items())},
                "train_counts": {str(k): int(v) for k, v in sorted(counts.items())},
            },
            f,
            ensure_ascii=False,
        )
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
