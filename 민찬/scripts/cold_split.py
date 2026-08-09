"""진짜 COLD(신규 아이템) 구간을 가진 데이터셋을 만든다.

무엇이 문제였나
    지금 데이터는 leave-one-out 분할이라 거의 모든 아이템이 학습에 등장한다.
    학습에 없는 아이템은 33개(0.3%)뿐이라 통계를 낼 수 없다.
    "신규 아이템 추천이 좋아졌다"를 재려면 신규 아이템이 충분히 있어야 한다.

왜 시간 기준 분할이 아닌가  ← 계획을 바꾼 이유
    전처리된 TFRecord 에는 타임스탬프가 없다. 아이템 id 도 시간을 담고 있지 않다.
    실측: 시퀀스 위치 vs 아이템 id 스피어만 상관 +0.07, 양의 상관 유저 비율 54.6%.
          시간을 담고 있다면 각각 1.0 / 100% 에 가까워야 한다. 사실상 무작위다.
    → 시간 기준 분할을 하려면 원본 Amazon 리뷰(unixReviewTime 포함)를 다시 받아
      ASIN 으로 이어 붙여야 한다. 그건 별도 다운로드가 필요한 일이다.

    그래서 타임스탬프 없이도 되는 **아이템 홀드아웃**으로 만든다.
    콜드스타트 논문에서 널리 쓰는 방식이고, 재는 대상("학습에 한 번도 없던
    아이템을 추천할 수 있는가")은 시간 분할과 같다.
    다른 점 하나는 명시해야 한다:
      시간 분할  = 신규 아이템 + 신규 시점의 유저 취향 변화가 섞임
      아이템 홀드아웃 = 신규 아이템 효과만 분리됨 (교란이 오히려 적다)

어떻게 만드나
    ① COLD 후보 = 테스트 정답으로 한 번 이상 등장하는 아이템
    ② 그중 목표 비율만큼 뽑는다 (--strategy 로 표본 방식 선택)
    ③ training / evaluation 시퀀스에서 COLD 아이템을 전부 지운다
       → 학습에서 완전히 사라진다. 이게 "신규"의 정의다.
    ④ testing 은 정답(마지막)은 남기고 그 앞 이력에서만 COLD 를 지운다
    ⑤ 너무 짧아진 시퀀스는 버린다
    ⑥ items/ (카탈로그)는 건드리지 않는다  ← 중요
       TIGER 는 아이템 임베딩만 있으면 학습에 없던 아이템에도 SID 를 붙일 수 있다.
       그 능력을 재는 것이 이 실험이므로 카탈로그에는 남아 있어야 한다.

사용:
    python scripts/cold_split.py \
        --data-dir data/amazon_data/beauty \
        --out-dir  work/cold/beauty \
        --cold-fraction 0.10 --strategy stratified --seed 42

    # 만든 뒤 구간 정의도 다시 만든다 (COLD 표시가 붙는다)
    python scripts/item_segments.py --data-dir work/cold/beauty \
        --out work/segments/beauty_cold.json --cold-items work/cold/beauty/cold_items.json
"""

import argparse
import glob
import json
import os
import shutil
from collections import Counter

import numpy as np
import tensorflow as tf

SPLITS = ["training", "evaluation", "testing"]


# ─────────────────────────────── TFRecord 읽고 쓰기 ───────────────────────────────


def partitions(split_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(split_dir, "*.tfrecord.gz")))
    if not files:
        raise FileNotFoundError(f"tfrecord 파일이 없습니다: {split_dir}")
    return files


def read_raw(path: str):
    """한 파티션의 레코드를 원본 바이트 그대로 하나씩 내준다."""
    for raw in tf.data.TFRecordDataset([path], compression_type="GZIP"):
        yield raw.numpy()


def parse_sequence(raw: bytes) -> tuple[tf.train.Example, list[int]]:
    """레코드를 풀고 아이템 id 만 파이썬 리스트로 꺼낸다.

    임베딩(길이 L×768)까지 파이썬 리스트로 옮기면 레코드 하나에 수천 개
    객체가 생겨 전체가 느려진다. 손댈 레코드인지 먼저 아이템 id 로만 판단하고,
    실제로 손댈 때만 나머지를 만진다.
    """
    example = tf.train.Example()
    example.ParseFromString(raw)
    return example, list(example.features.feature["sequence_data"].int64_list.value)


def rebuild(example: tf.train.Example, keep: list[int]) -> bytes:
    """시퀀스에서 keep 위치만 남긴 레코드를 다시 만든다.

    네 필드의 길이는 서로 묶여 있다.
        sequence_data  길이 L        아이템 id
        text           길이 L        아이템 제목
        embedding      길이 L×D      아이템 임베딩을 이어 붙인 것
        user_id        길이 1
    하나만 지우면 정렬이 어긋나므로 셋을 함께 자른다.
    """
    feature = example.features.feature
    sequence = list(feature["sequence_data"].int64_list.value)
    texts = list(feature["text"].bytes_list.value)
    embedding = np.asarray(feature["embedding"].float_list.value, dtype=np.float32)

    length = len(sequence)
    dim = len(embedding) // length if length else 0
    index = np.asarray(keep, dtype=np.int64)
    trimmed = embedding.reshape(length, dim)[index].reshape(-1) if dim else embedding[:0]

    built = {
        "user_id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(feature["user_id"].int64_list.value))
        ),
        "sequence_data": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[sequence[i] for i in keep])
        ),
        "text": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[texts[i] for i in keep])
        ),
        "embedding": tf.train.Feature(float_list=tf.train.FloatList(value=trimmed.tolist())),
    }
    return tf.train.Example(features=tf.train.Features(feature=built)).SerializeToString()


# ─────────────────────────────── COLD 아이템 고르기 ───────────────────────────────


def collect_stats(data_dir: str) -> tuple[Counter, Counter, int]:
    """(학습 등장 횟수, 테스트 정답 횟수, 카탈로그 아이템 수)."""
    train_counts: Counter = Counter()
    for path in partitions(os.path.join(data_dir, "training")):
        for raw in read_raw(path):
            _, sequence = parse_sequence(raw)
            train_counts.update(sequence)

    target_counts: Counter = Counter()
    for path in partitions(os.path.join(data_dir, "testing")):
        for raw in read_raw(path):
            _, sequence = parse_sequence(raw)
            target_counts[sequence[-1]] += 1

    n_items = 0
    for path in sorted(glob.glob(os.path.join(data_dir, "items", "*.tfrecord.gz"))):
        n_items += sum(1 for _ in tf.data.TFRecordDataset([path], compression_type="GZIP"))

    return train_counts, target_counts, n_items


def choose_cold_items(
    train_counts: Counter,
    target_counts: Counter,
    n_items: int,
    fraction: float,
    strategy: str,
    seed: int,
) -> tuple[list[int], dict]:
    """COLD 로 만들 아이템을 고른다.

    후보 조건은 하나뿐이다: **테스트 정답으로 한 번 이상 등장할 것.**
    정답으로 안 나오면 홀드아웃해도 평가할 수 없어서 표본이 되지 못한다.

    strategy
        stratified  HEAD/BODY/TAIL 비율을 유지해서 뽑는다 (기본).
                    COLD 구간이 카탈로그 전체를 닮아 "신규 아이템 일반"을 대표한다.
        tail        비인기(TAIL)에서만 뽑는다.
                    학습 데이터 손실이 가장 적다. 현실의 신규 아이템에 가깝다.
        random      후보 전체에서 균등하게.

    ⚠️ stratified 로 인기 아이템을 빼면 학습 상호작용이 크게 줄어든다.
       그 손실량을 아래 report 에 반드시 같이 낸다. 성능이 떨어졌을 때
       "COLD 라서"인지 "학습 데이터가 줄어서"인지 구분하려면 필요한 숫자다.
    """
    rng = np.random.default_rng(seed)
    candidates = np.array(sorted(target_counts), dtype=np.int64)
    n_target = int(round(n_items * fraction))

    if n_target > len(candidates):
        raise ValueError(
            f"목표 {n_target}개가 후보 {len(candidates)}개보다 많습니다. "
            f"--cold-fraction 을 {len(candidates)/n_items:.3f} 이하로 낮추세요."
        )

    # 후보를 학습 인기도 순으로 세 덩어리로 나눈다 (item_segments.py 와 같은 20/60/20).
    popularity = np.array([train_counts.get(int(i), 0) for i in candidates])
    order = np.lexsort((candidates, -popularity))
    ranked = candidates[order]
    head_end = int(round(len(ranked) * 0.2))
    tail_start = len(ranked) - int(round(len(ranked) * 0.2))
    buckets = {
        "HEAD": ranked[:head_end],
        "BODY": ranked[head_end:tail_start],
        "TAIL": ranked[tail_start:],
    }

    if strategy == "random":
        chosen = rng.choice(candidates, size=n_target, replace=False)
    elif strategy == "tail":
        pool = buckets["TAIL"]
        if n_target > len(pool):
            raise ValueError(
                f"tail 전략의 후보는 {len(pool)}개뿐입니다. "
                f"--cold-fraction 을 {len(pool)/n_items:.3f} 이하로 낮추거나 "
                f"--strategy stratified 를 쓰세요."
            )
        chosen = rng.choice(pool, size=n_target, replace=False)
    elif strategy == "stratified":
        chosen_parts = []
        allocated = 0
        names = ["HEAD", "BODY", "TAIL"]
        for i, name in enumerate(names):
            pool = buckets[name]
            if i == len(names) - 1:
                take = n_target - allocated          # 반올림 오차를 마지막에 몰아준다
            else:
                take = int(round(n_target * len(pool) / len(candidates)))
            take = min(take, len(pool))
            chosen_parts.append(rng.choice(pool, size=take, replace=False))
            allocated += take
        chosen = np.concatenate(chosen_parts)
    else:
        raise ValueError(f"모르는 strategy: {strategy}")

    cold = sorted(int(i) for i in chosen)
    cold_set = set(cold)

    total_interactions = sum(train_counts.values())
    lost_interactions = sum(train_counts.get(i, 0) for i in cold)
    report = {
        "strategy": strategy,
        "seed": seed,
        "cold_fraction_requested": fraction,
        "n_candidates": len(candidates),
        "n_cold_items": len(cold),
        "cold_item_share": len(cold) / n_items,
        "cold_test_users": sum(target_counts[i] for i in cold),
        "train_interactions_before": total_interactions,
        "train_interactions_lost": lost_interactions,
        "train_interaction_loss_share": lost_interactions / total_interactions
        if total_interactions
        else 0.0,
        "cold_by_popularity_bucket": {
            name: int(sum(1 for i in pool if int(i) in cold_set)) for name, pool in buckets.items()
        },
    }
    return cold, report


# ─────────────────────────────── 데이터셋 다시 쓰기 ───────────────────────────────


def rewrite_split(
    data_dir: str,
    out_dir: str,
    split: str,
    cold: set[int],
    min_length: int,
    keep_last: bool,
) -> dict:
    """한 split 을 COLD 아이템 없이 다시 쓴다.

    keep_last=True (testing) 이면 마지막 아이템(정답)은 COLD 여도 남긴다.
    그 앞 이력에서만 지운다. 그래야 "학습엔 없지만 정답으로는 나오는" 상황이 된다.
    """
    split_out = os.path.join(out_dir, split)
    os.makedirs(split_out, exist_ok=True)

    stats = {
        "records_in": 0,
        "records_out": 0,
        "records_dropped_too_short": 0,
        "items_removed": 0,
        "records_untouched": 0,
    }

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    for path in partitions(os.path.join(data_dir, split)):
        out_path = os.path.join(split_out, os.path.basename(path))
        with tf.io.TFRecordWriter(out_path, options=options) as writer:
            for raw in read_raw(path):
                stats["records_in"] += 1
                example, sequence = parse_sequence(raw)
                last = len(sequence) - 1

                keep = [
                    i
                    for i, item in enumerate(sequence)
                    if item not in cold or (keep_last and i == last)
                ]
                removed = len(sequence) - len(keep)
                if removed == 0:
                    # 손댈 게 없으면 원본 바이트를 그대로 옮긴다 (가장 빠른 길)
                    stats["records_untouched"] += 1
                    stats["records_out"] += 1
                    writer.write(raw)
                    continue

                stats["items_removed"] += removed
                if len(keep) < min_length:
                    stats["records_dropped_too_short"] += 1
                    continue
                writer.write(rebuild(example, keep))
                stats["records_out"] += 1

    return stats


def link_items(data_dir: str, out_dir: str) -> None:
    """카탈로그는 그대로 쓴다. COLD 아이템도 카탈로그에는 남아 있어야 한다."""
    src = os.path.abspath(os.path.join(data_dir, "items"))
    dst = os.path.join(out_dir, "items")
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            shutil.rmtree(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copytree(src, dst)


# ─────────────────────────────── 검증 ───────────────────────────────


def verify(out_dir: str, cold: set[int]) -> dict:
    """만든 데이터셋이 실제로 조건을 만족하는지 다시 읽어서 확인한다.

    ① training / evaluation 에 COLD 아이템이 0번 등장하는가
    ② testing 정답에 COLD 아이템이 충분히 남아 있는가
    ③ 필드 길이가 서로 맞는가 (sequence : text : embedding)
    """
    problems = []
    leaked = Counter()
    for split in ["training", "evaluation"]:
        for path in partitions(os.path.join(out_dir, split)):
            for raw in read_raw(path):
                _, sequence = parse_sequence(raw)
                leaked[split] += sum(1 for item in sequence if item in cold)
    leaked = {k: v for k, v in leaked.items() if v}
    if leaked:
        problems.append(f"COLD 아이템이 학습 쪽에 남아 있습니다: {leaked}")

    cold_targets = 0
    n_test_users = 0
    length_errors = 0
    history_leak = 0
    for path in partitions(os.path.join(out_dir, "testing")):
        for raw in read_raw(path):
            n_test_users += 1
            example, sequence = parse_sequence(raw)
            if sequence[-1] in cold:
                cold_targets += 1
            if any(item in cold for item in sequence[:-1]):
                history_leak += 1

            feature = example.features.feature
            n_text = len(feature["text"].bytes_list.value)
            n_embed = len(feature["embedding"].float_list.value)
            if n_text != len(sequence) or (sequence and n_embed % len(sequence) != 0):
                length_errors += 1

    if history_leak:
        problems.append(f"testing 이력에 COLD 아이템이 남은 레코드 {history_leak}건")
    if length_errors:
        problems.append(f"필드 길이가 어긋난 레코드 {length_errors}건")
    if cold_targets == 0:
        problems.append("testing 정답에 COLD 아이템이 하나도 없습니다")

    return {
        "ok": not problems,
        "problems": problems,
        "cold_target_users": cold_targets,
        "test_users": n_test_users,
        "cold_target_share": cold_targets / n_test_users if n_test_users else 0.0,
    }


# ─────────────────────────────── 실행 ───────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="원본 beauty / sports / toys 폴더")
    parser.add_argument("--out-dir", required=True, help="만들어 낼 데이터셋 폴더")
    parser.add_argument("--cold-fraction", type=float, default=0.10,
                        # argparse 가 help 문자열에 %-포맷을 적용하므로 %% 로 써야 한다.
                        help="카탈로그의 몇 %%를 COLD 로 만들지 (기본 0.10)")
    parser.add_argument("--strategy", default="stratified",
                        choices=["stratified", "tail", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-length", type=int, default=3,
                        help="COLD 를 뺀 뒤 이보다 짧아진 시퀀스는 버린다 (기본 3)")
    args = parser.parse_args()

    print(f"원본 읽는 중: {args.data_dir}")
    train_counts, target_counts, n_items = collect_stats(args.data_dir)
    print(f"  카탈로그 {n_items:,}개 · 학습 상호작용 {sum(train_counts.values()):,}건 "
          f"· 테스트 정답으로 등장하는 아이템 {len(target_counts):,}개")

    cold, report = choose_cold_items(
        train_counts, target_counts, n_items, args.cold_fraction, args.strategy, args.seed
    )
    cold_set = set(cold)
    print(f"\nCOLD 로 뽑은 아이템 {report['n_cold_items']:,}개 "
          f"({report['cold_item_share']:.1%}, 전략 {args.strategy})")
    print(f"  인기 구간별: {report['cold_by_popularity_bucket']}")
    print(f"  이 아이템들이 정답인 테스트 유저 {report['cold_test_users']:,}명")
    print(f"  🚨 학습 상호작용 손실 {report['train_interactions_lost']:,}건 "
          f"({report['train_interaction_loss_share']:.1%})")

    os.makedirs(args.out_dir, exist_ok=True)
    split_stats = {}
    for split in SPLITS:
        keep_last = split == "testing"
        print(f"\n{split} 다시 쓰는 중... (정답 보존: {keep_last})")
        stats = rewrite_split(
            args.data_dir, args.out_dir, split, cold_set, args.min_length, keep_last
        )
        split_stats[split] = stats
        print(f"  유저 {stats['records_in']:,} → {stats['records_out']:,} "
              f"(너무 짧아져 버림 {stats['records_dropped_too_short']:,}) "
              f"· 지운 아이템 {stats['items_removed']:,}건")

    link_items(args.data_dir, args.out_dir)
    print(f"\nitems/ 는 원본을 그대로 가리킵니다 (COLD 도 카탈로그에는 남아야 함)")

    print("\n검증 중...")
    check = verify(args.out_dir, cold_set)
    for problem in check["problems"]:
        print(f"  ❌ {problem}")
    if check["ok"]:
        print(f"  ✅ 학습/검증에 COLD 0건 · 테스트 정답이 COLD 인 유저 "
              f"{check['cold_target_users']:,}명 ({check['cold_target_share']:.1%})")

    meta = {
        "source_data_dir": os.path.abspath(args.data_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "min_length": args.min_length,
        "n_items": n_items,
        "selection": report,
        "splits": split_stats,
        "verification": check,
        "cold_items": cold,
    }
    meta_path = os.path.join(args.out_dir, "cold_items.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {meta_path}")
    print(f"데이터셋:   {os.path.abspath(args.out_dir)}")

    if not check["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
