"""cold_split 이 시퀀스를 자를 때 네 필드가 같이 움직이는지 검증한다.

가장 무서운 버그는 조용한 어긋남이다.
아이템 id 만 지우고 임베딩을 안 지우면 3번 아이템 자리에 4번 임베딩이 들어간다.
에러도 안 나고 학습도 되지만 결과가 통째로 틀린다. 그래서 이걸 먼저 막는다.

합성 데이터를 직접 만들어 쓴다. data/ 없이도 돌아가야 팀원 누구나 확인할 수 있다.
"""

import os
import shutil
import sys
import tempfile
from collections import Counter

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from cold_split import (  # noqa: E402
    choose_cold_items,
    parse_sequence,
    partitions,
    read_raw,
    rewrite_split,
    verify,
)

DIM = 4  # 임베딩 차원. 작게 잡아야 눈으로 확인할 수 있다.
N_ITEMS = 12

# 아이템 i 의 임베딩은 [i, i+0.1, i+0.2, i+0.3]. 값만 보면 어느 아이템인지 알 수 있다.
CATALOG = np.array([[i, i + 0.1, i + 0.2, i + 0.3] for i in range(N_ITEMS)], dtype=np.float32)

failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'OK  ' if ok else 'FAIL'} {label}{('  — ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(label)


def make_record(user_id: int, sequence: list[int]) -> bytes:
    embedding = CATALOG[sequence].reshape(-1)
    feature = {
        "user_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[user_id])),
        "sequence_data": tf.train.Feature(int64_list=tf.train.Int64List(value=sequence)),
        "text": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[f"item-{i}".encode() for i in sequence])
        ),
        "embedding": tf.train.Feature(float_list=tf.train.FloatList(value=embedding.tolist())),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def write_split(root: str, split: str, sequences: dict[int, list[int]]) -> None:
    os.makedirs(os.path.join(root, split), exist_ok=True)
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    path = os.path.join(root, split, "partition_0.tfrecord.gz")
    with tf.io.TFRecordWriter(path, options=options) as writer:
        for user_id, sequence in sorted(sequences.items()):
            writer.write(make_record(user_id, sequence))


def read_all(split_dir: str) -> list[dict]:
    out = []
    for path in partitions(split_dir):
        for raw in read_raw(path):
            example, sequence = parse_sequence(raw)
            feature = example.features.feature
            out.append(
                {
                    "user_id": int(feature["user_id"].int64_list.value[0]),
                    "sequence": sequence,
                    "text": [t.decode() for t in feature["text"].bytes_list.value],
                    "embedding": np.asarray(
                        feature["embedding"].float_list.value, dtype=np.float32
                    ).reshape(len(sequence), -1),
                }
            )
    return out


# ─────────────────────────── 합성 데이터셋 만들기 ───────────────────────────
# leave-one-out 구조를 흉내낸다. training = 앞부분, testing = 전체.

FULL = {
    0: [0, 1, 2, 3, 4],
    1: [5, 1, 6, 7, 8],
    2: [2, 9, 3, 10, 11],
    3: [0, 5, 2, 6, 9],
    4: [1, 3, 7, 10, 4],
}

root = tempfile.mkdtemp(prefix="cold_split_test_")
try:
    write_split(root, "training", {u: s[:-2] for u, s in FULL.items()})
    write_split(root, "evaluation", {u: s[:-1] for u, s in FULL.items()})
    write_split(root, "testing", dict(FULL))

    train_counts = Counter(i for s in FULL.values() for i in s[:-2])
    target_counts = Counter(s[-1] for s in FULL.values())

    # ── 아이템 고르기 ──
    cold, report = choose_cold_items(
        train_counts, target_counts, N_ITEMS, fraction=2 / N_ITEMS,
        strategy="random", seed=0,
    )
    check("고른 개수가 요청과 같다", report["n_cold_items"] == 2, str(report))
    check(
        "COLD 는 테스트 정답으로 등장하는 아이템에서만 고른다",
        all(i in target_counts for i in cold),
        f"cold={cold}, 후보={sorted(target_counts)}",
    )

    # 후보보다 많이 요청하면 조용히 줄이지 말고 멈춰야 한다.
    try:
        choose_cold_items(train_counts, target_counts, N_ITEMS, 1.0, "random", 0)
        check("후보보다 많이 요청하면 에러", False, "에러가 안 났다")
    except ValueError:
        check("후보보다 많이 요청하면 에러", True)

    # ── 자르기 ──
    # 결과가 눈에 보이도록 COLD 를 직접 지정한다. 4는 유저0의 정답, 8은 유저1의 정답.
    cold_set = {4, 8, 2}
    out_dir = os.path.join(root, "out")
    for split in ["training", "evaluation", "testing"]:
        rewrite_split(root, out_dir, split, cold_set, min_length=1,
                      keep_last=(split == "testing"))
    shutil.copytree(os.path.join(root, "training"), os.path.join(out_dir, "items"))

    train_out = {r["user_id"]: r for r in read_all(os.path.join(out_dir, "training"))}
    test_out = {r["user_id"]: r for r in read_all(os.path.join(out_dir, "testing"))}

    # ① 학습에서 COLD 가 사라졌는가
    leaked = [i for r in train_out.values() for i in r["sequence"] if i in cold_set]
    check("학습에 COLD 가 남아 있지 않다", not leaked, f"남은 것 {leaked}")

    # ② 테스트 정답은 COLD 여도 살아 있는가
    check("유저0의 정답 4(COLD)가 살아 있다", test_out[0]["sequence"][-1] == 4,
          str(test_out[0]["sequence"]))
    check("유저1의 정답 8(COLD)이 살아 있다", test_out[1]["sequence"][-1] == 8,
          str(test_out[1]["sequence"]))

    # ③ 테스트 이력(마지막 제외)에서는 COLD 가 지워졌는가
    #    유저0 [0,1,2,3,4] → 2는 이력이라 지우고 4는 정답이라 남긴다
    check("유저0 이력에서 2가 지워졌다", test_out[0]["sequence"] == [0, 1, 3, 4],
          str(test_out[0]["sequence"]))
    #    유저4 [1,3,7,10,4] → 정답 4가 COLD 이므로 남는다. 이력엔 COLD 없음
    check("유저4는 그대로", test_out[4]["sequence"] == [1, 3, 7, 10, 4],
          str(test_out[4]["sequence"]))

    # ④ 🚨 핵심: 남은 자리의 임베딩·텍스트가 그 아이템의 것인가
    misaligned = []
    for record in list(train_out.values()) + list(test_out.values()):
        for position, item in enumerate(record["sequence"]):
            if not np.allclose(record["embedding"][position], CATALOG[item]):
                misaligned.append((record["user_id"], position, item))
            if record["text"][position] != f"item-{item}":
                misaligned.append((record["user_id"], position, item, "text"))
    check("임베딩·텍스트가 아이템과 계속 짝이 맞는다", not misaligned, str(misaligned[:5]))

    # ⑤ 손대지 않은 레코드도 그대로인가 (원본 바이트 복사 경로)
    check("손대지 않은 레코드의 임베딩도 정확하다",
          np.allclose(test_out[4]["embedding"], CATALOG[[1, 3, 7, 10, 4]]))

    # ⑥ 너무 짧아진 시퀀스는 버린다
    strict_dir = os.path.join(root, "out_strict")
    stats = rewrite_split(root, strict_dir, "training", {0, 1, 2, 5, 9},
                          min_length=3, keep_last=False)
    check("min_length 미만은 버린다", stats["records_dropped_too_short"] > 0, str(stats))
    check("버린 만큼 출력이 줄어든다",
          stats["records_out"] == stats["records_in"] - stats["records_dropped_too_short"],
          str(stats))

    # ⑦ verify() 가 정상 데이터셋을 통과시키는가
    result = verify(out_dir, cold_set)
    check("verify 가 통과한다", result["ok"], str(result["problems"]))
    # 정답이 COLD 인 유저: 0(→4), 1(→8), 4(→4). 유저4는 이력이 안 잘렸어도 정답은 COLD 다.
    check("verify 가 COLD 정답 유저 수를 센다", result["cold_target_users"] == 3, str(result))

    # ⑧ verify() 가 오염된 데이터셋을 잡아내는가 (일부러 COLD 를 학습에 되돌린다)
    write_split(out_dir, "training", {9: [4, 0, 1]})
    broken = verify(out_dir, cold_set)
    check("verify 가 오염을 잡아낸다", not broken["ok"], "오염인데 통과했다")

finally:
    shutil.rmtree(root, ignore_errors=True)

print()
if failures:
    print(f"실패 {len(failures)}건: {failures}")
    sys.exit(1)
print("cold_split 검증: 전부 통과")
