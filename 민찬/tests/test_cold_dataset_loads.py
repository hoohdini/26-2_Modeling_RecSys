"""만들어 둔 COLD 데이터셋이 GRID 가 읽는 방식 그대로 읽히는지 확인한다.

왜 따로 두나
    tests/test_cold_split.py 는 자르는 로직이 맞는지를 합성 데이터로 본다.
    이 파일은 **실제로 만든 데이터셋**이 GRID 의 파서를 통과하는지를 본다.
    둘은 다른 질문이다. 로직이 맞아도 스키마가 어긋나면 학습이 시작조차 안 된다.

GRID 가 하는 일 (src/data/loading/components/iterators.py)
    ① 첫 레코드를 풀어 필드 종류를 추론한다      infer_feature_type
    ② 그 스키마로 tf.io.parse_example 을 돌린다   iter_batches
    같은 절차를 여기 옮겨 놨다. GRID 모듈을 직접 import 하면 순환 import 로
    죽는다(GRID 자체 문제이고, 정식 진입점으로 돌릴 때는 안 나타난다).

⚠️ GRID 프로세스를 새로 띄우지 않는다. 돌아가는 실험과 로그 폴더를 두고
   경합하지 않기 위해서다 — sweep_layers.newest() 주석 참고.

사용:
    python tests/test_cold_dataset_loads.py                     # 기본 두 데이터셋
    python tests/test_cold_dataset_loads.py work/cold/beauty_strat10
"""

import glob
import json
import os
import sys

import tensorflow as tf

EMBEDDING_DIM = 768
DEFAULT_DATASETS = ["work/cold/beauty_strat10", "work/cold/beauty_tail10"]

failures = []


def infer_feature_type(example_proto) -> dict:
    """GRID 의 infer_feature_type 과 같은 규칙."""
    description = {}
    for key, value in example_proto.items():
        if value.HasField("bytes_list"):
            description[key] = tf.io.RaggedFeature(tf.string)
        elif value.HasField("float_list"):
            description[key] = tf.io.RaggedFeature(tf.float32)
        elif value.HasField("int64_list"):
            description[key] = tf.io.RaggedFeature(tf.int64)
    return description


def check_split(data_dir: str, split: str, cold: set[int]) -> None:
    files = sorted(glob.glob(os.path.join(data_dir, split, "*.tfrecord.gz")))
    if not files:
        failures.append(f"{data_dir}/{split}: 파일 없음")
        return

    raw = tf.data.TFRecordDataset(files, compression_type="GZIP")
    example = tf.train.Example()
    example.ParseFromString(next(iter(raw)).numpy())
    spec = infer_feature_type(example.features.feature)

    expected = {"sequence_data", "text", "embedding", "user_id"}
    if set(spec) != expected:
        failures.append(f"{data_dir}/{split}: 필드가 다름 {sorted(spec)}")
        return

    # COLD 아이템 판정을 텐서 연산으로 한다. 파이썬 반복으로는 레코드가 2만 개라 느리다.
    cold_tensor = tf.constant(sorted(cold), dtype=tf.int64)
    is_testing = split == "testing"

    n_records = n_bad_length = n_leaked = n_cold_target = 0

    for batch in raw.ragged_batch(512, drop_remainder=False):
        parsed = tf.io.parse_example(batch, spec)
        sequences = parsed["sequence_data"]
        lengths = sequences.row_lengths()
        n_records += int(lengths.shape[0])

        # 세 필드의 길이가 서로 맞는가: text = L, embedding = L × 768
        n_bad_length += int(tf.reduce_sum(tf.cast(
            (parsed["text"].row_lengths() != lengths)
            | (parsed["embedding"].row_lengths() != lengths * EMBEDDING_DIM),
            tf.int32,
        )))

        flat = sequences.flat_values
        in_cold = tf.reduce_any(tf.equal(flat[:, None], cold_tensor[None, :]), axis=1)
        in_cold = tf.RaggedTensor.from_row_lengths(in_cold, lengths)

        if is_testing:
            # 정답(마지막)은 COLD 여도 되고, 그 앞 이력에는 있으면 안 된다
            n_cold_target += int(tf.reduce_sum(tf.cast(in_cold[:, -1:].flat_values, tf.int32)))
            n_leaked += int(tf.reduce_sum(tf.cast(
                tf.reduce_any(in_cold[:, :-1].to_tensor(False), axis=1), tf.int32)))
        else:
            n_leaked += int(tf.reduce_sum(tf.cast(in_cold.flat_values, tf.int32)))

    ok = n_bad_length == 0 and n_leaked == 0 and (not is_testing or n_cold_target > 0)
    extra = f"  COLD정답 {n_cold_target:,}" if is_testing else ""
    print(f"  {'OK  ' if ok else 'FAIL'} {split:<11} 레코드 {n_records:>6,}  "
          f"필드길이오류 {n_bad_length}  COLD유출 {n_leaked}{extra}")
    if not ok:
        failures.append(f"{data_dir}/{split}")


def main() -> None:
    datasets = sys.argv[1:] or DEFAULT_DATASETS
    for data_dir in datasets:
        meta_path = os.path.join(data_dir, "cold_items.json")
        if not os.path.exists(meta_path):
            print(f"건너뜀 (아직 안 만듦): {data_dir}")
            print("  먼저: python scripts/cold_split.py --data-dir data/amazon_data/beauty "
                  f"--out-dir {data_dir}")
            continue
        with open(meta_path) as f:
            cold = set(json.load(f)["cold_items"])
        print(f"===== {data_dir}  (COLD {len(cold):,}개) =====")
        for split in ["training", "evaluation", "testing"]:
            check_split(data_dir, split, cold)

    print()
    if failures:
        print(f"실패 {len(failures)}건: {failures}")
        sys.exit(1)
    print("COLD 데이터셋이 GRID 파싱 방식으로 정상적으로 읽힙니다.")


if __name__ == "__main__":
    main()
