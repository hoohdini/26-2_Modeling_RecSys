#!/usr/bin/env python3
"""
팀원 전처리 pkl(Beauty_split_A.pkl) → GRID 입력 TFRecord 변환

기존 to_grid.py와 동일한 출력 규격이지만, 입력 스키마가 다르다:
  - 구버전(build/Beauty.pkl): sequences/texts/n_item/temporal ...
  - 신버전(Beauty_split_A.pkl): uid/iid/item_text/loo_train/loo_val/loo_test (max_seq_len=20)

출력 (GRID configs 기준 폴더명 — training/evaluation/testing):
  <out>/items/       id(int64), text(bytes)
  <out>/training/    user_id(int64), sequence_data(int64[])  ← loo_train (마지막이 라벨)
  <out>/evaluation/  loo_train + [loo_val]                   ← 마지막(val)이 라벨
  <out>/testing/     loo_train + [loo_val, loo_test]         ← 마지막(test)이 라벨
  전부 GZIP 압축 TFRecord

사용:
  python3 to_grid_v2.py Beauty_split_A.pkl grid_data/beauty_A
"""
import os, sys, pickle, argparse
import tensorflow as tf

MAX_LEN = 30  # 아이템 개수 상한. SID 4단 기준 토큰 120개 = GRID 기본 sequence_length


def _int64(v):  return tf.train.Feature(int64_list=tf.train.Int64List(value=v))
def _bytes(v):  return tf.train.Feature(bytes_list=tf.train.BytesList(value=v))


def write_items(path, texts, n_items):
    os.makedirs(path, exist_ok=True)
    opt = tf.io.TFRecordOptions(compression_type="GZIP")
    n = 0
    with tf.io.TFRecordWriter(os.path.join(path, "items.tfrecord.gz"), opt) as w:
        for i in range(n_items):
            t = (texts.get(i) or "").strip() or f"item {i}"
            ex = tf.train.Example(features=tf.train.Features(feature={
                "id":   _int64([i]),
                "text": _bytes([t.encode("utf-8")]),
            }))
            w.write(ex.SerializeToString()); n += 1
    print(f"  items/      {n:,} 건")


def write_seqs(path, seqs, max_len=MAX_LEN):
    """seqs: {user_int_id: [item_id,...]} — 마지막 아이템이 정답(라벨)"""
    os.makedirs(path, exist_ok=True)
    opt = tf.io.TFRecordOptions(compression_type="GZIP")
    n = 0
    with tf.io.TFRecordWriter(os.path.join(path, "part-0.tfrecord.gz"), opt) as w:
        for u, items in seqs.items():
            s = list(items)[-max_len:]
            if len(s) < 2: continue
            ex = tf.train.Example(features=tf.train.Features(feature={
                "user_id":       _int64([int(u)]),
                "sequence_data": _int64([int(x) for x in s]),
            }))
            w.write(ex.SerializeToString()); n += 1
    print(f"  {os.path.basename(path)+'/':12s}{n:,} 건")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl"); ap.add_argument("out")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    a = ap.parse_args()

    D = pickle.load(open(a.pkl, "rb"))
    uid, iid = D["uid"], D["iid"]
    n_item = len(iid)
    print(f"유저 {len(uid):,} / 상품 {n_item:,} / max_seq_len(pkl)={D.get('max_seq_len')}")

    # 재전처리(2026-08-22) 이후 loo_* 의 키가 정수 user_id 다. 예전 pkl(문자열 키)도
    # 계속 열려야 하므로 자료형을 보고 분기한다.
    tr, ev, te = {}, {}, {}
    for u_str, hist in D["loo_train"].items():
        u = u_str if isinstance(u_str, int) else uid[u_str]
        v, t = D["loo_val"][u_str], D["loo_test"][u_str]
        tr[u] = list(hist)
        ev[u] = list(hist) + [v]
        te[u] = list(hist) + [v, t]

    os.makedirs(a.out, exist_ok=True)
    write_items(os.path.join(a.out, "items"), D["item_text"], n_item)
    write_seqs(os.path.join(a.out, "training"),   tr, a.max_len)
    write_seqs(os.path.join(a.out, "evaluation"), ev, a.max_len)
    write_seqs(os.path.join(a.out, "testing"),    te, a.max_len)

    # 검증 — 되읽어서 스키마·건수 확인
    for sub, feat in [("items", ("id", "text")),
                      ("training", ("user_id", "sequence_data")),
                      ("evaluation", ("user_id", "sequence_data")),
                      ("testing", ("user_id", "sequence_data"))]:
        d = os.path.join(a.out, sub)
        fs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".tfrecord.gz")]
        cnt = 0
        first = None
        for r in tf.data.TFRecordDataset(fs, compression_type="GZIP"):
            if first is None:
                ex = tf.train.Example(); ex.ParseFromString(r.numpy())
                first = {k: ("int64" if v.int64_list.value else "bytes")
                         for k, v in ex.features.feature.items()}
            cnt += 1
        assert set(first) == set(feat), f"{sub}: 스키마 불일치 {first}"
        print(f"  ✔ {sub}: {cnt:,} 건, 스키마 {first}")

    print(f"\n  → GRID 실행: python -m src.inference experiment=sem_embeds_inference_flat data_dir={os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
