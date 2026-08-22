#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split B(Temporal) 윈도우 → GRID/MaskGR 입력 TFRecord 변환.

Split A 와 무엇이 다른가
------------------------
Split A 는 유저당 정답이 **하나**다(loo_test). Split B 는 윈도우마다
`test_targets[user] = [item, item, ...]` 로 **여러 개**다. 그래서 "마지막 아이템이
라벨" 규약을 그대로 쓸 수 없다.

이 스크립트가 택한 방식
-----------------------
유저당 **한 번만 예측**하고, 생성된 후보를 **정답 집합 전체**와 대조한다.

    testing 시퀀스 = train_user_seq + [정답 하나(placeholder)]

붙이는 정답 하나는 **마스크 자리를 만들기 위한 것일 뿐**이다. NextKTokenMasking 이
그 자리를 가리므로 모델은 그 값을 보지 못하고, 생성 결과는 오직 히스토리에만 의존한다.
채점은 덤프의 `label_sid` 가 아니라 split 의 `test_targets` 전체 집합으로 한다
(`evaluate.Evaluator` 는 targets 를 집합으로 받으므로 다중 정답이 그대로 된다).

> ⚠️ 그래서 **모델 내부 지표(val/recall@5)는 우리 지표와 다릅니다.** 내부 지표는
> placeholder 하나만 정답으로 치기 때문이다. Temporal 트랙에서는 내부 지표를 믿지 말 것.

윈도우 스키마 가정
------------------
    windows: [{window_label, train_user_seq, val_targets, test_targets,
               cold_tiers, item_counts}, ...]
전처리 담당이 균등분할 → **밀도 가중 분할**로 바꾸는 중이므로 *내용*은 바뀐다.
이 스크립트는 위 **키 이름**에만 의존한다. 키가 바뀌면 여기만 고치면 된다.

사용:
  python3 to_grid_b.py Beauty_split_B.pkl grid_data/beauty_B          # 전 윈도우
  python3 to_grid_b.py Beauty_split_B.pkl grid_data/beauty_B -w W5    # 하나만
"""
import argparse
import os
import pickle
import sys

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from to_grid_v2 import MAX_LEN, write_items, write_seqs  # noqa: E402


def iter_windows(D):
    """스키마 두 가지를 모두 받아 (라벨, 윈도우dict) 를 순서대로 돌려준다.

    신(2026-08-22 재전처리)  windows = {라벨: {train_seq, val_targets, test_targets, ...}}
    구                        windows = [{window_label, train_user_seq, ...}]
    """
    w = D.get("windows")
    if isinstance(w, dict):
        for lbl in sorted(w):
            yield lbl, w[lbl]
    else:
        for x in (w or []):
            yield x["window_label"], x


def build_window(D, label, w, out_dir, max_len):
    """윈도우 하나 → <out_dir>/<label>/{items,training,evaluation,testing}"""
    uid = D["uid"]
    n_item = len(D["iid"])
    root = os.path.join(out_dir, label)

    tr_seq = w.get("train_seq", w.get("train_user_seq")) or {}
    val_t = w.get("val_targets") or {}
    test_t = w.get("test_targets") or {}

    def to_int(u):
        return u if isinstance(u, int) else uid[u]

    tr, ev, te = {}, {}, {}
    for u, hist in tr_seq.items():
        tr[to_int(u)] = list(hist)

    # evaluation / testing 은 학습 히스토리 + placeholder 정답 하나
    for src, dst in ((val_t, ev), (test_t, te)):
        for u, tgts in src.items():
            ui = to_int(u)
            hist = list(tr_seq.get(u, []))
            tg = list(tgts)
            if not tg:
                continue
            # 히스토리가 없는 유저(콜드 유저)는 시퀀스가 1개뿐이라 write_seqs 가 버린다.
            # 그건 의도한 것 — 히스토리 없이 다음 아이템을 맞히라는 건 다른 과제다.
            dst[ui] = hist + [tg[0]]

    print(f"\n=== {label}")
    print(f"  학습 유저 {len(tr):,} · 검증 유저 {len(ev):,} · 테스트 유저 {len(te):,}")
    n_multi = sum(1 for v in test_t.values() if len(v) > 1)
    print(f"  테스트 정답이 2개 이상인 유저 {n_multi:,} "
          f"(정답 총 {sum(len(v) for v in test_t.values()):,}개)")
    for k in ("n_train_interactions", "n_val_interactions", "n_test_interactions"):
        if k in w:
            print(f"  {k} = {w[k]:,}")
    if len(tr) < 1000:
        print(f"  [경고] 학습 유저가 {len(tr):,}명뿐입니다. 이 윈도우로는 학습이 성립하지 않을 수 있습니다.")

    os.makedirs(root, exist_ok=True)
    write_items(os.path.join(root, "items"), D["item_text"], n_item)
    write_seqs(os.path.join(root, "training"), tr, max_len)
    write_seqs(os.path.join(root, "evaluation"), ev, max_len)
    write_seqs(os.path.join(root, "testing"), te, max_len)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("out")
    ap.add_argument("-w", "--window", default=None, help="윈도우 라벨 하나만 (예: W5)")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    a = ap.parse_args()

    D = pickle.load(open(a.pkl, "rb"))
    if "windows" not in D:
        raise SystemExit("windows 키가 없습니다. Split B pkl 이 맞습니까?")
    wins = list(iter_windows(D))
    print(f"유저 {len(D['uid']):,} / 상품 {len(D['iid']):,} / 윈도우 {len(wins)}개 "
          f"({', '.join(l for l, _ in wins)})")

    made = []
    for lbl, w in wins:
        if a.window and lbl != a.window:
            continue
        made.append(build_window(D, lbl, w, a.out, a.max_len))
    if not made:
        raise SystemExit(f"해당 윈도우를 못 찾았습니다: {a.window}")

    print("\n=== 만든 데이터셋 ===")
    for p in made:
        print(f"  {os.path.abspath(p)}")
    print("\n  → MaskGR: sbatch --export=ALL,TAG=<태그>,SID=<sid.pt>,"
          f"DATA={os.path.abspath(made[0])} maskgr_train.sh")


if __name__ == "__main__":
    main()
