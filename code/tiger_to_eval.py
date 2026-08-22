#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TIGER 생성 결과 → 평가 하네스(evaluate.py) 연결 — 트랙 C 최소 목표.

하는 일은 셋뿐이다.
  1) TIGER 가 뱉은 semantic ID 를 item_id 로 되돌린다 (sid_tensor.pt 역매핑)
  2) {user: [item,...]} / {user: {target}} 두 dict 를 만든다
  3) Evaluator 에 넣어 정확도 + 다양성 + 콜드 버킷 지표를 뽑아 JSON 으로 저장한다

베이스라인(results/baseline_*_loo.json)과 같은 형식으로 떨어지므로 한 표에 놓을 수 있다.

사용:
  python tiger_to_eval.py --run clip_L4
  python tiger_to_eval.py --pred <경로.pkl> --label clip_L4_k50 --out results/xxx.json

주의 — 입력 예측 파일이 어떤 경로로 만들어졌는지 반드시 확인할 것.
  src.inference(predict_step) 산출물은 정답을 입력에서 가리지 않아 지표가 부풀려진다.
  --leaky 를 붙이면 결과 JSON 에 경고 플래그가 박힌다 (실수로 발표표에 들어가는 것 방지).
"""
import argparse
import io
import json
import os
import pickle
import sys

import numpy as np

if __name__ == "__main__":      # import 될 때 남의 stdout 을 갈아끼우면 안 된다
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import Evaluator, load_split_a, load_split_b, split_key_type  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------
# 1. SID -> item_id 역매핑
# ---------------------------------------------------------------
def build_sid2item(sid_path):
    """sid_tensor.pt (num_hierarchies, n_items) -> {(코드...): item_id}

    GRID 후처리가 전치해서 저장하므로 .t() 로 되돌려 행=item_id 로 만든다.
    마지막 자리는 충돌 구분자다. 전 자리가 맞아야 같은 아이템이다.
    """
    import torch
    sid = torch.load(sid_path, map_location="cpu", weights_only=False).t()  # (n_items, H)
    n_items, H = sid.shape
    sid2item = {tuple(int(x) for x in sid[i]): i for i in range(n_items)}
    if len(sid2item) != n_items:
        print(f"  [경고] SID 충돌: 고유 SID {len(sid2item):,} != 아이템 {n_items:,}")
    return sid2item, H, n_items


def decode_predictions(pred_path, sid2item, H):
    """merged_predictions.pkl -> ({user_idx: [item,...]}, 통계)

    반환 리스트는 생성 순서(=점수 내림차순)를 그대로 유지한다.
    실제 아이템에 매핑되지 않는 SID(무효 SID)는 버리고 개수를 센다.
    """
    import torch
    with open(pred_path, "rb") as f:
        recs = pickle.load(f)

    preds, labels, n_slot, n_invalid, n_dup, lens = {}, {}, 0, 0, 0, []
    for rec in recs:
        u = rec["user_id"]
        u = int(u.item()) if torch.is_tensor(u) else int(u)
        sids = rec["semantic_ids"]
        if torch.is_tensor(sids):
            sids = sids.view(-1, H)
        seen, lst = set(), []
        for row in sids:
            n_slot += 1
            it = sid2item.get(tuple(int(x) for x in row))
            if it is None:
                n_invalid += 1
                continue
            if it in seen:      # 서로 다른 SID 가 같은 아이템이 될 일은 없어야 하나 방어적으로
                n_dup += 1
                continue
            seen.add(it)
            lst.append(it)
        preds[u] = lst
        lens.append(len(lst))
        if "label_sid" in rec:      # eval_step 덤프에만 있음
            labels[u] = sid2item.get(tuple(int(x) for x in rec["label_sid"].reshape(-1)))

    stats = {
        "n_user_pred": len(preds),
        "n_generated_slot": n_slot,
        "n_slot_per_user": round(n_slot / max(1, len(preds)), 2),
        "n_invalid_sid": n_invalid,
        "invalid_sid_rate": round(n_invalid / max(1, n_slot), 6),
        "n_duplicate_item": n_dup,
        "len_after_filter_mean": round(float(np.mean(lens)), 3) if lens else 0.0,
        "len_after_filter_min": int(min(lens)) if lens else 0,
    }
    return preds, stats, labels


# ---------------------------------------------------------------
# 2. user 인덱스 -> 원본 user 키
# ---------------------------------------------------------------
def remap_users(preds_by_idx, uid, key_type=str):
    """예측(정수 user_id 키)을 split pkl 이 쓰는 키 자료형으로 맞춘다.

    2026-08-22 재전처리에서 split 의 유저 키가 reviewerID 문자열 -> 정수 user_id 로
    바뀌었다. 예전 pkl 도 계속 열려야 하므로 key_type 을 받아 분기한다.
      key_type is int -> 예측 키를 그대로 쓴다 (uid 에 있는 것만 남긴다)
      key_type is str -> 예전처럼 원본 문자열로 되돌린다
    """
    out, missing = {}, 0
    if key_type is int:
        valid = set(uid.values())
        for u, lst in preds_by_idx.items():
            if u not in valid:
                missing += 1
                continue
            out[u] = lst
        return out, missing
    idx2raw = {v: k for k, v in uid.items()}
    for u, lst in preds_by_idx.items():
        raw = idx2raw.get(u)
        if raw is None:
            missing += 1
            continue
        out[raw] = lst
    return out, missing


# ---------------------------------------------------------------
# 3. 평가
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="clip_L4",
                    help="tiger_runs/<TAG>/infer/merged_predictions.pkl 를 읽는다")
    ap.add_argument("--pred", default=None, help="예측 pkl 경로 직접 지정 (--run 무시)")
    ap.add_argument("--sid", default=os.path.join(ROOT, "sid", "L4", "sid_tensor.pt"))
    ap.add_argument("--split", default=os.path.join(ROOT, "Beauty_split_A.pkl"))
    ap.add_argument("--window", default=None,
                    help="Temporal 트랙: Split B 의 윈도우 라벨 (예: W5). 주면 Split B 로 읽는다")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--K", default="10,20,50")
    ap.add_argument("--exclude-seen", action="store_true",
                    help="학습 이력에 있는 아이템을 추천에서 제외 (고전 베이스라인과 같은 규칙)")
    ap.add_argument("--leaky", action="store_true",
                    help="predict_step 산출물처럼 정답 누출이 있는 예측임을 결과에 명시")
    a = ap.parse_args()

    pred_path = a.pred or os.path.join(ROOT, "tiger_runs", a.run, "infer", "merged_predictions.pkl")
    label = a.label or a.run
    track = "temporal" if a.window else "loo"
    suffix = f"_{a.window}" if a.window else ""
    out_path = a.out or os.path.join(ROOT, "results", f"tiger_{label}{suffix}_{track}.json")
    K = tuple(int(x) for x in a.K.split(","))

    print("=== TIGER -> 평가 하네스 ===")
    print(f"  예측 : {pred_path}")
    print(f"  SID  : {a.sid}")
    print(f"  분할 : {a.split}" + (f"  (윈도우 {a.window} · Temporal 트랙)" if a.window else "  (LOO 트랙)"))

    sid2item, H, n_items_sid = build_sid2item(a.sid)
    print(f"  SID {H}자리 · 아이템 {n_items_sid:,}개")

    preds_idx, stats, labels_idx = decode_predictions(pred_path, sid2item, H)
    print(f"  유저 {stats['n_user_pred']:,}명 · 유저당 생성 {stats['n_slot_per_user']}개")
    print(f"  무효 SID {stats['n_invalid_sid']:,}개 ({100 * stats['invalid_sid_rate']:.3f}%) 제거"
          f" -> 필터 후 평균 길이 {stats['len_after_filter_mean']}"
          f" (최소 {stats['len_after_filter_min']})")

    with open(a.split, "rb") as f:
        A = pickle.load(f)
    ktype = split_key_type(A)
    print(f"  유저 키 : {ktype.__name__}"
          + ("  (재전처리 이후 정수 user_id)" if ktype is int else "  (구 스키마 reviewerID 문자열)"))
    preds, missing = remap_users(preds_idx, A["uid"], ktype)
    if missing:
        print(f"  [경고] split 에 없는 user_id {missing:,}개 무시")

    cold_tiers = None
    if a.window:
        tc, hist, tgt, n_items, cold_tiers = load_split_b(a.split, a.window)
        n_tgt = sum(len(v) for v in tgt.values())
        print(f"  윈도우 {a.window}: 학습 유저 {len(hist):,} · 채점 유저 {len(tgt):,} "
              f"· 정답 {n_tgt:,}개 (유저당 {n_tgt / max(1, len(tgt)):.1f})")
    else:
        tc, hist, tgt, n_items = load_split_a(a.split)
    assert n_items == n_items_sid, f"아이템 수 불일치: split {n_items} vs SID {n_items_sid}"

    # 덤프에 정답 SID 가 함께 들어 있으면, 그것이 split 의 test 타깃과 같은지 대조한다.
    # 이게 맞아야 "모델이 본 정답"과 "우리가 채점하는 정답"이 같은 것이 증명된다.
    idx2raw = {v: k for k, v in A["uid"].items()} if ktype is str else None
    if labels_idx and a.window:
        print("  덤프 정답 SID 대조: Temporal 트랙에서는 건너뜁니다 — testing 시퀀스에 붙인 "
              "정답은 마스크 자리를 만들기 위한 placeholder 하나뿐이라 정답 집합과 1:1 대응하지 않습니다.")
    elif labels_idx:
        n_chk = n_ok = 0
        for u, lab in labels_idx.items():
            raw = u if ktype is int else idx2raw.get(u)
            if raw is None or lab is None:
                continue
            n_chk += 1
            n_ok += int(lab in tgt.get(raw, ()))
        stats["label_match_checked"] = n_chk
        stats["label_match_rate"] = round(n_ok / max(1, n_chk), 6)
        mark = "OK" if n_ok == n_chk else "불일치 있음 — 조인 키나 마스킹을 의심할 것"
        print(f"  덤프 정답 SID 대조: {n_ok:,}/{n_chk:,} 일치 ({mark})")

    if a.exclude_seen:
        before = sum(len(v) for v in preds.values())
        preds = {u: [i for i in lst if i not in set(hist.get(u, ()))] for u, lst in preds.items()}
        after = sum(len(v) for v in preds.values())
        stats["n_seen_removed"] = before - after
        stats["seen_rate"] = round((before - after) / max(1, before), 6)
        print(f"  이미 본 아이템 {before - after:,}개 제거 ({100 * stats['seen_rate']:.2f}%)")
    else:
        tot = sum(len(v) for v in preds.values())
        seen_hit = sum(1 for u, lst in preds.items() for i in lst if i in set(hist.get(u, ())))
        stats["n_seen_in_list"] = seen_hit
        stats["seen_rate"] = round(seen_hit / max(1, tot), 6)
        print(f"  (참고) 추천 목록 중 이미 본 아이템 {seen_hit:,}개"
              f" ({100 * stats['seen_rate']:.2f}%) — 제외하지 않음")

    maxK = max(K)
    stats["truncated_at"] = stats["n_slot_per_user"]
    if stats["n_slot_per_user"] < maxK:
        print(f"  [경고] 유저당 생성이 {stats['n_slot_per_user']}개뿐인데 @{maxK} 를 요구했다."
              f" @{maxK} 지표는 사실상 @{int(stats['n_slot_per_user'])} 값이며 과소평가된다."
              f" model.top_k_for_generation 을 {maxK} 이상으로 올려 재생성할 것.")

    ev = Evaluator(tc, n_items, tail_frac=0.5)
    m = ev.evaluate(preds, tgt, K=K)
    m["label"] = label
    m["_source"] = {
        "pred_file": os.path.relpath(pred_path, ROOT).replace("\\", "/"),
        "sid_file": os.path.relpath(a.sid, ROOT).replace("\\", "/"),
        "split_file": os.path.basename(a.split),
        "track": track,
        "window": a.window,
        "exclude_seen": bool(a.exclude_seen),
        "tail_frac": 0.5,
        **stats,
    }

    # 전처리 담당이 정의한 콜드 등급이 윈도우에 들어 있으면 그 기준으로도 한 번 더 쪼갠다.
    # 우리 BUCKETS(학습 등장 횟수)와는 다른 정의라 섞지 않고 따로 보고한다.
    if cold_tiers:
        from collections import Counter as _C
        tier_hit = {}
        topk = max(K)
        for u, gts in tgt.items():
            top = set(preds.get(u, [])[:topk])
            for g in gts:
                t = cold_tiers.get(g, "unknown")
                d = tier_hit.setdefault(t, [0, 0])
                d[1] += 1
                d[0] += int(g in top)
        m[f"tier_recall@{topk}"] = {t: {"n": d[1], "recall": round(d[0] / max(1, d[1]), 6)}
                                    for t, d in sorted(tier_hit.items())}
        print("")
        print(f"  콜드 등급별 recall@{topk} (전처리 담당 정의):")
        for t, d in sorted(tier_hit.items()):
            print(f"    {t:<10s} n={d[1]:>6,}  recall={d[0] / max(1, d[1]):.4f}")
    if a.leaky:
        m["_INVALID"] = ("predict_step 경로 산출물 — 정답이 입력에서 가려지지 않아 정확도가 "
                         "부풀려짐. 정확도 지표는 발표에 쓰지 말 것.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({label: m}, f, ensure_ascii=False, indent=1)

    print(f"\n=== 결과 ({label}) ===")
    for k in K:
        print(f"  @{k:<3d} recall {m[f'recall@{k}']:.4f}  ndcg {m[f'ndcg@{k}']:.4f}"
              f"  | APLT {m[f'aplt@{k}']:.4f}  cov {m[f'coverage@{k}']:.4f}"
              f"  gini {m[f'exposure_gini@{k}']:.4f}  tailExp {m[f'tail_exposure@{k}']:.4f}")
    print(f"  COLD_recall@{maxK} = {m[f'COLD_recall@{maxK}']:.4f}  (n={m['COLD_n_target']:,})")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
