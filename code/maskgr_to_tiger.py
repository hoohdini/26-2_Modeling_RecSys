#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaskGR 덤프를 TIGER 규약으로 변환한다.

왜 필요한가
-----------
두 덤프는 껍데기가 같다 — list[dict] 이고 키도 user_id / semantic_ids / label_sid / scores
로 동일하다. 그런데 **코드 값의 좌표계가 다르다.**

    TIGER   semantic_ids 범위 [0, 255]      계층별 원본 코드
    MaskGR  semantic_ids 범위 [0, 1284]     계층 오프셋이 더해진 값 (h * 257 + c)

MaskGR 은 마스크 확산이라 계층마다 코드 256 개 + 마스크 토큰 1 개 = **257 칸**을 쓴다.
그래서 stride 가 256 이 아니라 257 이다. (GRID/TIGER 는 마스크 토큰이 없어 256)

`tiger_to_eval.py` 는 TIGER 좌표계를 가정하므로, 변환 없이 넣으면 모든 SID 가 조회에
실패해 invalid_sid_rate = 1.0 · 전 지표 0 이 된다.

검증
----
실측(mg_text_B_W5): 오프셋 arange(H)*257 을 빼면 전 값이 [0,255] 에 들어가고,
**공통 유저 7,264 명 전원의 label_sid 가 TIGER 덤프와 정확히 일치**한다.
stride 256 · 258 은 각각 489/500, 500/500 이 범위를 벗어나 반증된다.

사용
----
    python maskgr_to_tiger.py <입력.pkl> <출력.pkl> [--stride 257]
"""
import argparse
import pickle
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--stride", type=int, default=257,
                    help="계층당 어휘 칸 수. MaskGR 은 256 코드 + 마스크 1 = 257")
    ap.add_argument("--width", type=int, default=256,
                    help="계층당 실제 코드 수. 변환 후 이 범위를 벗어나면 실패로 본다")
    a = ap.parse_args()

    rows = pickle.load(open(a.src, "rb"))
    if not isinstance(rows, list) or not rows:
        sys.exit(f"예상과 다른 덤프 구조: {type(rows).__name__}")

    H = rows[0]["semantic_ids"].shape[-1]
    off = torch.arange(H) * a.stride
    print(f"유저 {len(rows):,}명 · 계층 {H} · stride {a.stride}")

    bad = 0
    out = []
    for e in rows:
        sid = e["semantic_ids"] - off          # (C, H)
        lab = e["label_sid"] - off             # (H,)
        if not ((sid >= 0).all() and (sid < a.width).all()):
            bad += 1
        out.append({**e, "semantic_ids": sid, "label_sid": lab})

    if bad:
        sys.exit(f"변환 실패: {bad}명의 코드가 [0,{a.width}) 를 벗어납니다. "
                 f"stride 가 {a.stride} 가 맞는지 확인하세요.")

    lo = min(int(e["semantic_ids"].min()) for e in out)
    hi = max(int(e["semantic_ids"].max()) for e in out)
    print(f"변환 후 코드 범위 [{lo}, {hi}]  (기대: [0, {a.width - 1}])")

    pickle.dump(out, open(a.dst, "wb"))
    print("저장:", a.dst)


if __name__ == "__main__":
    main()
