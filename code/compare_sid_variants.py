"""여러 SID(.pt) 결과를 HANDOFF_graph_sid.md 2절과 같은 지표로 나란히 비교한다.

export_sid_csv.py와 달리 Beauty_split_A.pkl이나 고정 경로가 필요 없다 (item_text 조회 없이
sid_tensor.pt 하나만으로 계산되는 요약 통계만 뽑는다). 서버에서 sid_beauty.sh로 만든
결과를 scp로 받아온 뒤 바로 비교할 때 쓴다.

각 variant 디렉터리는 L3/sid_tensor.pt, L4/sid_tensor.pt를 담고 있어야 한다
(sid/, sid/gsid_a03/ 같은 기존 레이아웃과 동일).

    python code/compare_sid_variants.py \
        --variant baseline sid \
        --variant gsid_a03 sid/gsid_a03 \
        --variant gsid_a03_centered sid/gsid_a03_centered
"""

import argparse
from collections import Counter
from pathlib import Path

import torch


def summarize(sid_dir: Path, level: int) -> dict:
    t = torch.load(sid_dir / f"L{level}" / "sid_tensor.pt", map_location="cpu")
    sid = t.T if t.shape[0] == level + 1 else t  # (아이템, 레벨+1) 로 정렬
    n_items = sid.shape[0]

    codes = sid[:, :level]
    dedup = sid[:, level]
    tuples = [tuple(int(x) for x in row) for row in codes]
    cnt = Counter(tuples)
    is_collided = [cnt[tp] > 1 for tp in tuples]

    return {
        "level": level,
        "n_items": n_items,
        "unique_by_code_pct": round(100 * len(cnt) / n_items, 2),
        "collided_items": sum(is_collided),
        "collided_pct": round(100 * sum(is_collided) / n_items, 2),
        "max_collision_group": int(max(cnt.values())),
        "max_dedup_suffix": int(dedup.max()),
        "codes_used_per_level": "/".join(
            str(int(codes[:, lv].unique().numel())) for lv in range(level)
        ),
        "sid_all_unique": len(set(map(tuple, sid.tolist()))) == n_items,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant", nargs=2, action="append", metavar=("LABEL", "DIR"), required=True,
        help="비교할 SID. 여러 번 줄 수 있음. DIR 아래에 L3/, L4/ 가 있어야 함.",
    )
    p.add_argument("--levels", nargs="+", type=int, default=[3, 4])
    args = p.parse_args()

    rows = []
    for label, dir_str in args.variant:
        sid_dir = Path(dir_str)
        for level in args.levels:
            path = sid_dir / f"L{level}" / "sid_tensor.pt"
            if not path.exists():
                print(f"[skip] {label} L{level}: {path} 없음")
                continue
            row = {"variant": label}
            row.update(summarize(sid_dir, level))
            rows.append(row)

    if not rows:
        print("비교할 SID가 하나도 없습니다. --variant 경로를 확인하세요.")
        return

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    except ImportError:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
