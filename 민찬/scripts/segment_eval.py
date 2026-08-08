"""구간별(HEAD/BODY/TAIL) 추천 성능 평가 — 팀 공통 자.

왜 필요한가
    전체 평균 하나로는 "비인기 아이템이 더 잘 추천되는가"를 볼 수 없다.
    Ghost 논문이 그 예다. 비인기 노출을 17배 늘렸는데 전체 정확도는 7% 떨어져서,
    평균만 보면 "성능이 나빠졌다"로만 읽힌다.

무엇을 재는가
    정확도  구간별 HR@k, NDCG@k  (+ 표본 수 n 을 반드시 같이 낸다)
    노출    구간별 추천 횟수, Coverage, Gini
    진단    환각률, 코드북 활용률, SID 충돌률

두 가지 "구간"을 구분한다
    정확도는 정답 아이템이 어느 구간인지로 나눈다  (그 구간 유저를 맞혔나)
    노출은  추천된 아이템이 어느 구간인지로 나눈다  (그 구간을 얼마나 보여줬나)
    둘을 섞으면 해석이 무너진다.

GRID 에 묶여 있지 않다. 아래 4개만 있으면 어떤 구현이든 같은 표가 나온다.
    ① SID 맵     아이템 id → semantic ID
    ② 예측       유저 id → 추천 SID 상위 k개
    ③ 정답       유저 id → 정답 아이템 id
    ④ 구간 정의  item_segments.py 의 출력

사용:
    python scripts/segment_eval.py \
        --sid-map   <step4>/merged_predictions_tensor.pt \
        --predictions <step6>/merged_predictions.pkl \
        --segments  work/segments/beauty.json \
        --data-dir  data/amazon_data/beauty
"""

import argparse
import glob
import json
import os
import pickle
from collections import Counter

import numpy as np
import torch

SEGMENT_ORDER = ["HEAD", "BODY", "TAIL", "UNSEEN"]


# ─────────────────────────────── 입력 읽기 ───────────────────────────────


def load_sid_map(path: str) -> np.ndarray:
    """아이템 id → SID. (N, H) 형태로 돌려준다.

    GRID 4단계는 후처리에서 텐서를 전치해 (H, N) 으로 저장한다. H(자릿수)는
    보통 3~8, N(아이템 수)은 수천~수만이라 짧은 쪽이 자릿수다.
    """
    tensor = torch.load(path, weights_only=False)
    array = tensor.numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if array.ndim != 2:
        raise ValueError(f"SID 맵이 2차원이 아닙니다: {array.shape}")
    if array.shape[0] < array.shape[1]:
        array = array.T
    return array.astype(np.int64)


def load_predictions(path: str, key: str = "user_id", value: str = "semantic_ids") -> dict:
    """유저 id → 추천 SID 목록 (상위 k개, 각각 길이 H)."""
    with open(path, "rb") as f:
        rows = pickle.load(f)

    predictions = {}
    for row in rows:
        user_id = int(np.asarray(row[key]).reshape(-1)[0])
        beams = np.asarray(row[value])
        # (k, H) 로 맞춘다. 모델이 (k*H,) 로 납작하게 뱉는 경우도 있다.
        if beams.ndim == 1:
            raise ValueError(
                f"추천 결과가 1차원입니다({beams.shape}). (k, H) 형태가 필요합니다."
            )
        predictions[user_id] = beams.reshape(beams.shape[0], -1).astype(np.int64)
    return predictions


def load_targets(data_dir: str, split: str = "testing") -> dict:
    """유저 id → 정답 아이템 id (시퀀스의 마지막 아이템)."""
    import tensorflow as tf

    files = sorted(glob.glob(os.path.join(data_dir, split, "*.tfrecord.gz")))
    if not files:
        raise FileNotFoundError(f"tfrecord 파일이 없습니다: {data_dir}/{split}")

    targets = {}
    dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")
    for raw in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        feature = example.features.feature
        user_id = int(feature["user_id"].int64_list.value[0])
        sequence = list(feature["sequence_data"].int64_list.value)
        targets[user_id] = int(sequence[-1])
    return targets


# ─────────────────────────────── 지표 ───────────────────────────────


def gini(counts: np.ndarray) -> float:
    """노출 쏠림 정도. 0 = 완전 균등, 1 = 한 아이템이 독점."""
    if counts.size == 0 or counts.sum() == 0:
        return 0.0
    sorted_counts = np.sort(counts.astype(np.float64))
    n = sorted_counts.size
    index = np.arange(1, n + 1)
    return float((2 * (index * sorted_counts).sum()) / (n * sorted_counts.sum()) - (n + 1) / n)


def dcg_at_rank(rank: int) -> float:
    """정답이 1개일 때의 NDCG. rank 는 0부터 센다."""
    return 1.0 / np.log2(rank + 2)


def evaluate(
    predictions: dict,
    targets: dict,
    sid_map: np.ndarray,
    segments: dict,
    ks: list[int],
) -> dict:
    n_items, n_digits = sid_map.shape

    # SID(자릿수 튜플) → 아이템 id. 4단계에서 중복 제거 자릿수를 붙였으므로
    # 카탈로그 안에서는 1:1 이다.
    sid_to_item = {tuple(sid_map[i].tolist()): i for i in range(n_items)}

    max_k = max(ks)
    hits = {k: Counter() for k in ks}          # 구간 → 맞힌 유저 수
    ndcg = {k: Counter() for k in ks}          # 구간 → NDCG 합
    n_users = Counter()                        # 구간 → 평가한 유저 수
    exposure = np.zeros(n_items, dtype=np.int64)   # 아이템별 추천 횟수 (상위 max_k)
    n_generated = 0
    n_hallucinated = 0
    used_codewords = [set() for _ in range(n_digits)]

    skipped = 0
    for user_id, target in targets.items():
        beams = predictions.get(user_id)
        if beams is None:
            skipped += 1
            continue

        segment = segments.get(target, "UNSEEN")
        n_users[segment] += 1

        ranked_items = []
        for beam in beams[:max_k]:
            n_generated += 1
            for digit, token in enumerate(beam):
                used_codewords[digit].add(int(token))
            item = sid_to_item.get(tuple(beam.tolist()))
            if item is None:
                # 카탈로그에 없는 SID = 환각. 순위는 소비하되 노출에는 안 잡힌다.
                n_hallucinated += 1
                ranked_items.append(None)
            else:
                ranked_items.append(item)
                exposure[item] += 1

        for k in ks:
            for rank, item in enumerate(ranked_items[:k]):
                if item == target:
                    hits[k][segment] += 1
                    ndcg[k][segment] += dcg_at_rank(rank)
                    break

    # ── 구간별 표 ──
    rows = []
    for name in SEGMENT_ORDER + ["ALL"]:
        if name == "ALL":
            n = sum(n_users.values())
            seg_items = list(range(n_items))
        else:
            n = n_users[name]
            seg_items = [i for i, s in segments.items() if s == name]

        seg_exposure = exposure[seg_items] if seg_items else np.zeros(0, dtype=np.int64)
        row = {
            "segment": name,
            "n_users": n,
            "n_items": len(seg_items),
            "exposure": int(seg_exposure.sum()),
            "coverage": float((seg_exposure > 0).mean()) if len(seg_items) else 0.0,
        }
        for k in ks:
            hit = hits[k][name] if name != "ALL" else sum(hits[k].values())
            gain = ndcg[k][name] if name != "ALL" else sum(ndcg[k].values())
            row[f"HR@{k}"] = hit / n if n else 0.0
            row[f"NDCG@{k}"] = gain / n if n else 0.0
        rows.append(row)

    # ── 진단 ──
    catalog_codewords = [len(set(sid_map[:, d].tolist())) for d in range(n_digits)]
    unique_prefix = len({tuple(sid_map[i, :-1].tolist()) for i in range(n_items)})

    diagnostics = {
        "hallucination_rate": n_hallucinated / n_generated if n_generated else 0.0,
        "n_generated": n_generated,
        "n_hallucinated": n_hallucinated,
        "gini_exposure": gini(exposure),
        "codebook_used_generated": [len(s) for s in used_codewords],
        "codebook_used_catalog": catalog_codewords,
        "sid_collision_rate": 1.0 - unique_prefix / n_items,
        "users_evaluated": sum(n_users.values()),
        "users_without_prediction": skipped,
    }
    return {"rows": rows, "diagnostics": diagnostics, "ks": ks}


# ─────────────────────────────── 출력 ───────────────────────────────


def render(result: dict) -> str:
    ks = result["ks"]
    primary = max(ks)
    lines = []
    header = f"{'구간':<7}{'n':>8}{'HR@'+str(primary):>10}{'NDCG@'+str(primary):>11}{'노출수':>10}{'Coverage':>10}"
    rule = "─" * 56
    lines.append(header)
    lines.append(rule)
    for row in result["rows"]:
        if row["segment"] == "ALL":
            lines.append(rule)
        lines.append(
            f"{row['segment']:<7}{row['n_users']:>8,}{row[f'HR@{primary}']:>10.4f}"
            f"{row[f'NDCG@{primary}']:>11.4f}{row['exposure']:>10,}{row['coverage']:>10.1%}"
        )

    d = result["diagnostics"]
    lines.append("")
    lines.append(f"Gini(노출 쏠림):  {d['gini_exposure']:.3f}   (0=균등, 1=독점)")
    lines.append(
        f"환각률:           {d['hallucination_rate']:.2%}"
        f"   ({d['n_hallucinated']:,}/{d['n_generated']:,} — 카탈로그에 없는 SID)"
    )
    lines.append(f"SID 충돌률:       {d['sid_collision_rate']:.2%}   (중복 제거 자릿수 빼고 봤을 때)")
    lines.append(f"코드북 활용(생성): {d['codebook_used_generated']}")
    lines.append(f"코드북 활용(카탈로그): {d['codebook_used_catalog']}")
    if d["users_without_prediction"]:
        lines.append(f"⚠️ 추천 결과가 없는 유저 {d['users_without_prediction']:,}명은 제외했습니다")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid-map", required=True, help="3단계 출력 merged_predictions_tensor.pt")
    parser.add_argument("--predictions", required=True, help="추론 출력 merged_predictions.pkl")
    parser.add_argument("--segments", required=True, help="item_segments.py 의 출력 .json")
    parser.add_argument("--data-dir", required=True, help="정답을 읽을 데이터 폴더")
    parser.add_argument("--split", default="testing", help="정답 split (기본 testing)")
    parser.add_argument("--ks", default="5,10", help="쉼표로 구분한 k 목록")
    parser.add_argument("--out", default=None, help="결과를 저장할 .json (선택)")
    args = parser.parse_args()

    ks = [int(k) for k in args.ks.split(",")]

    with open(args.segments) as f:
        segments_file = json.load(f)
    segments = {int(k): v for k, v in segments_file["segments"].items()}

    sid_map = load_sid_map(args.sid_map)
    predictions = load_predictions(args.predictions)
    targets = load_targets(args.data_dir, args.split)

    result = evaluate(predictions, targets, sid_map, segments, ks)
    print(render(result))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
