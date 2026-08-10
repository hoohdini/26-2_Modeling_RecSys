"""구간별(HEAD/BODY/TAIL) 추천 성능 평가 — 팀 공통 자.

왜 필요한가
    전체 평균 하나로는 "비인기 아이템이 더 잘 추천되는가"를 볼 수 없다.
    Ghost 논문이 그 예다. 비인기 노출을 17배 늘렸는데 전체 정확도는 7% 떨어져서,
    평균만 보면 "성능이 나빠졌다"로만 읽힌다.

무엇을 재는가
    정확도  구간별 HR@k, NDCG@k  (+ 표본 수 n 과 95% 신뢰구간을 반드시 같이 낸다)
    노출    구간별 추천 횟수, 노출 배율(lift), Coverage, Gini
    진단    환각률, 코드북 활용률, SID 충돌률

🚨 신뢰구간을 왜 같이 내는가
    TAIL/COLD 는 표본이 HEAD 의 1/3 이하다. 같은 0.01 차이라도 HEAD 에서는
    확실한 차이지만 TAIL 에서는 우연일 수 있다. 신뢰구간이 없으면
    "비인기가 좋아졌다"를 잡음에서 구분할 수 없다.
    두 실행을 비교할 때는 --compare 를 쓴다. 같은 유저를 짝지어 재므로
    각각의 신뢰구간을 눈으로 겹쳐 보는 것보다 훨씬 민감하다.

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
    # 한 실행 평가
    python scripts/segment_eval.py \
        --sid-map   <step4>/merged_predictions_tensor.pt \
        --predictions <step6>/merged_predictions.pkl \
        --segments  work/segments/beauty.json \
        --data-dir  data/amazon_data/beauty \
        --out work/eval/L3.json

    # 두 실행 비교 (같은 유저를 짝지어 검정)
    python scripts/segment_eval.py --compare work/eval/L3.json work/eval/L4.json
"""

import argparse
import glob
import json
import os
import pickle
from collections import Counter

import numpy as np
import torch

# COLD 는 cold_split.py 로 일부러 학습에서 빼낸 신규 아이템,
# UNSEEN 은 우연히 학습에 안 나온 아이템이다. 절대 합치지 않는다.
SEGMENT_ORDER = ["HEAD", "BODY", "TAIL", "COLD", "UNSEEN"]
N_BOOTSTRAP = 2000


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


# 층 하나가 이 비트 미만이면 "무너졌다"고 본다.
# 코드워드 256개를 고르게 쓰면 8비트. 4비트면 실질적으로 16개만 쓰는 셈이다.
COLLAPSE_BITS = 4.0


def level_entropy(column: np.ndarray) -> float:
    """SID 한 자리가 실제로 담고 있는 정보량(비트).

    코드워드를 몇 개 "썼는가"로는 붕괴가 안 잡힌다. 256개를 다 쓰더라도
    그중 하나가 대부분을 삼키면 그 자리는 아이템을 거의 구분하지 못한다.
    고르게 쓰면 log2(256)=8비트, 한 곳에 몰리면 0비트에 가깝다.
    """
    _, counts = np.unique(column, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> tuple:
    """유저 단위로 재표본해서 평균의 95% 신뢰구간을 낸다.

    유저 n명의 점수(맞히면 1, 아니면 0 같은 값)에서 n명을 복원추출로 다시 뽑아
    평균을 구하는 일을 2,000번 반복하고, 그 분포의 2.5%/97.5% 지점을 쓴다.

    정규분포를 가정하지 않는다. HR 은 0/1 이라 n이 작으면 정규근사가 어긋난다.
    TAIL/COLD 처럼 표본이 적은 구간에서 특히 그렇다.
    """
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[index].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOTSTRAP,
                     seed: int = 0) -> dict:
    """같은 유저에 대한 두 점수의 차이를 검정한다.

    왜 짝지어 재는가
        두 실행은 같은 유저를 평가한다. 어려운 유저는 양쪽 다 틀린다.
        유저별 차이(b-a)만 보면 그 공통 난이도가 상쇄돼서, 각각의 신뢰구간을
        따로 그려 겹치는지 보는 것보다 훨씬 작은 차이도 잡아낸다.

    p 값은 양측이다. 재표본한 차이 평균이 0을 넘어가는 비율로 계산한다.
    """
    if a.size == 0:
        return {"delta": 0.0, "ci": (0.0, 0.0), "p_value": 1.0, "n": 0}
    diff = b - a
    rng = np.random.default_rng(seed)
    index = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[index].mean(axis=1)
    observed = float(diff.mean())
    # 재표본 분포가 0의 반대편에 얼마나 몰려 있는가.
    # 아래로 1/n_boot: 재표본이 한 번도 0을 안 넘었다고 p=0 이라 할 수는 없다.
    # 위로 1.0: 차이가 정확히 0이면 양쪽 비율이 모두 1이라 2배가 1을 넘어 버린다.
    tail = min((means <= 0).mean(), (means >= 0).mean())
    p_value = min(1.0, max(2 * tail, 1.0 / n_boot))
    return {
        "delta": observed,
        "ci": (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
        "p_value": float(p_value),
        "n": int(diff.size),
    }


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
    exposure = np.zeros(n_items, dtype=np.int64)   # 아이템별 추천 횟수 (상위 max_k)
    n_generated = 0
    n_hallucinated = 0
    used_codewords = [set() for _ in range(n_digits)]

    # 유저 단위 점수를 그대로 들고 있는다. 신뢰구간과 두 실행 비교에 필요하다.
    user_ids: list[int] = []
    user_segments: list[str] = []
    user_hits = {k: [] for k in ks}
    user_ndcg = {k: [] for k in ks}

    skipped = 0
    for user_id, target in targets.items():
        beams = predictions.get(user_id)
        if beams is None:
            skipped += 1
            continue

        segment = segments.get(target, "UNSEEN")
        user_ids.append(int(user_id))
        user_segments.append(segment)

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
            hit, gain = 0.0, 0.0
            for rank, item in enumerate(ranked_items[:k]):
                if item == target:
                    hit, gain = 1.0, dcg_at_rank(rank)
                    break
            user_hits[k].append(hit)
            user_ndcg[k].append(gain)

    segment_array = np.array(user_segments)
    scores = {
        k: {"hit": np.array(user_hits[k]), "ndcg": np.array(user_ndcg[k])} for k in ks
    }
    total_exposure = int(exposure.sum())

    # ── 구간별 표 ──
    rows = []
    for name in SEGMENT_ORDER + ["ALL"]:
        if name == "ALL":
            mask = np.ones(len(segment_array), dtype=bool)
            seg_items = list(range(n_items))
        else:
            mask = segment_array == name
            seg_items = [i for i, s in segments.items() if s == name]

        n = int(mask.sum())
        seg_exposure = exposure[seg_items] if seg_items else np.zeros(0, dtype=np.int64)
        seg_exposure_sum = int(seg_exposure.sum())
        item_share = len(seg_items) / n_items if n_items else 0.0
        exposure_share = seg_exposure_sum / total_exposure if total_exposure else 0.0

        row = {
            "segment": name,
            "n_users": n,
            "n_items": len(seg_items),
            "exposure": seg_exposure_sum,
            "coverage": float((seg_exposure > 0).mean()) if len(seg_items) else 0.0,
            "item_share": item_share,
            "exposure_share": exposure_share,
            # 배율 1.0 = 아이템 수에 걸맞은 만큼 노출됐다는 뜻.
            # 2.0 이면 제 몫의 두 배, 0.1 이면 10분의 1만 보여졌다.
            "exposure_lift": exposure_share / item_share if item_share else 0.0,
            "exposure_per_item": seg_exposure_sum / len(seg_items) if seg_items else 0.0,
        }
        for k in ks:
            for metric, values in (("HR", scores[k]["hit"]), ("NDCG", scores[k]["ndcg"])):
                selected = values[mask]
                row[f"{metric}@{k}"] = float(selected.mean()) if n else 0.0
                row[f"{metric}@{k}_ci"] = bootstrap_ci(selected) if n else (0.0, 0.0)
        rows.append(row)

    # ── 진단 ──
    catalog_codewords = [len(set(sid_map[:, d].tolist())) for d in range(n_digits)]
    unique_prefix = len({tuple(sid_map[i, :-1].tolist()) for i in range(n_items)})

    # 층별 정보량. 코드워드를 몇 개 썼는지로는 붕괴가 안 잡힌다.
    # 256개를 다 쓰더라도 그중 하나가 대부분을 삼키면 그 층은 아무것도 구분하지 못한다.
    # 실제로 실행의 20% 정도에서 일어난다 — docs/09_층수_충돌률_결과.md 참고.
    level_bits = [level_entropy(sid_map[:, d]) for d in range(n_digits - 1)]
    collapsed = [d for d, b in enumerate(level_bits) if b < COLLAPSE_BITS]

    by_segment = {r["segment"]: r for r in rows}
    head_per_item = by_segment.get("HEAD", {}).get("exposure_per_item", 0.0)
    tail_per_item = by_segment.get("TAIL", {}).get("exposure_per_item", 0.0)

    diagnostics = {
        "hallucination_rate": n_hallucinated / n_generated if n_generated else 0.0,
        "n_generated": n_generated,
        "n_hallucinated": n_hallucinated,
        "gini_exposure": gini(exposure),
        # Ghost 논문이 43:1 → 1.6:1 로 보고한 것과 같은 값이다.
        # 아이템 하나당 노출로 재야 아이템 수 차이에 속지 않는다.
        "head_tail_exposure_ratio": head_per_item / tail_per_item if tail_per_item else None,
        "codebook_used_generated": [len(s) for s in used_codewords],
        "codebook_used_catalog": catalog_codewords,
        "level_bits": [round(b, 2) for b in level_bits],
        "effective_bits": round(sum(level_bits), 2),
        "collapsed_levels": collapsed,
        "sid_collision_rate": 1.0 - unique_prefix / n_items,
        "users_evaluated": len(user_ids),
        "users_without_prediction": skipped,
    }
    return {
        "rows": rows,
        "diagnostics": diagnostics,
        "ks": ks,
        # --compare 가 같은 유저를 짝짓는 데 쓴다.
        "per_user": {
            "user_ids": user_ids,
            "segments": user_segments,
            "hit": {str(k): [float(v) for v in user_hits[k]] for k in ks},
            "ndcg": {str(k): [float(v) for v in user_ndcg[k]] for k in ks},
        },
    }


# ─────────────────────────────── 출력 ───────────────────────────────


def render(result: dict) -> str:
    ks = result["ks"]
    primary = max(ks)
    lines = []
    rule = "─" * 74

    lines.append("■ 정확도 — 정답 아이템이 어느 구간인지로 나눔 (그 구간 유저를 맞혔나)")
    lines.append(f"{'구간':<7}{'n':>8}{'HR@'+str(primary):>10}"
                 f"{'95% 신뢰구간':>22}{'NDCG@'+str(primary):>12}")
    lines.append(rule)
    for row in result["rows"]:
        if row["segment"] == "ALL":
            lines.append(rule)
        low, high = row[f"HR@{primary}_ci"]
        lines.append(
            f"{row['segment']:<7}{row['n_users']:>8,}{row[f'HR@{primary}']:>10.4f}"
            f"{f'[{low:.4f}, {high:.4f}]':>22}{row[f'NDCG@{primary}']:>12.4f}"
        )

    lines.append("")
    lines.append("■ 노출 — 추천된 아이템이 어느 구간인지로 나눔 (그 구간을 얼마나 보여줬나)")
    lines.append(f"{'구간':<7}{'아이템수':>9}{'노출수':>10}{'노출점유':>10}"
                 f"{'배율':>8}{'개당노출':>10}{'Coverage':>10}")
    lines.append(rule)
    for row in result["rows"]:
        if row["segment"] == "ALL":
            lines.append(rule)
        lines.append(
            f"{row['segment']:<7}{row['n_items']:>9,}{row['exposure']:>10,}"
            f"{row['exposure_share']:>10.1%}{row['exposure_lift']:>8.2f}"
            f"{row['exposure_per_item']:>10.2f}{row['coverage']:>10.1%}"
        )
    lines.append("배율 1.00 = 아이템 수에 걸맞은 만큼 노출됨. 2.00 = 제 몫의 두 배.")

    d = result["diagnostics"]
    lines.append("")
    lines.append("■ 진단")
    lines.append(f"Gini(노출 쏠림):   {d['gini_exposure']:.3f}   (0=균등, 1=독점)")
    if d.get("head_tail_exposure_ratio"):
        lines.append(
            f"HEAD:TAIL 노출비:  {d['head_tail_exposure_ratio']:.1f} : 1"
            f"   (아이템 하나당. Ghost 논문은 43:1 → 1.6:1 로 보고)"
        )
    lines.append(
        f"환각률:            {d['hallucination_rate']:.2%}"
        f"   ({d['n_hallucinated']:,}/{d['n_generated']:,} — 카탈로그에 없는 SID)"
    )
    lines.append(f"SID 충돌률:        {d['sid_collision_rate']:.2%}   (중복 제거 자릿수 빼고 봤을 때)")
    lines.append(f"코드북 활용(생성): {d['codebook_used_generated']}")
    lines.append(f"코드북 활용(카탈로그): {d['codebook_used_catalog']}")
    lines.append(
        f"층별 정보량(비트): {d['level_bits']}   합계 {d['effective_bits']}"
        f"   (자리마다 8.0 이 최대)"
    )
    if d["collapsed_levels"]:
        lines.append("")
        lines.append(
            f"🚨 코드북이 무너졌습니다 — {len(d['collapsed_levels'])}개 자리가 "
            f"{COLLAPSE_BITS}비트 미만입니다 (자리 번호 {d['collapsed_levels']})."
        )
        lines.append(
            "   그 자리는 SID 를 거의 구분하지 못합니다. 이 실행의 수치는 신뢰할 수 없습니다."
        )
        lines.append(
            "   코드북 학습(Step 3)을 다시 돌리세요. 실행의 20% 정도에서 일어납니다 "
            "— docs/09_층수_충돌률_결과.md"
        )
    if d["users_without_prediction"]:
        lines.append(f"⚠️ 추천 결과가 없는 유저 {d['users_without_prediction']:,}명은 제외했습니다")
    return "\n".join(lines)


def to_markdown(result: dict, title: str = "구간별 평가") -> str:
    """노션·보고서에 그대로 붙일 수 있는 표."""
    ks = result["ks"]
    lines = [f"### {title}", "", "| 구간 | n | " +
             " | ".join(f"HR@{k} | NDCG@{k}" for k in ks) +
             " | 노출 배율 | Coverage |"]
    lines.append("|---|---:|" + "---:|---:|" * len(ks) + "---:|---:|")
    for row in result["rows"]:
        cells = [row["segment"], f"{row['n_users']:,}"]
        for k in ks:
            low, high = row[f"HR@{k}_ci"]
            cells.append(f"{row[f'HR@{k}']:.4f}<br><sub>[{low:.4f}, {high:.4f}]</sub>")
            cells.append(f"{row[f'NDCG@{k}']:.4f}")
        cells.append(f"{row['exposure_lift']:.2f}")
        cells.append(f"{row['coverage']:.1%}")
        lines.append("| " + " | ".join(cells) + " |")

    d = result["diagnostics"]
    lines += ["", f"Gini {d['gini_exposure']:.3f} · 환각률 {d['hallucination_rate']:.2%} "
                  f"· SID 충돌률 {d['sid_collision_rate']:.2%}"]
    if d.get("head_tail_exposure_ratio"):
        lines.append(f"HEAD:TAIL 노출비 {d['head_tail_exposure_ratio']:.1f} : 1 (아이템 하나당)")
    return "\n".join(lines)


# ─────────────────────────────── 두 실행 비교 ───────────────────────────────


def compare(base: dict, other: dict, ks: list[int] | None = None) -> dict:
    """두 평가 결과를 같은 유저끼리 짝지어 비교한다.

    구간은 base 쪽 기준을 쓴다. 두 실행의 학습 데이터가 다르면(예: COLD 분할)
    구간 정의도 달라질 수 있는데, 그때 기준이 흔들리면 비교가 무의미해진다.
    """
    a_users = base["per_user"]["user_ids"]
    b_users = other["per_user"]["user_ids"]
    b_index = {u: i for i, u in enumerate(b_users)}
    common = [(i, b_index[u]) for i, u in enumerate(a_users) if u in b_index]
    if not common:
        raise ValueError("두 결과에 공통 유저가 없습니다. 같은 테스트 셋인지 확인하세요.")

    a_pos = np.array([i for i, _ in common])
    b_pos = np.array([j for _, j in common])
    segment_array = np.array(base["per_user"]["segments"])[a_pos]

    ks = ks or sorted(set(base["ks"]) & set(other["ks"]))
    rows = []
    for name in SEGMENT_ORDER + ["ALL"]:
        mask = np.ones(len(a_pos), dtype=bool) if name == "ALL" else segment_array == name
        if not mask.any():
            continue
        row = {"segment": name, "n_users": int(mask.sum())}
        for k in ks:
            for metric in ("hit", "ndcg"):
                a_values = np.array(base["per_user"][metric][str(k)])[a_pos][mask]
                b_values = np.array(other["per_user"][metric][str(k)])[b_pos][mask]
                label = ("HR" if metric == "hit" else "NDCG") + f"@{k}"
                row[label] = {
                    "base": float(a_values.mean()),
                    "other": float(b_values.mean()),
                    **paired_bootstrap(a_values, b_values),
                }
        rows.append(row)

    return {
        "ks": ks,
        "n_common_users": len(common),
        "n_base_only": len(a_users) - len(common),
        "n_other_only": len(b_users) - len(common),
        "rows": rows,
    }


def render_compare(result: dict, base_name: str, other_name: str) -> str:
    primary = max(result["ks"])
    lines = [
        f"■ {base_name}  →  {other_name}   (공통 유저 {result['n_common_users']:,}명 짝지어 비교)",
        "",
        f"{'구간':<7}{'n':>8}{'기준':>10}{'비교':>10}{'차이':>11}"
        f"{'95% 신뢰구간':>22}{'p':>9}",
        "─" * 78,
    ]
    for row in result["rows"]:
        if row["segment"] == "ALL":
            lines.append("─" * 78)
        cell = row[f"HR@{primary}"]
        low, high = cell["ci"]
        # 신뢰구간이 0을 안 걸치면 방향이 확실하다는 뜻이다.
        mark = "  " if low <= 0 <= high else (" ↑" if cell["delta"] > 0 else " ↓")
        lines.append(
            f"{row['segment']:<7}{row['n_users']:>8,}{cell['base']:>10.4f}"
            f"{cell['other']:>10.4f}{cell['delta']:>+11.4f}"
            f"{f'[{low:+.4f}, {high:+.4f}]':>22}{cell['p_value']:>8.3f}{mark}"
        )
    lines += [
        f"(HR@{primary} 기준. ↑↓ 는 95% 신뢰구간이 0을 걸치지 않는다는 표시)",
        "",
        "⚠️ 구간이 5개라 검정도 5번이다. 어느 하나가 p<0.05 인 것만으로는 약하다.",
        "   미리 정한 구간(우리 연구에서는 TAIL·COLD)에서 나온 결과인지 밝혀야 한다.",
    ]
    if result["n_base_only"] or result["n_other_only"]:
        lines.append(
            f"⚠️ 한쪽에만 있는 유저는 제외했습니다 "
            f"(기준 전용 {result['n_base_only']:,}명 / 비교 전용 {result['n_other_only']:,}명)"
        )
    return "\n".join(lines)


# ─────────────────────────────── 실행 ───────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs=2, metavar=("기준.json", "비교.json"),
                        help="이미 저장한 평가 결과 두 개를 짝지어 비교한다")
    parser.add_argument("--sid-map", help="3단계 출력 merged_predictions_tensor.pt")
    parser.add_argument("--predictions", help="추론 출력 merged_predictions.pkl")
    parser.add_argument("--segments", help="item_segments.py 의 출력 .json")
    parser.add_argument("--data-dir", help="정답을 읽을 데이터 폴더")
    parser.add_argument("--split", default="testing", help="정답 split (기본 testing)")
    parser.add_argument("--ks", default="5,10", help="쉼표로 구분한 k 목록")
    parser.add_argument("--out", default=None, help="결과를 저장할 .json (선택)")
    parser.add_argument("--markdown", default=None, help="표를 저장할 .md (선택)")
    args = parser.parse_args()

    if args.compare:
        base_path, other_path = args.compare
        with open(base_path) as f:
            base = json.load(f)
        with open(other_path) as f:
            other = json.load(f)
        for name, data in ((base_path, base), (other_path, other)):
            if "per_user" not in data:
                raise SystemExit(
                    f"❌ {name} 에 per_user 가 없습니다. 예전 형식입니다.\n"
                    f"   segment_eval.py 로 다시 평가해서 저장해 주세요."
                )
        result = compare(base, other)
        print(render_compare(result, os.path.basename(base_path), os.path.basename(other_path)))
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n저장 완료: {args.out}")
        return

    missing = [n for n in ("sid_map", "predictions", "segments", "data_dir")
               if not getattr(args, n)]
    if missing:
        parser.error("다음 인자가 필요합니다: " + ", ".join("--" + n.replace("_", "-")
                                                       for n in missing))

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
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w") as f:
            f.write(to_markdown(result))
        print(f"저장 완료: {args.markdown}")


if __name__ == "__main__":
    main()
